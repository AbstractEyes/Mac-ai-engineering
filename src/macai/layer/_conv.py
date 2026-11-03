"""The linear-attention layer's convolution path as one Metal pass each way: the depthwise causal convolution (kernel K = 4,
no bias) and its SiLU over the input projection's [B, T, C] output read in place, writing the rule's inputs directly:

    u[t, c] = sum_j w[c, j] x[t - (K - 1) + j, c]       (zero before the sequence start; float32 accumulation)
    y = u sigmoid(u)                                      (on the float32 u, one rounding)
    q, k = y's first two key_dim channel blocks as [B, T, H_v, D_k] in x's dtype, each key head written once for each of the
           H_v / H_k value heads it serves (transformers' repeat_interleave of q and k at 9B, without the copy);
    v    = the last value_dim channels as [B, T, H_v, D_v] in float32 (the rule computes in float32, so its drop-in reads it as
           it is: no cast and no copy)

transformers instead transposes the projection's output to [B, C, T], convolves (PyTorch's depthwise kernel first copies the
view into channel-first order), rounds u, applies SiLU as its own pass, transposes back, splits and repeats.

The backward recomputes u in registers from the saved x (nothing else is kept) and gathers dx with one writer an element:

    du = dy sigmoid(u) (1 + u (1 - sigmoid(u)))          (dy of q and k summed over the value heads a key head serves)
    dx[t, c] = sum_j w[c, j] du[t + (K - 1) - j, c]       (du = 0 past the sequence end)
    dw[c, j] = sum_{b, t} du[t, c] x[t - (K - 1) + j, c]  (only when the weight needs a gradient: float32 partials a tile,
                                                           then a second pass)

dx is written as [B, T, C] in x's dtype: the projection's gradient arrives in its own layout (transformers' path hands it the
transpose of a [B, C, T] tensor). Each thread owns 4 adjacent channels and walks L time steps, keeping the last K - 1 inputs
(and, in the backward, the last K - 1 du) in registers: no threadgroup memory, no barriers.
"""
import torch

from . import _lib

K = 4

_SOURCE = r"""
#define L __L__
#define KW 4

struct Taps { float4 w0, w1, w2, w3; };     // tap j of the thread's 4 channels

template <typename TW>
inline Taps load_taps(const device TW* w, uint c) {
  Taps t;
  float4 r[4];
  for (uint i = 0; i < 4; ++i) r[i] = ld4(w + (ulong)(c + i) * KW);   // w[c + i, 0..3]
  t.w0 = float4(r[0].x, r[1].x, r[2].x, r[3].x);
  t.w1 = float4(r[0].y, r[1].y, r[2].y, r[3].y);
  t.w2 = float4(r[0].z, r[1].z, r[2].z, r[3].z);
  t.w3 = float4(r[0].w, r[1].w, r[2].w, r[3].w);
  return t;
}

// gid.x: 4 channels; gid.y: a tile of L time steps; gid.z: the batch row. q, k: [B, T, KD * REP] (H_v heads of DKH);
// v: [B, T, C - 2 KD] float
template <typename T, typename TW>
inline void conv_fwd_impl(const device T* x, const device TW* w, device T* qo, device T* ko, device float* vo,
                          uint Tn, uint C, uint KD, uint DKH, uint REP, uint3 gid) {
  const uint c = gid.x * 4;
  if (c >= C) return;
  const uint t0 = gid.y * L, b = gid.z;
  const Taps tp = load_taps(w, c);
  const device T* xb = x + (ulong)b * Tn * C + c;
  float4 xm3 = (t0 >= 3) ? ld4(xb + (ulong)(t0 - 3) * C) : float4(0.0f);
  float4 xm2 = (t0 >= 2) ? ld4(xb + (ulong)(t0 - 2) * C) : float4(0.0f);
  float4 xm1 = (t0 >= 1) ? ld4(xb + (ulong)(t0 - 1) * C) : float4(0.0f);
  const uint te = min(t0 + L, Tn);
  // where this thread's channels go: q or k (each key head written REP times), or v
  const bool isv = c >= 2 * KD;
  const uint cc = isv ? c - 2 * KD : (c >= KD ? c - KD : c);
  device T* qk = (c >= KD) ? ko : qo;
  const uint h = cc / DKH, d = cc % DKH;
  const ulong qrow = (ulong)KD * REP, vrow = C - 2 * KD;
  const ulong qcol = (ulong)h * REP * DKH + d;
  for (uint t = t0; t < te; ++t) {
    const float4 xt = ld4(xb + (ulong)t * C);
    const float4 u = fma(tp.w3, xt, fma(tp.w2, xm1, fma(tp.w1, xm2, tp.w0 * xm3)));
    const float4 y = u / (1.0f + exp(-u));
    const ulong bt = (ulong)b * Tn + t;
    if (isv) {
      *((device float4*)(vo + bt * vrow + cc)) = y;
    } else {
      for (uint r = 0; r < REP; ++r) st4(qk + bt * qrow + qcol + (ulong)r * DKH, y);
    }
    xm3 = xm2; xm2 = xm1; xm1 = xt;
  }
}

template <typename T, typename TW>
inline void conv_bwd_impl(const device T* x, const device TW* w, const device T* dq, const device T* dk,
                          const device float* dv, device T* dx, device float* dwp, uint Tn, uint C, uint KD, uint DKH,
                          uint REP, uint want_dw, uint ntiles, uint3 gid) {
  const uint c = gid.x * 4;
  if (c >= C) return;
  const uint t0 = gid.y * L, b = gid.z;
  const Taps tp = load_taps(w, c);
  const device T* xb = x + (ulong)b * Tn * C + c;
  device T* dxb = dx + (ulong)b * Tn * C + c;
  const bool isv = c >= 2 * KD;
  const uint cc = isv ? c - 2 * KD : (c >= KD ? c - KD : c);
  const device T* dqk = (c >= KD) ? dk : dq;
  const uint h = cc / DKH, d = cc % DKH;
  const ulong qrow = (ulong)KD * REP, vrow = C - 2 * KD;
  const ulong qcol = (ulong)h * REP * DKH + d;
  float4 xm3 = (t0 >= 3) ? ld4(xb + (ulong)(t0 - 3) * C) : float4(0.0f);
  float4 xm2 = (t0 >= 2) ? ld4(xb + (ulong)(t0 - 2) * C) : float4(0.0f);
  float4 xm1 = (t0 >= 1) ? ld4(xb + (ulong)(t0 - 1) * C) : float4(0.0f);
  float4 dm1 = float4(0.0f), dm2 = float4(0.0f), dm3 = float4(0.0f);   // du[t - 1], du[t - 2], du[t - 3]
  float4 a0 = float4(0.0f), a1 = float4(0.0f), a2 = float4(0.0f), a3 = float4(0.0f);
  const uint tl = min(t0 + L, Tn);          // this tile's outputs: [t0, tl)
  // walk t from t0 to tl + 2: du[t] needs x[t - 3 .. t]; dx[t - 3] is complete once du[t] is known
  for (uint t = t0; t < tl + 3; ++t) {
    float4 du = float4(0.0f), xt = float4(0.0f);
    if (t < Tn) {
      xt = ld4(xb + (ulong)t * C);
      const float4 u = fma(tp.w3, xt, fma(tp.w2, xm1, fma(tp.w1, xm2, tp.w0 * xm3)));
      const float4 s = 1.0f / (1.0f + exp(-u));
      const ulong bt = (ulong)b * Tn + t;
      float4 g;
      if (isv) {
        g = *((const device float4*)(dv + bt * vrow + cc));
      } else {
        g = float4(0.0f);
        for (uint r = 0; r < REP; ++r) g += ld4(dqk + bt * qrow + qcol + (ulong)r * DKH);
      }
      du = g * s * (1.0f + u * (1.0f - s));
      if (want_dw && t < tl) {
        a0 = fma(du, xm3, a0); a1 = fma(du, xm2, a1); a2 = fma(du, xm1, a2); a3 = fma(du, xt, a3);
      }
    }
    if (t >= t0 + 3) {
      st4(dxb + (ulong)(t - 3) * C, fma(tp.w0, du, fma(tp.w1, dm1, fma(tp.w2, dm2, tp.w3 * dm3))));
    }
    xm3 = xm2; xm2 = xm1; xm1 = xt;
    dm3 = dm2; dm2 = dm1; dm1 = du;
  }
  if (want_dw) {
    // partial [B * ntiles, C, 4]: this tile's sums for its 4 channels
    device float* p = dwp + ((ulong)(b * ntiles + gid.y) * C + c) * KW;
    for (uint i = 0; i < 4; ++i) *((device float4*)(p + i * KW)) = float4(a0[i], a1[i], a2[i], a3[i]);
  }
}

#define CONV(T, TW, SUF)                                                                                               \
kernel void conv_fwd_##SUF(device const T* x [[buffer(0)]], device const TW* w [[buffer(1)]], device T* qo [[buffer(2)]], \
                           device T* ko [[buffer(3)]], device float* vo [[buffer(4)]], constant uint& Tn [[buffer(5)]],  \
                           constant uint& C [[buffer(6)]], constant uint& KD [[buffer(7)]], constant uint& DKH [[buffer(8)]], \
                           constant uint& REP [[buffer(9)]], uint3 gid [[thread_position_in_grid]]) {                   \
  conv_fwd_impl<T, TW>(x, w, qo, ko, vo, Tn, C, KD, DKH, REP, gid);                                                    \
}                                                                                                                      \
kernel void conv_bwd_##SUF(device const T* x [[buffer(0)]], device const TW* w [[buffer(1)]],                           \
                           device const T* dq [[buffer(2)]], device const T* dk [[buffer(3)]],                          \
                           device const float* dv [[buffer(4)]], device T* dx [[buffer(5)]], device float* dwp [[buffer(6)]], \
                           constant uint& Tn [[buffer(7)]], constant uint& C [[buffer(8)]], constant uint& KD [[buffer(9)]], \
                           constant uint& DKH [[buffer(10)]], constant uint& REP [[buffer(11)]],                         \
                           constant uint& want_dw [[buffer(12)]], constant uint& ntiles [[buffer(13)]],                  \
                           uint3 gid [[thread_position_in_grid]]) {                                                     \
  conv_bwd_impl<T, TW>(x, w, dq, dk, dv, dx, dwp, Tn, C, KD, DKH, REP, want_dw, ntiles, gid);                          \
}

__INSTANCES__

// dw [C * 4] = the sum over the P partial tiles: a simdgroup an element, each lane every 32nd tile, then a tree
kernel void dw_sum(device const float* p [[buffer(0)]], device float* out [[buffer(1)]], constant uint& P [[buffer(2)]],
                   constant uint& n [[buffer(3)]], uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint i = gid / 32;
  if (i >= n) return;
  float s = 0.0f;
  for (uint j = lane; j < P; j += 32) s += p[(ulong)j * n + i];
  s = simd_sum(s);
  if (lane == 0) out[i] = s;
}
"""

TILE = 32            # time steps a thread walks (L)
GROUP = 256          # threads a threadgroup (along the channels)


def library():
    return _lib.library(_SOURCE.replace("__INSTANCES__", _lib.instances("CONV", True)), (("L", TILE),))


def supported(x: torch.Tensor, w: torch.Tensor, kd: int, dk: int, rep: int, dv: int) -> bool:
    if x.dim() != 3 or not _lib.supported(x, w):
        return False
    C = x.shape[-1]
    vd = C - 2 * kd
    return (w.numel() == C * K and w.shape[-1] == K and C % 4 == 0 and kd % dk == 0 and dk % 4 == 0 and vd > 0
            and dv > 0 and vd % dv == 0 and (kd // dk) * rep == vd // dv and x.numel() < 2 ** 31
            and x.numel() // C * kd * rep < 2 ** 31)


class ConvQKV(torch.autograd.Function):
    """(q, k, v) = the convolution path of x [B, T, C]: see the module's docstring."""

    @staticmethod
    def forward(ctx, x, weight, kd: int, dk: int, rep: int, dv: int):
        B, T, C = x.shape
        xc = _lib.aligned(x)
        w = _lib.aligned(weight.reshape(C, K))
        hv = kd // dk * rep
        q = torch.empty((B, T, hv, dk), dtype=x.dtype, device=x.device)
        k = torch.empty((B, T, hv, dk), dtype=x.dtype, device=x.device)
        v = torch.empty((B, T, hv, dv), dtype=torch.float32, device=x.device)
        if B * T:
            _lib.kernel(library(), "conv_fwd", x.dtype, w.dtype)(
                xc, w, q, k, v, T, C, kd, dk, rep,
                threads=(C // 4, -(-T // TILE), B), group_size=(min(GROUP, C // 4), 1, 1))
        ctx.save_for_backward(xc, w)
        ctx.dims, ctx.wshape, ctx.wdtype = (kd, dk, rep, dv), weight.shape, weight.dtype
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk_, dv_):
        xc, w = ctx.saved_tensors
        kd, dk, rep, dv = ctx.dims
        B, T, C = xc.shape
        hv = kd // dk * rep
        want_dw = ctx.needs_input_grad[1]
        dev = xc.device
        dq = (torch.zeros((B, T, hv, dk), dtype=xc.dtype, device=dev) if dq is None
              else _lib.aligned(dq.to(xc.dtype)))
        dk_ = (torch.zeros((B, T, hv, dk), dtype=xc.dtype, device=dev) if dk_ is None
               else _lib.aligned(dk_.to(xc.dtype)))
        dv_ = (torch.zeros((B, T, hv, dv), dtype=torch.float32, device=dev) if dv_ is None
               else _lib.aligned(dv_.to(torch.float32)))
        dx = torch.empty_like(xc)
        ntiles = -(-T // TILE)
        dwp = (torch.empty((B * ntiles, C, K), dtype=torch.float32, device=dev) if want_dw
               else torch.empty(4, dtype=torch.float32, device=dev))
        lib = library()
        if B * T:
            _lib.kernel(lib, "conv_bwd", xc.dtype, w.dtype)(
                xc, w, dq, dk_, dv_, dx, dwp, T, C, kd, dk, rep, int(want_dw), ntiles,
                threads=(C // 4, ntiles, B), group_size=(min(GROUP, C // 4), 1, 1))
        dw = None
        if want_dw:
            out = torch.zeros(C * K, dtype=torch.float32, device=dev)
            if B * T:
                lib.dw_sum(dwp, out, B * ntiles, C * K, threads=(32 * C * K, 1, 1), group_size=(256, 1, 1))
            dw = out.view(ctx.wshape).to(ctx.wdtype)
        return (dx if ctx.needs_input_grad[0] else None), dw, None, None, None, None


def conv_qkv(x: torch.Tensor, weight: torch.Tensor, kd: int, dk: int, rep: int, dv: int):
    return ConvQKV.apply(x, weight, kd, dk, rep, dv)
