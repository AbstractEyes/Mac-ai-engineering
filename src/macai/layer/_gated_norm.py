"""Qwen3.5's gated RMSNorm (transformers' Qwen3_5RMSNormGated, rows of one value head) as one Metal pass each way, with
transformers' rounding points:

    x_hat = x * rsqrt(mean(x^2) + eps)        (float32)
    h     = w * x_hat.to(x.dtype)              (rounded to x's dtype when w has x's dtype, as PyTorch's type promotion does)
    o     = (h * silu(z.float())).to(x.dtype)

and its backward in float32 (one rounding of each gradient), with s = silu(z), sig = sigmoid(z):

    dz = do h sig (1 + z (1 - sig))
    dx = r (dxh - x_hat <x_hat, dxh> / D),  dxh = do s w
    dw = sum over rows of do s x_hat.to(x.dtype)   (only when the weight needs a gradient: float32 partials, a second pass)

One simdgroup a row (4 elements a lane at D = 128: a sum is one simd_sum, no threadgroup memory, no barriers), R rows a
threadgroup. The forward saves the inputs; the backward recomputes r.
"""
import torch

from . import _lib

_SOURCE = r"""
#define D __D__
#define R __R__
#define CH ((D + 127) / 128)
#define FULL ((D % 128) == 0)
#define IDX(j) (((j) * 32 + lane) * 4)

template <typename T, typename TW>
inline void gnorm_fwd_impl(const device T* x, const device T* z, const device TW* w, device T* o, uint N, float eps,
                           uint round_h, uint tg, uint sg, uint lane) {
  const uint row = tg * R + sg;
  if (row >= N) return;
  const device T* xr = x + (ulong)row * D;
  const device T* zr = z + (ulong)row * D;
  float4 xv[CH];
  float acc = 0.0f;
  for (uint j = 0; j < CH; ++j) {
    const uint i = IDX(j);
    xv[j] = (FULL || i < D) ? ld4(xr + i) : float4(0.0f);
    acc += dot(xv[j], xv[j]);
  }
  const float r = precise::rsqrt(simd_sum(acc) / float(D) + eps);
  device T* orow = o + (ulong)row * D;
  for (uint j = 0; j < CH; ++j) {
    const uint i = IDX(j);
    if (FULL || i < D) {
      float4 h = ld4(w + i) * rnd4<T>(xv[j] * r);
      if (round_h) h = rnd4<T>(h);
      const float4 zf = ld4(zr + i);
      st4(orow + i, h * (zf / (1.0f + exp(-zf))));
    }
  }
}

template <typename T, typename TW>
inline void gnorm_bwd_impl(const device T* x, const device T* z, const device TW* w, const device T* dout, device T* dx,
                           device T* dz, device float* dwp, uint N, float eps, uint round_h, uint want_dw, uint ngroups,
                           uint tg, uint sg, uint lane) {
  float4 dwa[CH];
  for (uint j = 0; j < CH; ++j) dwa[j] = float4(0.0f);
  for (uint row = tg * R + sg; row < N; row += ngroups * R) {
    const device T* xr = x + (ulong)row * D;
    const device T* zr = z + (ulong)row * D;
    const device T* gr = dout + (ulong)row * D;
    float4 xv[CH], dxh[CH];
    float acc = 0.0f;
    for (uint j = 0; j < CH; ++j) {
      const uint i = IDX(j);
      xv[j] = (FULL || i < D) ? ld4(xr + i) : float4(0.0f);
      acc += dot(xv[j], xv[j]);
    }
    const float r = precise::rsqrt(simd_sum(acc) / float(D) + eps);
    float acc2 = 0.0f;
    for (uint j = 0; j < CH; ++j) {
      const uint i = IDX(j);
      dxh[j] = float4(0.0f);
      if (FULL || i < D) {
        const float4 wv = ld4(w + i);
        const float4 xr4 = rnd4<T>(xv[j] * r);
        float4 h = wv * xr4;
        if (round_h) h = rnd4<T>(h);
        const float4 zf = ld4(zr + i);
        const float4 sg4 = sigmoid4(zf);
        const float4 g = ld4(gr + i);
        st4(dz + (ulong)row * D + i, g * h * sg4 * (1.0f + zf * (1.0f - sg4)));
        const float4 gs = g * (zf * sg4);
        dxh[j] = gs * wv;
        acc2 += dot(xv[j] * r, dxh[j]);
        if (want_dw) dwa[j] += gs * xr4;
      }
    }
    const float c = simd_sum(acc2) / float(D);
    for (uint j = 0; j < CH; ++j) {
      const uint i = IDX(j);
      if (FULL || i < D) st4(dx + (ulong)row * D + i, r * (dxh[j] - (xv[j] * r) * c));
    }
  }
  if (want_dw) {
    device float* pr = dwp + (ulong)(tg * R + sg) * D;
    for (uint j = 0; j < CH; ++j) {
      const uint i = IDX(j);
      if (FULL || i < D) *((device float4*)(pr + i)) = dwa[j];
    }
  }
}

#define GNORM(T, TW, SUF)                                                                                              \
kernel void gnorm_fwd_##SUF(device const T* x [[buffer(0)]], device const T* z [[buffer(1)]],                          \
                            device const TW* w [[buffer(2)]], device T* o [[buffer(3)]], constant uint& N [[buffer(4)]], \
                            constant float& eps [[buffer(5)]], constant uint& round_h [[buffer(6)]],                     \
                            uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],        \
                            uint lane [[thread_index_in_simdgroup]]) {                                                  \
  gnorm_fwd_impl<T, TW>(x, z, w, o, N, eps, round_h, tg, sg, lane);                                                    \
}                                                                                                                      \
kernel void gnorm_bwd_##SUF(device const T* x [[buffer(0)]], device const T* z [[buffer(1)]],                          \
                            device const TW* w [[buffer(2)]], device const T* dout [[buffer(3)]], device T* dx [[buffer(4)]], \
                            device T* dz [[buffer(5)]], device float* dwp [[buffer(6)]], constant uint& N [[buffer(7)]], \
                            constant float& eps [[buffer(8)]], constant uint& round_h [[buffer(9)]],                     \
                            constant uint& want_dw [[buffer(10)]], constant uint& ngroups [[buffer(11)]],                \
                            uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],        \
                            uint lane [[thread_index_in_simdgroup]]) {                                                  \
  gnorm_bwd_impl<T, TW>(x, z, w, dout, dx, dz, dwp, N, eps, round_h, want_dw, ngroups, tg, sg, lane);                  \
}

__INSTANCES__

// the column sums of the partials [P, D] (float32): a simdgroup a column, each lane every 32nd row, then a tree
kernel void col_sum(device const float* p [[buffer(0)]], device float* out [[buffer(1)]], constant uint& P [[buffer(2)]],
                    uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint c = gid / 32;
  if (c >= D) return;
  float s = 0.0f;
  for (uint i = lane; i < P; i += 32) s += p[(ulong)i * D + c];
  s = simd_sum(s);
  if (lane == 0) out[c] = s;
}
"""

R = 8                 # rows (simdgroups) a threadgroup
MAX_D = 1024
DW_GROUPS = 256


def library(d: int):
    return _lib.library(_SOURCE.replace("__INSTANCES__", _lib.instances("GNORM", True)), (("D", d), ("R", R)))


def supported(x: torch.Tensor, z: torch.Tensor, w: torch.Tensor) -> bool:
    d = x.shape[-1] if x.dim() else 0
    return (_lib.supported(x, z, w) and x.dtype == z.dtype and x.shape == z.shape and w.dim() == 1 and w.shape[0] == d
            and d % 4 == 0 and 0 < d <= MAX_D and (w.dtype == x.dtype or w.dtype == torch.float32) and x.numel() < 2 ** 31)


class GatedRMSNorm(torch.autograd.Function):
    """o = (w * rmsnorm(x).to(x.dtype)) * silu(z.float()), cast to x's dtype (transformers' Qwen3_5RMSNormGated)."""

    @staticmethod
    def forward(ctx, x, z, w, eps: float):
        d = x.shape[-1]
        xs, zs, wc = _lib.aligned(x), _lib.aligned(z), _lib.aligned(w)
        n = xs.numel() // d
        o = torch.empty_like(xs)
        round_h = int(w.dtype == x.dtype)
        if n:
            groups = -(-n // R)
            _lib.kernel(library(d), "gnorm_fwd", x.dtype, w.dtype)(
                xs, zs, wc, o, n, float(eps), round_h, threads=(groups * R * 32, 1, 1), group_size=(R * 32, 1, 1))
        ctx.save_for_backward(xs, zs, wc)
        ctx.eps, ctx.round_h, ctx.wdtype = float(eps), round_h, w.dtype
        return o

    @staticmethod
    def backward(ctx, do):
        xs, zs, wc = ctx.saved_tensors
        d = xs.shape[-1]
        n = xs.numel() // d
        want_dw = ctx.needs_input_grad[2]
        g = _lib.aligned(do.to(xs.dtype))
        dx, dz = torch.empty_like(xs), torch.empty_like(zs)
        groups = -(-n // R)
        if want_dw:
            groups = min(groups, DW_GROUPS)
            dwp = torch.empty((groups * R, d), dtype=torch.float32, device=xs.device)
        else:
            dwp = torch.empty(4, dtype=torch.float32, device=xs.device)
        lib = library(d)
        if n:
            _lib.kernel(lib, "gnorm_bwd", xs.dtype, wc.dtype)(
                xs, zs, wc, g, dx, dz, dwp, n, ctx.eps, ctx.round_h, int(want_dw), groups,
                threads=(groups * R * 32, 1, 1), group_size=(R * 32, 1, 1))
        dw = None
        if want_dw:
            out = torch.zeros(d, dtype=torch.float32, device=xs.device)
            if n:
                lib.col_sum(dwp, out, groups * R, threads=(32 * d, 1, 1), group_size=(256, 1, 1))
            dw = out.to(ctx.wdtype)
        return dx, dz, dw, None


def gated_rms_norm(x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return GatedRMSNorm.apply(x, z, w, eps)
