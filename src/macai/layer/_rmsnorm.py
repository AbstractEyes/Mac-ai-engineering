"""Qwen3.5's zero-centered RMSNorm (transformers' Qwen3_5RMSNorm) as one Metal pass each way:

    y = x * rsqrt(mean(x^2) + eps) * (1 + w)          (float32 arithmetic, one rounding to x's dtype at the end)

and its backward, with r = rsqrt(mean(x^2) + eps), x_hat = x r, w' = 1 + w:

    dx = r (w' dy - x_hat <x_hat, w' dy> / D) = r (w' dy - x r^2 <x, w' dy> / D)
    dw = sum over rows of dy x_hat                      (only when the weight needs a gradient)

The forward saves x itself (no float32 copy) and the backward recomputes r. Rows of any width D (a multiple of 4) up to 8,192,
read in place with any row stride (the full-attention q / k norms read every other 256-element half of the query
projection's rows). A row is spread over S simdgroups (S = 1 for D <= 512), R rows a threadgroup. The weight's gradient, when
one is needed, is summed as float32 partials (one partial row a row-group of the threadgroups, which then stride over the rows)
and a second pass over the partials: no float atomics.
"""
import torch

from . import _lib

_SOURCE = r"""
#define D __D__
#define S __S__
#define R __R__
#define CH ((D + 128 * S - 1) / (128 * S))
#define FULL ((D % (128 * S)) == 0)
#define IDX(j) ((((j) * S + ss) * 32 + lane) * 4)

// the sum of v over the S simdgroups of row rr of the threadgroup; every thread of the threadgroup calls it
inline float2 row_sum2(float2 v, threadgroup float2* buf, uint rr, uint ss, uint lane) {
  v = simd_sum(v);
#if S > 1
  if (lane == 0) buf[rr * S + ss] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float2 t = buf[rr * S];
  for (uint i = 1; i < S; ++i) t += buf[rr * S + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return t;
#else
  return v;
#endif
}

template <typename T, typename TW>
inline void zrms_fwd_impl(const device T* x, const device TW* w, device T* y, uint N, uint sx, float eps,
                          threadgroup float2* buf, uint tg, uint sg, uint lane) {
  const uint rr = sg / S, ss = sg % S;
  const uint row = tg * R + rr;
  const device T* xr = x + (ulong)min(row, N - 1) * sx;
  float4 v[CH];
  float acc = 0.0f;
  for (uint j = 0; j < CH; ++j) {
    const uint i = IDX(j);
    v[j] = (FULL || i < D) ? ld4(xr + i) : float4(0.0f);
    acc += dot(v[j], v[j]);
  }
  const float tot = row_sum2(float2(acc, 0.0f), buf, rr, ss, lane).x;
  if (row >= N) return;
  const float r = precise::rsqrt(tot / float(D) + eps);
  device T* yr = y + (ulong)row * D;
  for (uint j = 0; j < CH; ++j) {
    const uint i = IDX(j);
    if (FULL || i < D) st4(yr + i, (v[j] * r) * (1.0f + ld4(w + i)));
  }
}

template <typename T, typename TW>
inline void zrms_bwd_impl(const device T* x, const device TW* w, const device T* dy, device T* dx, device float* dwp,
                          uint N, uint sx, uint sdy, float eps, uint want_dw, uint ngroups,
                          threadgroup float2* buf, uint tg, uint sg, uint lane) {
  const uint rr = sg / S, ss = sg % S;
  float4 wv[CH], dwa[CH];
  for (uint j = 0; j < CH; ++j) {
    const uint i = IDX(j);
    wv[j] = (FULL || i < D) ? 1.0f + ld4(w + i) : float4(0.0f);
    dwa[j] = float4(0.0f);
  }
  // with want_dw the threadgroups stride over the rows (ngroups of them), each row-group summing its own partial row
  for (uint base = tg * R; base < N; base += ngroups * R) {
    const uint row = base + rr;
    const uint rrow = min(row, N - 1);
    const device T* xr = x + (ulong)rrow * sx;
    const device T* gr = dy + (ulong)rrow * sdy;
    float4 xv[CH], gv[CH];
    float2 acc = float2(0.0f);
    for (uint j = 0; j < CH; ++j) {
      const uint i = IDX(j);
      const bool in = FULL || i < D;
      xv[j] = in ? ld4(xr + i) : float4(0.0f);
      gv[j] = in ? ld4(gr + i) : float4(0.0f);
      acc.x += dot(xv[j], xv[j]);
      acc.y += dot(xv[j], wv[j] * gv[j]);
    }
    const float2 tot = row_sum2(acc, buf, rr, ss, lane);
    if (row < N) {
      const float r = precise::rsqrt(tot.x / float(D) + eps);
      const float k = r * r * tot.y / float(D);
      device T* dr = dx + (ulong)row * D;
      for (uint j = 0; j < CH; ++j) {
        const uint i = IDX(j);
        if (FULL || i < D) st4(dr + i, r * (wv[j] * gv[j] - xv[j] * k));
        if (want_dw) dwa[j] += gv[j] * (xv[j] * r);
      }
    }
  }
  if (want_dw) {
    device float* pr = dwp + (ulong)(tg * R + rr) * D;
    for (uint j = 0; j < CH; ++j) {
      const uint i = IDX(j);
      if (FULL || i < D) *((device float4*)(pr + i)) = dwa[j];
    }
  }
}

#define ZRMS(T, TW, SUF)                                                                                               \
kernel void zrms_fwd_##SUF(device const T* x [[buffer(0)]], device const TW* w [[buffer(1)]], device T* y [[buffer(2)]], \
                           constant uint& N [[buffer(3)]], constant uint& sx [[buffer(4)]],                              \
                           constant float& eps [[buffer(5)]], uint tg [[threadgroup_position_in_grid]],                  \
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {        \
  threadgroup float2 buf[R * S];                                                                                       \
  zrms_fwd_impl<T, TW>(x, w, y, N, sx, eps, buf, tg, sg, lane);                                                        \
}                                                                                                                      \
kernel void zrms_bwd_##SUF(device const T* x [[buffer(0)]], device const TW* w [[buffer(1)]],                           \
                           device const T* dy [[buffer(2)]], device T* dx [[buffer(3)]], device float* dwp [[buffer(4)]], \
                           constant uint& N [[buffer(5)]], constant uint& sx [[buffer(6)]], constant uint& sdy [[buffer(7)]], \
                           constant float& eps [[buffer(8)]], constant uint& want_dw [[buffer(9)]],                      \
                           constant uint& ngroups [[buffer(10)]], uint tg [[threadgroup_position_in_grid]],              \
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {        \
  threadgroup float2 buf[R * S];                                                                                       \
  zrms_bwd_impl<T, TW>(x, w, dy, dx, dwp, N, sx, sdy, eps, want_dw, ngroups, buf, tg, sg, lane);                       \
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

MAX_D = 8192
DW_GROUPS = 256          # threadgroups (at most) when the weight's gradient is summed


def config(d: int):
    """(S simdgroups a row, R rows a threadgroup) for rows of width d."""
    if d <= 512:
        return 1, 8
    if d <= 1024:
        return 2, 4
    if d <= 2048:
        return 4, 2
    return 8, 1


def library(d: int):
    s, r = config(d)
    return _lib.library(_SOURCE.replace("__INSTANCES__", _lib.instances("ZRMS", True)), (("D", d), ("S", s), ("R", r)))


def supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    d = x.shape[-1] if x.dim() else 0
    return (_lib.supported(x, w) and w.dim() == 1 and w.shape[0] == d and d % 4 == 0 and 0 < d <= MAX_D
            and x.numel() < 2 ** 31)


class ZeroCenteredRMSNorm(torch.autograd.Function):
    """y = rmsnorm(x) * (1 + w) over the last dimension (transformers' Qwen3_5RMSNorm), one Metal pass each way."""

    @staticmethod
    def forward(ctx, x, w, eps: float):
        d = x.shape[-1]
        rows, sx = _lib.as_rows(x, d)
        n = rows.shape[0]
        wc = _lib.aligned(w)
        y = torch.empty((n, d), dtype=x.dtype, device=x.device)
        if n:
            s, r = config(d)
            groups = -(-n // r)
            _lib.kernel(library(d), "zrms_fwd", x.dtype, w.dtype)(
                rows, wc, y, n, sx, float(eps), threads=(groups * r * s * 32, 1, 1), group_size=(r * s * 32, 1, 1))
        ctx.save_for_backward(rows, wc)
        ctx.eps, ctx.shape, ctx.wdtype = float(eps), x.shape, w.dtype
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        rows, wc = ctx.saved_tensors
        n, d = rows.shape
        want_dw = ctx.needs_input_grad[1]
        if n == 0 or not (ctx.needs_input_grad[0] or want_dw):
            return None, (torch.zeros_like(wc) if want_dw else None), None
        sx = rows.stride(0) if n > 1 else d
        g, sdy = _lib.as_rows(dy.to(rows.dtype), d)
        dx = torch.empty((n, d), dtype=rows.dtype, device=rows.device)
        s, r = config(d)
        groups = -(-n // r)
        if want_dw:
            groups = min(groups, DW_GROUPS)
            dwp = torch.empty((groups * r, d), dtype=torch.float32, device=rows.device)
        else:
            dwp = torch.empty(4, dtype=torch.float32, device=rows.device)
        lib = library(d)
        _lib.kernel(lib, "zrms_bwd", rows.dtype, wc.dtype)(
            rows, wc, g, dx, dwp, n, sx, sdy, ctx.eps, int(want_dw), groups,
            threads=(groups * r * s * 32, 1, 1), group_size=(r * s * 32, 1, 1))
        dw = None
        if want_dw:
            out = torch.empty(d, dtype=torch.float32, device=rows.device)
            lib.col_sum(dwp, out, groups * r, threads=(32 * d, 1, 1), group_size=(256, 1, 1))
            dw = out.to(ctx.wdtype)
        return (dx.view(ctx.shape) if ctx.needs_input_grad[0] else None), dw, None


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return ZeroCenteredRMSNorm.apply(x, w, eps)
