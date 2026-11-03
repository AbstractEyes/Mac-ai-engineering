"""The rule's query / key prologue as one Metal pass each way, for the drop-in's use_qk_l2norm_in_kernel=True:

    y = x * rsqrt(sum(x^2) + 1e-6) * scale        (float32 out, from float32, float16 or bfloat16 in; scale = DK^-1/2 for queries)

which is transformers' (and flash-linear-attention's) l2norm in float32 followed by the query scaling, here without the separate
casts and the dozen elementwise operations (forward and backward) that PyTorch runs for it. The backward, with r the saved
rsqrt and x_hat = x r:

    dx = scale r (dy - x_hat (x_hat . dy))        (written in the input's dtype, as autograd expects)
"""
import functools

import torch

EPS = 1e-6
ROWS_A_GROUP = 8      # one simdgroup a row (a head vector), eight rows a threadgroup

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define EPS 1e-6f

#define L2N_KERNELS(T, NAME)                                                                                        \
kernel void l2n_fwd_##NAME(device const T* x [[buffer(0)]], device float* y [[buffer(1)]],                          \
                           device float* rinv [[buffer(2)]], device const float* scale [[buffer(3)]],                \
                           constant uint& D [[buffer(4)]], constant uint& N [[buffer(5)]],                           \
                           uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {           \
  const uint row = gid / 32;                                                                                          \
  if (row >= N) return;                                                                                               \
  const ulong base = (ulong)row * D;                                                                                  \
  float ss = 0.0f;                                                                                                    \
  for (uint i = lane; i < D; i += 32) {                                                                               \
    const float v = float(x[base + i]);                                                                               \
    ss = fma(v, v, ss);                                                                                               \
  }                                                                                                                   \
  ss = simd_sum(ss);                                                                                                  \
  const float r = precise::rsqrt(ss + EPS);                                                                           \
  for (uint i = lane; i < D; i += 32) y[base + i] = (float(x[base + i]) * r) * scale[0];                              \
  if (lane == 0) rinv[row] = r;                                                                                       \
}                                                                                                                     \
kernel void l2n_bwd_##NAME(device const float* dy [[buffer(0)]], device const T* x [[buffer(1)]],                    \
                           device const float* rinv [[buffer(2)]], device const float* scale [[buffer(3)]],          \
                           device T* dx [[buffer(4)]], constant uint& D [[buffer(5)]], constant uint& N [[buffer(6)]], \
                           uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {           \
  const uint row = gid / 32;                                                                                          \
  if (row >= N) return;                                                                                               \
  const ulong base = (ulong)row * D;                                                                                  \
  const float r = rinv[row];                                                                                          \
  float dot = 0.0f;                                                                                                   \
  for (uint i = lane; i < D; i += 32) dot = fma(float(x[base + i]), dy[base + i], dot);                               \
  dot = simd_sum(dot) * r;                                                                                            \
  const float sr = scale[0] * r;                                                                                      \
  for (uint i = lane; i < D; i += 32) dx[base + i] = T(sr * (dy[base + i] - (float(x[base + i]) * r) * dot));         \
}

L2N_KERNELS(float, f32)
L2N_KERNELS(half, f16)
L2N_KERNELS(bfloat, bf16)
"""

_SUFFIX = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}


@functools.lru_cache(maxsize=None)
def library():
    return torch.mps.compile_shader(_SOURCE)


def supported(x: torch.Tensor) -> bool:
    return x.device.type == "mps" and x.dtype in _SUFFIX


def _launch(rows: int) -> dict:
    groups = -(-rows // ROWS_A_GROUP)
    return dict(threads=(groups * ROWS_A_GROUP * 32, 1, 1), group_size=(ROWS_A_GROUP * 32, 1, 1))


class L2NormScale(torch.autograd.Function):
    """y = l2norm(x.float()) * scale over the last dimension, as one kernel each way."""

    @staticmethod
    def forward(ctx, x, scale: float):
        x = x.contiguous()
        D = x.shape[-1]
        rows = x.numel() // D
        y = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        r = torch.empty(rows, device=x.device, dtype=torch.float32)
        s = torch.full((1,), float(scale), device=x.device, dtype=torch.float32)
        getattr(library(), "l2n_fwd_" + _SUFFIX[x.dtype])(x, y, r, s, D, rows, **_launch(rows))
        ctx.save_for_backward(x, r, s)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, r, s = ctx.saved_tensors
        D = x.shape[-1]
        rows = r.numel()
        dx = torch.empty_like(x)
        getattr(library(), "l2n_bwd_" + _SUFFIX[x.dtype])(dy.to(torch.float32).contiguous(), x, r, s, dx, D, rows, **_launch(rows))
        return dx, None
