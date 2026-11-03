"""The MLP's SwiGLU, silu(gate) * up, as one Metal pass each way with transformers' rounding points (silu rounded to the input
dtype before the product, the product rounded):

    h = (silu(e).to(dtype) * u).to(dtype)
    du = dh silu(e)                     de = dh u sigmoid(e) (1 + e (1 - sigmoid(e)))      (float32, one rounding each)

The forward reads e and u and writes h; the backward reads dh, e and u and writes de and du (silu recomputed). Elementwise,
8 elements a thread in two 4-element vector loads.
"""
import torch

from . import _lib

_SOURCE = r"""
template <typename T>
inline void swiglu_fwd_impl(const device T* e, const device T* u, device T* h, uint n8, uint i) {
  if (i >= n8) return;
  for (uint k = 0; k < 2; ++k) {
    const ulong o = (ulong)i * 8 + k * 4;
    const float4 ef = ld4(e + o);
    const float4 s = rnd4<T>(ef / (1.0f + exp(-ef)));
    st4(h + o, s * ld4(u + o));
  }
}

template <typename T>
inline void swiglu_bwd_impl(const device T* e, const device T* u, const device T* dh, device T* de, device T* du, uint n8,
                            uint i) {
  if (i >= n8) return;
  for (uint k = 0; k < 2; ++k) {
    const ulong o = (ulong)i * 8 + k * 4;
    const float4 ef = ld4(e + o);
    const float4 sg = sigmoid4(ef);
    const float4 g = ld4(dh + o);
    st4(du + o, g * rnd4<T>(ef / (1.0f + exp(-ef))));      // the forward's silu, as rounded there
    st4(de + o, g * ld4(u + o) * sg * (1.0f + ef * (1.0f - sg)));
  }
}

#define SWIGLU(T, TW, SUF)                                                                                             \
kernel void swiglu_fwd_##SUF(device const T* e [[buffer(0)]], device const T* u [[buffer(1)]], device T* h [[buffer(2)]], \
                             constant uint& n8 [[buffer(3)]], uint i [[thread_position_in_grid]]) {                     \
  swiglu_fwd_impl<T>(e, u, h, n8, i);                                                                                  \
}                                                                                                                      \
kernel void swiglu_bwd_##SUF(device const T* e [[buffer(0)]], device const T* u [[buffer(1)]], device const T* dh [[buffer(2)]], \
                             device T* de [[buffer(3)]], device T* du [[buffer(4)]], constant uint& n8 [[buffer(5)]],    \
                             uint i [[thread_position_in_grid]]) {                                                      \
  swiglu_bwd_impl<T>(e, u, dh, de, du, n8, i);                                                                         \
}

__INSTANCES__
"""

GROUP = 256


def library():
    return _lib.library(_SOURCE.replace("__INSTANCES__", _lib.instances("SWIGLU", False)), ())


def supported(e: torch.Tensor, u: torch.Tensor) -> bool:
    return _lib.supported(e, u) and e.dtype == u.dtype and e.shape == u.shape and e.numel() % 8 == 0 and e.numel() < 2 ** 31


class SwiGLU(torch.autograd.Function):
    """h = silu(e) * u with transformers' rounding (silu rounded to the input dtype first)."""

    @staticmethod
    def forward(ctx, e, u):
        ec, uc = _lib.aligned(e), _lib.aligned(u)
        h = torch.empty_like(ec)
        n8 = ec.numel() // 8
        if n8:
            _lib.kernel(library(), "swiglu_fwd", e.dtype)(ec, uc, h, n8, threads=(n8, 1, 1), group_size=(GROUP, 1, 1))
        ctx.save_for_backward(ec, uc)
        return h

    @staticmethod
    def backward(ctx, dh):
        ec, uc = ctx.saved_tensors
        g = _lib.aligned(dh.to(ec.dtype))
        de, du = torch.empty_like(ec), torch.empty_like(uc)
        n8 = ec.numel() // 8
        if n8:
            _lib.kernel(library(), "swiglu_bwd", ec.dtype)(ec, uc, g, de, du, n8, threads=(n8, 1, 1), group_size=(GROUP, 1, 1))
        return de, du


def swiglu(e: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return SwiGLU.apply(e, u)
