"""Shared helpers of the layer kernels: compiling a Metal source with its constants, the dtype instantiations, the row layout
of a tensor."""
import functools

import torch

SUFFIX = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}
METAL = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}

# vector loads and stores of 4 elements, conversions through float (Metal's float -> bfloat / half conversion rounds to the
# nearest even, as PyTorch's .to() does)
HEADER = r"""
#include <metal_stdlib>
using namespace metal;

template <typename T> inline float4 ld4(const device T* p) { return float4(*((const device vec<T, 4>*)p)); }
template <typename T> inline void st4(device T* p, float4 v) { *((device vec<T, 4>*)p) = vec<T, 4>(v); }
template <typename T> inline float4 rnd4(float4 v) { return float4(vec<T, 4>(v)); }
inline float4 sigmoid4(float4 x) { return 1.0f / (1.0f + exp(-x)); }
"""


def instances(macro: str, pairs: bool) -> str:
    """The macro invoked for every input dtype (and, with pairs, every weight dtype): MACRO(T, TW, SUFFIX)."""
    out = []
    for dt, t in METAL.items():
        if pairs:
            for wdt, tw in METAL.items():
                out.append(f"{macro}({t}, {tw}, {SUFFIX[dt]}_{SUFFIX[wdt]})")
        else:
            out.append(f"{macro}({t}, {t}, {SUFFIX[dt]})")
    return "\n".join(out)


@functools.lru_cache(maxsize=None)
def library(source: str, consts: tuple):
    """The source with each __NAME__ replaced by its value, compiled once per distinct set of constants."""
    for k, v in consts:
        source = source.replace(f"__{k}__", str(v))
    return torch.mps.compile_shader(HEADER + source)


def kernel(lib, name: str, *dtypes) -> object:
    return getattr(lib, name + "_" + "_".join(SUFFIX[d] for d in dtypes))


def supported(*xs) -> bool:
    """Every tensor on the Apple GPU in a dtype the kernels take."""
    return all(x.device.type == "mps" and x.dtype in SUFFIX for x in xs)


def as_rows(x: torch.Tensor, d: int):
    """x as [N, d] rows the kernels read in place: element stride 1, row stride and storage offset multiples of 4 elements
    (a view when x's layout allows, else a contiguous copy). Returns (rows, row stride)."""
    try:
        r = x.view(-1, d)
    except RuntimeError:
        r = x.reshape(-1, d)
    if r.shape[0] > 0 and (r.stride(1) != 1 or r.stride(0) % 4 or r.storage_offset() % 4):
        r = r.contiguous()
    return r, (r.stride(0) if r.shape[0] > 1 else d)


def aligned(x: torch.Tensor) -> torch.Tensor:
    """x contiguous at a storage offset that is a multiple of 4 elements (the kernels' vector loads)."""
    x = x.contiguous()
    return x if x.storage_offset() % 4 == 0 else x.clone()
