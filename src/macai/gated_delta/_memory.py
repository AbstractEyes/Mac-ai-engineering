"""Where the gated delta rule's buffers come from (docs/series_memory.md).

PyTorch's Apple-GPU allocator (2.15) cuts buffers of 10-512 MiB out of shared 1 GiB heaps (best fit, splitting the free range
it picks) and gives a heap back to the system only when every buffer in it is free. In training every batch has a new sequence
length, so every buffer whose size follows the length takes a new size: a freed buffer seldom fits the next request exactly,
the remainders pile up, and buffers that live from the forward to the backward keep whole heaps from being released.

Size classes (op.SIZE_CLASSES): a buffer of n bytes is cut as the first n bytes of a buffer of the smallest class holding n,
the classes being 2^(j/4) bytes rounded up to a 16 KiB page (four classes a doubling: about 9% over on average; at most 19%
from 10 MiB up, the sizes the 1 GiB heaps hold; the page rounding adds more below 2 MiB). The tensor has the requested shape,
contiguous at storage offset 0; only its storage is larger (resize_ to a smaller size keeps the storage, and the tensor is not
a view), so every kernel reads and writes exactly the elements it did, and the buffers of nearby lengths share one size the
allocator can hand back as it is.
"""
import math

import torch

CLASSES_A_DOUBLING = 4
PAGE = 16384        # bytes; every class is a whole number of pages


def class_bytes(nbytes: int) -> int:
    """The smallest size class holding nbytes."""
    if nbytes <= 0:
        return 0
    # start just below the class holding nbytes (the floating-point logarithm may round either way), then step up
    j = max(0, math.floor(CLASSES_A_DOUBLING * math.log2(nbytes)) - 1)
    while True:
        c = -(-math.ceil(2.0 ** (j / CLASSES_A_DOUBLING)) // PAGE) * PAGE
        if c >= nbytes:
            return c
        j += 1


def empty(shape, dtype: torch.dtype, device, classes: bool) -> torch.Tensor:
    """torch.empty(shape), contiguous; with classes its storage is the size class of its bytes."""
    if not classes:
        return torch.empty(shape, dtype=dtype, device=device)
    n = math.prod(shape) * dtype.itemsize
    return torch.empty(class_bytes(n) // dtype.itemsize, dtype=dtype, device=device).resize_(shape)


def contiguous_as(x: torch.Tensor, dtype: torch.dtype, classes: bool) -> torch.Tensor:
    """x converted to dtype, contiguous, at a storage offset that is a multiple of 4 elements (the kernels read pairs of
    floats): x itself when it already is so; else a copy, size-classed with classes. No autograd (inside a backward)."""
    if x.dtype == dtype and x.is_contiguous() and x.storage_offset() % 4 == 0:
        return x
    if classes:
        y = empty(x.shape, dtype, x.device, True)
        y.copy_(x)
        return y
    y = x.to(dtype).contiguous()
    return y if y.storage_offset() % 4 == 0 else y.clone()


class ClassedCast(torch.autograd.Function):
    """x.to(dtype).contiguous() into a size-classed buffer; the gradient converted back to x's dtype the same way. The values
    are those of x.to(dtype) and of its backward (both conversions are copy_ either way)."""

    @staticmethod
    def forward(ctx, x, dtype):
        ctx.dtype = x.dtype
        y = empty(x.shape, dtype, x.device, True)
        y.copy_(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        if dy.dtype == ctx.dtype:
            return dy, None
        dx = empty(dy.shape, ctx.dtype, dy.device, True)
        dx.copy_(dy)
        return dx, None


def cast(x: torch.Tensor, dtype: torch.dtype, classes: bool) -> torch.Tensor:
    """x as a contiguous tensor of dtype inside autograd: x itself when it already is one; else x.to(dtype).contiguous(), its
    buffer (and its gradient's) size-classed with classes."""
    if x.dtype == dtype and x.is_contiguous():
        return x
    if classes:
        return ClassedCast.apply(x, dtype)
    return x.to(dtype).contiguous()
