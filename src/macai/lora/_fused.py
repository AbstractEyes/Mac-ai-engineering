"""The LoRA adapter path as one autograd function (series design D5): what PEFT's vanilla LoRA adds to a linear layer's
output, forward and backward, with the passes around the rank-r products folded into two Metal kernels.

With x [..., d_in], the base output R [..., d_out], A [r, d_in], B [d_out, r], the scale s = alpha / r, the dropout p and its
mask M (drawn by torch's own dropout call), all as PEFT computes them under autocast (x cast to the adapters' dtype for the
dropout, the products in the compute dtype):

    forward    x_d = M x / (1 - p)        torch.native_dropout, the call nn.Dropout makes: the same mask from the same
                                          generator state as PEFT's draw
               P   = x_d A^T              PyTorch's product, [N, r]
               Y   = R + s P B^T          one Metal kernel: R read and Y written once, the product of rank r in float32
    backward   G   = s dY B               PyTorch's product, the scale applied to the [N, r] result
               dB  = s dY^T P             PyTorch's product, the scale applied to the [d_out, r] result
               dA  = G^T x_d              PyTorch's product
               dx  = M (G A) / (1 - p)    one Metal kernel, the mask and the scale in its epilogue; autograd adds it to the
                                          base's input gradient as before
               dR  = dY

The compute dtype is autocast's when autocast is on for the device, else the adapters' own (PEFT's two cases). The dropout runs
on x in its own dtype when that is the compute dtype and the adapters are float32 (the dropped bf16 / fp16 values are then bit
for bit PEFT's dropped float32 copy after autocast's cast); otherwise on x cast to the adapters' dtype, as PEFT runs it.
"""
import torch

from . import _kernels

_SHORTCUT = (torch.bfloat16, torch.float16)


def dropout(x: torch.Tensor, p: float):
    """torch's own dropout call on the Apple GPU (what nn.Dropout runs in training): (x dropped and scaled, the bool mask)."""
    return torch.native_dropout(x, p, True)


class AdapterPath(torch.autograd.Function):
    """Y = R + s * dropout(x) A^T B^T (see the module's docstring); returns a new tensor, R itself is not modified."""

    @staticmethod
    def forward(ctx, x, result, a, b, p: float, s: float, cdt):
        d_in, d_out = x.shape[-1], result.shape[-1]
        n = result.numel() // d_out
        adt = a.dtype
        with torch.autocast(x.device.type, enabled=False):
            mask = None
            if p > 0:
                if x.dtype == cdt and cdt in _SHORTCUT and adt == torch.float32:
                    xd, mask = dropout(x, p)
                else:
                    xd, mask = dropout(x if x.dtype == adt else x.to(adt), p)
                    xd = xd.to(cdt)
            elif x.dtype == cdt and (adt == cdt or (adt == torch.float32 and cdt in _SHORTCUT)):
                xd = x                                    # PEFT's float32 copy and autocast's cast back give x's own bits
            else:
                xd = (x if x.dtype == adt else x.to(adt)).to(cdt)
            xd2 = xd.reshape(n, d_in)
            ac, bc = a.to(cdt), b.to(cdt)
            pm = torch.mm(xd2, ac.t())
            y = _kernels.up(pm, bc, result.reshape(n, d_out), s)
        need_x, need_a, need_b = ctx.needs_input_grad[0], ctx.needs_input_grad[2], ctx.needs_input_grad[3]
        ctx.save_for_backward(xd2 if need_a else None, mask if need_x else None, pm if need_b else None,
                              ac if need_x else None, bc if (need_x or need_a) else None)
        ctx.n, ctx.d_out, ctx.s, ctx.cdt = n, d_out, float(s), cdt
        ctx.dscale = float(1.0 / (1.0 - p)) if p > 0 else 1.0
        ctx.x_shape, ctx.x_dtype, ctx.a_dtype, ctx.b_dtype = x.shape, x.dtype, a.dtype, b.dtype
        return y.view(result.shape)

    @staticmethod
    def backward(ctx, dy):
        xd2, mask, pm, ac, bc = ctx.saved_tensors
        need_x, need_r, need_a, need_b = (ctx.needs_input_grad[i] for i in range(4))
        dx = da = db = None
        with torch.autocast(dy.device.type, enabled=False):
            dy2 = dy.reshape(ctx.n, ctx.d_out)
            if dy2.dtype != ctx.cdt:
                dy2 = dy2.to(ctx.cdt)
            g = None
            if need_x or need_a:
                g = torch.mm(dy2, bc)
                if ctx.s != 1.0:
                    g.mul_(ctx.s)
            if need_a:
                da = torch.mm(g.t(), xd2).to(ctx.a_dtype)
            if need_b:
                db = torch.mm(dy2.t(), pm)
                if ctx.s != 1.0:
                    db.mul_(ctx.s)
                db = db.to(ctx.b_dtype)
            if need_x:
                dx = _kernels.down(g, ac, mask, ctx.dscale, ctx.x_dtype).view(ctx.x_shape)
        return dx, (dy if need_r else None), da, db, None, None, None


def adapter_path(x, result, a, b, p: float, s: float, cdt) -> torch.Tensor:
    """result + s * dropout(x, p) a^T b^T, forward and backward as AdapterPath."""
    return AdapterPath.apply(x, result, a, b, p, s, cdt)
