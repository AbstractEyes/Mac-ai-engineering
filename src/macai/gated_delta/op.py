"""The gated delta rule on the Mac's GPU: an autograd function over the Metal kernels, and a drop-in for transformers'
torch_chunk_gated_delta_rule (same arguments, same returns)."""
import math

import torch

from ._metal import BLK, library


KERNEL = "auto"       # forward: "auto" = one threadgroup a head where the shape allows (DK % 8, DV % 4, DV <= 128); "column" = 0.1.0's
KERNEL_BWD = "column"  # backward: "column" (32 lanes a column, float atomics; the faster on an M3 Max) or "head"


def use_head_kernel(dk: int, dv: int) -> bool:
    return KERNEL == "auto" and dk % 8 == 0 and dv % 4 == 0 and dv <= 128 and 8 * dv >= dk + 2


class GatedDeltaRule(torch.autograd.Function):
    """o, S_T = recurrence(q, k, v, g, beta, S_0) on [B, T, H, D] float32 tensors (q already normalized and scaled)."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, s0, save: bool):
        B, T, H, DK = q.shape
        DV = v.shape[-1]
        if DV % 32 != 0:
            raise ValueError(f"the value head width must be a multiple of 32, got {DV}")
        if save and DK > 128:
            # the backward sums 32 columns x DK rows in threadgroup memory: DK 256 needs 33,024 bytes of the 32,768 allowed
            raise NotImplementedError(f"the backward supports key head widths up to 128 (got {DK}); the forward runs any multiple of 32")
        lib = library(DK)
        o = torch.empty(B, T, H, DV, device=q.device, dtype=torch.float32)
        sT = torch.empty(B, H, DK, DV, device=q.device, dtype=torch.float32)
        nb = math.ceil(T / BLK)
        ckpt = torch.empty(B, H, nb, DK, DV, device=q.device, dtype=torch.float32) if save else torch.empty(1, device=q.device)
        head = use_head_kernel(DK, DV)
        if head:
            lib.gdr_fwd8(q, k, v, g, beta, s0, o, sT, ckpt, T, H, DV, int(save),
                         threads=(8 * DV, H, B), group_size=(8 * DV, 1, 1))
        else:
            lib.gdr_fwd(q, k, v, g, beta, s0, o, sT, ckpt, T, H, DV, int(save),
                        threads=(DV // 32 * 1024, H, B), group_size=(1024, 1, 1))
        if save:
            ctx.save_for_backward(q, k, v, g, beta, ckpt)
            ctx.head = head and KERNEL_BWD == "head"   # both forwards save the same state layout, so either backward reads it
        return o, sT

    @staticmethod
    def backward(ctx, do, dsT):
        q, k, v, g, beta, ckpt = ctx.saved_tensors
        B, T, H, DK = q.shape
        DV = v.shape[-1]
        lib = library(DK)
        do = torch.zeros_like(v) if do is None else do.to(torch.float32).contiguous()
        dsT = (torch.zeros(B, H, DK, DV, device=q.device, dtype=torch.float32) if dsT is None
               else dsT.to(torch.float32).contiguous())
        make = torch.empty_like if ctx.head else torch.zeros_like     # the head kernel writes every element once
        dq, dk, dg, dbeta = make(q), make(k), make(g), make(beta)
        dv = torch.empty_like(v)
        ds0 = torch.empty(B, H, DK, DV, device=q.device, dtype=torch.float32)
        if ctx.head:
            lib.gdr_bwd8(q, k, v, g, beta, ckpt, do, dsT, dq, dk, dv, dg, dbeta, ds0, T, H, DV,
                         threads=(8 * DV, H, B), group_size=(8 * DV, 1, 1))
        else:
            lib.gdr_bwd(q, k, v, g, beta, ckpt, do, dsT, dq, dk, dv, dg, dbeta, ds0, T, H, DV,
                        threads=(DV // 32 * 1024, H, B), group_size=(1024, 1, 1))
        return dq, dk, dv, dg, dbeta, ds0, None


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size: int = 64, initial_state=None, output_final_state: bool = False,
                           use_qk_l2norm_in_kernel: bool = False, **kwargs):
    """Drop-in for transformers' torch_chunk_gated_delta_rule: query, key [B, T, H, DK]; value [B, T, H, DV]; g (log decay, <= 0)
    and beta [B, T, H]; initial_state [B, H, DK, DV] or None. Returns the output [B, T, H, DV] in the query's dtype and the final
    state (float32) when output_final_state, else None. chunk_size is accepted and unused (the kernel steps one token at a time)."""
    if kwargs.get("cu_seqlens") is not None:
        # packed rows (several sequences in one row, their boundaries in cu_seqlens): the state must restart at each
        # boundary; not implemented yet, and running without it would silently carry state across sequences
        raise NotImplementedError("macai's gated delta rule does not take packed rows (cu_seqlens) yet")
    dtype = query.dtype
    q, k, v = query.float(), key.float(), value.float()
    if use_qk_l2norm_in_kernel:
        q, k = l2norm(q), l2norm(k)
    q = q * q.shape[-1] ** -0.5
    B, T, H, DK = q.shape
    DV = v.shape[-1]
    s0 = (torch.zeros(B, H, DK, DV, device=q.device, dtype=torch.float32) if initial_state is None
          else initial_state.to(torch.float32))
    tensors = (q, k, v, g.float(), beta.float(), s0)
    save = torch.is_grad_enabled() and any(t.requires_grad for t in tensors)
    o, sT = GatedDeltaRule.apply(*(t.contiguous() for t in tensors), save)
    return o.to(dtype), (sT if output_final_state else None)


def recurrent_reference(q, k, v, g, beta, s0):
    """The recurrence step by step in plain PyTorch (float32, [B, T, H, D] inputs, q already normalized and scaled): the parity
    reference for the kernels."""
    B, T, H, DK = q.shape
    s = s0.clone()
    outs = []
    for t in range(T):
        a = g[:, t].exp()[..., None, None]
        st = s * a
        u = (st * k[:, t, :, :, None]).sum(-2)
        d = beta[:, t, :, None] * (v[:, t] - u)
        s = st + k[:, t, :, :, None] * d[:, :, None, :]
        outs.append((s * q[:, t, :, :, None]).sum(-2))
    return torch.stack(outs, 1), s
