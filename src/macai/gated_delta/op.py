"""The gated delta rule on the Mac's GPU: an autograd function over the Metal kernels, and a drop-in for transformers'
torch_chunk_gated_delta_rule (same arguments, same returns)."""
import math

import torch

from . import _chunked
from ._metal import BLK, library


# forward: "chunked" = the chunked kernels (_chunked.py); "auto" = 0.2.0's one threadgroup a head where the shape allows
# (DK % 8, DV % 4, DV <= 128), else 0.1.0's; "column" = 0.1.0's
KERNEL = "auto"
# backward: "chunked" = the chunked kernels; "column" = 0.1.0's (32 lanes a column, float atomics); "head" = 0.2.0's
# (one threadgroup a head, used only after the "auto" forward). Any forward pairs with any backward: each forward saves
# what the chosen backward reads (see GatedDeltaRule.forward).
KERNEL_BWD = "column"


def head_shape(dk: int, dv: int) -> bool:
    """Shapes the one-threadgroup-a-head kernels run (8 lanes a value column, DV / 4 simdgroups)."""
    return dk % 8 == 0 and dv % 4 == 0 and dv <= 128 and 8 * dv >= dk + 2


def use_head_kernel(dk: int, dv: int) -> bool:
    return KERNEL == "auto" and head_shape(dk, dv)


def _even(t: torch.Tensor) -> torch.Tensor:
    """The kernels read pairs of floats: a tensor whose storage starts at an odd float is copied to an aligned one."""
    return t if t.storage_offset() % 4 == 0 else t.clone()


def _prep(lib, q, k, g, beta):
    """Per chunk: the cumulative log decay G, the inverse A, the decayed Q K^T (M) and K K^T (KG)."""
    B, T, H, _ = q.shape
    nt = math.ceil(T / _chunked.C)
    c = _chunked.C
    gc = torch.empty(B, H, nt, c, device=q.device, dtype=torch.float32)
    a, m, kg = (torch.empty(B, H, nt, c, c, device=q.device, dtype=torch.float32) for _ in range(3))
    lib.gdc_prep(q, k, g, beta, gc, a, m, kg, T, H, threads=(nt * 128, H, B), group_size=(128, 1, 1))
    return gc, a, m, kg


def _state_grid(dv: int, H: int, Z: int):
    per = 8 * _chunked.NCB * _chunked.NSG
    if dv % per != 0:
        raise ValueError(f"the chunked kernels need the value head width to be a multiple of {per}, got {dv}")
    return dict(threads=(dv // per * 32 * _chunked.NSG, H, Z), group_size=(32 * _chunked.NSG, 1, 1))


class GatedDeltaRule(torch.autograd.Function):
    """o, S_T = recurrence(q, k, v, g, beta, S_0) on [B, T, H, D] float32 tensors (q already normalized and scaled)."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, s0, save: bool):
        q, k, v, g, beta, s0 = (_even(t) for t in (q, k, v, g, beta, s0))
        B, T, H, DK = q.shape
        DV = v.shape[-1]
        if DV % 32 != 0:
            raise ValueError(f"the value head width must be a multiple of 32, got {DV}")
        chunked_bwd = save and KERNEL_BWD == "chunked"
        if save and DK > 128 and not chunked_bwd:
            # the token-at-a-time backwards sum 32 columns x DK rows in threadgroup memory: DK 256 needs 33,024 bytes of the
            # 32,768 allowed; the chunked backward has no such limit
            raise NotImplementedError(f"the token-at-a-time backward supports key head widths up to 128 (got {DK}); "
                                      "the forward runs any multiple of 32, the chunked backward any multiple of 32")
        dev = q.device
        o = torch.empty(B, T, H, DV, device=dev, dtype=torch.float32)
        sT = torch.empty(B, H, DK, DV, device=dev, dtype=torch.float32)
        none = torch.empty(4, device=dev, dtype=torch.float32)
        nb = math.ceil(T / BLK)
        ctx.dims = (B, T, H, DK, DV)
        if KERNEL == "chunked":
            lib = _chunked.library(DK)
            nt = math.ceil(T / _chunked.C)
            gc, a, m, kg = _prep(lib, q, k, g, beta)
            sc = vn = xu = ckpt = none
            flags = 0
            if chunked_bwd:
                sc = torch.empty(B, H, nt, DK, DV, device=dev, dtype=torch.float32)
                vn = torch.empty(B, H, nt * _chunked.C, DV, device=dev, dtype=torch.float32)
                xu = torch.empty_like(vn)
                flags = 1
            elif save:
                ckpt = torch.empty(B, H, nb, DK, DV, device=dev, dtype=torch.float32)
                flags = 2
            lib.gdc_fwd(q, k, v, beta, s0, gc, a, m, o, sT, sc, vn, xu, ckpt, T, H, DV, flags, 1, nt, **_state_grid(DV, H, B))
            if chunked_bwd:
                ctx.mode = "chunked"
                ctx.save_for_backward(q, k, beta, gc, a, kg, m, sc, vn, xu)
            elif save:
                ctx.mode = "head" if KERNEL_BWD == "head" and head_shape(DK, DV) else "column"
                ctx.save_for_backward(q, k, v, g, beta, ckpt)
            return o, sT
        lib = library(DK)
        ckpt = torch.empty(B, H, nb, DK, DV, device=dev, dtype=torch.float32) if save else none
        head = use_head_kernel(DK, DV)
        if head:
            lib.gdr_fwd8(q, k, v, g, beta, s0, o, sT, ckpt, T, H, DV, int(save),
                         threads=(8 * DV, H, B), group_size=(8 * DV, 1, 1))
        else:
            lib.gdr_fwd(q, k, v, g, beta, s0, o, sT, ckpt, T, H, DV, int(save),
                        threads=(DV // 32 * 1024, H, B), group_size=(1024, 1, 1))
        if save:
            ctx.save_for_backward(q, k, v, g, beta, ckpt)
            # both token-at-a-time forwards save the same state layout (every BLK tokens), so any backward reads it
            ctx.mode = "chunked_from_checkpoints" if chunked_bwd else ("head" if head and KERNEL_BWD == "head" else "column")
        return o, sT

    @staticmethod
    def backward(ctx, do, dsT):
        B, T, H, DK, DV = ctx.dims
        dev = ctx.saved_tensors[0].device
        do = torch.zeros(B, T, H, DV, device=dev, dtype=torch.float32) if do is None else _even(do.to(torch.float32).contiguous())
        dsT = (torch.zeros(B, H, DK, DV, device=dev, dtype=torch.float32) if dsT is None
               else _even(dsT.to(torch.float32).contiguous()))
        if ctx.mode.startswith("chunked"):
            lib = _chunked.library(DK)
            nt = math.ceil(T / _chunked.C)
            if ctx.mode == "chunked":
                q, k, beta, gc, a, kg, m, sc, vn, xu = ctx.saved_tensors
                sstride, nstates = 1, nt
            else:
                # after a token-at-a-time forward: its checkpoints every BLK tokens hold the chunk states (every C / BLK-th);
                # the per-chunk matrices and Vn, Xu are recomputed, every chunk in parallel
                q, k, v, g, beta, ckpt = ctx.saved_tensors
                gc, a, m, kg = _prep(lib, q, k, g, beta)
                vn = torch.empty(B, H, nt * _chunked.C, DV, device=dev, dtype=torch.float32)
                xu = torch.empty_like(vn)
                none = torch.empty(4, device=dev, dtype=torch.float32)
                sc, sstride, nstates = ckpt, _chunked.C // BLK, ckpt.shape[2]
                lib.gdc_fwd(q, k, v, beta, none, gc, a, m, none, none, sc, vn, xu, none, T, H, DV, 4, sstride, nstates,
                            **_state_grid(DV, H, B * nt))
            dsn = torch.empty(B, H, nt, DK, DV, device=dev, dtype=torch.float32)
            dvn = torch.empty(B, H, nt * _chunked.C, DV, device=dev, dtype=torch.float32)
            ds0 = torch.empty(B, H, DK, DV, device=dev, dtype=torch.float32)
            lib.gdc_bwd_state(q, k, beta, do, dsT, gc, a, m, dsn, dvn, ds0, T, H, DV, **_state_grid(DV, H, B))
            dq, dk = torch.empty_like(q), torch.empty_like(k)
            dv = torch.empty(B, T, H, DV, device=dev, dtype=torch.float32)
            dg, dbeta = torch.empty_like(beta), torch.empty_like(beta)
            lib.gdc_bwd_chunk(q, k, beta, do, gc, a, kg, m, sc, dsn, vn, xu, dvn, dq, dk, dv, dg, dbeta, T, H, DV, sstride, nstates,
                              threads=(nt * 128, H, B), group_size=(128, 1, 1))
            return dq, dk, dv, dg, dbeta, ds0, None
        q, k, v, g, beta, ckpt = ctx.saved_tensors
        lib = library(DK)
        head = ctx.mode == "head"
        make = torch.empty_like if head else torch.zeros_like     # the head kernel writes every element once
        dq, dk, dg, dbeta = make(q), make(k), make(g), make(beta)
        dv = torch.empty_like(v)
        ds0 = torch.empty(B, H, DK, DV, device=dev, dtype=torch.float32)
        if head:
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
    state (float32) when output_final_state, else None. chunk_size is accepted and unused (the chunked kernels use their own
    chunk length, _chunked.C; the others step one token at a time)."""
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
