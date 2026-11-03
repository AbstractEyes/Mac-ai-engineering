"""Phase 1 profile of the gated delta rule on the Apple GPU: transformers' chunked reference stage by stage (forward), its
forward and backward totals and operator counts, and the baseline Metal kernels (D0) with their pipeline limits and rooflines.

    python bench/profile_gated_delta.py      # writes bench/results/profile_gated_delta_<stamp>.json
"""
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from macai.gated_delta import l2norm, op  # noqa: E402
from macai.gated_delta._metal import BLK, library  # noqa: E402
from macai.instruments import gpu_ms, pipeline, roofline  # noqa: E402

SHAPES = {"qwen3.5-0.8b": (4, 1024, 16), "qwen3.5-9b": (2, 1024, 32)}
DK = DV = 128
C = 64


def inputs(B, T, H, seed=0):
    g0 = torch.Generator().manual_seed(seed)
    q = torch.randn(B, T, H, DK, generator=g0).to("mps", torch.bfloat16)
    k = torch.randn(B, T, H, DK, generator=g0).to("mps", torch.bfloat16)
    v = torch.randn(B, T, H, DV, generator=g0).to("mps", torch.bfloat16)
    g = (-torch.rand(B, T, H, generator=g0)).to("mps")
    beta = torch.rand(B, T, H, generator=g0).to("mps")
    return q, k, v, g, beta


def reference_stages(q, k, v, g, beta):
    """transformers' torch_chunk_gated_delta_rule (5.22.1), split into its stages, each timed by Metal events (forward only)."""
    marks = {}
    ev = lambda: torch.mps.Event(enable_timing=True)  # noqa: E731
    e = [ev()]
    e[0].record()

    def mark(name):
        x = ev()
        x.record()
        e.append(x)
        marks[name] = len(e) - 1
    B, T, H, _ = k.shape
    query, key, value, b_, decay = [x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format) for x in (q, k, v, beta, g)]
    query, key = l2norm(query), l2norm(key)
    query = query * DK ** -0.5
    pad = (C - T % C) % C
    query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
    b_, decay = (F.pad(x, (0, pad)) for x in (b_, decay))
    mark("1 prologue: transpose, cast, l2norm, scale, pad")
    v_beta, k_beta = value * b_.unsqueeze(-1), key * b_.unsqueeze(-1)
    query, key, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, C, x.shape[-1]) for x in (query, key, k_beta, v_beta)]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, C)
    mask = torch.ones(C, C, dtype=torch.bool, device=q.device).triu(1)
    cum = decay.cumsum(dim=3)
    pair = (cum.unsqueeze(4) - cum.unsqueeze(3)).masked_fill(mask, float("-inf")).exp()
    mark("2 beta products, chunk cumsum, pairwise decay")
    ut = (k_beta @ key.transpose(-1, -2)) * pair
    intra = (query @ key.transpose(-1, -2)) * pair
    dkb = k_beta * cum.exp().unsqueeze(-1)
    mark("3 K K^T and Q K^T in chunks")
    nv = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
    kcd = torch.linalg.solve_triangular(ut, dkb, upper=False, unitriangular=True)
    mark("4 the two triangular solves (UT transform)")
    query = query * cum.exp().unsqueeze(-1)
    key = key * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    chunk_decay = cum[..., -1].exp()[..., None, None]
    S = torch.zeros(B, H, DK, DV, device=q.device)
    out = torch.zeros_like(nv)
    mark("5 decay rescaling")
    for i in range(nv.shape[2]):
        vn = nv[:, :, i] - kcd[:, :, i] @ S
        out[:, :, i] = query[:, :, i] @ S + intra[:, :, i] @ vn
        S = S * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ vn
    mark("6 the sequential pass over chunks")
    out = out.reshape(B, H, -1, DV)[:, :, :T].transpose(1, 2).to(q.dtype, memory_format=torch.contiguous_format)
    mark("7 epilogue")
    torch.mps.synchronize()
    names = sorted(marks, key=marks.get)
    return {n: round(e[marks[n] - 1].elapsed_time(e[marks[n]]), 3) for n in names}


def op_counts(fn):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        fn()
        torch.mps.synchronize()
    return sum(1 for evt in prof.events() if evt.name.startswith("aten::"))


def main():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    ref = getattr(M.torch_chunk_gated_delta_rule, "__wrapped__", M.torch_chunk_gated_delta_rule)
    lib = library(DK)
    out = {"chunk": C, "blk": BLK, "pipelines": {name: pipeline(getattr(lib, name)) for name in ("gdr_fwd", "gdr_bwd", "gdr_fwd8", "gdr_bwd8")},
           "shapes": {}}
    for name, (B, T, H) in SHAPES.items():
        xs = inputs(B, T, H)
        r = {"batch": B, "tokens": T, "heads": H}
        reference_stages(*xs)                                   # warm-up: the first call builds its graphs
        runs = [reference_stages(*xs) for _ in range(3)]
        r["reference_stages_fwd_ms"] = {n: round(sorted(x[n] for x in runs)[1], 3) for n in runs[0]}   # the median of 3
        def ref_fwd():
            with torch.no_grad():
                ref(*xs, use_qk_l2norm_in_kernel=True)
        def ref_fb():
            ys = [x.detach().clone().requires_grad_(True) for x in xs]
            ref(*ys, use_qk_l2norm_in_kernel=True)[0].float().sum().backward()
        r["reference_fwd"] = gpu_ms(ref_fwd)
        r["reference_fwd_bwd"] = gpu_ms(ref_fb)
        r["reference_aten_ops_fwd"] = op_counts(ref_fwd)
        r["reference_aten_ops_fwd_bwd"] = op_counts(ref_fb)
        # D0: the kernels alone on float32 inputs prepared as the drop-in prepares them
        q = l2norm(xs[0].float()).contiguous() * DK ** -0.5
        k = l2norm(xs[1].float()).contiguous()
        v, g, beta = xs[2].float().contiguous(), xs[3].contiguous(), xs[4].contiguous()
        s0 = torch.zeros(B, H, DK, DV, device="mps")
        o = torch.empty(B, T, H, DV, device="mps")
        sT = torch.empty_like(s0)
        nb = math.ceil(T / BLK)
        ckpt = torch.empty(B, H, nb, DK, DV, device="mps")
        dout = torch.randn(B, T, H, DV, device="mps")
        grads = [torch.zeros_like(q), torch.zeros_like(k), torch.empty_like(v), torch.zeros_like(g), torch.zeros_like(beta), torch.empty_like(s0)]
        flops_fwd = B * T * H * DV * DK * 8.0           # per token a column: decay, u, the update, o: ~4 FMAs a row
        bytes_fwd = 4.0 * (B * T * H * (2 * DK + 2 * DV + 2))
        bytes_ckpt = 4.0 * B * H * nb * DK * DV
        for kname, threads, group in (("gdr_fwd", (DV // 32 * 1024, H, B), (1024, 1, 1)), ("gdr_fwd8", (8 * DV, H, B), (8 * DV, 1, 1))):
            kern = getattr(lib, kname)
            for save in (0, 1):
                t = gpu_ms(lambda: kern(q, k, v, g, beta, s0, o, sT, ckpt, T, H, DV, save, threads=threads, group_size=group))
                r[f"{kname}_save{save}"] = dict(t, roofline=roofline(flops_fwd, bytes_fwd + save * bytes_ckpt, t["median_ms"]))
        lib.gdr_fwd8(q, k, v, g, beta, s0, o, sT, ckpt, T, H, DV, 1, threads=(8 * DV, H, B), group_size=(8 * DV, 1, 1))
        flops_bwd = flops_fwd * 3.5                      # recompute the block, then ~2.5x the forward's work a token
        for kname, threads, group in (("gdr_bwd", (DV // 32 * 1024, H, B), (1024, 1, 1)), ("gdr_bwd8", (8 * DV, H, B), (8 * DV, 1, 1))):
            kern = getattr(lib, kname)
            def run():
                for x in grads[:2] + grads[3:5]:
                    x.zero_()
                kern(q, k, v, g, beta, ckpt, dout, sT, *grads, T, H, DV, threads=threads, group_size=group)
            t = gpu_ms(run)
            r[kname] = dict(t, roofline=roofline(flops_bwd, bytes_fwd * 2 + bytes_ckpt, t["median_ms"]))
        print(name, json.dumps(r)[:1500], flush=True)
        out["shapes"][name] = r
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"profile_gated_delta_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path, "\npipelines", json.dumps(out["pipelines"]))


if __name__ == "__main__":
    main()
