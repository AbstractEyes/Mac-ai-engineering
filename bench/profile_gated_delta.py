"""Profile of the gated delta rule on the Apple GPU: transformers' chunked reference stage by stage (forward), its forward and
backward totals and operator counts, the baseline Metal kernels (D0) and the chunked kernels (D1 forward, D2 backward) with
their pipeline limits and rooflines, and the rule's forward + backward through the autograd function and through the drop-in
under each kernel pairing.

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
from macai.gated_delta import _chunked, chunk_gated_delta_rule, l2norm, op  # noqa: E402
from macai.gated_delta._metal import BLK, library  # noqa: E402
from macai.instruments import gpu_ms, pipeline, roofline, warm_gpu  # noqa: E402

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


MMA = 2 * 8 * 8 * 8        # FLOPs of one 8 x 8 x 8 simdgroup product


def chunked_work(B, T, H, dk, dv, c):
    """FLOPs (the 8 x 8 x 8 products each kernel runs) and the least device bytes (every input read once, every output
    written once) of the chunked kernels, by kernel."""
    nt = math.ceil(T / c)
    n = B * H * nt
    R, Kb, Vb = c // 8, dk // 8, dv // 8
    L = R * (R + 1) // 2
    merges = sum((R - d) * (d + 1) for d in range(1, R))
    w2 = sum(bj + 1 for bi in range(R) for bj in range(bi + 1)) + sum(R - bi for bi in range(R) for bj in range(bi + 1))
    f4 = 4.0
    out = {
        "gdc_prep": ((2 * L * Kb + merges) * MMA, f4 * (2 * c * dk + 2 * c + c + 3 * c * c)),
        "gdc_fwd_save0": ((3 * Kb * R * Vb + 2 * L * Vb) * MMA, f4 * (2 * c * dk + c * dv + 2 * c * c + 2 * c + c * dv)),
        "gdc_fwd_save1": ((3 * Kb * R * Vb + 2 * L * Vb) * MMA,
                          f4 * (2 * c * dk + c * dv + 2 * c * c + 2 * c + c * dv + dk * dv + 2 * c * dv)),
        "gdc_bwd_state": ((3 * Kb * R * Vb + 2 * L * Vb) * MMA, f4 * (2 * c * dk + c * dv + 2 * c * c + 2 * c + dk * dv + c * dv)),
        "gdc_bwd_chunk": ((2 * L * Vb + w2 + 3 * R * Vb * Kb + 5 * L * Kb + L * Vb) * MMA,
                          f4 * (2 * c * dk + 4 * c * dv + 2 * dk * dv + 3 * c * c + 2 * c + 2 * c * dk + c * dv + 2 * c)),
    }
    return {k: (fl * n, by * n) for k, (fl, by) in out.items()}


def chunked_profile(B, T, H):
    """The chunked kernels alone (back-to-back times) on float32 inputs prepared as the drop-in prepares them."""
    lib = _chunked.library(DK)
    c = _chunked.C
    nt = math.ceil(T / c)
    f32 = dict(device="mps", dtype=torch.float32)
    g0 = torch.Generator().manual_seed(1)
    q = (l2norm(torch.randn(B, T, H, DK, generator=g0)) * DK ** -0.5).to("mps")
    k = l2norm(torch.randn(B, T, H, DK, generator=g0)).to("mps")
    v = torch.randn(B, T, H, DV, generator=g0).to("mps")
    g = (-torch.rand(B, T, H, generator=g0)).to("mps")
    beta = torch.rand(B, T, H, generator=g0).to("mps")
    s0 = torch.zeros(B, H, DK, DV, **f32)
    dout = torch.randn(B, T, H, DV, generator=g0).to("mps")
    dsT = torch.zeros(B, H, DK, DV, **f32)
    gc = torch.empty(B, H, nt, c, **f32)
    a, m, kg = (torch.empty(B, H, nt, c, c, **f32) for _ in range(3))
    o, sT = torch.empty(B, T, H, DV, **f32), torch.empty(B, H, DK, DV, **f32)
    sc = torch.empty(B, H, nt, DK, DV, **f32)
    vn = torch.empty(B, H, nt * c, DV, **f32)
    xu = torch.empty_like(vn)
    none = torch.empty(4, **f32)
    dsn, dvn = torch.empty_like(sc), torch.empty_like(vn)
    ds0 = torch.empty(B, H, DK, DV, **f32)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    dg, dbeta = torch.empty_like(beta), torch.empty_like(beta)
    grid = op._state_grid(DV, H, B)
    nb = 32 * _chunked.NSGB
    run = {
        "gdc_prep": lambda: lib.gdc_prep(q, k, g, beta, gc, a, m, kg, T, H, threads=(nt * 128, H, B), group_size=(128, 1, 1)),
        "gdc_fwd_save0": lambda: lib.gdc_fwd(q, k, v, beta, s0, gc, a, m, o, sT, none, none, none, none, T, H, DV, 0, 1, nt, **grid),
        "gdc_fwd_save1": lambda: lib.gdc_fwd(q, k, v, beta, s0, gc, a, m, o, sT, sc, vn, xu, none, T, H, DV, 1, 1, nt, **grid),
        "gdc_bwd_state": lambda: lib.gdc_bwd_state(q, k, beta, dout, dsT, gc, a, m, dsn, dvn, ds0, T, H, DV, **grid),
        "gdc_bwd_chunk": lambda: lib.gdc_bwd_chunk(q, k, beta, dout, gc, a, kg, m, sc, dsn, vn, xu, dvn, dq, dk, dv, dg, dbeta,
                                                   T, H, DV, 1, nt, threads=(nt * nb, H, B), group_size=(nb, 1, 1)),
    }
    for name in ("gdc_prep", "gdc_fwd_save1", "gdc_bwd_state", "gdc_bwd_chunk"):
        run[name]()
    work = chunked_work(B, T, H, DK, DV, c)
    r = {"chunk": c, "state_pass_threadgroups": grid["threads"][0] // grid["group_size"][0] * H * B,
         "state_pass_threads_a_threadgroup": grid["group_size"][0]}
    for name, fn in run.items():
        t = gpu_ms(fn, reps=10, warm=2, inner=5)
        r[name] = dict(t, roofline=roofline(*work[name], t["median_ms"]))
    combos = {"forward_with_states": (("gdc_prep", "gdc_fwd_save1")), "forward_without_states": ("gdc_prep", "gdc_fwd_save0"),
              "backward": ("gdc_bwd_state", "gdc_bwd_chunk")}
    for name, parts in combos.items():
        t = gpu_ms(lambda: [run[p]() for p in parts], reps=10, warm=2, inner=5)
        fl = sum(work[p][0] for p in parts)
        by = sum(work[p][1] for p in parts)
        r[name] = dict(t, roofline=roofline(fl, by, t["median_ms"]))
    return r


def rule_through_autograd(B, T, H):
    """GatedDeltaRule (prepared float32 inputs, every input needing its gradient) and the drop-in (bf16 inputs, l2 norm in the
    call), forward + backward, under each kernel pairing."""
    g0 = torch.Generator().manual_seed(2)
    q = (l2norm(torch.randn(B, T, H, DK, generator=g0)) * DK ** -0.5).to("mps")
    k = l2norm(torch.randn(B, T, H, DK, generator=g0)).to("mps")
    v = torch.randn(B, T, H, DV, generator=g0).to("mps")
    g = (-torch.rand(B, T, H, generator=g0)).to("mps")
    beta = torch.rand(B, T, H, generator=g0).to("mps")
    s0 = torch.zeros(B, H, DK, DV, device="mps")
    dout = torch.randn(B, T, H, DV, generator=g0).to("mps")
    raw = inputs(B, T, H, seed=3)
    out = {}
    for label, values in PAIRINGS.items():
        old = {k: getattr(op, k) for k in values}
        for key_, val in values.items():
            setattr(op, key_, val)
        try:
            xs = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta, s0)]

            def fn_fwd():
                with torch.no_grad():
                    op.GatedDeltaRule.apply(*xs, False)

            def fn_fb():
                y, _ = op.GatedDeltaRule.apply(*xs, True)
                y.backward(dout)

            def drop_fwd():
                with torch.no_grad():
                    chunk_gated_delta_rule(*raw, use_qk_l2norm_in_kernel=True)

            def drop_fb():
                ys = [x.detach().clone().requires_grad_(True) for x in raw]
                chunk_gated_delta_rule(*ys, use_qk_l2norm_in_kernel=True)[0].float().sum().backward()
            out[label] = {"function_fwd": gpu_ms(fn_fwd), "function_fwd_bwd": gpu_ms(fn_fb),
                          "dropin_fwd": gpu_ms(drop_fwd), "dropin_fwd_bwd": gpu_ms(drop_fb)}
        finally:
            for key_, val in old.items():
                setattr(op, key_, val)
    return out


# op settings, as in bench_gated_delta.py: the 0.2.0 drop-in; the chunked kernels; the chunked kernels with the fused prologue
# (the chunked ones at Phase 3's chunking: chunk 16 forwards, no value split)
PAIRINGS = {"d0_head_fwd_column_bwd": dict(KERNEL="auto", KERNEL_BWD="column", FUSED_NORM=False),
            "chunked": dict(KERNEL="chunked", KERNEL_BWD="chunked", FUSED_NORM=False, FWD_CHUNK=16, VALUE_SPLIT=1),
            "chunked_fused_norm": dict(KERNEL="chunked", KERNEL_BWD="chunked", FUSED_NORM=True, FWD_CHUNK=16, VALUE_SPLIT=1)}


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
    clib = _chunked.library(DK)
    out = {"chunk": C, "blk": BLK, "pipelines": {name: pipeline(getattr(lib, name)) for name in ("gdr_fwd", "gdr_bwd", "gdr_fwd8", "gdr_bwd8")},
           "chunked_settings": {"C": _chunked.C, "NCB": _chunked.NCB, "NSG": _chunked.NSG, "STAGE": _chunked.STAGE,
                                "HWLOAD": _chunked.HWLOAD, "NSGB": _chunked.NSGB, "KGRP": _chunked.KGRP},
           "shapes": {}}
    out["pipelines"].update({name: pipeline(getattr(clib, name)) for name in ("gdc_prep", "gdc_fwd", "gdc_bwd_state", "gdc_bwd_chunk")})
    warm_gpu(2.0)
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
        del ckpt, grads
        r["chunked"] = chunked_profile(B, T, H)
        r["rule_fwd_bwd"] = rule_through_autograd(B, T, H)
        print(name, json.dumps(r)[:1500], flush=True)
        out["shapes"][name] = r
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"profile_gated_delta_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path, "\npipelines", json.dumps(out["pipelines"]))


if __name__ == "__main__":
    main()
