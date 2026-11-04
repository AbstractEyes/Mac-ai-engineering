"""Series design D3 (the rule's chunking and tiling; docs/series_layer.md): the gated delta rule timed as the Qwen3.5 layer calls
it, under named variants of macai.gated_delta's settings, in interleaved rounds.

    python bench/bench_d3.py --name d3_baseline --variants base
    python bench/bench_d3.py --name d3_bench --variants base,fwd8,dbuf,fwd8+dbuf,split2,split4,split8 --kernels

The call: macai.gated_delta.chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
with q, k bfloat16 and v, g, beta float32 [B, T, H, 128] (the layer's convolution and gates paths hand the rule exactly these),
Metal events around back-to-back calls (macai.instruments.gpu_ms), three modes:

  fwd        the forward without gradients (a prefill; keeps nothing)
  train_fwd  the forward with every input needing its gradient (the training forward: under the recomputed states it keeps
             only its inputs), no backward
  fwd_bwd    the training forward and its backward (a fixed bfloat16 output gradient; the final state's gradient None, as in
             training)

Shapes: the bench shapes (4 x 1,024 x 16 heads, 2 x 1,024 x 32); every micro-batch of the step profiles with D4 + D8 + D5
(bench/layer_logs/profile_d5_0.8b_2x8.json: 2 rows x 16 heads, 24 micro-batches; profile_d5_9b_1x16.json: 1 row x 32 heads,
32 micro-batches), read as "the rule a micro-batch" (each micro-batch's call times its linear-attention layers, 18 / 24,
averaged over the micro-batches: the profiles' rule rows, 75.8 / 99.2 ms); the long sequences (1 row, 16 and 32 heads,
4,096 / 8,192 / 16,384 / 32,768 tokens). --kernels adds the state passes and the per-chunk backward alone (the kernels on
prepared float32 inputs, back-to-back) at the bench shapes, a mean micro-batch of each model and two long shapes.

--order shape (the default since the record d3_bench) times every variant at one shape before the next shape, so the variants
compared at a shape run seconds apart (the variants' order rotating with the round and the shape), with --pause-s seconds idle
between rounds; --order variant times one variant at every shape before the next (the baseline and the exploration run d3_bench_run2;
a run of record that spanned minutes between variants drifted, d3_bench_drifted). A figure is the median of its rounds' medians.
Writes bench/layer_logs/<name>.json. The first line printed is the process id.
"""
import os
import sys

print(f"[bench_d3] pid {os.getpid()}", flush=True)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from macai.gated_delta import _chunked, chunk_gated_delta_rule, l2norm, op  # noqa: E402
from macai.instruments import gpu_ms, warm_gpu  # noqa: E402

LOGS = os.path.join(HERE, "layer_logs")
D = 128
# the variants: settings of macai.gated_delta.op ("op") and macai.gated_delta._chunked ("chunked"); "base" sets nothing
VARIANTS = {
    "base": {},
    "fwd8": {"op": {"FWD_CHUNK": 8}},                                   # (a1)
    "dbuf": {"chunked": {"DBUF": 1}},                                   # (a2), two buffers, one barrier a chunk
    "dbuf2": {"chunked": {"DBUF": 2}},                                  # (a2), one buffer, the next chunk read ahead
    "fwd8+dbuf": {"op": {"FWD_CHUNK": 8}, "chunked": {"DBUF": 1}},
    "split2": {"op": {"VALUE_SPLIT": 2}},                               # (b)
    "split4": {"op": {"VALUE_SPLIT": 4}},
    "split8": {"op": {"VALUE_SPLIT": 8}},
    "split2+dbuf": {"op": {"VALUE_SPLIT": 2}, "chunked": {"DBUF": 1}},
    "split4+dbuf": {"op": {"VALUE_SPLIT": 4}, "chunked": {"DBUF": 1}},
    "fwd8+split2": {"op": {"FWD_CHUNK": 8, "VALUE_SPLIT": 2}},
    "fwd8+split4": {"op": {"FWD_CHUNK": 8, "VALUE_SPLIT": 4}},
    "auto": {"op": {"VALUE_SPLIT": "auto"}},
    "fwd8+auto": {"op": {"FWD_CHUNK": 8, "VALUE_SPLIT": "auto"}},
}
PROFILES = {"0.8b": ("profile_d5_0.8b_2x8", 16, 18), "9b": ("profile_d5_9b_1x16", 32, 24)}
BENCH_SHAPES = {"0.8b_bench": (4, 1024, 16), "9b_bench": (2, 1024, 32)}
LONG_T = (1024, 4096, 8192, 16384, 32768)
LONG_H = (16, 32)
KERNEL_SHAPES = {"0.8b_bench": (4, 1024, 16), "9b_bench": (2, 1024, 32), "0.8b_mean": (2, 811, 16), "9b_mean": (1, 768, 32),
                 "long_8k_h16": (1, 8192, 16), "long_8k_h32": (1, 8192, 32), "long_32k_h16": (1, 32768, 16)}


class Settings:
    """Apply a variant's settings inside a block; a setting the code does not have is an error (never a silent base run)."""

    def __init__(self, spec):
        self.spec = spec

    def __enter__(self):
        self.old = []
        for modname, values in self.spec.items():
            mod = {"op": op, "chunked": _chunked}[modname]
            for k, v in values.items():
                if not hasattr(mod, k):
                    raise AttributeError(f"macai.gated_delta.{modname} has no setting {k}")
                self.old.append((mod, k, getattr(mod, k)))
                setattr(mod, k, v)

    def __exit__(self, *exc):
        for mod, k, v in reversed(self.old):
            setattr(mod, k, v)


def git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def micro_batch_tokens():
    """Each profiled micro-batch's (rows, tokens), by model, from the step profiles' records."""
    out = {}
    for model, (name, heads, layers) in PROFILES.items():
        with open(os.path.join(LOGS, name + ".json"), encoding="utf-8") as fh:
            mbs = json.load(fh)["profile"]["micro_batches"]
        out[model] = {"heads": heads, "layers": layers, "rows_tokens": [tuple(m["shapes"]["input_ids"]) for m in mbs]}
    return out


def inputs(B, T, H, seed):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, T, H, D, generator=gen).to("mps", torch.bfloat16)
    k = torch.randn(B, T, H, D, generator=gen).to("mps", torch.bfloat16)
    v = torch.randn(B, T, H, D, generator=gen).to("mps")
    g = (-torch.rand(B, T, H, generator=gen)).to("mps")
    beta = torch.rand(B, T, H, generator=gen).to("mps")
    dout = torch.randn(B, T, H, D, generator=gen).to("mps", torch.bfloat16)
    return (q, k, v, g, beta), dout


def call(xs):
    return chunk_gated_delta_rule(xs[0], xs[1], xs[2], g=xs[3], beta=xs[4], output_final_state=True,
                                  use_qk_l2norm_in_kernel=True)


def modes(base, dout):
    def fwd():
        with torch.no_grad():
            call(base)

    def train_fwd():
        call([x.detach().requires_grad_(True) for x in base])

    def fwd_bwd():
        o, _ = call([x.detach().requires_grad_(True) for x in base])
        o.backward(dout)
    return {"fwd": fwd, "train_fwd": train_fwd, "fwd_bwd": fwd_bwd}


def time_modes(base, dout, est_ms: float, reps: int):
    """The three modes, back-to-back calls between two events so a measured span is >= ~6 ms (the GPU's ramp after each
    synchronization stays outside short calls' figures)."""
    out = {}
    for mode, fn in modes(base, dout).items():
        per = est_ms * (3.0 if mode == "fwd_bwd" else 1.0)
        inner = max(1, min(8, math.ceil(6.0 / max(per, 0.05))))
        out[mode] = gpu_ms(fn, reps=reps, warm=1, inner=inner)["median_ms"]
    return out


# ---- the kernels alone ------------------------------------------------------------------------------------------------------

def kernel_parts(B, T, H, seed=1):
    """The state passes and the per-chunk backward on prepared float32 inputs, as the rule launches them under the current
    settings: the forward's state pass (no states kept: at the forward's chunk length), the backward's recompute pass (flags 1|8,
    chunk 16), the backward state pass and the per-chunk backward. Back-to-back Metal-event medians, ms."""
    g0 = torch.Generator().manual_seed(seed)
    f32 = dict(device="mps", dtype=torch.float32)
    q = (l2norm(torch.randn(B, T, H, D, generator=g0)) * D ** -0.5).to("mps")
    k = l2norm(torch.randn(B, T, H, D, generator=g0)).to("mps")
    v = torch.randn(B, T, H, D, generator=g0).to("mps")
    g = (-torch.rand(B, T, H, generator=g0)).to("mps")
    beta = torch.rand(B, T, H, generator=g0).to("mps")
    dout = torch.randn(B, T, H, D, generator=g0).to("mps")
    s0 = torch.zeros(B, H, D, D, **f32)
    none = torch.empty(4, **f32)
    o, sT = torch.empty(B, T, H, D, **f32), torch.empty(B, H, D, D, **f32)
    res = {}
    plan = op.plan(B, T, H, D, D, save=False) if hasattr(op, "plan") else None
    # the forward's state pass (flags 0), at the forward's own chunk length and launch
    if plan is not None:
        lib_f, grid_f, cf = plan["fwd_lib"], plan["fwd_grid"], plan["fwd_chunk"]
    else:
        lib_f, grid_f, cf = _chunked.library(D), op._state_grid(D, H, B), _chunked.C
    ntf = math.ceil(T / cf)
    prep_f = op._prep(lib_f, q, k, g, beta, False, cf) if plan is not None else op._prep(lib_f, q, k, g, beta, False)
    gcf, af, mf, _ = prep_f
    res["fwd_state_pass"] = gpu_ms(lambda: lib_f.gdc_fwd(q, k, v, beta, s0, gcf, af, mf, o, sT, none, none, none, none, T, H, D, 0,
                                                         1, ntf, **grid_f), reps=5, warm=1, inner=3)["median_ms"]
    del prep_f, gcf, af, mf
    # the backward's passes at chunk 16, under the current launch (split, double buffering)
    if plan is not None:
        lib, grid = plan["bwd_lib"], plan["bwd_grid"]
    else:
        lib, grid = _chunked.library(D), op._state_grid(D, H, B)
    c = _chunked.C
    nt = math.ceil(T / c)
    gc, a, m, kg = op._prep(lib, q, k, g, beta, False)
    sc = torch.empty(B, H, nt, D, D, **f32)
    vn, xu = torch.empty(B, H, nt * c, D, **f32), torch.empty(B, H, nt * c, D, **f32)
    res["prep16"] = gpu_ms(lambda: op._prep(lib, q, k, g, beta, False), reps=5, warm=1, inner=3)["median_ms"]
    res["recompute_state_pass"] = gpu_ms(lambda: lib.gdc_fwd(q, k, v, beta, s0, gc, a, m, none, none, sc, vn, xu, none, T, H, D,
                                                             1 | 8, 1, nt, **grid), reps=5, warm=1, inner=3)["median_ms"]
    dsn, dvn = torch.empty_like(sc), torch.empty_like(vn)
    ds0, dsT = torch.empty(B, H, D, D, **f32), torch.zeros(B, H, D, D, **f32)
    res["bwd_state_pass"] = gpu_ms(lambda: lib.gdc_bwd_state(q, k, beta, dout, dsT, gc, a, m, dsn, dvn, ds0, T, H, D, **grid),
                                   reps=5, warm=1, inner=3)["median_ms"]
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    dg, dbeta = torch.empty_like(beta), torch.empty_like(beta)
    nb = 32 * _chunked.NSGB
    res["bwd_chunk"] = gpu_ms(lambda: lib.gdc_bwd_chunk(q, k, beta, dout, gc, a, kg, m, sc, dsn, vn, xu, dvn, dq, dk, dv, dg, dbeta,
                                                        T, H, D, 1, nt, threads=(nt * nb, H, B), group_size=(nb, 1, 1)),
                              reps=5, warm=1, inner=3)["median_ms"]
    res["state_passes"] = round(res["fwd_state_pass"] + res["recompute_state_pass"] + res["bwd_state_pass"], 3)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--variants", default="base")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--kernels", action="store_true")
    ap.add_argument("--no-long", action="store_true")
    ap.add_argument("--no-readers", action="store_true")
    ap.add_argument("--order", choices=("shape", "variant"), default="shape")
    ap.add_argument("--pause-s", type=float, default=0.0, help="seconds idle between rounds (--order shape)")
    a = ap.parse_args()
    names = a.variants.split(",")
    for n in names:
        if n not in VARIANTS:
            raise SystemExit(f"unknown variant {n}")
    mbs = micro_batch_tokens()
    shapes = dict(BENCH_SHAPES)
    if not a.no_readers:
        for model, spec in mbs.items():
            for (B, T) in sorted(set(spec["rows_tokens"])):
                shapes[f"{model}_mb_{B}x{T}"] = (B, T, spec["heads"])
    if not a.no_long:
        for H in LONG_H:
            for T in LONG_T:
                shapes[f"long_h{H}_{T}"] = (1, T, H)
    rec = {"name": a.name, "pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "repo_commit": git("rev-parse", "--short", "HEAD"), "repo_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
           "torch": torch.__version__, "variants": {n: VARIANTS[n] for n in names}, "rounds": a.rounds, "reps": a.reps,
           "defaults": {"op": {k: getattr(op, k) for k in ("KERNEL", "KERNEL_BWD", "FUSED_NORM", "RECOMPUTE_STATES", "SIZE_CLASSES",
                                                          "FWD_CHUNK", "VALUE_SPLIT") if hasattr(op, k)},
                        "chunked": {k: getattr(_chunked, k) for k in ("C", "NSG", "STAGE", "NSGB", "DBUF") if hasattr(_chunked, k)}},
           "micro_batches": mbs, "shapes": {k: list(v) for k, v in shapes.items()}, "times": {}, "kernels": {},
           "order": a.order, "pause_s": a.pause_s}
    print(f"[bench_d3] {len(names)} variants x {len(shapes)} shapes x 3 modes x {a.rounds} rounds; commit {rec['repo_commit']}"
          f" dirty {rec['repo_dirty']}", flush=True)
    warm_gpu(2.0)
    data = {}
    for s, (B, T, H) in shapes.items():
        data[s] = inputs(B, T, H, seed=zlib.crc32(s.encode()) % 1000)
    # a first estimate of each shape's forward (for the back-to-back counts), with the defaults
    est = {}
    for s, (base, dout) in data.items():
        def f(b=base):
            with torch.no_grad():
                call(b)
        est[s] = gpu_ms(f, reps=2, warm=1, inner=1)["median_ms"]
    times = {n: {s: {m: [] for m in ("fwd", "train_fwd", "fwd_bwd")} for s in shapes} for n in names}
    kern = {n: {s: [] for s in KERNEL_SHAPES} for n in names} if a.kernels else {}
    t_start = time.time()
    if a.order == "shape":
        # every variant's libraries compiled before the first timed call
        first = next(iter(data))
        for n in names:
            with Settings(VARIANTS[n]):
                time_modes(data[first][0], data[first][1], est[first], 1)
                if a.kernels:
                    kernel_parts(*KERNEL_SHAPES["9b_mean"])
        for r in range(a.rounds):
            if r and a.pause_s:
                time.sleep(a.pause_s)
            warm_gpu(1.0)
            t0 = time.time()
            for i, (s, (base, dout)) in enumerate(data.items()):
                k = (r + i) % len(names)
                for n in names[k:] + names[:k]:
                    reps = a.reps if not s.startswith("long") else max(2, a.reps - 1)
                    with Settings(VARIANTS[n]):
                        for m, v in time_modes(base, dout, est[s], reps).items():
                            times[n][s][m].append(v)
                torch.mps.empty_cache()
            if a.kernels:
                for i, (s, (B, T, H)) in enumerate(KERNEL_SHAPES.items()):
                    k = (r + i) % len(names)
                    for n in names[k:] + names[:k]:
                        with Settings(VARIANTS[n]):
                            kern[n][s].append(kernel_parts(B, T, H))
                    torch.mps.empty_cache()
            spent = time.time() - t_start
            print(f"[bench_d3] round {r + 1}/{a.rounds}: {time.time() - t0:.1f} s; spent {spent / 60:.1f} min, left "
                  f"~{spent / (r + 1) * (a.rounds - r - 1) / 60:.1f} min", flush=True)
    for r in range(a.rounds if a.order == "variant" else 0):
        order = names[r % len(names):] + names[:r % len(names)]
        for n in order:
            t0 = time.time()
            with Settings(VARIANTS[n]):
                # the variant's libraries compiled and the GPU's clocks up before the first timed shape
                first = next(iter(data.values()))
                time_modes(first[0], first[1], est[next(iter(data))], 2)
                warm_gpu(0.3)
                for s, (base, dout) in data.items():
                    reps = a.reps if not s.startswith("long") else max(2, a.reps - 1)
                    for m, v in time_modes(base, dout, est[s], reps).items():
                        times[n][s][m].append(v)
                    torch.mps.empty_cache()
                if a.kernels:
                    for s, (B, T, H) in KERNEL_SHAPES.items():
                        kern[n][s].append(kernel_parts(B, T, H))
                        torch.mps.empty_cache()
            spent = time.time() - t_start
            done = r * len(names) + order.index(n) + 1
            left = spent / done * (a.rounds * len(names) - done)
            print(f"[bench_d3] round {r + 1}/{a.rounds} {n}: {time.time() - t0:.1f} s; spent {spent / 60:.1f} min, "
                  f"left ~{left / 60:.1f} min", flush=True)
    med = statistics.median
    for n in names:
        rec["times"][n] = {s: {m: {"median_ms": round(med(v), 3), "rounds": v} for m, v in ms.items()} for s, ms in times[n].items()}
        if a.kernels:
            rec["kernels"][n] = {s: {p: {"median_ms": round(med(x[p] for x in rows), 3), "rounds": [x[p] for x in rows]}
                                     for p in rows[0]} for s, rows in kern[n].items()}
        # the rule a micro-batch: each profiled micro-batch's call x its model's linear-attention layers, averaged
        rec.setdefault("micro_batch_ms", {})[n] = {}
        if not a.no_readers:
            for model, spec in mbs.items():
                row = {}
                for m in ("fwd", "train_fwd", "fwd_bwd"):
                    vals = [rec["times"][n][f"{model}_mb_{B}x{T}"][m]["median_ms"] * spec["layers"] for B, T in spec["rows_tokens"]]
                    row[m] = round(sum(vals) / len(vals), 2)
                rec["micro_batch_ms"][n][model] = row
    rec["ended"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    rec["wall_s"] = round(time.time() - t_start, 1)
    path = os.path.join(LOGS, a.name + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    for n in names:
        print(f"[bench_d3] {n} micro-batch {json.dumps(rec['micro_batch_ms'][n])}", flush=True)
        for s in shapes:
            if not s.split("_mb_")[0] in mbs:
                print(f"[bench_d3] {n} {s} " + " ".join(f"{m} {rec['times'][n][s][m]['median_ms']}" for m in rec["times"][n][s]),
                      flush=True)
    print(f"[bench_d3] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
