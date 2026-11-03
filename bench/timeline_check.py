"""Check the timeline instrument (bench/timeline.py) on known cases before it profiles a training step (docs/series_layer.md,
step 0). Each case runs with the instrument's fence (a tiny kernel before each event, the default) and without it.

    python bench/timeline_check.py     # writes bench/layer_logs/timeline_check.json

1. Chains of matrix products (bf16, 512 / 1,024 / 2,048 cubed) with a mark before and after each product: each product's
   segment against its time from macai.instruments.gpu_ms, the marked chain's span against the wall clock, and the marked
   chain's wall clock against the same chain without marks (the marks' cost on the GPU's timeline).
2. Chains of short operations (elementwise adds of 1 M and 1 K elements, a mark before and after each): segments far
   shorter than the products', and whether a segment whose GPU time follows the CPU's issue is seen as dispatch-bound (CPU issue
   time >= 80% of the event time).
3. Forward and backward through a stack of linear layers of known sizes under bf16 autocast (frozen weights, the gradient to
   the input) with module marks and their backward nodes: each layer's attributed forward and backward time against the same
   product timed alone; the sum of the components against the wall clock; the marked step against the unmarked one.
4. The CPU cost of a mark with no GPU work, and of resolving it afterwards.
"""
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from macai.instruments import gpu_ms, warm_gpu  # noqa: E402
from timeline import Instrument, Timeline  # noqa: E402

DEV = "mps"
ROUNDS = 5


def wall(fn, reps=ROUNDS):
    """Wall-clock ms of fn() (synchronized before and after), median of reps after one warm call."""
    fn()
    torch.mps.synchronize()
    out = []
    for _ in range(reps):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(out)


def q(xs):
    s = sorted(xs)
    return {"median": round(statistics.median(s), 4), "p10": round(s[len(s) // 10], 4), "p90": round(s[9 * len(s) // 10], 4),
            "min": round(s[0], 4), "max": round(s[-1], 4), "negative": sum(1 for x in s if x < 0), "n": len(s)}


def marked_chain(op, k, fence):
    """op() k times, each between an enter and an exit mark: (wall ms per round, product segments, gaps, cpu ms per round)."""
    tl = Timeline(fence=fence)

    def marked():
        for _ in range(k):
            tl.enter("op")
            op()
            tl.exit("op")

    tl.start()
    marked()
    torch.mps.synchronize()
    tl.release()
    walls, segs, gaps, cpu, spans = [], [], [], [], []
    for _ in range(ROUNDS):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        marked()
        torch.mps.synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
        t = tl.resolve()
        segs += [t[i + 1] - t[i] for i in range(0, len(t) - 1, 2)]
        gaps += [t[i + 1] - t[i] for i in range(1, len(t) - 1, 2)]
        cpu += [(tl.cpu[i + 1] - tl.cpu[i]) / 1e6 for i in range(0, len(t) - 1, 2)]
        spans.append(t[-1] - t[0])
        tl.release()
    tl.stop()
    return walls, segs, gaps, cpu, spans


def chain_of_products(n, fence, k=40):
    x = torch.randn(n, n, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(n, n, device=DEV, dtype=torch.bfloat16) * n ** -0.5
    y = torch.empty(n, n, device=DEV, dtype=torch.bfloat16)
    op = lambda: torch.mm(x, w, out=y)  # noqa: E731
    single = gpu_ms(op, reps=10, warm=3, inner=20)

    def plain():
        for _ in range(k):
            op()

    w_plain = wall(plain)
    walls, segs, gaps, cpu, spans = marked_chain(op, k, fence)
    seg = q(segs)
    return {"shape": [n, n, n], "dtype": "bfloat16", "products": k, "fence": fence,
            "gpu_ms_one_product": single["median_ms"], "tflops": round(2 * n ** 3 / single["median_ms"] / 1e9, 2),
            "product_segment_ms": seg, "gap_between_marks_ms": q(gaps),
            "segment_over_gpu_ms": round(seg["median"] / single["median_ms"], 4),
            "chain_wall_ms_unmarked": round(w_plain, 3), "chain_wall_ms_marked": round(statistics.median(walls), 3),
            "marked_over_unmarked": round(statistics.median(walls) / w_plain, 4),
            "events_span_over_wall": round(statistics.median(spans) / statistics.median(walls), 4),
            "cpu_issue_ms_per_product": round(statistics.median(cpu), 4)}


def chain_of_short_ops(numel, fence, k=400):
    a = torch.randn(numel, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(numel, device=DEV, dtype=torch.bfloat16)
    out = torch.empty_like(a)
    op = lambda: torch.add(a, b, out=out)  # noqa: E731
    single = gpu_ms(op, reps=10, warm=3, inner=50)

    def plain():
        for _ in range(k):
            op()

    w_plain = wall(plain)
    walls, segs, gaps, cpu, spans = marked_chain(op, k, fence)
    seg = q(segs)
    cpu_med = statistics.median(cpu)
    return {"numel": numel, "ops": k, "fence": fence, "gpu_ms_one_op_back_to_back": single["median_ms"],
            "op_segment_ms": seg, "gap_between_marks_ms": q(gaps),
            "wall_ms_unmarked": round(w_plain, 3), "wall_ms_marked": round(statistics.median(walls), 3),
            "added_ms_per_mark": round((statistics.median(walls) - w_plain) / (2 * k), 5),
            "cpu_issue_ms_per_op": round(cpu_med, 5),
            "cpu_over_event": round(cpu_med / seg["median"], 3) if seg["median"] > 0 else None,
            "dispatch_bound_by_rule": bool(seg["median"] > 0 and cpu_med >= 0.8 * seg["median"])}


class Stack(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x


def stack_forward_backward(fence, tokens=4096, dims=(1024, 4096, 1024, 8192, 1024, 2048, 1024, 256, 1024)):
    model = Stack(dims).to(DEV, torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    x0 = torch.randn(tokens, dims[0], device=DEV, dtype=torch.bfloat16)
    gy = torch.randn(tokens, dims[-1], device=DEV, dtype=torch.bfloat16)

    def step():
        x = x0.detach().requires_grad_(True)
        with torch.autocast(DEV, dtype=torch.bfloat16):
            y = model(x)
        y.backward(gy)

    alone = []          # each layer alone: its forward, and its backward (the gradient to its input: dX = dY W)
    for m, a, b in zip(model.layers, dims[:-1], dims[1:]):
        xi = torch.randn(tokens, a, device=DEV, dtype=torch.bfloat16)
        gi = torch.randn(tokens, b, device=DEV, dtype=torch.bfloat16)
        f = gpu_ms(lambda: torch.nn.functional.linear(xi, m.weight), reps=10, warm=3, inner=10)
        bw = gpu_ms(lambda: torch.mm(gi, m.weight), reps=10, warm=3, inner=10)
        alone.append({"in": a, "out": b, "fwd_ms": f["median_ms"], "bwd_ms": bw["median_ms"]})

    w_plain = wall(step)
    tl = Timeline(fence=fence)
    inst = Instrument(tl)
    for i, m in enumerate(model.layers):
        inst.module(m, f"layer{i}")

    def marked_step():
        tl.enter("step")
        x = x0.detach().requires_grad_(True)
        with torch.autocast(DEV, dtype=torch.bfloat16):
            y = model(x)
        tl.enter("backward:bwd")
        y.backward(gy)
        tl.exit("backward:bwd")
        tl.exit("step")

    tl.start()
    marked_step()
    torch.mps.synchronize()
    tl.release()
    per, walls, sums, checks = {}, [], [], []
    for _ in range(ROUNDS):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        marked_step()
        torch.mps.synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
        rows = tl.attribute()
        checks.append(rows.pop("_checks"))
        sums.append(checks[-1]["sum_ms"])
        for k, v in rows.items():
            per.setdefault(f"{k[0]}|{k[1]}", []).append(v)
        tl.release()
    tl.stop()
    inst.remove()
    layers = []
    for i, a in enumerate(alone):
        f = statistics.median(v[0] for v in per.get(f"layer{i}|fwd", [[0, 0, 0]]))
        b = statistics.median(v[0] for v in per.get(f"layer{i}|bwd", [[0, 0, 0]]))
        layers.append(dict(a, attributed_fwd_ms=round(f, 4), attributed_bwd_ms=round(b, 4),
                           fwd_over_alone=round(f / a["fwd_ms"], 4), bwd_over_alone=round(b / a["bwd_ms"], 4)))
    return {"fence": fence, "tokens": tokens, "dims": list(dims), "layers": layers,
            "components_median_ms": {k: round(statistics.median(v[0] for v in vs), 4) for k, vs in per.items()},
            "components_cpu_median_ms": {k: round(statistics.median(v[1] for v in vs), 4) for k, vs in per.items()},
            "wall_ms_unmarked": round(w_plain, 3), "wall_ms_marked": round(statistics.median(walls), 3),
            "marked_over_unmarked": round(statistics.median(walls) / w_plain, 4),
            "components_sum_ms": round(statistics.median(sums), 3),
            "components_sum_over_wall": round(statistics.median(sums) / statistics.median(walls), 4),
            "checks": checks[-1]}


def mark_cpu_cost(fence, n=2000):
    tl = Timeline(fence=fence)
    tl.start()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        tl.enter("x")
    t1 = time.perf_counter()
    times = tl.resolve()
    t2 = time.perf_counter()
    tl.release()
    tl.stop()
    return {"fence": fence, "marks": n, "cpu_us_per_mark": round((t1 - t0) / n * 1e6, 2),
            "resolve_us_per_mark": round((t2 - t1) / n * 1e6, 2),
            "gpu_span_ms_of_marks_alone": round(times[-1] - times[0], 3)}


def main():
    print(f"[timeline_check] pid {os.getpid()}", flush=True)
    out = {"torch": torch.__version__, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "rounds": ROUNDS}
    warm_gpu(2.0)
    for fence in (False, True):
        key = "fence" if fence else "no_fence"
        r = out[key] = {}
        r["mark_cpu_cost"] = mark_cpu_cost(fence)
        print(key, "mark cost", json.dumps(r["mark_cpu_cost"]), flush=True)
        r["products"] = [chain_of_products(n, fence) for n in (512, 1024, 2048)]
        for x in r["products"]:
            print(key, "products", json.dumps(x), flush=True)
        r["short_ops"] = [chain_of_short_ops(n, fence) for n in (1 << 20, 1 << 10)]
        for x in r["short_ops"]:
            print(key, "short ops", json.dumps(x), flush=True)
        r["stack"] = stack_forward_backward(fence)
        print(key, "stack", json.dumps({k: v for k, v in r["stack"].items() if k != "components_cpu_median_ms"}), flush=True)
    out["ended"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    path = os.path.join(HERE, "layer_logs", "timeline_check.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
