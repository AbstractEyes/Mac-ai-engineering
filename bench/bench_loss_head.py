"""Series designs D8 / D8b (docs/series_layer.md): the LM head and the loss timed part by part at the training micro-batch's
shapes, on every position (before D8) and on the labelled positions only (D8), Metal events, the two alternating in
interleaved rounds; and a fused cross-entropy's least memory traffic (the conditional design D8b's floor).

    python bench/bench_loss_head.py                 # bench/layer_logs/d8_loss_head.json

The shapes are the step profile's averages (docs/series_layer.md, step 0): Qwen3.5-0.8B 2 rows of 811 positions (1,622), 312 of
them labelled; 9B 1 row of 768 positions, 163 labelled; vocabulary 248,320; the head weight bf16 and frozen; bf16 autocast as
the Trainer runs it. The parts, as the model's forward and its loss function run them:

  head            F.linear(h [N, H], W [V, H]) -> logits [N, V] in bf16
  cast            the loss's upcast of the logits to float32
  ce forward      F.cross_entropy on the float32 logits (log-softmax and the pick), the mean over the labelled rows
  ce backward     its gradient [N, V] in float32
  cast back       that gradient to bf16 (the upcast's backward)
  head backward   dE = dZ W [N, H] (the frozen head: no weight gradient)
  output copy     accelerate's conversion of the returned logits to float32 (the model's output)
  whole           the head and the loss forward and backward as one autograd graph, the output copy included

The pid is printed first.
"""
import json
import os
import statistics
import sys
import time

print(f"[bench_loss_head] pid {os.getpid()}", flush=True)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from macai.instruments import gpu_ms, warm_gpu  # noqa: E402

DEV, BF = "mps", torch.bfloat16
V = 248320
ROUNDS = 5
SHAPES = {"0.8b": dict(H=1024, positions=1622, labelled=312), "9b": dict(H=4096, positions=768, labelled=163)}
COPY_GBPS = 356.1          # bench/hw_ceilings.py's copy rate (macai.instruments.CEILINGS)


def inner_for(fn, target_ms=10.0) -> int:
    fn()
    torch.mps.synchronize()
    t = gpu_ms(fn, reps=3, warm=1)["median_ms"]
    return max(1, int(target_ms / max(t, 1e-3)))


def parts(H, n, labelled, gen):
    """The callables of one shape: n rows go through the head; labelled of them carry a label (the rest ignored)."""
    W = (torch.randn(V, H, generator=gen) * 0.02).to(DEV, BF)
    h = (torch.randn(n, H, generator=gen)).to(DEV, BF)
    y = torch.randint(0, V, (n,), generator=gen)
    y[labelled:] = -100                                   # (on the labelled rows only: no row ignored)
    y = y[torch.randperm(n, generator=gen)].to(DEV)
    with torch.autocast(DEV, dtype=BF):
        z = F.linear(h, W)
    z32 = z.float()
    g32 = torch.autograd.grad(F.cross_entropy(z32.detach().requires_grad_(True), y, ignore_index=-100),
                              z32.detach().requires_grad_(True), allow_unused=True)
    leaf32 = z32.detach().requires_grad_(True)
    dz32 = torch.autograd.grad(F.cross_entropy(leaf32, y, ignore_index=-100), leaf32)[0]
    dz = dz32.to(BF)
    del g32

    def head():
        with torch.autocast(DEV, dtype=BF):
            return F.linear(h, W)

    def cast():
        return z.float()

    def ce_fwd():
        return F.cross_entropy(z32, y, ignore_index=-100)

    def ce_fwd_bwd():
        x = z32.detach().requires_grad_(True)
        return torch.autograd.grad(F.cross_entropy(x, y, ignore_index=-100), x)[0]

    def cast_back():
        return dz32.to(BF)

    def head_bwd():
        return dz @ W

    hh = h.detach().requires_grad_(True)

    def whole():
        with torch.autocast(DEV, dtype=BF):
            logits = F.linear(hh, W)
        out = logits.float()                               # accelerate's conversion of the returned logits
        loss = F.cross_entropy(logits.float(), y, ignore_index=-100)
        g = torch.autograd.grad(loss, hh)[0]
        return out, g
    return {"head": head, "cast": cast, "ce_fwd": ce_fwd, "ce_fwd_bwd": ce_fwd_bwd, "cast_back": cast_back,
            "head_bwd": head_bwd, "output_copy": cast, "whole": whole}


def main():
    warm_gpu(2.0)
    gen = torch.Generator().manual_seed(0)
    out = {"vocab": V, "shapes": SHAPES, "rounds": ROUNDS, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "torch": torch.__version__, "models": {}}
    for model, s in SHAPES.items():
        variants = {"every_position": parts(s["H"], s["positions"], s["labelled"], gen),
                    "labelled_only": parts(s["H"], s["labelled"], s["labelled"], gen)}
        inner = {v: {k: inner_for(fn) for k, fn in fns.items()} for v, fns in variants.items()}
        rounds = {v: {k: [] for k in fns} for v, fns in variants.items()}
        for r in range(ROUNDS):
            order = list(variants) if r % 2 == 0 else list(reversed(variants))
            for v in order:
                for k, fn in variants[v].items():
                    rounds[v][k].append(gpu_ms(fn, reps=7, warm=1, inner=inner[v][k])["median_ms"])
        res = {}
        for v in variants:
            ms = {k: round(statistics.median(x), 3) for k, x in rounds[v].items()}
            ms["ce_bwd"] = round(ms["ce_fwd_bwd"] - ms["ce_fwd"], 3)
            ms["loss_own_passes"] = round(ms["cast"] + ms["ce_fwd_bwd"] + ms["cast_back"], 3)
            ms["head_products"] = round(ms["head"] + ms["head_bwd"], 3)
            ms["parts_sum"] = round(ms["head_products"] + ms["loss_own_passes"] + ms["output_copy"], 3)
            res[v] = {"ms": ms, "rounds_ms": rounds[v], "inner": inner[v]}
        n = s["labelled"]
        # a fused cross-entropy (D8b): the loss forward reads the bf16 logits once, the backward reads them again and writes
        # the bf16 gradient over them; no float32 copy of the logits, no output copy (least traffic, at the copy rate)
        fused_bytes = 3 * n * V * 2
        res["d8b_floor"] = {"bytes": fused_bytes, "ms": round(fused_bytes / COPY_GBPS / 1e6, 3),
                            "replaces_ms": round(res["labelled_only"]["ms"]["loss_own_passes"]
                                                 + res["labelled_only"]["ms"]["output_copy"], 3)}
        out["models"][model] = res
        print(f"[bench_loss_head] {model}: " + json.dumps({v: res[v]["ms"] for v in variants}), flush=True)
        del variants
        torch.mps.empty_cache()
    path = os.path.join(HERE, "layer_logs", "d8_loss_head.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"[bench_loss_head] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
