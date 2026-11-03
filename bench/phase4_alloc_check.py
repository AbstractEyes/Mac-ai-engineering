"""Series phase 4 diagnostic: why the kernels' runs hold a larger Metal allocator pool than the reference's while their live
tensors are fewer. The training pattern of one linear-attention stack, in isolation: per micro-batch a new sequence length, the
rule called by 24 layers (every call's saved tensors alive until the backward), then the backward through all 24; the
allocator's pool (driver memory) and its live tensors read after each micro-batch; the cache emptied between each micro-
batch's forward and backward, as in the training runs. The 9B shape (1 x T x 32 heads x 128).

    python bench/phase4_alloc_check.py              # prints and writes bench/phase4_logs/alloc_check.json
    python bench/phase4_alloc_check.py --designs    # with the memory designs as arms: bench/memory_logs/alloc_check.json

Arms: the reference; the kernels; the kernels with every length padded up to a multiple of 64 tokens before the call (the
reference pads to its chunk of 64 inside the call), to see whether the number of distinct buffer sizes is what grows the pool;
with --designs also the kernels under each memory design (size classes, the states recomputed, both; docs/series_memory.md).
"""
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from macai.gated_delta import chunk_gated_delta_rule, op  # noqa: E402

H, D, LAYERS, MICRO = 32, 128, 24, 24
GB = 2 ** 30


def padded64(fn):
    def call(q, k, v, g, beta, **kw):
        T = q.shape[1]
        pad = (-T) % 64
        if pad:   # trailing tokens with k = 0 and beta = 0 leave the state untouched; their outputs are dropped
            q, k, v = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v))
            g, beta = (F.pad(x, (0, 0, 0, pad)) for x in (g, beta))
        o, s = fn(q, k, v, g, beta, **kw)
        return o[:, :T], s
    return call


def run(fn, lengths):
    torch.mps.synchronize()
    torch.mps.empty_cache()
    base_driver, base_live = torch.mps.driver_allocated_memory(), torch.mps.current_allocated_memory()
    rows = []
    for T in lengths:
        gen = torch.Generator().manual_seed(T)
        leaves = [torch.randn(1, T, H, D, generator=gen).to("mps", torch.bfloat16).requires_grad_(True) for _ in range(3)]
        g = (-torch.rand(1, T, H, generator=gen)).to("mps").requires_grad_(True)
        beta = torch.rand(1, T, H, generator=gen).to("mps", torch.bfloat16).requires_grad_(True)
        outs = [fn(*leaves, g, beta, use_qk_l2norm_in_kernel=True)[0] for _ in range(LAYERS)]
        live_fwd = torch.mps.current_allocated_memory()
        torch.mps.empty_cache()          # as the training runs do (the Trainer's torch_empty_cache_steps=1)
        sum(o.float().sum() for o in outs).backward()
        del outs, leaves, g, beta
        torch.mps.synchronize()
        rows.append({"T": T, "live_at_forward_end_gb": round((live_fwd - base_live) / GB, 3),
                     "pool_gb": round((torch.mps.driver_allocated_memory() - base_driver) / GB, 3)})
    return rows


def main():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    lengths = random.Random(0).sample(range(400, 1300), MICRO)
    out = {"shape": {"heads": H, "head_dim": D, "layers": LAYERS, "micro_batches": MICRO}, "lengths": lengths, "arms": {}}
    designs = "--designs" in sys.argv[1:]
    stored = dict(SIZE_CLASSES=False, RECOMPUTE_STATES=False)
    arms = [("reference", M.torch_chunk_gated_delta_rule, stored), ("kernels", chunk_gated_delta_rule, stored),
            ("kernels, lengths padded to 64", padded64(chunk_gated_delta_rule), stored)]
    if designs:
        # the kernels' memory designs (docs/series_memory.md), op's settings in force for the whole arm
        arms += [("kernels, size classes", chunk_gated_delta_rule, dict(SIZE_CLASSES=True, RECOMPUTE_STATES=False)),
                 ("kernels, states recomputed", chunk_gated_delta_rule, dict(SIZE_CLASSES=False, RECOMPUTE_STATES=True)),
                 ("kernels, both", chunk_gated_delta_rule, dict(SIZE_CLASSES=True, RECOMPUTE_STATES=True))]
    for name, fn, values in arms:
        old = {k: getattr(op, k) for k in values}
        for k, v in values.items():
            setattr(op, k, v)
        try:
            rows = run(fn, lengths)
        finally:
            for k, v in old.items():
                setattr(op, k, v)
        out["arms"][name] = rows
        print(f"{name:30s} live at the forward's end {min(r['live_at_forward_end_gb'] for r in rows):.2f}-"
              f"{max(r['live_at_forward_end_gb'] for r in rows):.2f} GiB; pool after micro-batch 1 / 8 / {MICRO}: "
              f"{rows[0]['pool_gb']:.2f} / {rows[7]['pool_gb']:.2f} / {rows[-1]['pool_gb']:.2f} GiB", flush=True)
    folder = os.path.join(HERE, "memory_logs" if designs else "phase4_logs")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "alloc_check.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
