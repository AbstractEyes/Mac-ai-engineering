"""Series design D8 (docs/series_layer.md): does the Apple GPU's bf16 product give a row the same bits whatever the number of rows
in the product? The LM head on the rows gathered at random positions (19% of them, as the labelled share) against the head on
every position, at Qwen3.5-0.8B's (2 x 811, hidden 1,024) and 9B's (1 x 768, hidden 4,096) micro-batch shapes, vocabulary
248,320, bf16 autocast; and the gradient to the head's input, dE = dZ W, with dZ zero at the other rows. Prints the elements
that differ. The record of 2026-11-03 15:07 PST: bench/layer_logs/d8_head_rows.log (run under the guard, before D8's code).

    python bench/probe_head_rows.py
"""
import os
import time
print(f"[probe] pid {os.getpid()}", flush=True)
import torch
import torch.nn.functional as F

torch.manual_seed(0)
dev = "mps"
for H, T, B in ((1024, 811, 2), (4096, 768, 1)):
    V = 248320
    W = (torch.randn(V, H) * 0.02).to(dev, torch.bfloat16)
    x = torch.randn(B, T, H).to(dev, torch.bfloat16)
    keep = torch.rand(B, T, device=dev) < 0.19
    with torch.autocast(dev, torch.bfloat16):
        full = F.linear(x, W)                 # [B, T, V]
        part = F.linear(x[keep], W)           # [N, V]
    a = full[keep]
    diff = (a != part)
    print(f"H {H} B {B} T {T}: N {int(keep.sum())}; forward rows: differing elements {int(diff.sum())} of {diff.numel()}; "
          f"max abs diff {float((a.float() - part.float()).abs().max()):.3g}", flush=True)
    # the gradient to the input: dZ (zeros at the other rows) @ W
    dz = torch.zeros(B, T, V, device=dev, dtype=torch.bfloat16)
    g = (torch.randn(int(keep.sum()), V, device=dev) * 1e-4).to(torch.bfloat16)
    dz[keep] = g
    de_full = dz.reshape(-1, V) @ W
    de_part = g @ W
    b = de_full.reshape(B, T, H)[keep]
    d2 = (b != de_part)
    other = de_full.reshape(B, T, H)[~keep]
    print(f"   dE rows: differing {int(d2.sum())} of {d2.numel()}; max abs diff {float((b.float() - de_part.float()).abs().max()):.3g};"
          f" other rows all zero {bool((other == 0).all())}", flush=True)
    del W, x, full, part, dz, de_full
    torch.mps.empty_cache()
