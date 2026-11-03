"""Series phase 4 diagnostic: how exact each gated delta rule is inside the training loop. The Trainer runs the model's forward
under torch.autocast("mps", bfloat16) (bf16 mixed precision): the reference implementation's chunk products then run in
bfloat16, while the Metal kernels compute in float32 whatever autocast says. Both arms against a float64 step-by-step reference,
with and without autocast, on the same inputs (float32 copies of bf16 values, so no final bf16 rounding hides the difference).

    python bench/phase4_autocast_check.py       # prints and writes bench/phase4_logs/autocast_check.json
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from macai.gated_delta import chunk_gated_delta_rule, l2norm, recurrent_reference  # noqa: E402

B, T, H, D = 2, 512, 16, 128
NAMES = ("o", "dq", "dk", "dv", "dg", "dbeta")


def main():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    gen = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(B, T, H, D, generator=gen).bfloat16().float() for _ in range(3))
    g = -torch.rand(B, T, H, generator=gen) * 0.5
    beta = torch.rand(B, T, H, generator=gen).bfloat16().float()
    do = torch.randn(B, T, H, D, generator=gen)
    xs = (q, k, v, g, beta)
    x64 = [x.double().requires_grad_(True) for x in xs]
    o64, _ = recurrent_reference(l2norm(x64[0]) * D ** -0.5, l2norm(x64[1]), x64[2], x64[3], x64[4],
                                 torch.zeros(B, H, D, D, dtype=torch.float64))
    (o64 * do.double()).sum().backward()
    truth = [o64.detach()] + [x.grad for x in x64]

    def errors(fn, autocast):
        xm = [x.to("mps").requires_grad_(True) for x in xs]
        with torch.autocast("mps", dtype=torch.bfloat16, enabled=autocast):
            o, _ = fn(*xm, use_qk_l2norm_in_kernel=True)
        (o.float() * do.to("mps")).sum().backward()
        got = [o.detach().cpu().double()] + [x.grad.cpu().double() for x in xm]
        return {n: float((a - b).abs().max() / b.abs().max()) for n, a, b in zip(NAMES, got, truth)}

    out = {"shape": [B, T, H, D], "torch": torch.__version__, "measure": "max abs error / max abs value, against float64",
           "results": {}}
    for name, fn in (("reference", M.torch_chunk_gated_delta_rule), ("kernels", chunk_gated_delta_rule)):
        for ac in (False, True):
            e = errors(fn, ac)
            out["results"][f"{name}, autocast {'on' if ac else 'off'}"] = e
            print(f"{name:9s} autocast {'on ' if ac else 'off'}: " + ", ".join(f"{n} {x:.1e}" for n, x in e.items()), flush=True)
    os.makedirs(os.path.join(HERE, "phase4_logs"), exist_ok=True)
    with open(os.path.join(HERE, "phase4_logs", "autocast_check.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
