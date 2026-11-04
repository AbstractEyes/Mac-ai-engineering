"""Series design D5 (b), the LoRA adapter path fused (macai.lora; docs/series_layer.md): PEFT's LoRA linear layer with the fused
path off and on at every adapted projection's shape of Qwen3.5-0.8B and 9B, at the token counts of the readers' micro-batches
(0.8B 2 x 811 = 1,622 positions, 9B 1 x 768: the step profiles' averages), forward and forward + backward, timed by Metal events
in interleaved rounds.

    python bench/bench_lora.py                     # bench/layer_logs/d5_adapter.json

Each projection is one PEFT LoRA layer as the readers train it: an nn.Linear base in bf16 (frozen), rank 32, alpha 64, dropout
0.05 (training mode: a fresh mask each call), the adapters in float32 with B drawn (nonzero, as after the first steps), x in bf16
requiring a gradient, under bf16 autocast; the backward gives the gradients to x, A and B. Three variants alternate in every
round, their order reversed each round: PEFT's forward ('peft'), the fused path ('fused'), and the base product alone ('base':
F.linear and its input gradient). The adapter path is the layer less its base: peft - base, fused - base. Each timing is
back-to-back calls filling at least ~10 ms, the median of 7 (bench/profile_layer.py's timed); the reading is the median over the
rounds. The totals weight each projection by its calls in a micro-batch (its layers).

Beside them, at each shape: the fused path's own parts (torch's dropout call; the up kernel; the down kernel) and the rank-32
products it leaves to PyTorch (P = x_d A^T, G = dY B, dB = dY^T P, dA = G^T x_d) with their rates, against the large-square
rate measured at the start and the end of the session (the registered prediction: the rank-32 products <= 30% of it).
"""
import argparse
import json
import os
import statistics
import sys
import time

print(f"[bench_lora] pid {os.getpid()}", flush=True)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import profile_layer as PL  # noqa: E402
import profile_products as PP  # noqa: E402
from macai import lora, patch  # noqa: E402
from macai.instruments import gpu_ms, warm_gpu  # noqa: E402
from macai.lora import _fused, _kernels  # noqa: E402

DEV, BF = "mps", torch.bfloat16
ROUNDS = 5
RANK, ALPHA, P_DROP = 32, 64, 0.05
SHAPES = {"0.8b": (2, 811), "9b": (1, 768)}


def projections(tc):
    """(name, d_in, d_out, calls a micro-batch) for every adapted projection, from the model's config."""
    h, i = tc["hidden_size"], tc["intermediate_size"]
    kd = tc["linear_key_head_dim"] * tc["linear_num_key_heads"]
    vd = tc["linear_value_head_dim"] * tc["linear_num_value_heads"]
    nq, nkv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
    gate = 2 if tc.get("attn_output_gate", True) else 1
    n_lin = sum(t == "linear_attention" for t in tc["layer_types"])
    n_full = len(tc["layer_types"]) - n_lin
    return [("lin.in_proj_qkv", h, 2 * kd + vd, n_lin), ("lin.in_proj_z", h, vd, n_lin), ("lin.out_proj", vd, h, n_lin),
            ("lin.gate_proj", h, i, n_lin), ("lin.up_proj", h, i, n_lin), ("lin.down_proj", i, h, n_lin),
            ("full.q_proj", h, nq * hd * gate, n_full), ("full.k_proj", h, nkv * hd, n_full), ("full.v_proj", h, nkv * hd, n_full),
            ("full.o_proj", nq * hd, h, n_full), ("full.gate_proj", h, i, n_full), ("full.up_proj", h, i, n_full),
            ("full.down_proj", i, h, n_full)]


class Net(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj = torch.nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        return self.proj(x)


def build(d_in, d_out, B, T, seed):
    from peft import LoraConfig, inject_adapter_in_model
    torch.manual_seed(seed)
    net = Net(d_in, d_out)
    torch.nn.init.normal_(net.proj.weight, std=0.02)
    net = inject_adapter_in_model(LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=P_DROP, target_modules=["proj"]), net)
    for n, p in net.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.02)
    net = net.to(DEV)
    net.proj.base_layer.to(BF)
    for n, p in net.named_parameters():
        p.requires_grad_("lora_" in n)
    net.train()
    gen = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(B, T, d_in, generator=gen).to(DEV, BF).requires_grad_(True)
    gy = (torch.randn(B, T, d_out, generator=gen) * 0.1).to(DEV, BF)
    return net, x, gy


class Variant:
    def __init__(self, fn, wants, gy):
        self.fn, self.wants, self.gy = fn, wants, gy

    def run(self, back):
        y = self.fn()
        if back:
            torch.autograd.grad(y, self.wants, self.gy)


def interleaved(variants: dict, rounds: int) -> dict:
    names = list(variants)
    got = {n: {"fwd": [], "fwd_bwd": []} for n in names}
    for r in range(rounds):
        for n in (names if r % 2 == 0 else names[::-1]):
            v = variants[n]
            got[n]["fwd"].append(PL.timed(lambda: v.run(False))[0])
            got[n]["fwd_bwd"].append(PL.timed(lambda: v.run(True))[0])
    return {n: {"fwd_ms": round(statistics.median(g["fwd"]), 4), "fwd_bwd_ms": round(statistics.median(g["fwd_bwd"]), 4),
                "rounds": g} for n, g in got.items()}


def shape_case(d_in, d_out, B, T, rounds, seed):
    net, x, gy = build(d_in, d_out, B, T, seed)
    m = net.proj
    a, b = m.lora_A["default"].weight, m.lora_B["default"].weight
    w = m.base_layer.weight

    def module(on):
        def fn():
            with lora.configured(FUSED=on), torch.autocast(DEV, dtype=BF):
                return m(x)
        return fn

    def base():
        with torch.autocast(DEV, dtype=BF):
            return F.linear(x, w)
    r = interleaved({"peft": Variant(module(False), [x, a, b], gy), "fused": Variant(module(True), [x, a, b], gy),
                     "base": Variant(base, [x], gy)}, rounds)
    for k in ("fwd_ms", "fwd_bwd_ms"):
        r["peft_adapter_" + k] = round(r["peft"][k] - r["base"][k], 4)
        r["fused_adapter_" + k] = round(r["fused"][k] - r["base"][k], 4)
    r["adapter_speedup_fwd_bwd"] = round(r["peft_adapter_fwd_bwd_ms"] / r["fused_adapter_fwd_bwd_ms"], 3)
    r["adapter_speedup_fwd"] = round(r["peft_adapter_fwd_ms"] / r["fused_adapter_fwd_ms"], 3)

    # the fused path's parts, and the rank-32 products it leaves to PyTorch (bf16, as under autocast)
    n = B * T
    with torch.no_grad():
        xb = x.detach()
        xd, mask = _fused.dropout(xb, P_DROP)
        xd2 = xd.reshape(n, d_in)
        ac, bc = a.detach().to(BF), b.detach().to(BF)
        pm = torch.mm(xd2, ac.t())
        res = F.linear(xb, w).reshape(n, d_out)
        dy2 = gy.reshape(n, d_out)
        g = torch.mm(dy2, bc)
        parts = {"dropout (torch.native_dropout)": lambda: _fused.dropout(xb, P_DROP),
                 "up kernel (R + s P B^T)": lambda: _kernels.up(pm, bc, res, 2.0),
                 "down kernel (M (G A) / (1 - p))": lambda: _kernels.down(g, ac, mask, 1 / (1 - P_DROP), BF)}
        prods = {"P = x_d A^T": (lambda: torch.mm(xd2, ac.t()), 2 * n * d_in * RANK),
                 "G = dY B": (lambda: torch.mm(dy2, bc), 2 * n * d_out * RANK),
                 "dB = dY^T P": (lambda: torch.mm(dy2.t(), pm), 2 * n * d_out * RANK),
                 "dA = G^T x_d": (lambda: torch.mm(g.t(), xd2), 2 * n * d_in * RANK)}
        r["parts_ms"] = {k: PL.timed(fn)[0] for k, fn in parts.items()}
        r["products"] = {}
        for k, (fn, fl) in prods.items():
            ms = PL.timed(fn)[0]
            r["products"][k] = {"ms": ms, "flops": fl, "tflops": round(fl / ms / 1e9, 3)}
        # each kernel's least traffic: up reads R and writes Y; down writes dx and reads the mask (P, G and the weights are small)
        el = res.element_size()
        r["floors_ms"] = {"up kernel": round(2 * n * d_out * el / 356.1e6, 4),
                          "down kernel": round(n * d_in * (el + 1) / 356.1e6, 4)}
    del net, x, gy
    torch.mps.empty_cache()
    return r


def squares():
    out = {}
    for n in (2048, 4096):
        x, w = torch.randn(n, n, device=DEV, dtype=BF), torch.randn(n, n, device=DEV, dtype=BF)
        ms = gpu_ms(lambda: F.linear(x, w), reps=7, warm=2, inner=5)["median_ms"]
        out[str(n)] = round(2 * n ** 3 / ms / 1e9, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="0.8b,9b")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--out", default=os.path.join(HERE, "layer_logs", "d5_adapter.json"))
    a = ap.parse_args()
    patch.install()
    rec = {"torch": torch.__version__, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "rounds": a.rounds, "shapes": SHAPES,
           "lora": dict(rank=RANK, alpha=ALPHA, dropout=P_DROP), "lora_patches": patch.lora_patches(), "models": {}}
    import peft
    rec["peft"] = peft.__version__
    warm_gpu(2.0)
    rec["squares_start"] = squares()
    print("large squares (F.linear) TFLOP/s", rec["squares_start"], flush=True)
    for model in a.models.split(","):
        B, T = SHAPES[model]
        tc = PP.text_config(PP.MODELS[model])
        done, rows = {}, []
        for name, d_in, d_out, calls in projections(tc):
            if (d_in, d_out) not in done:
                done[(d_in, d_out)] = shape_case(d_in, d_out, B, T, a.rounds, seed=len(done))
                r = done[(d_in, d_out)]
                print(f"{model} {d_in}->{d_out}: peft {r['peft']['fwd_bwd_ms']:.3f} fused {r['fused']['fwd_bwd_ms']:.3f} base "
                      f"{r['base']['fwd_bwd_ms']:.3f} ms; adapter path {r['peft_adapter_fwd_bwd_ms']:.3f} -> "
                      f"{r['fused_adapter_fwd_bwd_ms']:.3f} ms (x{r['adapter_speedup_fwd_bwd']}); parts {r['parts_ms']}; "
                      f"products {[v['tflops'] for v in r['products'].values()]} TFLOP/s", flush=True)
            rows.append({"projection": name, "d_in": d_in, "d_out": d_out, "calls": calls, "shape": f"{d_in}->{d_out}"})
        tot = {k: 0.0 for k in ("peft_adapter_fwd_bwd_ms", "fused_adapter_fwd_bwd_ms", "peft_adapter_fwd_ms",
                                "fused_adapter_fwd_ms", "peft_fwd_bwd_ms", "fused_fwd_bwd_ms", "base_fwd_bwd_ms")}
        for row in rows:
            r = done[(row["d_in"], row["d_out"])]
            row.update({k: r[k] for k in ("peft_adapter_fwd_bwd_ms", "fused_adapter_fwd_bwd_ms", "peft_adapter_fwd_ms",
                                          "fused_adapter_fwd_ms", "adapter_speedup_fwd_bwd")})
            for k in ("peft_adapter_fwd_bwd_ms", "fused_adapter_fwd_bwd_ms", "peft_adapter_fwd_ms", "fused_adapter_fwd_ms"):
                tot[k] += row["calls"] * r[k]
            tot["peft_fwd_bwd_ms"] += row["calls"] * r["peft"]["fwd_bwd_ms"]
            tot["fused_fwd_bwd_ms"] += row["calls"] * r["fused"]["fwd_bwd_ms"]
            tot["base_fwd_bwd_ms"] += row["calls"] * r["base"]["fwd_bwd_ms"]
        tot = {k: round(v, 2) for k, v in tot.items()}
        tot["adapter_speedup_fwd_bwd"] = round(tot["peft_adapter_fwd_bwd_ms"] / tot["fused_adapter_fwd_bwd_ms"], 3)
        tot["adapter_speedup_fwd"] = round(tot["peft_adapter_fwd_ms"] / tot["fused_adapter_fwd_ms"], 3)
        rec["models"][model] = {"micro_batch": f"{B}x{T}", "projections": rows,
                                "shapes": {f"{k[0]}->{k[1]}": v for k, v in done.items()}, "per_micro_batch": tot}
        print(f"{model} a micro-batch: the adapter path {tot['peft_adapter_fwd_bwd_ms']} -> {tot['fused_adapter_fwd_bwd_ms']} ms "
              f"(x{tot['adapter_speedup_fwd_bwd']}); forward {tot['peft_adapter_fwd_ms']} -> {tot['fused_adapter_fwd_ms']} ms",
              flush=True)
    rec["squares_end"] = squares()
    print("large squares at the end", rec["squares_end"], flush=True)
    rec["ended"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
