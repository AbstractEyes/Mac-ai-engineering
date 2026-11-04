"""Time the gated delta rule on the Mac's GPU: the function alone and a whole Qwen3.5 linear-attention decoder layer, transformers'
reference path against the Metal kernels (each kernel choice: the token-at-a-time D0 pair and the chunked D1 + D2 pair),
forward and forward + backward, at LoRA-training shapes (the layer's weights frozen, the gradient flowing to its input).

    python bench/bench_gated_delta.py            # writes bench/results/gated_delta_<stamp>.json
    python bench/bench_gated_delta.py --memory   # the memory designs only: bench/memory_logs/function_cost.json

The layer is timed with the rule's kernels alone (macai.patch.installed(layer=False)): the layer parts of macai.layer are
measured by bench/bench_layer.py.

--memory times the rule (forward, forward + backward) under each memory design of the default kernels (op.SIZE_CLASSES,
op.RECOMPUTE_STATES; docs/series_memory.md) in interleaved rounds, and measures what one forward keeps for its backward.
"""
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from macai import patch  # noqa: E402
from macai.gated_delta import chunk_gated_delta_rule, op  # noqa: E402
from macai.instruments import warm_gpu  # noqa: E402

SHAPES = {"qwen3.5-0.8b": dict(model="Qwen/Qwen3.5-0.8B", batch=4), "qwen3.5-9b": dict(model="Qwen/Qwen3.5-9B", batch=2)}
T, WARM, REPS = 1024, 2, 5
# op settings: the 0.2.0 drop-in (token-at-a-time kernels, the l2 normalization as PyTorch operations); the chunked kernels
# (D1 + D2) with the same PyTorch normalization; the chunked kernels with the fused query / key prologue. The chunked choices keep
# Phase 3's chunking (chunk 16 forwards, no value split; series design D3's defaults are measured by bench/bench_d3.py)
PHASE3 = dict(FWD_CHUNK=16, VALUE_SPLIT=1)
CHOICES = {"d0": dict(KERNEL="auto", KERNEL_BWD="column", FUSED_NORM=False),
           "chunked": dict(KERNEL="chunked", KERNEL_BWD="chunked", FUSED_NORM=False, **PHASE3),
           "chunked_fused_norm": dict(KERNEL="chunked", KERNEL_BWD="chunked", FUSED_NORM=True, **PHASE3)}
# the memory designs, each on the default kernels (chunked, fused prologue)
MEMORY = {name: dict(CHOICES["chunked_fused_norm"], SIZE_CLASSES=c, RECOMPUTE_STATES=r)
          for name, (c, r) in {"stored": (False, False), "classes": (True, False), "recompute": (False, True),
                               "classes+recompute": (True, True)}.items()}
MEMORY_ROUNDS = 5


def timed(fn, x0):
    res = {}
    for back in (False, True):
        def once():
            x = x0.detach().clone().requires_grad_(back)
            if back:
                fn(x).float().sum().backward()
            else:
                with torch.no_grad():
                    fn(x)
        for _ in range(WARM):
            once()
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            once()
        torch.mps.synchronize()
        res["fwd_bwd_ms" if back else "fwd_ms"] = round((time.perf_counter() - t0) / REPS * 1e3, 2)
    return res


def timed_function(fn, base):
    """The function with every input needing its gradient, as inside a training layer."""
    res = {}
    for back in (False, True):
        def once():
            xs = [x.detach().clone().requires_grad_(back) for x in base]
            if back:
                fn(*xs, use_qk_l2norm_in_kernel=True)[0].float().sum().backward()
            else:
                with torch.no_grad():
                    fn(*xs, use_qk_l2norm_in_kernel=True)
        for _ in range(WARM):
            once()
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            once()
        torch.mps.synchronize()
        res["fwd_bwd_ms" if back else "fwd_ms"] = round((time.perf_counter() - t0) / REPS * 1e3, 2)
    return res


class settings:
    """Set op's module settings (KERNEL, KERNEL_BWD, FUSED_NORM) inside a block."""

    def __init__(self, values):
        self.values = values

    def __enter__(self):
        self.old = {k: getattr(op, k) for k in self.values}
        for k, v in self.values.items():
            setattr(op, k, v)

    def __exit__(self, *exc):
        for k, v in self.old.items():
            setattr(op, k, v)


def text_config(model_id):
    from huggingface_hub import hf_hub_download
    cfg = json.load(open(hf_hub_download(model_id, "config.json")))
    return cfg.get("text_config", cfg)


def kept_by_forward(base) -> dict:
    """The allocator's live bytes one forward adds and keeps until its backward (its output held), and the driver's."""
    xs = [x.detach().clone().requires_grad_(True) for x in base]
    torch.mps.synchronize()
    torch.mps.empty_cache()
    live0, drv0 = torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()
    o = chunk_gated_delta_rule(*xs, use_qk_l2norm_in_kernel=True)[0]
    torch.mps.synchronize()
    out = {"kept_mib": round((torch.mps.current_allocated_memory() - live0) / 2 ** 20, 1),
           "driver_growth_mib": round((torch.mps.driver_allocated_memory() - drv0) / 2 ** 20, 1)}
    o.float().sum().backward()
    del o, xs
    torch.mps.synchronize()
    return out


def memory_main():
    """The rule's cost and what its forward keeps under each memory design, at the training shapes."""
    out = {"tokens": T, "torch": torch.__version__, "designs": MEMORY, "rounds": MEMORY_ROUNDS, "warm": WARM, "reps": REPS,
           "shapes": {}}
    warm_gpu(2.0)
    for name, spec in SHAPES.items():
        tc = text_config(spec["model"])
        B, H = spec["batch"], tc["linear_num_value_heads"]
        DK, DV = tc["linear_key_head_dim"], tc["linear_value_head_dim"]
        gen = torch.Generator().manual_seed(0)
        q0, k0 = (torch.randn(B, T, H, DK, generator=gen).to("mps", torch.bfloat16) for _ in range(2))
        v0 = torch.randn(B, T, H, DV, generator=gen).to("mps", torch.bfloat16)
        g0 = (-torch.rand(B, T, H, generator=gen)).to("mps")
        b0 = torch.rand(B, T, H, generator=gen).to("mps", torch.bfloat16)
        base = (q0, k0, v0, g0, b0)
        r = {"batch": B, "heads": H, "dk": DK, "dv": DV, "rounds": {d: [] for d in MEMORY}}
        for _ in range(MEMORY_ROUNDS):
            for design, values in MEMORY.items():
                with settings(values):
                    r["rounds"][design].append(timed_function(chunk_gated_delta_rule, base))
        for design, values in MEMORY.items():
            rows = r["rounds"][design]
            r[design] = {k: round(statistics.median(x[k] for x in rows), 2) for k in ("fwd_ms", "fwd_bwd_ms")}
            with settings(values):
                r[design].update(kept_by_forward(base))
        for design in MEMORY:
            r[design]["fwd_bwd_over_stored"] = round(r[design]["fwd_bwd_ms"] / r["stored"]["fwd_bwd_ms"], 3)
        print(name, json.dumps({d: r[d] for d in MEMORY}), flush=True)
        out["shapes"][name] = r
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_logs", "function_cost.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


def main():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    reference = M.torch_chunk_gated_delta_rule
    out = {"tokens": T, "torch": torch.__version__, "choices": CHOICES, "shapes": {}}
    warm_gpu(2.0)
    for name, spec in SHAPES.items():
        tc = text_config(spec["model"])
        B, H = spec["batch"], tc["linear_num_value_heads"]
        DK, DV = tc["linear_key_head_dim"], tc["linear_value_head_dim"]
        gen = torch.Generator().manual_seed(0)
        k0 = torch.randn(B, T, H, DK, generator=gen).to("mps", torch.bfloat16)
        v0 = torch.randn(B, T, H, DV, generator=gen).to("mps", torch.bfloat16)
        g0 = (-torch.rand(B, T, H, generator=gen)).to("mps")
        b0 = torch.rand(B, T, H, generator=gen).to("mps")
        r = {"batch": B, "heads": H, "dk": DK, "dv": DV}
        q0 = torch.randn(B, T, H, DK, generator=gen).to("mps", torch.bfloat16)
        r["function_reference"] = timed_function(reference, (q0, k0, v0, g0, b0))
        for choice, values in CHOICES.items():
            with settings(values):
                r[f"function_{choice}"] = timed_function(chunk_gated_delta_rule, (q0, k0, v0, g0, b0))
        cfg = Qwen3_5TextConfig(**tc)
        cfg._attn_implementation = "sdpa"
        layer = M.Qwen3_5DecoderLayer(cfg, cfg.layer_types.index("linear_attention")).to("mps", torch.bfloat16).train()
        for p in layer.parameters():
            p.requires_grad_(False)
        x0 = torch.randn(B, T, cfg.hidden_size, generator=gen).to("mps", torch.bfloat16)
        r["layer_reference"] = timed(lambda x: layer(x, position_embeddings=None), x0)
        for choice, values in CHOICES.items():
            with patch.installed(layer=False), settings(values):      # the rule alone (the layer parts off)
                r[f"layer_{choice}"] = timed(lambda x: layer(x, position_embeddings=None), x0)
        for part in ("function", "layer"):
            for choice in CHOICES:
                r[f"{part}_speedup_{choice}"] = {k: round(r[f"{part}_reference"][k] / r[f"{part}_{choice}"][k], 2)
                                                 for k in ("fwd_ms", "fwd_bwd_ms")}
        print(name, json.dumps(r), flush=True)
        out["shapes"][name] = r
        del layer
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"gated_delta_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    memory_main() if "--memory" in sys.argv[1:] else main()
