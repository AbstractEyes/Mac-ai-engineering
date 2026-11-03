"""Series D3-D5, step 0 (c): the matrix products of the readers' training step, PyTorch on the Apple GPU (bf16 in, bf16 out, as
autocast runs them) against MLX (mx.matmul, bf16), read against the large-square rate measured in the same session
(docs/series_layer.md).

    python bench/profile_products.py --shapes 0.8b=4x1024,2x903 --shapes 9b=2x1024,1x859          # PyTorch: products_torch.json
    python bench/profile_products.py --mlx --shapes 0.8b=4x1024,2x903 --shapes 9b=2x1024,1x859    # MLX (its own process, after)
    python bench/profile_products.py --square-forms      # the large-square rate per form (X W, X W^T): products_square_forms.json

Each shape is batch x tokens: the bench's (first) and the training micro-batch's median padded one. At each, for every
projection (in -> out) of both models, the products autograd runs in LoRA training with frozen weights:

  base fwd       Y = X W^T                         F.linear(X [B, T, in], W [out, in])
  base bwd       dX = dY W                         the linear's backward to its input (no weight gradient: frozen)
  lora_A fwd     H = X A^T                         rank 32
  lora_B fwd     Y = H B^T
  lora_B bwd     dH = dY B and dB = dY^T H         the gradients to its input and its weight
  lora_A bwd     dX = dH A and dA = dH^T X

and the LM head (forward, and the backward to its input, against 248,320 words). For in_proj_qkv, whose output the layer
transposes for the convolution path, the backward products also run with the gradient in the layout the layer hands them (the
transpose of a contiguous [B, out, T] tensor). The large-square products (2,048, 4,096 and 8,192 cubed, at the start and the
end of the session) are the control; the fastest of them is the rate the shapes are read against.

PyTorch: Metal events (macai.instruments.gpu_ms: back-to-back calls between two events) and the wall clock around the same
calls. MLX: the wall clock around mx.eval of the same number of calls, the backward products written as the matmuls autograd
runs. Each case is called often enough to fill at least ~10 ms, and at least 3 times back to back; the median of 7 rounds.
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {"0.8b": "Qwen/Qwen3.5-0.8B", "9b": "Qwen/Qwen3.5-9B"}
RANK = 32
SQUARES = (2048, 4096, 8192)
FILL_MS = 10.0
REPS = 7


def text_config(model_id):
    from huggingface_hub import hf_hub_download
    cfg = json.load(open(hf_hub_download(model_id, "config.json")))
    return cfg.get("text_config", cfg)


def projections(tc) -> list:
    """(name, in, out, LoRA target, layers using it) for both decoder layer types, and the LM head."""
    h, i = tc["hidden_size"], tc["intermediate_size"]
    kd = tc["linear_key_head_dim"] * tc["linear_num_key_heads"]
    vd = tc["linear_value_head_dim"] * tc["linear_num_value_heads"]
    hv = tc["linear_num_value_heads"]
    nq, nkv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
    gate = 2 if tc.get("attn_output_gate", True) else 1
    n_lin = sum(t == "linear_attention" for t in tc["layer_types"])
    n_full = len(tc["layer_types"]) - n_lin
    n = len(tc["layer_types"])
    return [("in_proj_qkv", h, 2 * kd + vd, True, n_lin), ("in_proj_z", h, vd, True, n_lin),
            ("in_proj_a / in_proj_b", h, hv, False, 2 * n_lin), ("out_proj", vd, h, True, n_lin),
            ("q_proj", h, nq * hd * gate, True, n_full), ("k_proj / v_proj", h, nkv * hd, True, 2 * n_full),
            ("o_proj", nq * hd, h, True, n_full), ("gate_proj / up_proj", h, i, True, 2 * n), ("down_proj", i, h, True, n),
            ("lm_head", h, tc["vocab_size"], False, 1)]


def cases(tc) -> list:
    """(projection, product, in, out, layout, uses a step): the products of one call."""
    out = []
    for name, k_in, n_out, lora, uses in projections(tc):
        out.append((name, "base fwd", k_in, n_out, "plain", uses))
        out.append((name, "base bwd", k_in, n_out, "plain", uses))
        if name == "in_proj_qkv":
            out.append((name, "base bwd", k_in, n_out, "layer", uses))
        if lora:
            for prod in ("lora_A fwd", "lora_B fwd", "lora_B bwd", "lora_A bwd"):
                out.append((name, prod, k_in, n_out, "plain", uses))
            if name == "in_proj_qkv":
                out.append((name, "lora_B bwd", k_in, n_out, "layer", uses))
    return out


def flops(prod, m, k_in, n_out):
    if prod in ("base fwd", "base bwd"):
        return 2 * m * k_in * n_out
    if prod in ("lora_A fwd",):
        return 2 * m * k_in * RANK
    if prod in ("lora_B fwd",):
        return 2 * m * RANK * n_out
    if prod == "lora_B bwd":
        return 4 * m * RANK * n_out
    if prod == "lora_A bwd":
        return 4 * m * RANK * k_in
    raise ValueError(prod)


def fill(once_ms):
    return max(3, min(400, int(FILL_MS / max(once_ms, 1e-3)) + 1))


# ---- PyTorch ----
def torch_main(a):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, os.path.join(HERE, "..", "src"))
    from macai.instruments import gpu_ms, warm_gpu
    dev, bf = "mps", torch.bfloat16
    gen = torch.Generator().manual_seed(0)

    def rnd(*s, grad=False):
        return ((torch.randn(*s, generator=gen) * 0.5).to(dev, bf)).requires_grad_(grad)

    def grad_out(B, T, n, layout):
        return rnd(B, n, T).transpose(1, 2) if layout == "layer" else rnd(B, T, n)

    def build(prod, B, T, k_in, n_out, layout):
        if prod == "base fwd":
            x, w = rnd(B, T, k_in), rnd(n_out, k_in)
            return lambda: F.linear(x, w)
        if prod == "base bwd":
            x, w = rnd(B, T, k_in, grad=True), rnd(n_out, k_in)
            y, g = F.linear(x, w), grad_out(B, T, n_out, layout)
            return lambda: torch.autograd.grad(y, x, g, retain_graph=True)
        if prod == "lora_A fwd":
            x, w = rnd(B, T, k_in), rnd(RANK, k_in)
            return lambda: F.linear(x, w)
        if prod == "lora_B fwd":
            h, w = rnd(B, T, RANK), rnd(n_out, RANK)
            return lambda: F.linear(h, w)
        if prod == "lora_B bwd":
            h, w = rnd(B, T, RANK, grad=True), rnd(n_out, RANK, grad=True)
            y, g = F.linear(h, w), grad_out(B, T, n_out, layout)
            return lambda: torch.autograd.grad(y, (h, w), g, retain_graph=True)
        if prod == "lora_A bwd":
            x, w = rnd(B, T, k_in, grad=True), rnd(RANK, k_in, grad=True)
            y, g = F.linear(x, w), rnd(B, T, RANK)
            return lambda: torch.autograd.grad(y, (x, w), g, retain_graph=True)
        raise ValueError(prod)

    def time_case(fn, fl):
        for _ in range(3):
            fn()
        torch.mps.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        inner = fill((time.perf_counter() - t0) * 1e3)
        ev = gpu_ms(fn, reps=REPS, warm=1, inner=inner)
        walls = []
        for _ in range(REPS):
            torch.mps.synchronize()
            t0 = time.perf_counter()
            for _ in range(inner):
                fn()
            torch.mps.synchronize()
            walls.append((time.perf_counter() - t0) * 1e3 / inner)
        w = statistics.median(walls)
        return {"events_ms": ev["median_ms"], "wall_ms": round(w, 4), "inner": inner,
                "tflops_events": round(fl / ev["median_ms"] / 1e9, 3), "tflops_wall": round(fl / w / 1e9, 3)}

    out = {"framework": f"torch {torch.__version__}", "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "reps": REPS,
           "rank": RANK, "squares": {}, "models": {}}
    warm_gpu(2.0)

    def squares():
        res = {}
        for n in SQUARES:
            x, w = rnd(n, n), rnd(n, n)
            res[str(n)] = time_case(lambda: F.linear(x, w), 2 * n ** 3)
            del x, w
        return res

    out["squares"] = squares()
    print("squares", json.dumps(out["squares"]), flush=True)
    ref = max(v["tflops_events"] for v in out["squares"].values())
    out["large_square_tflops"] = ref
    for model, shapes in a.shapes.items():
        tc = text_config(MODELS[model])
        out["models"][model] = {}
        for B, T in shapes:
            rows = []
            for name, prod, k_in, n_out, layout, uses in cases(tc):
                fn = build(prod, B, T, k_in, n_out, layout)
                fl = flops(prod, B * T, k_in, n_out)
                r = time_case(fn, fl)
                r.update(projection=name, product=prod, layout=layout, in_features=k_in, out_features=n_out, calls_a_pass=uses,
                         flops=fl, of_large_square=round(r["tflops_events"] / ref, 4))
                rows.append(r)
                print(model, f"{B}x{T}", name, prod, layout, json.dumps({k: r[k] for k in (
                    "events_ms", "wall_ms", "tflops_events", "of_large_square")}), flush=True)
                del fn
                torch.mps.empty_cache()
            out["models"][model][f"{B}x{T}"] = rows
    out["squares_end"] = squares()
    print("squares at the end", json.dumps(out["squares_end"]), flush=True)
    return out


# ---- MLX ----
def mlx_main(a):
    import mlx.core as mx
    mx.random.seed(0)
    bf = mx.bfloat16

    def rnd(*s):
        return (mx.random.normal(s) * 0.5).astype(bf)

    def grad_out(B, T, n, layout):
        return mx.swapaxes(rnd(B, n, T), 1, 2) if layout == "layer" else rnd(B, T, n)

    def build(prod, B, T, k_in, n_out, layout):
        M = B * T
        if prod == "base fwd":
            x, w = rnd(B, T, k_in), rnd(n_out, k_in)
            return lambda: mx.matmul(x, w.T)
        if prod == "base bwd":
            g, w = grad_out(B, T, n_out, layout), rnd(n_out, k_in)
            return lambda: mx.matmul(g, w)
        if prod == "lora_A fwd":
            x, w = rnd(B, T, k_in), rnd(RANK, k_in)
            return lambda: mx.matmul(x, w.T)
        if prod == "lora_B fwd":
            h, w = rnd(B, T, RANK), rnd(n_out, RANK)
            return lambda: mx.matmul(h, w.T)
        if prod == "lora_B bwd":
            h, w, g = rnd(B, T, RANK), rnd(n_out, RANK), grad_out(B, T, n_out, layout)
            return lambda: [mx.matmul(g, w), mx.matmul(g.reshape(M, n_out).T, h.reshape(M, RANK))]
        if prod == "lora_A bwd":
            x, w, g = rnd(B, T, k_in), rnd(RANK, k_in), rnd(B, T, RANK)
            return lambda: [mx.matmul(g, w), mx.matmul(g.reshape(M, RANK).T, x.reshape(M, k_in))]
        raise ValueError(prod)

    def ev(o):
        mx.eval(o)

    def time_case(fn, fl):
        for _ in range(3):
            ev(fn())
        t0 = time.perf_counter()
        ev(fn())
        inner = fill((time.perf_counter() - t0) * 1e3)
        walls = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            ev([fn() for _ in range(inner)])
            walls.append((time.perf_counter() - t0) * 1e3 / inner)
        w = statistics.median(walls)
        return {"wall_ms": round(w, 4), "inner": inner, "tflops_wall": round(fl / w / 1e9, 3)}

    out = {"framework": f"mlx {mx.__version__}", "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "reps": REPS, "rank": RANK,
           "models": {}}

    def squares():
        res = {}
        for n in SQUARES:
            x, w = rnd(n, n), rnd(n, n)
            res[str(n)] = time_case(lambda: mx.matmul(x, w.T), 2 * n ** 3)
        return res

    out["squares"] = squares()
    print("squares", json.dumps(out["squares"]), flush=True)
    ref = max(v["tflops_wall"] for v in out["squares"].values())
    out["large_square_tflops"] = ref
    for model, shapes in a.shapes.items():
        tc = text_config(MODELS[model])
        out["models"][model] = {}
        for B, T in shapes:
            rows = []
            for name, prod, k_in, n_out, layout, uses in cases(tc):
                fn = build(prod, B, T, k_in, n_out, layout)
                fl = flops(prod, B * T, k_in, n_out)
                r = time_case(fn, fl)
                r.update(projection=name, product=prod, layout=layout, in_features=k_in, out_features=n_out, calls_a_pass=uses,
                         flops=fl, of_large_square=round(r["tflops_wall"] / ref, 4))
                rows.append(r)
                print(model, f"{B}x{T}", name, prod, layout, json.dumps({k: r[k] for k in (
                    "wall_ms", "tflops_wall", "of_large_square")}), flush=True)
            out["models"][model][f"{B}x{T}"] = rows
    out["squares_end"] = squares()
    print("squares at the end", json.dumps(out["squares_end"]), flush=True)
    return out


def square_forms_main():
    """The large-square rate in the forms PyTorch can run it: X W (torch.mm, a new output or a preallocated one), X W^T
    (F.linear, the form nn.Linear runs) and X W^T through torch.mm on the transposed view. Metal events, back-to-back calls."""
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, os.path.join(HERE, "..", "src"))
    from macai.instruments import gpu_ms, warm_gpu
    warm_gpu(2.0)
    out = {"framework": f"torch {torch.__version__}", "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "reps": REPS, "forms": {}}
    for n in SQUARES:
        x = torch.randn(n, n, device="mps", dtype=torch.bfloat16)
        w = torch.randn(n, n, device="mps", dtype=torch.bfloat16) * n ** -0.5
        y = torch.empty(n, n, device="mps", dtype=torch.bfloat16)
        forms = {"X W (torch.mm)": lambda: torch.mm(x, w), "X W (torch.mm, out=)": lambda: torch.mm(x, w, out=y),
                 "X W^T (F.linear)": lambda: F.linear(x, w), "X W^T (torch.mm on w.t())": lambda: torch.mm(x, w.t())}
        res = {}
        for _ in range(2):                  # two rounds, interleaved
            for name, fn in forms.items():
                fn()
                torch.mps.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.mps.synchronize()
                inner = fill((time.perf_counter() - t0) * 1e3)
                ms = gpu_ms(fn, reps=REPS, warm=1, inner=inner)["median_ms"]
                res.setdefault(name, []).append(round(2 * n ** 3 / ms / 1e9, 3))
        out["forms"][str(n)] = res
        print(n, json.dumps(res), flush=True)
        del x, w, y
        torch.mps.empty_cache()
    out["ended"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    path = os.path.join(HERE, "layer_logs", "products_square_forms.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path, flush=True)


def main():
    print(f"[products] pid {os.getpid()}", flush=True)
    if "--square-forms" in sys.argv[1:]:
        return square_forms_main()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx", action="store_true")
    ap.add_argument("--shapes", action="append", required=True, help="model=BxT,BxT (the bench's shape first)")
    a = ap.parse_args()
    a.shapes = {k: [tuple(int(x) for x in s.split("x")) for s in v.split(",")] for k, v in (t.split("=") for t in a.shapes)}
    out = mlx_main(a) if a.mlx else torch_main(a)
    out["shapes"] = {k: [f"{b}x{t}" for b, t in v] for k, v in a.shapes.items()}
    out["ended"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    path = os.path.join(HERE, "layer_logs", "products_mlx.json" if a.mlx else "products_torch.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
