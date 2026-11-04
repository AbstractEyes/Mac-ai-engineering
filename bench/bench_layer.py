"""Series design D4 (the layer's surroundings fused; docs/series_layer.md): each part's PyTorch path against its kernel, and the
whole decoder layers with the parts off and on, forward and forward + backward, timed by Metal events in interleaved rounds.

    python bench/bench_layer.py                      # bench/layer_logs/d4_parts.json

At the bench shapes (Qwen3.5-0.8B 4 x 1,024 tokens, 9B 2 x 1,024), on decoder layers built from each model's config as
bench/profile_layer.py builds them (bf16 weights drawn at the scale of trained ones, frozen; bf16 autocast, as the Trainer runs
them; the gradient to the part's inputs). Each part runs on inputs laid out as the layer hands them:

  the zero-centered RMSNorm   the input norm on [B, T, hidden]; the full-attention layers' q_norm (on every other half of the
                              query projection's rows, a strided view) and k_norm
  the convolution path        the projection's output [B, T, C] to the rule's inputs as its kernels read them. The PyTorch path:
                              transformers' transpose, causal_conv1d_fn, transpose, split, reshape and repeat (9B), then what the
                              rule's drop-in does to them before its kernels (q and k made contiguous, v converted to float32);
                              also timed alone: causal_conv1d_fn on the transposed view (step 0's part)
  the gated RMSNorm           the rule's output and z as [B T H_v, 128] rows
  beta and g                  in_proj_b's and in_proj_a's outputs [B, T, H_v] to float32 beta and g (the PyTorch path includes
                              the drop-in's conversion of beta to float32)
  SwiGLU                      gate_proj's and up_proj's outputs [B, T, I]
  in_proj_qkv with LoRA       (a check of the layout) at the 0.8B micro-batch shape 2 x 903: its forward + backward with the output's
                              gradient as transformers' path hands it (the transpose of a [B, C, T] tensor) and as the fused
                              convolution path hands it ([B, T, C])

Each timing: back-to-back calls filling at least ~10 ms, the median of 7 (macai.instruments.gpu_ms, as profile_layer.py); the
PyTorch path and the kernel alternate within every round, the order flipped each round (step 0 saw the GPU's rate move by up to
~20% over minutes of continuous work); the reading is the median over the rounds. Floors: step 0's (bench/layer_logs/
layer_parts.json: each part's least traffic at 356 GB/s). The pid is printed first.
"""
import argparse
import json
import os
import statistics
import sys
import time

print(f"[bench_layer] pid {os.getpid()}", flush=True)

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import profile_layer as PL  # noqa: E402
from macai import layer, patch  # noqa: E402
from macai.instruments import warm_gpu  # noqa: E402

DEV, BF = "mps", torch.bfloat16
ROUNDS = 5
SHAPES = {"0.8b": (4, 1024), "9b": (2, 1024)}
STEP0 = {"input_layernorm": "input_layernorm", "conv": "conv path (causal_conv1d_fn + SiLU)",
         "glue": "glue (transposes, split, reshape, beta, g, repeat)", "gated_norm": "gated norm (RMSNormGated)",
         "swiglu": "SwiGLU (act(gate) * up)", "q_norm": "q_norm", "k_norm": "k_norm"}


def step0_parts():
    r = json.load(open(os.path.join(HERE, "layer_logs", "layer_parts.json"), encoding="utf-8"))
    out = {}
    for model, (B, T) in SHAPES.items():
        res = r["models"][model][f"frozen {B}x{T}"]
        rows = {p["part"]: p for kind in ("linear_attention", "full_attention") for p in res[kind]["parts"]}
        out[model] = {k: {"fwd_ms": rows[v]["fwd_ms"], "fwd_bwd_ms": rows[v]["fwd_bwd_ms"],
                          "floor_fwd_bwd_ms": rows[v]["floor_fwd_bwd_ms"]} for k, v in STEP0.items()}
        out[model]["layer_linear"] = {"fwd_ms": res["linear_attention"]["whole_fwd_ms"],
                                      "fwd_bwd_ms": res["linear_attention"]["whole_fwd_bwd_ms"]}
        out[model]["layer_full"] = {"fwd_ms": res["full_attention"]["whole_fwd_ms"],
                                    "fwd_bwd_ms": res["full_attention"]["whole_fwd_bwd_ms"]}
    return out


class Variant:
    """fn(*inputs) -> output(s), run under bf16 autocast; back: the gradient to every input that needs one."""

    def __init__(self, fn, inputs, settings=None):
        self.fn, self.inputs, self.settings = fn, inputs, settings or {}
        self.grads = None

    def run(self, back):
        with layer.configured(**self.settings), torch.autocast(DEV, dtype=BF):
            out = self.fn(*self.inputs)
        if not back:
            return
        outs = [o for o in (out if isinstance(out, (tuple, list)) else (out,)) if o.requires_grad]
        if self.grads is None:
            gen = torch.Generator().manual_seed(1)
            self.grads = [(torch.randn(o.shape, generator=gen) * 0.1).to(DEV, o.dtype) for o in outs]
        want = [x for x in self.inputs if isinstance(x, torch.Tensor) and x.requires_grad]
        torch.autograd.grad(outs, want, self.grads, allow_unused=True)


def interleaved(variants: dict, rounds=ROUNDS) -> dict:
    """{name: {fwd_ms, fwd_bwd_ms, rounds}}: each round times every variant forward and forward + backward, the variants'
    order reversed every other round."""
    names = list(variants)
    got = {n: {"fwd": [], "fwd_bwd": []} for n in names}
    for r in range(rounds):
        for n in (names if r % 2 == 0 else names[::-1]):
            v = variants[n]
            got[n]["fwd"].append(PL.timed(lambda: v.run(False))[0])
            got[n]["fwd_bwd"].append(PL.timed(lambda: v.run(True))[0])
    return {n: {"fwd_ms": round(statistics.median(g["fwd"]), 4), "fwd_bwd_ms": round(statistics.median(g["fwd_bwd"]), 4),
                "rounds": g} for n, g in got.items()}


def leaf(t, grad=True):
    return t.detach().clone().requires_grad_(grad)


def parts(model, B, T, tc):
    cfg, M, lin, full = PL.build_layers(model, tc, False)
    a = lin.linear_attn
    gen = torch.Generator().manual_seed(0)
    x = (torch.randn(B, T, cfg.hidden_size, generator=gen)).to(DEV, BF)
    kd, vd, dk, dv = a.key_dim, a.value_dim, a.head_k_dim, a.head_v_dim
    rep = a.num_v_heads // a.num_k_heads
    out = {}
    with torch.no_grad(), torch.autocast(DEV, dtype=BF):
        xn = lin.input_layernorm(x)
        mixed = a.in_proj_qkv(xn)
        z = a.in_proj_z(xn)
        bb = a.in_proj_b(xn)
        aa = a.in_proj_a(xn)
        e = lin.mlp.gate_proj(xn)
        u = lin.mlp.up_proj(xn)
    # the zero-centered RMSNorm (the input norm)
    xi = leaf(x)
    out["input_layernorm"] = {"pytorch": Variant(lin.input_layernorm, [xi], {"RMSNORM": False}),
                              "kernel": Variant(lin.input_layernorm, [xi], {"RMSNORM": True})}
    # the convolution path
    w = a.conv1d.weight

    def torch_conv(m_, full_path=True):
        y = M.causal_conv1d_fn(m_.transpose(1, 2), w.squeeze(1), a.conv1d.bias, activation=a.activation)
        if not full_path:
            return y
        y = y.transpose(1, 2)
        q, k, v = torch.split(y, [kd, kd, vd], dim=-1)
        q, k, v = q.reshape(B, T, -1, dk), k.reshape(B, T, -1, dk), v.reshape(B, T, -1, dv)
        if rep > 1:
            q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        return q.contiguous(), k.contiguous(), v.float().contiguous()       # as the rule's drop-in hands them to its kernels

    ml = leaf(mixed)
    out["conv"] = {"pytorch": Variant(torch_conv, [ml]),
                   "kernel": Variant(lambda m_: layer.conv_qkv(m_, w, kd, dk, rep, dv), [ml]),
                   "pytorch_conv_only": Variant(lambda m_: torch_conv(m_, False), [ml])}
    # beta and g
    A_log, dt_bias = a.A_log, a.dt_bias

    def torch_gates(b_, a_):
        beta = b_.sigmoid()
        g = -A_log.float().exp() * torch.nn.functional.softplus(a_.float() + dt_bias)
        return beta.float(), g                    # the drop-in converts beta to float32

    bl, al = leaf(bb), leaf(aa)
    out["gates"] = {"pytorch": Variant(torch_gates, [bl, al]),
                    "kernel": Variant(lambda b_, a_: layer.gates(b_, a_, A_log, dt_bias), [bl, al])}
    # the gated RMSNorm: the rule's output (bf16) and z as rows of one value head
    core = (torch.randn(B * T * a.num_v_heads, dv, generator=gen) * 0.3).to(DEV, BF)
    zr = z.reshape(-1, dv)
    cl, zl = leaf(core), leaf(zr)
    out["gated_norm"] = {"pytorch": Variant(a.norm, [cl, zl], {"GATED_NORM": False}),
                         "kernel": Variant(a.norm, [cl, zl], {"GATED_NORM": True})}
    # SwiGLU
    act = lin.mlp.act_fn
    el, ul = leaf(e), leaf(u)
    out["swiglu"] = {"pytorch": Variant(lambda e_, u_: act(e_) * u_, [el, ul]), "kernel": Variant(layer.swiglu, [el, ul])}
    # the full-attention layer's q / k norms on the layout transformers hands them
    fa = full.self_attn
    hd = fa.head_dim
    with torch.no_grad(), torch.autocast(DEV, dtype=BF):
        xf = full.input_layernorm(x)
        qg = fa.q_proj(xf)
        kk = fa.k_proj(xf)

    # the query norm's input as step 0 gave it: the strided view (every other half of the projection's rows) as a leaf
    ql = torch.chunk(qg.view(B, T, -1, 2 * hd), 2, dim=-1)[0].detach().requires_grad_(True)
    kl = leaf(kk.view(B, T, -1, hd))
    out["q_norm"] = {"pytorch": Variant(fa.q_norm, [ql], {"RMSNORM": False}),
                     "kernel": Variant(fa.q_norm, [ql], {"RMSNORM": True})}
    out["k_norm"] = {"pytorch": Variant(fa.k_norm, [kl], {"RMSNORM": False}), "kernel": Variant(fa.k_norm, [kl], {"RMSNORM": True})}
    return cfg, M, lin, full, x, out


def lora_layout_check(tc):
    """in_proj_qkv with LoRA (rank 32) at 0.8B's micro-batch shape: its output's gradient as transformers' path hands it (the
    transpose of a contiguous [B, C, T]) against [B, T, C] contiguous (as the fused convolution path hands it)."""
    B, T = 2, 903
    cfg, M, lin, full = PL.build_layers("0.8b", tc, True)
    m = lin.linear_attn.in_proj_qkv
    gen = torch.Generator().manual_seed(3)
    xn = leaf((torch.randn(B, T, cfg.hidden_size, generator=gen)).to(DEV, BF))
    params = [p for n, p in m.named_parameters() if "lora_" in n]

    class V(Variant):
        def __init__(self, transposed):
            super().__init__(None, [xn])
            self.transposed = transposed

        def run(self, back):
            with torch.autocast(DEV, dtype=BF):
                y = m(xn)
            if not back:
                return
            if self.grads is None:
                g = (torch.randn(B, y.shape[-1], T, generator=torch.Generator().manual_seed(4)) * 0.1).to(DEV, y.dtype)
                self.grads = [g.transpose(1, 2) if self.transposed else g.transpose(1, 2).contiguous()]
            torch.autograd.grad([y], [xn] + params, self.grads, allow_unused=True)
    return interleaved({"transposed_gradient": V(True), "contiguous_gradient": V(False)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="0.8b,9b")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--out", default=os.path.join(HERE, "layer_logs", "d4_parts.json"))
    ap.add_argument("--layer-parts", default="RMSNORM,CONV,GATED_NORM,GATES,SWIGLU",
                    help="the parts on in the whole-layer timing's 'on' variant")
    a = ap.parse_args()
    s0 = step0_parts()
    rec = {"torch": torch.__version__, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "rounds": a.rounds,
           "shapes": SHAPES, "step0": s0, "models": {}}
    warm_gpu(2.0)
    patch.install(lora=False)            # PEFT's own LoRA layers (the LoRA-B layout check measured them; series design D5 fuses them)
    rec["patches"] = patch.layer_patches()
    on = {p: p in a.layer_parts.split(",") for p in layer.PARTS}
    off = dict.fromkeys(layer.PARTS, False)
    rec["layer_on"] = on
    for model in a.models.split(","):
        B, T = SHAPES[model]
        tc = PL.text_config(PL.MODELS[model])
        cfg, M, lin, full, x, pv = parts(model, B, T, tc)
        res = {}
        for part, variants in pv.items():
            r = interleaved(variants, a.rounds)
            fl = s0[model][part]["floor_fwd_bwd_ms"] if part in s0[model] else None
            if part == "gates":
                fl = None                     # step 0's glue part held the layout copies as well (reported beside)
            r["speedup_fwd"] = round(r["pytorch"]["fwd_ms"] / r["kernel"]["fwd_ms"], 3)
            r["speedup_fwd_bwd"] = round(r["pytorch"]["fwd_bwd_ms"] / r["kernel"]["fwd_bwd_ms"], 3)
            if "pytorch_conv_only" in r:
                r["speedup_fwd_bwd_against_conv_only"] = round(r["pytorch_conv_only"]["fwd_bwd_ms"] / r["kernel"]["fwd_bwd_ms"], 3)
            r["floor_fwd_bwd_ms"] = fl
            r["kernel_over_floor"] = round(r["kernel"]["fwd_bwd_ms"] / fl, 2) if fl else None
            r["step0"] = s0[model].get(part)
            res[part] = r
            print(f"{model} {part:16s} pytorch {r['pytorch']['fwd_ms']:7.3f} / {r['pytorch']['fwd_bwd_ms']:7.3f} ms  kernel "
                  f"{r['kernel']['fwd_ms']:7.3f} / {r['kernel']['fwd_bwd_ms']:7.3f} ms  x{r['speedup_fwd_bwd']}  "
                  f"floor {fl} ({r['kernel_over_floor']}x)", flush=True)
        # the whole layers: every part off against the parts on
        xw = leaf(x)
        for kind, lyr in (("linear", lin), ("full", full)):
            if kind == "linear":
                fn = lambda t: lyr(t, position_embeddings=None)                        # noqa: E731
            else:
                rot = M.Qwen3_5TextRotaryEmbedding(config=cfg).to(DEV)
                pos = torch.arange(T, device=DEV).view(1, 1, -1).expand(3, B, -1)
                with torch.no_grad():
                    pe = rot(x, pos)
                fn = lambda t, pe=pe, pos=pos: lyr(t, position_embeddings=pe, attention_mask=None,      # noqa: E731
                                                   position_ids=pos[0])
            r = interleaved({"off": Variant(fn, [xw], off), "on": Variant(fn, [xw], on)}, a.rounds)
            r["speedup_fwd_bwd"] = round(r["off"]["fwd_bwd_ms"] / r["on"]["fwd_bwd_ms"], 3)
            r["saved_fwd_bwd_ms"] = round(r["off"]["fwd_bwd_ms"] - r["on"]["fwd_bwd_ms"], 3)
            res[f"layer_{kind}"] = r
            print(f"{model} layer {kind:6s} off {r['off']['fwd_ms']:7.2f} / {r['off']['fwd_bwd_ms']:7.2f} ms  on "
                  f"{r['on']['fwd_ms']:7.2f} / {r['on']['fwd_bwd_ms']:7.2f} ms  x{r['speedup_fwd_bwd']}", flush=True)
        rec["models"][model] = res
        del lin, full, pv
        torch.mps.empty_cache()
    if "0.8b" in a.models.split(","):
        r = lora_layout_check(PL.text_config(PL.MODELS["0.8b"]))
        r["transposed_over_contiguous_fwd_bwd"] = round(r["transposed_gradient"]["fwd_bwd_ms"]
                                                        / r["contiguous_gradient"]["fwd_bwd_ms"], 3)
        rec["in_proj_qkv_lora_layout_0.8b_2x903"] = r
        print(f"in_proj_qkv LoRA 2x903: transposed gradient {r['transposed_gradient']['fwd_bwd_ms']} ms, contiguous "
              f"{r['contiguous_gradient']['fwd_bwd_ms']} ms (x{r['transposed_over_contiguous_fwd_bwd']})", flush=True)
    rec["ended"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
