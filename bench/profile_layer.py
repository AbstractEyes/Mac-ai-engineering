"""Series D3-D5, step 0 (b): each part of Qwen3.5's two decoder layers timed in isolation, with its bytes, FLOPs and floors
(docs/series_layer.md).

    python bench/profile_layer.py --shapes 0.8b=4x1024,2x903 --shapes 9b=2x1024,1x903     # layer_parts.json (after the products)

At each shape (batch x tokens: the bench's, then the training micro-batch's median padded one), for the linear-attention and
the full-attention decoder layer of both models: every part forward and forward + backward (the gradient to the part's input,
the weights frozen), in two variants: frozen, and with LoRA rank 32 (alpha 64, dropout 0.05) on the record's targets (PEFT's
own LoRA layers, their weights in float32 as PEFT keeps them). bf16 weights and activations under bf16 autocast, as the
Trainer runs them; the Metal kernels for the rule (macai.patch); each part's inputs laid out as the layer hands them (the
convolution path's transposed views, the rule's strided q / k / v). Metal events (macai.instruments.gpu_ms), back-to-back
calls filling at least ~10 ms, the median of 7 rounds. The whole layer the same way, against the sum of its parts.

Each part's bytes are the least traffic it needs as one pass (each input read once, each output written once; the backward
reads the output's gradient and what it needs of the inputs, and writes the inputs' gradients), its FLOPs the products' and
the elementwise work's. Its floor is the larger of bytes / 356 GB/s (the copy rate measured by bench/hw_ceilings.py) and
FLOPs / the product rate given by --rate (TFLOP/s; the large-square bf16 rate of bench/profile_products.py in the same session).
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from macai import patch  # noqa: E402
from macai.instruments import CEILINGS, gpu_ms, warm_gpu  # noqa: E402

MODELS = {"0.8b": "Qwen/Qwen3.5-0.8B", "9b": "Qwen/Qwen3.5-9B"}
LORA = dict(r=32, alpha=64, dropout=0.05)
TARGETS = ("in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
BW = CEILINGS["copy_gbps"]
FILL_MS = 10.0
REPS = 7
DEV, BF = "mps", torch.bfloat16
PAD_ROW_FRACTION = 0.6      # at a micro-batch shape with 2 rows, the second row's real length (the rest is padding)


def text_config(model_id):
    from huggingface_hub import hf_hub_download
    cfg = json.load(open(hf_hub_download(model_id, "config.json")))
    return cfg.get("text_config", cfg)


def fill(once_ms):
    return max(3, min(200, int(FILL_MS / max(once_ms, 1e-3)) + 1))


def timed(fn):
    for _ in range(2):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.mps.synchronize()
    inner = fill((time.perf_counter() - t0) * 1e3)
    return gpu_ms(fn, reps=REPS, warm=1, inner=inner)["median_ms"], inner


class Part:
    """One part: run(back) does its forward (and the backward to its inputs and LoRA weights when back)."""

    def __init__(self, name, fwd, inputs, params=(), fbytes=0, bbytes=0, fflops=0, bflops=0, note=""):
        self.name, self.fwd, self.inputs, self.params = name, fwd, list(inputs), list(params)
        self.fbytes, self.bbytes, self.fflops, self.bflops, self.note = fbytes, bbytes, fflops, bflops, note
        self.grads = None

    def run(self, back):
        with torch.autocast(DEV, dtype=BF):
            out = self.fwd(*self.inputs)
        if not back:
            return
        outs = out if isinstance(out, (tuple, list)) else (out,)
        outs = [o for o in outs if isinstance(o, torch.Tensor) and o.requires_grad]
        if self.grads is None:
            gen = torch.Generator().manual_seed(1)
            self.grads = [(torch.randn(o.shape, generator=gen) * 0.1).to(DEV, o.dtype) for o in outs]
        want = [x for x in self.inputs + self.params if isinstance(x, torch.Tensor) and x.requires_grad]
        torch.autograd.grad(outs, want, self.grads, allow_unused=True)


def leaf(t, grad=True):
    """A leaf with t's values and t's strides (the layout the layer hands the part)."""
    return t.detach().requires_grad_(grad)


def lora_params(mod):
    return [p for n, p in mod.named_parameters() if "lora_" in n]


def build_layers(model, tc, with_lora):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    cfg = Qwen3_5TextConfig(**tc)
    cfg._attn_implementation = "sdpa"
    lin = M.Qwen3_5DecoderLayer(cfg, cfg.layer_types.index("linear_attention"))
    full = M.Qwen3_5DecoderLayer(cfg, cfg.layer_types.index("full_attention"))
    torch.manual_seed(0)
    for layer in (lin, full):
        for p in layer.parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, std=0.02)
    if with_lora:
        from peft import LoraConfig, inject_adapter_in_model
        conf = LoraConfig(r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=LORA["dropout"], bias="none",
                          target_modules=list(TARGETS))
        lin = inject_adapter_in_model(conf, lin)
        full = inject_adapter_in_model(conf, full)
    out = []
    for layer in (lin, full):
        layer = layer.to(DEV, BF).train()
        for n, p in layer.named_parameters():
            if "lora_" in n:
                p.data = p.data.float()            # PEFT keeps the adapters in float32 (autocast_adapter_dtype)
                if "lora_B" in n:
                    torch.nn.init.normal_(p, std=0.02)     # nonzero, as after the first steps
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
        out.append(layer)
    return cfg, M, out[0], out[1]


def proj_bytes(n, k_in, n_out, lora):
    """(fwd bytes, bwd bytes, fwd flops, bwd flops) of a projection, with its adapter fused when lora."""
    fb = 2 * n * k_in + 2 * k_in * n_out + 2 * n * n_out
    bb = 2 * n * n_out + 2 * k_in * n_out + 2 * n * k_in
    ff = bf = 2 * n * k_in * n_out
    if lora:
        # fused with the base product, the adapter adds its weights (float32) to the traffic and, in the backward, a read of
        # X for dA = dH^T X; dropout's mask is drawn in place
        r = LORA["r"]
        fb += 8 * r * (k_in + n_out)
        bb += 2 * n * k_in + 8 * r * (k_in + n_out)
        ff += 2 * n * r * (k_in + n_out)
        bf += 4 * n * r * (k_in + n_out)
    return fb, bb, ff, bf


def linear_parts(cfg, M, layer, B, T, mask2d, with_lora):
    a = layer.linear_attn
    H, D = cfg.hidden_size, cfg.hidden_size
    N = B * T
    kd, vd = a.key_dim, a.value_dim
    C = 2 * kd + vd
    Hv, dk, dv = a.num_v_heads, a.head_k_dim, a.head_v_dim
    rep = a.num_v_heads // a.num_k_heads
    I = cfg.intermediate_size
    gen = torch.Generator().manual_seed(0)

    def rnd(*s, dtype=BF, scale=1.0):
        return (torch.randn(*s, generator=gen) * scale).to(DEV, dtype)

    x = rnd(B, T, H)
    parts = []
    lp = (lambda m: lora_params(m)) if with_lora else (lambda m: [])

    def is_lora(m):
        return with_lora and hasattr(m, "lora_A")

    # 1. input_layernorm
    xi = leaf(x)
    parts.append(Part("input_layernorm", layer.input_layernorm, [xi], fbytes=4 * N * D, bbytes=6 * N * D,
                      fflops=4 * N * D, bflops=8 * N * D))
    with torch.no_grad(), torch.autocast(DEV, dtype=BF):
        xn = layer.input_layernorm(x)
    if mask2d is not None:
        parts.append(Part("padding mask (apply_mask_to_padding_states)", lambda t: M.apply_mask_to_padding_states(t, mask2d),
                          [leaf(xn)], fbytes=4 * N * D, bbytes=4 * N * D, fflops=N * D, bflops=N * D))
    # 2. projections
    for name, k_in, n_out in (("in_proj_qkv", D, C), ("in_proj_z", D, vd), ("in_proj_b", D, Hv), ("in_proj_a", D, Hv)):
        m = getattr(a, name)
        fb, bb, ff, bf = proj_bytes(N, k_in, n_out, is_lora(m))
        parts.append(Part(name, m, [leaf(xn)], lp(m), fb, bb, ff, bf))
    # in_proj_qkv as the layer runs it: its output transposed for the convolution path, the gradient coming back transposed
    m = a.in_proj_qkv
    fb, bb, ff, bf = proj_bytes(N, D, C, is_lora(m))
    parts.append(Part("in_proj_qkv, output transposed as the layer uses it", lambda t: m(t).transpose(1, 2), [leaf(xn)], lp(m),
                      fb, bb, ff, bf, note="the gradient arrives as the transpose of a contiguous [B, C, T] tensor"))
    with torch.no_grad(), torch.autocast(DEV, dtype=BF):
        mixed = a.in_proj_qkv(xn)
        z = a.in_proj_z(xn)
        bb_ = a.in_proj_b(xn)
        aa = a.in_proj_a(xn)
    # 3. the convolution path (depthwise causal conv1d, kernel 4, SiLU) on the transposed view
    w_conv = a.conv1d.weight.squeeze(1)
    mixed_l = leaf(mixed)
    parts.append(Part("conv path (causal_conv1d_fn + SiLU)", lambda t: M.causal_conv1d_fn(
        t.transpose(1, 2), w_conv, a.conv1d.bias, activation=a.activation), [mixed_l],
        fbytes=4 * N * C, bbytes=6 * N * C, fflops=(2 * a.conv_kernel_size + 4) * N * C,
        bflops=(4 * a.conv_kernel_size + 8) * N * C))
    with torch.no_grad(), torch.autocast(DEV, dtype=BF):
        conv_out = M.causal_conv1d_fn(mixed.transpose(1, 2), w_conv, a.conv1d.bias, activation=a.activation)

    # 4. the glue: transposes, split, reshape, beta, g, repeat
    def glue(co, a_, b_, z_):
        mq = co.transpose(1, 2)
        q, k, v = torch.split(mq, [kd, kd, vd], dim=-1)
        q = q.reshape(B, T, -1, dk)
        k = k.reshape(B, T, -1, dk)
        v = v.reshape(B, T, -1, dv)
        beta = b_.sigmoid()
        g = -a.A_log.float().exp() * F.softplus(a_.float() + a.dt_bias)
        if rep > 1:
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
        return q, k, v, g, beta, z_.reshape(B, T, -1, dv)

    # q, k, v leave as views of the convolution's output and z as a view of in_proj_z's: the forward's least traffic is beta and
    # g from a and b (and, at 9B, the repeated q and k); the backward gathers dq, dk, dv into the convolution output's gradient
    small = N * Hv * (2 + 2 + 4 + 2)                       # a, b read; g (float32) and beta written
    rep_f = 2 * N * 2 * kd * (1 + rep) if rep > 1 else 0   # q, k read, their repeats written
    glue_f = small + rep_f
    glue_b = 2 * N * (2 * kd * rep + vd) + 2 * N * C + small
    parts.append(Part("glue (transposes, split, reshape, beta, g, repeat)", glue,
                      [leaf(conv_out), leaf(aa), leaf(bb_), leaf(z)], fbytes=glue_f, bbytes=glue_b,
                      fflops=12 * N * Hv, bflops=24 * N * Hv,
                      note="q, k, v and z are views: the forward writes only beta, g (and the repeats); the backward gathers "
                           "the gradients into the convolution output's"))
    with torch.no_grad(), torch.autocast(DEV, dtype=BF):
        q, k, v, g, beta, z4 = glue(conv_out, aa, bb_, z)
    # 5. the rule (macai's kernels, through the drop-in), on the strided q / k / v the glue hands it
    Cc = 16
    rule_f = 2 * N * Hv * (3 * dk * dv + 2 * Cc * dk + Cc * dv)
    rule_bytes = 2 * N * Hv * (2 * dk + dv) + 8 * N * Hv + 2 * N * Hv * dv
    parts.append(Part("rule (gated delta, chunked kernels)", lambda q_, k_, v_, g_, b_: M.torch_chunk_gated_delta_rule(
        q_, k_, v_, g=g_, beta=b_, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)[0],
        [leaf(q), leaf(k), leaf(v), leaf(g), leaf(beta)], fbytes=rule_bytes, bbytes=2 * rule_bytes,
        fflops=rule_f, bflops=3 * rule_f, note="FLOPs: the chunk-16 algorithm, roughly; the backward recomputes the states"))
    with torch.no_grad(), torch.autocast(DEV, dtype=BF):
        core = M.torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
                                              use_qk_l2norm_in_kernel=True)[0]
    # 6. the gated norm
    parts.append(Part("gated norm (RMSNormGated)", a.norm, [leaf(core.reshape(-1, dv)), leaf(z4.reshape(-1, dv))],
                      fbytes=6 * N * vd, bbytes=10 * N * vd, fflops=10 * N * vd, bflops=20 * N * vd))
    with torch.no_grad(), torch.autocast(DEV, dtype=BF):
        normed = a.norm(core.reshape(-1, dv), z4.reshape(-1, dv)).reshape(B, T, -1)
    # 7. out_proj
    fb, bb, ff, bf = proj_bytes(N, vd, D, is_lora(a.out_proj))
    parts.append(Part("out_proj", a.out_proj, [leaf(normed)], lp(a.out_proj), fb, bb, ff, bf))
    parts += mlp_parts(cfg, layer, B, T, x, lp, is_lora)
    whole_params = lp(layer)
    return parts, x, whole_params


def mlp_parts(cfg, layer, B, T, x, lp, is_lora):
    N, D, I = B * T, cfg.hidden_size, cfg.intermediate_size
    m = layer.mlp
    x2 = x.flip(0).roll(1, dims=1).contiguous()     # the add's second operand: other values in other memory, as in the layer
    out = [Part("residual add", lambda s, t: s + t, [leaf(x), leaf(x2)], fbytes=6 * N * D, bbytes=0, fflops=N * D),
           Part("post_attention_layernorm", layer.post_attention_layernorm, [leaf(x)], fbytes=4 * N * D, bbytes=6 * N * D,
                fflops=4 * N * D, bflops=8 * N * D)]
    with torch.no_grad(), torch.autocast(DEV, dtype=BF):
        xn = layer.post_attention_layernorm(x)
        g = m.gate_proj(xn)
        u = m.up_proj(xn)
    for name, mod in (("gate_proj", m.gate_proj), ("up_proj", m.up_proj)):
        fb, bb, ff, bf = proj_bytes(N, D, I, is_lora(mod))
        out.append(Part(name, mod, [leaf(xn)], lp(mod), fb, bb, ff, bf))
    out.append(Part("SwiGLU (act(gate) * up)", lambda a_, b_: m.act_fn(a_) * b_, [leaf(g), leaf(u)], fbytes=6 * N * I,
                    bbytes=10 * N * I, fflops=6 * N * I, bflops=12 * N * I))
    with torch.no_grad(), torch.autocast(DEV, dtype=BF):
        h = m.act_fn(g) * u
    fb, bb, ff, bf = proj_bytes(N, I, D, is_lora(m.down_proj))
    out.append(Part("down_proj", m.down_proj, [leaf(h)], lp(m.down_proj), fb, bb, ff, bf))
    out.append(Part("residual add", lambda s, t: s + t, [leaf(x), leaf(x2)], fbytes=6 * N * D, bbytes=0, fflops=N * D))
    return out


def full_parts(cfg, M, layer, B, T, pos_emb, mask4d, with_lora):
    a = layer.self_attn
    N, D = B * T, cfg.hidden_size
    nq, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, a.head_dim
    gen = torch.Generator().manual_seed(0)
    x = (torch.randn(B, T, D, generator=gen)).to(DEV, BF)
    lp = (lambda m: lora_params(m)) if with_lora else (lambda m: [])

    def is_lora(m):
        return with_lora and hasattr(m, "lora_A")

    parts = [Part("input_layernorm", layer.input_layernorm, [leaf(x)], fbytes=4 * N * D, bbytes=6 * N * D,
                  fflops=4 * N * D, bflops=8 * N * D)]
    with torch.no_grad(), torch.autocast(DEV, dtype=BF):
        xn = layer.input_layernorm(x)
    for name, n_out in (("q_proj", nq * hd * 2), ("k_proj", nkv * hd), ("v_proj", nkv * hd)):
        m = getattr(a, name)
        fb, bb, ff, bf = proj_bytes(N, D, n_out, is_lora(m))
        parts.append(Part(name, m, [leaf(xn)], lp(m), fb, bb, ff, bf))
    with torch.no_grad(), torch.autocast(DEV, dtype=BF):
        qg = a.q_proj(xn)
        kk = a.k_proj(xn)
        vv = a.v_proj(xn)
    shp = (B, T, -1, hd)

    def pre(qg_):
        qs, gate = torch.chunk(qg_.view(B, T, -1, hd * 2), 2, dim=-1)
        return qs.reshape(shp), gate.reshape(B, T, -1)

    # the query half stays a view; the gate's reshape copies it; the backward gathers dq and dgate into q_proj's gradient
    parts.append(Part("attention glue before (chunk, views)", pre, [leaf(qg)], fbytes=4 * N * nq * hd,
                      bbytes=8 * N * nq * hd, note="the query a view, the gate copied by its reshape"))
    with torch.no_grad(), torch.autocast(DEV, dtype=BF):
        qs, gate = pre(qg)
    parts.append(Part("q_norm", a.q_norm, [leaf(qs)], fbytes=4 * N * nq * hd, bbytes=6 * N * nq * hd,
                      fflops=4 * N * nq * hd, bflops=8 * N * nq * hd))
    parts.append(Part("k_norm", a.k_norm, [leaf(kk.view(shp))], fbytes=4 * N * nkv * hd, bbytes=6 * N * nkv * hd,
                      fflops=4 * N * nkv * hd, bflops=8 * N * nkv * hd))
    with torch.no_grad(), torch.autocast(DEV, dtype=BF):
        qn = a.q_norm(qs).transpose(1, 2)
        kn = a.k_norm(kk.view(shp)).transpose(1, 2)
        vt = vv.view(shp).transpose(1, 2)
    cos, sin = pos_emb
    rot = (nq + nkv) * hd
    parts.append(Part("rotary (apply_rotary_pos_emb)", lambda q_, k_: M.apply_rotary_pos_emb(q_, k_, cos, sin),
                      [leaf(qn), leaf(kn)], fbytes=4 * N * rot, bbytes=4 * N * rot, fflops=3 * N * rot, bflops=3 * N * rot))
    with torch.no_grad(), torch.autocast(DEV, dtype=BF):
        qr, kr = M.apply_rotary_pos_emb(qn, kn, cos, sin)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    attn_fn = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
    att_f = 4 * B * nq * T * T * hd // 2          # the causal half of QK^T and PV
    parts.append(Part("attention (scaled dot-product, sdpa)", lambda q_, k_, v_: attn_fn(
        a, q_, k_, v_, mask4d, dropout=0.0, scaling=a.scaling)[0], [leaf(qr), leaf(kr), leaf(vt)],
        fbytes=2 * N * hd * (2 * nq + 2 * nkv), bbytes=2 * N * hd * (4 * nq + 4 * nkv), fflops=att_f, bflops=int(2.5 * att_f),
        note="bytes: the flash-style minimum (no T x T matrix); FLOPs: the causal half"))
    with torch.no_grad(), torch.autocast(DEV, dtype=BF):
        ao = attn_fn(a, qr, kr, vt, mask4d, dropout=0.0, scaling=a.scaling)[0]

    def post(ao_, gate_):
        return ao_.reshape(B, T, -1).contiguous() * torch.sigmoid(gate_)

    parts.append(Part("attention glue after (reshape, output gate)", post, [leaf(ao), leaf(gate)], fbytes=6 * N * nq * hd,
                      bbytes=10 * N * nq * hd, fflops=5 * N * nq * hd, bflops=10 * N * nq * hd))
    with torch.no_grad(), torch.autocast(DEV, dtype=BF):
        gated = post(ao, gate)
    fb, bb, ff, bf = proj_bytes(N, nq * hd, D, is_lora(a.o_proj))
    parts.append(Part("o_proj", a.o_proj, [leaf(gated)], lp(a.o_proj), fb, bb, ff, bf))
    parts += mlp_parts(cfg, layer, B, T, x, lp, is_lora)
    return parts, x, lp(layer)


def measure(parts, whole_fn, x, whole_params, rate):
    rows = []
    for p in parts:
        f, fi = timed(lambda: p.run(False))
        fb, fbi = timed(lambda: p.run(True))
        floor_f = max(p.fbytes / BW / 1e6, p.fflops / rate / 1e9)
        floor_fb = max((p.fbytes + p.bbytes) / BW / 1e6, (p.fflops + p.bflops) / rate / 1e9)
        rows.append({"part": p.name, "fwd_ms": f, "fwd_bwd_ms": fb, "bwd_ms": round(fb - f, 4),
                     "bytes_fwd": p.fbytes, "bytes_bwd": p.bbytes, "flops_fwd": p.fflops, "flops_bwd": p.bflops,
                     "floor_fwd_ms": round(floor_f, 4), "floor_fwd_bwd_ms": round(floor_fb, 4),
                     "over_floor_fwd_bwd": round(fb / floor_fb, 2) if floor_fb > 0 else None,
                     "bound": "bandwidth" if (p.fbytes + p.bbytes) / BW / 1e6 >= (p.fflops + p.bflops) / rate / 1e9 else "compute",
                     "note": p.note, "inner": [fi, fbi]})
        print(f"   {p.name:55s} f {f:8.3f} f+b {fb:8.3f} ms  floor {floor_fb:7.3f} ({rows[-1]['over_floor_fwd_bwd']}x)",
              flush=True)
    xw = leaf(x)
    gen = torch.Generator().manual_seed(2)
    gy = (torch.randn(x.shape, generator=gen) * 0.1).to(DEV, BF)

    def whole(back):
        with torch.autocast(DEV, dtype=BF):
            y = whole_fn(xw)
        if back:
            torch.autograd.grad(y, [xw] + list(whole_params), gy, allow_unused=True)

    wf, _ = timed(lambda: whole(False))
    wfb, _ = timed(lambda: whole(True))
    # the projection parts' "output transposed" variant is a second reading of in_proj_qkv, not a part of the sum
    summed = [r for r in rows if "output transposed" not in r["part"]]
    sf = sum(r["fwd_ms"] for r in summed)
    sfb = sum(r["fwd_bwd_ms"] for r in summed)
    return {"parts": rows, "whole_fwd_ms": wf, "whole_fwd_bwd_ms": wfb, "parts_sum_fwd_ms": round(sf, 3),
            "parts_sum_fwd_bwd_ms": round(sfb, 3), "parts_sum_over_whole_fwd_bwd": round(sfb / wfb, 4),
            "floor_sum_fwd_bwd_ms": round(sum(r["floor_fwd_bwd_ms"] for r in summed), 3)}


def main():
    print(f"[profile_layer] pid {os.getpid()}", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", action="append", required=True, help="model=BxT,BxT (the bench's shape first)")
    ap.add_argument("--rate", default="auto", help="the product rate for the compute floors, TFLOP/s; 'auto': the highest "
                                                  "large-square bf16 rate PyTorch reached in the session's records "
                                                  "(bench/layer_logs/products_torch.json, products_square_forms.json)")
    ap.add_argument("--variants", default="frozen,lora")
    a = ap.parse_args()
    rate_source = "--rate"
    if a.rate == "auto":
        best = []
        p = os.path.join(HERE, "layer_logs", "products_torch.json")
        if os.path.exists(p):
            r = json.load(open(p, encoding="utf-8"))
            best += [v["tflops_events"] for k in ("squares", "squares_end") for v in r.get(k, {}).values()]
        p = os.path.join(HERE, "layer_logs", "products_square_forms.json")
        if os.path.exists(p):
            r = json.load(open(p, encoding="utf-8"))
            best += [x for forms in r["forms"].values() for xs in forms.values() for x in xs]
        a.rate = max(best)
        rate_source = "the highest large-square rate in products_torch.json / products_square_forms.json"
    a.rate = float(a.rate)
    shapes = {k: [tuple(int(x) for x in s.split("x")) for s in v.split(",")] for k, v in (t.split("=") for t in a.shapes)}
    out = {"torch": torch.__version__, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "rate_tflops": a.rate,
           "rate_source": rate_source, "bandwidth_gbps": BW, "reps": REPS, "lora": LORA, "lora_targets": TARGETS,
           "pad_row_fraction": PAD_ROW_FRACTION, "models": {}}
    warm_gpu(2.0)
    patch.install(layer=False)      # the rule's kernels; the layer parts off (step 0 measured transformers' parts)
    for model, shp in shapes.items():
        tc = text_config(MODELS[model])
        out["models"][model] = {}
        for variant in a.variants.split(","):
            cfg, M, lin, full = build_layers(model, tc, variant == "lora")
            for B, T in shp:
                key = f"{variant} {B}x{T}"
                print(f"== {model} {key}", flush=True)
                mask2d = None
                am = torch.ones(B, T, dtype=torch.long, device=DEV)
                if B > 1 and T != 1024:                      # a micro-batch shape: the second row padded
                    am[1, int(PAD_ROW_FRACTION * T):] = 0
                    mask2d = am
                emb = torch.zeros(B, T, cfg.hidden_size, device=DEV, dtype=BF)
                pos = torch.arange(T, device=DEV).view(1, 1, -1).expand(3, B, -1)
                mask4d = M.create_causal_mask(config=cfg, inputs_embeds=emb, attention_mask=am, past_key_values=None,
                                              position_ids=pos[0])
                lin_mask = M.create_recurrent_attention_mask(config=cfg, inputs_embeds=emb, attention_mask=am,
                                                             past_key_values=None)
                rot = M.Qwen3_5TextRotaryEmbedding(config=cfg).to(DEV)
                with torch.no_grad():
                    pe = rot(emb, pos)
                res = {}
                print(" linear-attention layer", flush=True)
                parts, x, wp = linear_parts(cfg, M, lin, B, T, lin_mask, variant == "lora")
                res["linear_attention"] = measure(parts, lambda t: lin(t, position_embeddings=None, attention_mask=lin_mask),
                                                  x, wp, a.rate)
                print(" full-attention layer", flush=True)
                parts, x, wp = full_parts(cfg, M, full, B, T, pe, mask4d, variant == "lora")
                res["full_attention"] = measure(parts, lambda t: full(t, position_embeddings=pe, attention_mask=mask4d,
                                                                      position_ids=pos[0]), x, wp, a.rate)
                res["masks"] = {"linear": None if lin_mask is None else list(lin_mask.shape),
                                "full": None if mask4d is None else list(mask4d.shape)}
                for kind in ("linear_attention", "full_attention"):
                    r = res[kind]
                    print(f" {kind}: parts {r['parts_sum_fwd_bwd_ms']} ms against the whole layer {r['whole_fwd_bwd_ms']} ms "
                          f"({r['parts_sum_over_whole_fwd_bwd']}); floors {r['floor_sum_fwd_bwd_ms']} ms", flush=True)
                out["models"][model][key] = res
                torch.mps.empty_cache()
            del lin, full
            torch.mps.empty_cache()
    out["ended"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    path = os.path.join(HERE, "layer_logs", "layer_parts.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
