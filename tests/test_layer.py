"""The layer parts (macai.layer; series design D4) against float64 on the CPU, beside transformers' own path (PyTorch on the
Apple GPU) against the same float64 reference: the forward and every gradient of each part, at the real head counts and widths
of Qwen3.5-0.8B and 9B, odd sequence lengths, more than one row, float32 / bfloat16 / float16 inputs. The bar: the part's error
at most twice the PyTorch path's own error, each tensor separately (error = the largest absolute difference over the reference's
largest magnitude). Inputs and upstream gradients are drawn in the run's dtype, so every path, the reference included, sees the
same values.

Also: the installation (each method replaced only while its source matches; every part off gives transformers' bits; the
projections and their dropout draws in the original order), and one end-to-end check: whole decoder layers of both kinds with
LoRA (rank 32, dropout 0.05 with fixed masks) at the bench shapes, the patched layer and the unpatched one against a float64
run of the same layer.

MACAI_PARITY_LOG=<file> appends every measured case (errors of both paths) to that file as JSON lines.
"""
import contextlib
import copy
import json
import os
import warnings

import pytest
import torch
import torch.nn.functional as F

import macai.patch
from macai import layer, lora
from macai.layer import _conv, _gates, _qwen3_5, _swiglu

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")
M = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
DTYPES = [torch.float32, torch.bfloat16, torch.float16]
DT = {torch.float32: "f32", torch.bfloat16: "bf16", torch.float16: "f16"}
LOG = os.environ.get("MACAI_PARITY_LOG")
EPS = 1e-6
# the real widths: (key_dim, value_dim, key head width, repeat of q / k over value heads, value head width)
LIN = {"0.8b": dict(kd=2048, vd=2048, dk=128, rep=1, dv=128, hv=16, hidden=1024, inter=3584, nq=8, nkv=2),
       "9b": dict(kd=2048, vd=4096, dk=128, rep=2, dv=128, hv=32, hidden=4096, inter=12288, nq=16, nkv=4)}
SHAPES = [(1, 1), (2, 7), (3, 33), (2, 129)]          # (rows, tokens): odd lengths, more than one row


@pytest.fixture(scope="module", autouse=True)
def patched():
    macai.patch.install()
    yield
    macai.patch.uninstall()


def rel(a: torch.Tensor, r: torch.Tensor) -> float:
    return ((a - r).abs().max() / r.abs().max().clamp_min(1e-300)).item()


def draw(shape, dtype, scale=1.0, seed=0, shift=0.0):
    """float64 values exactly representable in dtype."""
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=gen, dtype=torch.float64) * scale + shift).to(dtype).double()


def run(fn, xs, dtypes, needs, gouts, device):
    leaves = [x.detach().to(device, d).clone().requires_grad_(n) for x, d, n in zip(xs, dtypes, needs)]
    outs = fn(*leaves)
    outs = outs if isinstance(outs, tuple) else (outs,)
    torch.autograd.backward(outs, [g.to(device, o.dtype) for g, o in zip(gouts, outs)])
    return ([o.detach().cpu().double() for o in outs],
            [x.grad.detach().cpu().double() if n else None for x, n in zip(leaves, needs)])


def parity(case, ref_fn, torch_fn, fused_fn, xs, dtypes, needs, out_names, in_names, gdtype):
    """Each output and each needed input gradient: the fused path's and the PyTorch path's errors against float64; the bar
    (fused <= 2 x PyTorch) for each."""
    with torch.no_grad():
        shapes = [o.shape for o in (lambda o: o if isinstance(o, tuple) else (o,))(ref_fn(*xs))]
    gouts = [draw(s, gdtype, seed=100 + i) for i, s in enumerate(shapes)]
    ro, rg = run(ref_fn, xs, [torch.float64] * len(xs), needs, gouts, "cpu")
    to, tg = run(torch_fn, xs, dtypes, needs, gouts, "mps")
    fo, fg = run(fused_fn, xs, dtypes, needs, gouts, "mps")
    rows = {}
    for name, r, t, f in zip(out_names, ro, to, fo):
        rows[name] = (rel(t, r), rel(f, r))
    for name, r, t, f, n in zip(in_names, rg, tg, fg, needs):
        if n:
            rows["d" + name] = (rel(t, r), rel(f, r))
    if LOG:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"case": case, "errors": {k: {"pytorch": v[0], "fused": v[1]} for k, v in rows.items()}}) + "\n")
    bad = {k: v for k, v in rows.items() if not v[1] <= 2 * v[0]}
    assert not bad, f"{case}: fused error over twice the PyTorch path's (pytorch, fused): {bad}"
    return rows


# ---- the zero-centered RMSNorm -----------------------------------------------------------------------------------------

def ref_rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1 + w)


def norm_module(d, w, device, dtype):
    m = M.Qwen3_5RMSNorm(d, eps=EPS)
    m.weight.data = w.to(dtype)
    return m.to(device)


@mps
@pytest.mark.parametrize("dtype", DTYPES, ids=DT.get)
@pytest.mark.parametrize("bt", SHAPES, ids=str)
@pytest.mark.parametrize("d", [1024, 4096])
def test_rmsnorm(d, bt, dtype):
    x = draw((*bt, d), dtype, seed=1)
    w = draw((d,), dtype, scale=0.1, seed=2)
    mods = {}

    def path(on):
        def fn(x_):
            m = mods.setdefault(on, norm_module(d, w, "mps", dtype))
            with layer.configured(RMSNORM=on):
                return m(x_)
        return fn
    parity(f"rmsnorm d{d} {bt} {DT[dtype]}", lambda x_: ref_rms(x_, w), path(False), path(True), [x], [dtype], [True],
           ["y"], ["x"], dtype)


@mps
@pytest.mark.parametrize("dtype", DTYPES, ids=DT.get)
@pytest.mark.parametrize("model", ["0.8b", "9b"])
def test_rmsnorm_query_key_heads(model, dtype):
    """The full-attention layers' q / k norms: rows of 256, the query norm reading every other half of the projection's rows
    (a strided view, as transformers hands it), the key norm a contiguous [B, T, heads, 256]."""
    c = LIN[model]
    B, T, hd = 2, 37, 256
    qg = draw((B, T, c["nq"], 2 * hd), dtype, seed=3)
    w = draw((hd,), dtype, scale=0.1, seed=4)

    def path(on):
        m = norm_module(hd, w, "mps", dtype)

        def fn(qg_):
            q, _gate = torch.chunk(qg_, 2, dim=-1)
            with layer.configured(RMSNORM=on):
                return m(q)
        return fn
    parity(f"rmsnorm q_norm {model} {DT[dtype]}", lambda qg_: ref_rms(torch.chunk(qg_, 2, -1)[0], w), path(False), path(True),
           [qg], [dtype], [True], ["y"], ["qg"], dtype)
    k = draw((B, T, c["nkv"], hd), dtype, seed=5)

    def kpath(on):
        m = norm_module(hd, w, "mps", dtype)

        def fn(k_):
            with layer.configured(RMSNORM=on):
                return m(k_)
        return fn
    parity(f"rmsnorm k_norm {model} {DT[dtype]}", lambda k_: ref_rms(k_, w), kpath(False), kpath(True), [k], [dtype], [True],
           ["y"], ["k"], dtype)


@mps
@pytest.mark.parametrize("dtype", DTYPES, ids=DT.get)
def test_rmsnorm_trained_weight(dtype):
    """The weight's gradient when the weight trains (float32 partials and a second pass)."""
    B, T, d = 3, 333, 1024
    x = draw((B, T, d), dtype, seed=6)
    w = draw((d,), dtype, scale=0.1, seed=7)

    def path(on):
        m = norm_module(d, w, "mps", dtype)

        def fn(x_, w_):
            with layer.configured(RMSNORM=on):
                return torch.func.functional_call(m, {"weight": w_}, (x_,))
        return fn
    parity(f"rmsnorm trained weight {DT[dtype]}", ref_rms, path(False), path(True), [x, w], [dtype, dtype], [True, True],
           ["y"], ["x", "w"], dtype)


# ---- the gated RMSNorm --------------------------------------------------------------------------------------------------

def ref_gated(x, z, w):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)) * F.silu(z)


def check_gated_norm(model, bt, dtype, trained):
    c = LIN[model]
    rows = bt[0] * bt[1] * c["hv"]
    x = draw((rows, c["dv"]), dtype, seed=8)
    z = draw((rows, c["dv"]), dtype, seed=9)
    w = draw((c["dv"],), dtype, scale=0.1, seed=10, shift=1.0)

    def path(on):
        m = M.Qwen3_5RMSNormGated(c["dv"], eps=EPS)
        m.weight.data = w.to(dtype)
        m = m.to("mps")

        def fn(x_, z_, w_):
            with layer.configured(GATED_NORM=on):
                return torch.func.functional_call(m, {"weight": w_}, (x_, z_))
        return fn
    parity(f"gated_norm {model} {bt} {DT[dtype]}{' trained' if trained else ''}", ref_gated, path(False), path(True),
           [x, z, w], [dtype] * 3, [True, True, trained], ["o"], ["x", "z", "w"], dtype)


@mps
@pytest.mark.parametrize("dtype", DTYPES, ids=DT.get)
@pytest.mark.parametrize("bt", SHAPES, ids=str)
@pytest.mark.parametrize("model", ["0.8b", "9b"])
def test_gated_norm(model, bt, dtype):
    check_gated_norm(model, bt, dtype, False)


@mps
@pytest.mark.parametrize("dtype", DTYPES, ids=DT.get)
@pytest.mark.parametrize("model", ["0.8b", "9b"])
def test_gated_norm_trained_weight(model, dtype):
    """The weight's gradient when the weight trains (float32 partials and a second pass)."""
    check_gated_norm(model, (3, 33), dtype, True)


# ---- the convolution path -----------------------------------------------------------------------------------------------

def conv_ref_layout(y, c, B, T):
    q, k, v = torch.split(y, [c["kd"], c["kd"], c["vd"]], dim=-1)
    q = q.reshape(B, T, -1, c["dk"])
    k = k.reshape(B, T, -1, c["dk"])
    v = v.reshape(B, T, -1, c["dv"])
    if c["rep"] > 1:
        q = q.repeat_interleave(c["rep"], dim=2)
        k = k.repeat_interleave(c["rep"], dim=2)
    return q, k, v


def check_conv_path(model, bt, dtype, trained):
    """transformers' path: transpose, causal_conv1d_fn (depthwise conv1d, SiLU), transpose, split, reshape, repeat; against
    the fused kernel writing q, k (repeated) and v (float32) directly."""
    c = LIN[model]
    B, T = bt
    C = 2 * c["kd"] + c["vd"]
    x = draw((B, T, C), dtype, seed=11)
    w = draw((C, 1, 4), dtype, scale=0.3, seed=12)

    def ref(x_, w_):
        u = F.conv1d(x_.transpose(1, 2), w_, padding=3, groups=C)[:, :, :T]
        return conv_ref_layout(F.silu(u).transpose(1, 2), c, B, T)

    def torch_path(x_, w_):
        y = M.causal_conv1d_fn(x_.transpose(1, 2), w_.squeeze(1), None, activation="silu").transpose(1, 2)
        return conv_ref_layout(y, c, B, T)

    def fused(x_, w_):
        return _conv.conv_qkv(x_, w_, c["kd"], c["dk"], c["rep"], c["dv"])
    parity(f"conv {model} {bt} {DT[dtype]}{' trained' if trained else ''}", ref, torch_path, fused, [x, w], [dtype, dtype],
           [True, trained], ["q", "k", "v"], ["x", "w"], dtype)


@mps
@pytest.mark.parametrize("dtype", DTYPES, ids=DT.get)
@pytest.mark.parametrize("bt", [(1, 1), (2, 3), (2, 33), (3, 100)], ids=str)
@pytest.mark.parametrize("model", ["0.8b", "9b"])
def test_conv_path(model, bt, dtype):
    check_conv_path(model, bt, dtype, False)


@mps
@pytest.mark.parametrize("dtype", DTYPES, ids=DT.get)
@pytest.mark.parametrize("model", ["0.8b", "9b"])
def test_conv_path_trained_weight(model, dtype):
    """The convolution weight's gradient when it trains (float32 partials a tile, then a second pass)."""
    check_conv_path(model, (2, 33), dtype, True)


# ---- beta and g ---------------------------------------------------------------------------------------------------------

def check_gates(model, bt, dtype, trained):
    hv = LIN[model]["hv"]
    b = draw((*bt, hv), dtype, scale=2.0, seed=13)
    a = draw((*bt, hv), dtype, scale=3.0, seed=14)
    gen = torch.Generator().manual_seed(15)
    A_log = torch.log(torch.rand(hv, generator=gen, dtype=torch.float64) * 15.99 + 0.01).to(dtype).double()
    dt_bias = draw((hv,), dtype, seed=16)

    def ref(b_, a_, A_, d_):
        return torch.sigmoid(b_), -A_.exp() * F.softplus(a_ + d_)

    def torch_path(b_, a_, A_, d_):
        return b_.sigmoid(), -A_.float().exp() * F.softplus(a_.float() + d_)

    def fused(b_, a_, A_, d_):
        return _gates.gates(b_, a_, A_, d_)
    parity(f"gates {model} {bt} {DT[dtype]}{' trained' if trained else ''}", ref, torch_path, fused, [b, a, A_log, dt_bias],
           [dtype] * 4, [True, True, trained, trained], ["beta", "g"], ["b", "a", "A_log", "dt_bias"], torch.float32)


@mps
@pytest.mark.parametrize("dtype", DTYPES, ids=DT.get)
@pytest.mark.parametrize("bt", SHAPES, ids=str)
@pytest.mark.parametrize("model", ["0.8b", "9b"])
def test_gates(model, bt, dtype):
    check_gates(model, bt, dtype, False)


@mps
@pytest.mark.parametrize("dtype", DTYPES, ids=DT.get)
@pytest.mark.parametrize("model", ["0.8b", "9b"])
def test_gates_trained(model, dtype):
    """A_log's and dt_bias's gradients when they train."""
    check_gates(model, (3, 33), dtype, True)


@mps
def test_gates_softplus_threshold():
    """Past softplus's threshold (a + dt_bias > 20) PyTorch returns the input itself; deep in the negative range, log1p."""
    a = torch.tensor([[-80.0, -30.0, -5.0, 0.0, 19.5, 20.5, 40.0, 3.0]], dtype=torch.float32)
    b = torch.zeros_like(a)
    A_log = torch.zeros(8)
    dtb = torch.zeros(8)
    beta, g = _gates.gates(*(t.to("mps") for t in (b, a, A_log, dtb)))
    ref = -F.softplus(a.double())
    assert ((g.cpu().double() - ref).abs() / ref.abs()).max() < 1e-6


# ---- SwiGLU -------------------------------------------------------------------------------------------------------------

@mps
@pytest.mark.parametrize("dtype", DTYPES, ids=DT.get)
@pytest.mark.parametrize("bt", SHAPES, ids=str)
@pytest.mark.parametrize("model", ["0.8b", "9b"])
def test_swiglu(model, bt, dtype):
    inter = LIN[model]["inter"]
    e = draw((*bt, inter), dtype, scale=2.0, seed=17)
    u = draw((*bt, inter), dtype, seed=18)
    act = M.ACT2FN["silu"]
    parity(f"swiglu {model} {bt} {DT[dtype]}", lambda e_, u_: F.silu(e_) * u_, lambda e_, u_: act(e_) * u_, _swiglu.swiglu,
           [e, u], [dtype, dtype], [True, True], ["h"], ["e", "u"], dtype)


# ---- the installation ---------------------------------------------------------------------------------------------------

def tiny_config(model="0.8b", layers=4):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    c = LIN[model]
    cfg = Qwen3_5TextConfig(hidden_size=c["hidden"], intermediate_size=c["inter"], num_hidden_layers=layers,
                            num_attention_heads=c["nq"], num_key_value_heads=c["nkv"], head_dim=256,
                            linear_num_key_heads=c["kd"] // c["dk"], linear_num_value_heads=c["hv"], linear_key_head_dim=128,
                            linear_value_head_dim=128, linear_conv_kernel_dim=4, rms_norm_eps=EPS, hidden_act="silu",
                            vocab_size=256)
    cfg._attn_implementation = "sdpa"
    return cfg


def test_patches_installed():
    """Every source hashes as in the version written against (checked on transformers' own methods), and every patch is in
    force."""
    assert set(macai.patch.layer_patches()) == set(_qwen3_5.PATCHES)
    macai.patch.uninstall()
    try:
        assert all(_qwen3_5.check_sources(M).values()), _qwen3_5.check_sources(M)
    finally:
        macai.patch.install()
    assert set(macai.patch.layer_patches()) == set(_qwen3_5.PATCHES)


def test_source_mismatch_leaves_original(monkeypatch):
    """A method whose source differs from the version written against stays transformers' own, with a warning."""
    macai.patch.uninstall()
    try:
        monkeypatch.setitem(_qwen3_5.SOURCES, "Qwen3_5MLP.forward", "0" * 16)
        before = M.Qwen3_5MLP.__dict__["forward"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            macai.patch.install()
        assert M.Qwen3_5MLP.__dict__["forward"] is before
        assert "mlp" not in macai.patch.layer_patches() and "rmsnorm" in macai.patch.layer_patches()
        assert any("Qwen3_5MLP.forward" in str(x.message) for x in w)
    finally:
        macai.patch.uninstall()
        monkeypatch.undo()
        macai.patch.install()
    assert "mlp" in macai.patch.layer_patches()


@contextlib.contextmanager
def recorded_calls(mod):
    """The order in which mod's submodules are called (forward pre-hooks)."""
    calls, hooks = [], []
    for name, m in mod.named_modules():
        if name:
            hooks.append(m.register_forward_pre_hook(lambda _m, _a, n=name: calls.append(n)))
    try:
        yield calls
    finally:
        for h in hooks:
            h.remove()


@mps
@pytest.mark.parametrize("model", ["0.8b", "9b"])
def test_parts_off_bit_identical_and_call_order(model):
    """Every part off: the replaced methods give transformers' bits. Every part on: the same submodules are called in the
    same order (PEFT's LoRA layers and their dropout draws included)."""
    from peft import LoraConfig, inject_adapter_in_model
    cfg = tiny_config(model)
    torch.manual_seed(0)
    lin = M.Qwen3_5DecoderLayer(cfg, 0)
    lin = inject_adapter_in_model(LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=[
        "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj"]), lin)
    lin = lin.to("mps", torch.bfloat16).train()
    x = (torch.randn(2, 37, cfg.hidden_size) * 0.5).to("mps", torch.bfloat16)
    outs, orders, masks = {}, {}, {}
    try:
        for name in ("original", "off", "on"):
            # original: the rule installed, the layer's methods transformers' own
            macai.patch.uninstall()
            macai.patch.install(layer=name != "original")
            drops = []
            # where each dropout zeroed its input (positions whose input is itself zero cannot tell, and are left out below)
            hooks = [m.register_forward_hook(lambda _m, i, o: drops.append(((o == 0).cpu(), (i[0] != 0).cpu())))
                     for n, m in lin.named_modules() if n.endswith("lora_dropout.default")]
            try:
                # PEFT's own LoRA forward (macai.lora's fused path calls none of PEFT's dropout, lora_A and lora_B modules)
                with (layer.configured(**dict.fromkeys(layer.PARTS, name == "on")), recorded_calls(lin) as calls,
                      lora.configured(FUSED=False), torch.autocast("mps", torch.bfloat16)):
                    torch.manual_seed(1)
                    outs[name] = lin(x, position_embeddings=None).float().cpu()
                orders[name] = list(calls)
                masks[name] = drops
            finally:
                for h in hooks:
                    h.remove()
    finally:
        macai.patch.uninstall()
        macai.patch.install()
    assert torch.equal(outs["original"], outs["off"])
    # the SwiGLU part replaces the MLP's activation module (it draws no random numbers); every other call in the same order
    keep = [n for n in orders["original"] if not n.endswith("act_fn")]
    assert orders["original"] == orders["off"] and keep == orders["on"]
    assert len(masks["on"]) == len(masks["original"]) == 6
    for (a, ia), (b, ib) in zip(masks["original"], masks["on"]):
        both = ia & ib
        assert both.float().mean() > 0.99 and torch.equal(a[both], b[both])
    assert (outs["on"] - outs["original"]).abs().max() / outs["original"].abs().max() < 0.05


@mps
@pytest.mark.parametrize("model", ["0.8b", "9b"])
def test_fresh_cache_takes_the_parts(model, monkeypatch):
    """Training forwards carry a fresh cache (transformers' Trainer sets the model's use_cache from its arguments): with a cache
    holding no state of the layer the parts run, and the cache ends as transformers' own forward leaves it (the convolution's
    state bit for bit, the final recurrent state to the rule's precision); a decode step from either cache runs transformers'
    path and gives the same output."""
    from transformers.cache_utils import DynamicCache
    cfg = tiny_config(model)
    torch.manual_seed(0)
    gdn = M.Qwen3_5DecoderLayer(cfg, 0).linear_attn
    for n, p in gdn.named_parameters():
        if "conv1d" in n:
            torch.nn.init.normal_(p, std=0.3)
        elif p.dim() > 1:
            torch.nn.init.normal_(p, std=0.02)
    gdn = gdn.to("mps", torch.bfloat16)
    x = (torch.randn(2, 45, cfg.hidden_size) * 0.5).to("mps", torch.bfloat16)
    x1 = (torch.randn(2, 1, cfg.hidden_size) * 0.5).to("mps", torch.bfloat16)
    calls = []
    real = _qwen3_5.conv_path
    monkeypatch.setattr(_qwen3_5, "conv_path", lambda *a, **k: calls.append(1) or real(*a, **k))
    out, caches = {}, {}
    for on in (False, True):
        cache = DynamicCache(config=cfg)
        with layer.configured(**dict.fromkeys(layer.PARTS, on)), torch.no_grad():
            y = gdn(x, cache_params=cache)
            y1 = gdn(x1, cache_params=cache)
        out[on], caches[on] = (y.float().cpu(), y1.float().cpu()), cache
    assert len(calls) == 1                     # the prefill took the fused path; the decode step did not
    for i in range(2):
        assert (out[True][i] - out[False][i]).abs().max() / out[False][i].abs().max() < 0.03
    # after the prefill and the decode step both caches hold the same convolution window and close recurrent states
    a, b = (c.get_conv_state(0, state_idx=0) for c in (caches[False], caches[True]))
    assert torch.equal(a, b)
    a, b = (c.get_recurrent_state(0, state_idx=0).float() for c in (caches[False], caches[True]))
    assert (a - b).abs().max() / a.abs().max() < 0.03


@mps
def test_mlp_and_full_attention_dispatch():
    """The MLP and the full-attention layer's norms through the replaced methods: close to transformers' path in bf16."""
    cfg = tiny_config("0.8b")
    torch.manual_seed(0)
    full = M.Qwen3_5DecoderLayer(cfg, 3).to("mps", torch.bfloat16)
    assert full.block_type == "full_attention"
    x = torch.randn(2, 19, cfg.hidden_size).to("mps", torch.bfloat16)
    rot = M.Qwen3_5TextRotaryEmbedding(config=cfg).to("mps")
    pos = torch.arange(19, device="mps").view(1, 1, -1).expand(3, 2, -1)
    pe = rot(x, pos)
    out = {}
    for on in (False, True):
        with layer.configured(**dict.fromkeys(layer.PARTS, on)):
            out[on] = full(x, position_embeddings=pe, attention_mask=None, position_ids=pos[0]).float().cpu()
    assert (out[True] - out[False]).abs().max() / out[False].abs().max() < 0.03


# ---- end to end: whole decoder layers with LoRA, against float64 ------------------------------------------------------

class FixedDropout(torch.nn.Module):
    """Dropout with a fixed mask a shape (the same draws on every device and dtype)."""

    def __init__(self, p, seed):
        super().__init__()
        self.p, self.seed, self.masks = p, seed, {}

    def forward(self, x):
        m = self.masks.get(tuple(x.shape))
        if m is None:
            gen = torch.Generator().manual_seed(self.seed)
            m = self.masks[tuple(x.shape)] = (torch.rand(x.shape, generator=gen) >= self.p).double()
        return x * m.to(x.device, x.dtype) / (1 - self.p)


@contextlib.contextmanager
def float64_everywhere():
    """transformers' layer code casts to float32 in places (x.float(), .to(torch.float32)); for the float64 reference those
    casts keep float64 tensors as they are (CPU only, inside this block)."""
    orig_float, orig_to = torch.Tensor.float, torch.Tensor.to

    def f(self, *a, **k):
        return self if self.dtype == torch.float64 else orig_float(self, *a, **k)

    def to(self, *a, **k):
        if self.dtype == torch.float64:
            a = tuple(torch.float64 if x is torch.float32 else x for x in a)
            if k.get("dtype") is torch.float32:
                k["dtype"] = torch.float64
        return orig_to(self, *a, **k)
    torch.Tensor.float, torch.Tensor.to = f, to
    try:
        yield
    finally:
        torch.Tensor.float, torch.Tensor.to = orig_float, orig_to


def build_layer(model, kind, seed=0):
    from peft import LoraConfig, inject_adapter_in_model
    from huggingface_hub import hf_hub_download
    try:
        tc = json.load(open(hf_hub_download({"0.8b": "Qwen/Qwen3.5-0.8B", "9b": "Qwen/Qwen3.5-9B"}[model], "config.json")))
    except Exception:  # noqa: BLE001 - offline without the cache: the widths above
        tc = None
    if tc is not None:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        cfg = Qwen3_5TextConfig(**tc.get("text_config", tc))
        cfg._attn_implementation = "sdpa"
    else:
        cfg = tiny_config(model, layers=4)
    idx = cfg.layer_types.index("linear_attention" if kind == "linear" else "full_attention")
    torch.manual_seed(seed)
    lyr = M.Qwen3_5DecoderLayer(cfg, idx)
    for n, p in lyr.named_parameters():
        if "conv1d" in n:
            torch.nn.init.normal_(p, std=0.3)
        elif p.dim() > 1:
            torch.nn.init.normal_(p, std=0.02)
        elif "linear_attn.norm" in n:
            torch.nn.init.normal_(p, mean=1.0, std=0.1)          # the gated norm's weight multiplies directly
        elif "norm" in n:
            torch.nn.init.normal_(p, std=0.1)                    # the zero-centered norms' weights: 1 + w
    targets = ["in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lyr = inject_adapter_in_model(LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target_modules=targets), lyr)
    for i, (n, m) in enumerate(sorted(lyr.named_modules())):
        if n.endswith("lora_dropout") and isinstance(m, torch.nn.ModuleDict):
            m["default"] = FixedDropout(0.05, 1000 + i)
    for n, p in lyr.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.02)
        if "lora_" not in n:
            p.data = p.data.to(torch.bfloat16).float()          # the bf16 model's values
        p.requires_grad_("lora_" in n)
    return cfg, lyr


@mps
@pytest.mark.parametrize("kind", ["linear", "full"])
@pytest.mark.parametrize("model,bt", [("0.8b", (4, 1024)), ("9b", (2, 1024))], ids=["0.8b-4x1024", "9b-2x1024"])
def test_decoder_layer_end_to_end(model, bt, kind):
    """The whole decoder layer, LoRA on (rank 32, fixed dropout masks), bf16 weights and activations under bf16 autocast
    (as the Trainer runs it), the D2m rule: every D4 part on against every part off, both against a float64 CPU run of the
    same layer (transformers' code with its float32 casts kept at float64, transformers' own rule); the output, the input's
    gradient and every LoRA weight's gradient."""
    cfg, base = build_layer(model, kind)
    B, T = bt
    x = draw((B, T, cfg.hidden_size), torch.bfloat16, seed=20)
    gy = draw((B, T, cfg.hidden_size), torch.bfloat16, scale=0.1, seed=21)
    pos = torch.arange(T).view(1, 1, -1).expand(3, B, -1)
    rot = M.Qwen3_5TextRotaryEmbedding(config=cfg)
    with torch.no_grad():
        cos, sin = rot(torch.zeros(1, dtype=torch.float32), pos)
    cos, sin = cos.to(torch.bfloat16).double(), sin.to(torch.bfloat16).double()

    def go(lyr, device, dtype, autocast):
        xx = x.detach().to(device, dtype).clone().requires_grad_(True)
        kw = dict(position_embeddings=(cos.to(device, dtype), sin.to(device, dtype)), attention_mask=None)
        if kind == "full":
            kw["position_ids"] = pos[0].to(device)
        ctx = torch.autocast(device, dtype=torch.bfloat16) if autocast else contextlib.nullcontext()
        with ctx:
            y = lyr(xx, **kw)
        y.backward(gy.to(device, y.dtype))
        grads = {n: p.grad.detach().cpu().double() for n, p in lyr.named_parameters() if p.requires_grad}
        return y.detach().cpu().double(), xx.grad.detach().cpu().double(), grads

    # the float64 reference: transformers' code and rule on the CPU
    ref_layer = copy.deepcopy(base).double()
    rule = macai.patch._saved.get(M.__name__, M.torch_chunk_gated_delta_rule)
    rule = getattr(rule, "__wrapped__", rule)
    saved_rule = M.torch_chunk_gated_delta_rule
    M.torch_chunk_gated_delta_rule = rule
    try:
        with layer.configured(**dict.fromkeys(layer.PARTS, False)), float64_everywhere():
            ry, rdx, rgr = go(ref_layer, "cpu", torch.float64, False)
    finally:
        M.torch_chunk_gated_delta_rule = saved_rule
    res = {}
    for on in (False, True):
        lyr = copy.deepcopy(base)
        for n, p in lyr.named_parameters():
            if "lora_" not in n:
                p.data = p.data.to(torch.bfloat16)
        lyr = lyr.to("mps").train()
        with layer.configured(**dict.fromkeys(layer.PARTS, on)):
            res[on] = go(lyr, "mps", torch.bfloat16, True)
        del lyr
        torch.mps.empty_cache()
    rows = {"y": (rel(res[False][0], ry), rel(res[True][0], ry)), "dx": (rel(res[False][1], rdx), rel(res[True][1], rdx))}
    for n in rgr:
        rows["d " + n.replace(".default.weight", "")] = (rel(res[False][2][n], rgr[n]), rel(res[True][2][n], rgr[n]))
    case = f"end to end {model} {kind} {B}x{T}"
    if LOG:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"case": case, "errors": {k: {"pytorch": v[0], "fused": v[1]} for k, v in rows.items()}}) + "\n")
    bad = {k: v for k, v in rows.items() if not v[1] <= 2 * v[0]}
    assert not bad, f"{case}: fused error over twice the PyTorch path's (pytorch, fused): {bad}"
