"""The fused LoRA adapter path (macai.lora; series design D5) against float64 on the CPU, beside PEFT's own path (PyTorch on the
Apple GPU) against the same float64 reference: the forward and every gradient (to x, to A, to B) of one PEFT LoRA linear layer
at the real widths of Qwen3.5-0.8B's and 9B's adapted projections, odd lengths, more than one row; bfloat16 and float16 inputs
under autocast (the readers' configuration and its float16 twin), float32 inputs without autocast, and bfloat16 inputs without
autocast (the adapters' float32 as the compute dtype). Dropout on (0.05, training mode): both paths draw their masks from the
same generator state, and the float64 reference applies the mask the fused path drew. The bar: the fused path's error at most
twice PEFT's own, each tensor separately (error = the largest absolute difference over the reference's largest magnitude).
Cases with the base weight set to zero measure the adapter term alone (with a drawn base weight its rounding dominates both
paths' outputs).

Also: the masks (the same generator state gives the same mask; the dropped bf16 / fp16 values bit for bit PEFT's dropped float32
copy after the cast); the gradients of A and B the same bits as PEFT's under bf16 autocast (the same products on the same bits);
every condition under which PEFT's own forward runs, counted by reason, and its bits then; the installation (the method replaced
only while its source matches PEFT 0.21.2); whole decoder layers of both kinds with LoRA and dropout on, the fused path and
PEFT's against a float64 run of the same layer replaying the same masks.

MACAI_PARITY_LOG=<file> appends every measured case (errors of both paths) to that file as JSON lines.
"""
import contextlib
import copy
import json
import os
import warnings

import pytest
import torch

import macai.patch
from macai import layer, lora
from macai.lora import _fused, _kernels, _peft

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")
peft = pytest.importorskip("peft")
LOG = os.environ.get("MACAI_PARITY_LOG")
P_DROP = 0.05
DT = {torch.float32: "f32", torch.bfloat16: "bf16", torch.float16: "f16"}
# the adapted projections' widths (d_in, d_out): Qwen3.5-0.8B and 9B
WIDTHS = {"0.8b": [(1024, 6144), (1024, 2048), (2048, 1024), (1024, 3584), (3584, 1024), (1024, 4096), (1024, 512)],
          "9b": [(4096, 8192), (4096, 4096), (4096, 12288), (12288, 4096), (4096, 1024)]}
SHAPES = [(1, 1), (2, 7), (3, 33), (2, 129)]          # (rows, tokens): odd lengths, more than one row
# (x's dtype, autocast dtype or None): the readers' configuration first
CONFIGS = [(torch.bfloat16, torch.bfloat16), (torch.float16, torch.float16), (torch.float32, None), (torch.bfloat16, None)]
CFG_ID = {c: f"x-{DT[c[0]]}-autocast-{DT[c[1]] if c[1] else 'off'}" for c in CONFIGS}


@pytest.fixture(scope="module", autouse=True)
def patched():
    macai.patch.install()
    yield
    macai.patch.uninstall()


def rel(a: torch.Tensor, r: torch.Tensor) -> float:
    return ((a - r).abs().max() / r.abs().max().clamp_min(1e-300)).item()


def draw(shape, dtype, scale=1.0, seed=0):
    """float64 values exactly representable in dtype."""
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=gen, dtype=torch.float64) * scale).to(dtype).double()


class Net(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj = torch.nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        return self.proj(x)


def lora_net(d_in, d_out, base_dtype, *, rank=32, alpha=64, dropout=P_DROP, base_zero=False, seed=0, **conf):
    """One PEFT LoRA layer (the adapters in float32, B drawn so every gradient is nonzero) on the Apple GPU, training mode."""
    from peft import LoraConfig, inject_adapter_in_model
    net = Net(d_in, d_out)
    with torch.no_grad():
        net.proj.weight.copy_(torch.zeros(d_out, d_in) if base_zero else draw((d_out, d_in), base_dtype, 0.02, seed))
    net = inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=["proj"], **conf),
                                  net)
    with torch.no_grad():
        for n, p in net.named_parameters():
            if "lora_A" in n:
                p.copy_(draw(p.shape, torch.float32, 0.02, seed + 1))
            if "lora_B" in n:
                p.copy_(draw(p.shape, torch.float32, 0.02, seed + 2))
    net = net.to("mps")
    net.proj.base_layer.to(base_dtype)
    for n, p in net.named_parameters():
        p.requires_grad_("lora_" in n)
    return net.train()


@contextlib.contextmanager
def captured_masks():
    """The masks the fused path draws, in order (macai.lora._fused.dropout wrapped)."""
    got, orig = [], _fused.dropout

    def rec(x, p):
        out = orig(x, p)
        got.append(out[1].detach().cpu())
        return out
    _fused.dropout = rec
    try:
        yield got
    finally:
        _fused.dropout = orig


@contextlib.contextmanager
def peft_masks(net):
    """The masks PEFT's dropout modules draw, in order: where the dropped copy is nonzero, with x's own zeros kept apart."""
    got, hooks = [], []
    for n, m in net.named_modules():
        if isinstance(m, torch.nn.Dropout):
            hooks.append(m.register_forward_hook(lambda _m, i, o: got.append(((o != 0).cpu(), (i[0] == 0).cpu()))))
    try:
        yield got
    finally:
        for h in hooks:
            h.remove()


def run(net, x0, gy, on, x_dtype, ac_dtype, seed=1234):
    """The layer's forward and backward on the Apple GPU from a fixed generator state: y, dx, dA, dB (float64, CPU)."""
    x = x0.detach().to("mps", x_dtype).clone().requires_grad_(True)
    for p in net.parameters():
        p.grad = None
    torch.mps.manual_seed(seed)
    ctx = torch.autocast("mps", dtype=ac_dtype) if ac_dtype is not None else contextlib.nullcontext()
    with lora.configured(FUSED=on), ctx:
        y = net(x)
    y.backward(gy.to("mps", y.dtype))
    m = net.proj
    return (y.detach().cpu().double(), x.grad.detach().cpu().double(),
            m.lora_A["default"].weight.grad.detach().cpu().double(), m.lora_B["default"].weight.grad.detach().cpu().double())


def reference(net, x0, gy, mask, p=P_DROP):
    """The same layer in float64 on the CPU: y = x W^T + s (M x / (1 - p)) A^T B^T, its gradients."""
    m = net.proj
    W = m.base_layer.weight.detach().cpu().double()
    A = m.lora_A["default"].weight.detach().cpu().double().requires_grad_(True)
    B = m.lora_B["default"].weight.detach().cpu().double().requires_grad_(True)
    s = m.scaling["default"]
    x = x0.detach().clone().requires_grad_(True)
    xd = x * mask.double() / (1 - p) if mask is not None else x
    y = x @ W.t() + s * (xd @ A.t()) @ B.t()
    y.backward(gy)
    return y.detach(), x.grad, A.grad, B.grad


def check(case, rows):
    if LOG:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"case": case, "errors": {k: {"pytorch": v[0], "fused": v[1]} for k, v in rows.items()}}) + "\n")
    bad = {k: v for k, v in rows.items() if not v[1] <= 2 * v[0]}
    assert not bad, f"{case}: fused error over twice PEFT's (peft, fused): {bad}"


def parity_case(d_in, d_out, bt, cfg, base_zero, seed=0):
    x_dtype, ac_dtype = cfg
    base_dtype = x_dtype if ac_dtype is None else torch.bfloat16 if ac_dtype == torch.bfloat16 else torch.float16
    net = lora_net(d_in, d_out, base_dtype, base_zero=base_zero, seed=seed)
    x0 = draw((*bt, d_in), x_dtype, seed=seed + 10)
    out_dtype = ac_dtype if ac_dtype is not None else (torch.float32 if x_dtype == torch.float32 else x_dtype)
    gy = draw((*bt, d_out), out_dtype, scale=0.1, seed=seed + 11)
    with peft_masks(net) as pm:
        peft_res = run(net, x0, gy, False, x_dtype, ac_dtype)
    with captured_masks() as fm:
        fused_res = run(net, x0, gy, True, x_dtype, ac_dtype)
    assert len(fm) == 1 and len(pm) == 1
    kept, xzero = pm[0]
    mask = fm[0].reshape(kept.shape)
    assert torch.equal(mask[~xzero], kept[~xzero]), "the fused path's mask differs from PEFT's"
    ref = reference(net, x0, gy, mask)
    rows = {}
    for name, r, t, f in zip(("y", "dx", "dA", "dB"), ref, peft_res, fused_res):
        rows[name] = (rel(t, r), rel(f, r))
    case = f"lora {d_in}->{d_out} {bt} {CFG_ID[cfg]}{' base zero' if base_zero else ''}"
    check(case, rows)
    return peft_res, fused_res


@mps
@pytest.mark.parametrize("cfg", CONFIGS, ids=CFG_ID.get)
@pytest.mark.parametrize("bt", SHAPES, ids=str)
@pytest.mark.parametrize("width", WIDTHS["0.8b"][:2] + WIDTHS["0.8b"][4:5] + WIDTHS["9b"][3:4], ids=str)
def test_parity(width, bt, cfg):
    peft_res, fused_res = parity_case(*width, bt, cfg, base_zero=False)
    if cfg == (torch.bfloat16, torch.bfloat16):
        # under the readers' autocast the weight gradients are the same products on the same bits (s = 2)
        assert torch.equal(peft_res[2], fused_res[2]) and torch.equal(peft_res[3], fused_res[3])


@mps
@pytest.mark.parametrize("cfg", CONFIGS, ids=CFG_ID.get)
@pytest.mark.parametrize("bt", [(2, 7), (3, 33)], ids=str)
@pytest.mark.parametrize("model", ["0.8b", "9b"])
def test_parity_adapter_term_alone(model, bt, cfg):
    """The base weight zero: the output and the input gradient are the adapter path's alone, every projection width."""
    for i, width in enumerate(WIDTHS[model]):
        parity_case(*width, bt, cfg, base_zero=True, seed=100 * i)


@mps
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=DT.get)
@pytest.mark.parametrize("p", [0.05, 0.1, 0.5])
def test_dropout_in_own_dtype_is_peft_bits(dtype, p):
    """torch's dropout call on x in its own dtype: the same mask as on PEFT's float32 copy from the same generator state, and
    the dropped values bit for bit PEFT's dropped copy after autocast's cast (the fused path's shortcut)."""
    import torch.nn.functional as F
    for shape in [(2, 37, 1024), (1, 129, 3584), (3, 11, 4096)]:
        x = (torch.randn(shape, generator=torch.Generator().manual_seed(3)) * 2).to("mps", dtype)
        torch.mps.manual_seed(11)
        ref = F.dropout(x.float(), p, True).to(dtype)
        torch.mps.manual_seed(11)
        out, mask = _fused.dropout(x, p)
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
        assert torch.equal(mask, (ref != 0) | (x == 0) & mask)


@mps
def test_same_generator_state_same_masks_and_outputs():
    """A run of several layers in sequence (the RNG stream moves between them as in a model): the fused path and PEFT's draw
    the same masks in the same order from the same seed; the outputs agree to rounding, the weight gradients bit for bit."""
    from peft import LoraConfig, inject_adapter_in_model

    class Stack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(1024, 2048, bias=False)
            self.b = torch.nn.Linear(2048, 1024, bias=False)
            self.c = torch.nn.Linear(1024, 3584, bias=False)

        def forward(self, x):
            return self.c(self.b(self.a(x)))
    torch.manual_seed(0)
    net = Stack()
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.02)
    net = inject_adapter_in_model(LoraConfig(r=32, lora_alpha=64, lora_dropout=P_DROP, target_modules=["a", "b", "c"]), net)
    for n, p in net.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.02)
    net = net.to("mps")
    for n, m in net.named_modules():
        if isinstance(m, torch.nn.Linear) and n.endswith("base_layer"):
            m.to(torch.bfloat16)
    for n, p in net.named_parameters():
        p.requires_grad_("lora_" in n)
    x0 = torch.randn(2, 77, 1024).to("mps", torch.bfloat16)
    gy = (torch.randn(2, 77, 3584) * 0.1).to("mps", torch.bfloat16)
    res, masks = {}, {}
    for on in (False, True):
        x = x0.clone().requires_grad_(True)
        for p in net.parameters():
            p.grad = None
        with peft_masks(net) as pm, captured_masks() as fm:
            torch.mps.manual_seed(77)
            with lora.configured(FUSED=on), torch.autocast("mps", dtype=torch.bfloat16):
                y = net(x)
            y.backward(gy)
        masks[on] = list(pm) if not on else list(fm)
        res[on] = (y.detach().float().cpu(), x.grad.float().cpu(),
                   {n: p.grad.cpu().clone() for n, p in net.named_parameters() if p.requires_grad})
    assert len(masks[False]) == len(masks[True]) == 3
    for (kept, xzero), m in zip(masks[False], masks[True]):
        assert torch.equal(m.reshape(kept.shape)[~xzero], kept[~xzero])
    # past the first layer the inputs differ by rounding (the fused path rounds the output once): close, not the same bits
    assert rel(res[True][0], res[False][0]) < 1e-2 and rel(res[True][1], res[False][1]) < 1e-2
    for n in res[False][2]:
        assert rel(res[True][2][n].double(), res[False][2][n].double()) < 2e-2, n


@mps
def test_kernels_against_float64():
    """The two kernels alone: every dtype pair, ranks 8 / 32 / 64, widths that are not multiples of the 64-column tile."""
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        for cdt in (torch.float32, torch.bfloat16, torch.float16):
            for n, c, r in [(1, 8, 8), (37, 1032, 32), (130, 72, 64), (513, 4096, 32)]:
                gen = torch.Generator().manual_seed(n + c + r)
                P = torch.randn(n, r, generator=gen).to("mps", cdt)
                Bw = (torch.randn(c, r, generator=gen) * 0.05).to("mps", cdt)
                R = torch.randn(n, c, generator=gen).to("mps", dt)
                y = _kernels.up(P, Bw, R, 2.0)
                ref = R.cpu().double() + 2.0 * P.cpu().double() @ Bw.cpu().double().t()
                tol = {torch.float32: 1e-6, torch.bfloat16: 8e-3, torch.float16: 1e-3}[dt]
                assert rel(y.cpu().double(), ref) < tol
                G = torch.randn(n, r, generator=gen).to("mps", cdt)
                A = (torch.randn(r, c, generator=gen) * 0.05).to("mps", cdt)
                M = torch.rand(n, c, generator=gen).to("mps") > 0.05
                d = _kernels.down(G, A, M, 1 / 0.95, dt)
                refd = (G.cpu().double() @ A.cpu().double()) * M.cpu().double() / 0.95
                assert rel(d.cpu().double(), refd) < tol
                assert bool((d.cpu()[~M.cpu()] == 0).all())


# ---- the conditions: PEFT's own forward runs, counted ---------------------------------------------------------------

def _fallback_reason(net, x, **kw):
    lora.reset_stats()
    with lora.configured(FUSED=True), torch.autocast("mps", dtype=torch.bfloat16):
        net(x, **kw) if kw else net(x)
    st = lora.stats()
    return st["fused"], st["fallbacks"]


@mps
def test_fallbacks_counted_and_peft_bits():
    x = torch.randn(2, 9, 1024).to("mps", torch.bfloat16)

    def bits_same_as_peft(net, **kw):
        out = {}
        for on in (False, True):
            torch.mps.manual_seed(5)
            with lora.configured(FUSED=on), torch.autocast("mps", dtype=torch.bfloat16):
                out[on] = (net(x, **kw) if kw else net(x)).detach().cpu()
        return torch.equal(out[False].view(torch.int16), out[True].view(torch.int16))

    # a LoRA variant (DoRA)
    net = lora_net(1024, 2048, torch.bfloat16, use_dora=True)
    assert _fallback_reason(net, x) == (0, {"a LoRA variant": 1}) and bits_same_as_peft(net)
    # LoRA bias
    net = lora_net(1024, 2048, torch.bfloat16, lora_bias=True)
    assert _fallback_reason(net, x) == (0, {"LoRA bias": 1}) and bits_same_as_peft(net)
    # a rank the kernels do not take
    net = lora_net(1024, 2048, torch.bfloat16, rank=12, alpha=24)
    assert _fallback_reason(net, x) == (0, {"rank": 1}) and bits_same_as_peft(net)
    # adapters disabled; merged
    net = lora_net(1024, 2048, torch.bfloat16)
    net.proj.enable_adapters(False)
    assert _fallback_reason(net, x) == (0, {"adapters disabled": 1}) and bits_same_as_peft(net)
    net.proj.enable_adapters(True)
    net.proj.merge()
    assert _fallback_reason(net, x) == (0, {"merged": 1})
    net.proj.unmerge()
    # a dropout that is not nn.Dropout / nn.Identity
    net = lora_net(1024, 2048, torch.bfloat16)
    net.proj.lora_dropout["default"] = torch.nn.Dropout1d(0.05)
    assert _fallback_reason(net, x) == (0, {"dropout not nn.Dropout or nn.Identity": 1})
    # adapter_names (mixed-batch inference)
    net = lora_net(1024, 2048, torch.bfloat16).eval()
    assert _fallback_reason(net.proj, x, adapter_names=["default", "default"]) == (0, {"arguments beside x": 1})
    # two active adapters
    from peft import LoraConfig
    net = lora_net(1024, 2048, torch.bfloat16)
    net.proj.update_layer("other", 32, lora_alpha=64, config=LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05))
    net.proj.set_adapter(["default", "other"])
    net = net.to("mps")
    assert _fallback_reason(net, x) == (0, {"not one active adapter": 1})
    # on the CPU
    net = lora_net(1024, 2048, torch.bfloat16).cpu()
    lora.reset_stats()
    with lora.configured(FUSED=True):
        net(x.cpu())
    assert lora.stats() == {"fused": 0, "fallbacks": {"not on the Apple GPU": 1}}
    # the fused path: an Identity dropout (p = 0), and evaluation mode (the dropout off) take it
    net = lora_net(1024, 2048, torch.bfloat16, dropout=0.0)
    assert type(net.proj.lora_dropout["default"]) is torch.nn.Identity
    assert _fallback_reason(net, x) == (1, {})
    net = lora_net(1024, 2048, torch.bfloat16).eval()
    assert _fallback_reason(net, x) == (1, {})
    lora.reset_stats()


@mps
@pytest.mark.parametrize("cfg", CONFIGS, ids=CFG_ID.get)
def test_no_dropout_and_eval(cfg):
    """Without a mask (an Identity dropout, or nn.Dropout in evaluation mode) the fused path keeps PEFT's semantics: x cast as
    PEFT casts it, the same parity bar."""
    x_dtype, ac_dtype = cfg
    for dropout, train in ((0.0, True), (P_DROP, False)):
        base_dtype = x_dtype if ac_dtype is None else ac_dtype
        net = lora_net(1024, 3584, base_dtype, dropout=dropout, base_zero=True)
        net.train(train)
        x0 = draw((2, 33, 1024), x_dtype, seed=4)
        out_dtype = ac_dtype if ac_dtype is not None else x_dtype
        gy = draw((2, 33, 3584), out_dtype, scale=0.1, seed=5)
        peft_res = run(net, x0, gy, False, x_dtype, ac_dtype)
        fused_res = run(net, x0, gy, True, x_dtype, ac_dtype)
        ref = reference(net, x0, gy, None)
        check(f"lora no mask (dropout {dropout}, train {train}) {CFG_ID[cfg]}",
              {n: (rel(t, r), rel(f, r)) for n, r, t, f in zip(("y", "dx", "dA", "dB"), ref, peft_res, fused_res)})


@mps
def test_input_without_gradient_and_frozen_adapter():
    """x without a gradient (the first layer of a model whose embeddings are frozen); an adapter weight frozen."""
    net = lora_net(1024, 2048, torch.bfloat16)
    x = torch.randn(2, 9, 1024).to("mps", torch.bfloat16)
    for on in (False, True):
        for p in net.parameters():
            p.grad = None
        net.proj.lora_A["default"].weight.requires_grad_(False)
        torch.mps.manual_seed(3)
        with lora.configured(FUSED=on), torch.autocast("mps", dtype=torch.bfloat16):
            y = net(x)
        y.float().sum().backward()
        assert net.proj.lora_A["default"].weight.grad is None and net.proj.lora_B["default"].weight.grad is not None
        net.proj.lora_A["default"].weight.requires_grad_(True)


# ---- the installation ----------------------------------------------------------------------------------------------

def test_patch_installed_and_source():
    """PEFT's method hashes as in the version written against, and the replacement is in force."""
    assert macai.patch.lora_patches() == ["linear"]
    assert all(_peft.check_sources().values()), _peft.check_sources()
    from peft.tuners.lora import layer as L
    assert L.Linear.forward is _peft.linear_forward


def test_source_mismatch_leaves_original(monkeypatch):
    """A PEFT whose method differs from 0.21.2's keeps its own forward, with a warning."""
    from peft.tuners.lora import layer as L
    macai.patch.uninstall()
    try:
        monkeypatch.setitem(_peft.SOURCES, "Linear.forward", "0" * 16)
        before = L.Linear.__dict__["forward"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            macai.patch.install()
        assert L.Linear.__dict__["forward"] is before and macai.patch.lora_patches() == []
        assert any("lora.Linear.forward" in str(x.message) for x in w)
    finally:
        macai.patch.uninstall()
        monkeypatch.undo()
        macai.patch.install()
    assert macai.patch.lora_patches() == ["linear"]


@mps
def test_setting_off_is_peft():
    """FUSED off: PEFT's own forward, its bits."""
    net = lora_net(1024, 2048, torch.bfloat16)
    x = torch.randn(2, 9, 1024).to("mps", torch.bfloat16)
    lora.reset_stats()
    out = []
    for _ in range(2):
        torch.mps.manual_seed(9)
        with lora.configured(FUSED=False), torch.autocast("mps", dtype=torch.bfloat16):
            out.append(net(x).detach().cpu())
    macai.patch.uninstall()
    try:
        torch.mps.manual_seed(9)
        with torch.autocast("mps", dtype=torch.bfloat16):
            own = net(x).detach().cpu()
    finally:
        macai.patch.install()
    assert torch.equal(out[0].view(torch.int16), own.view(torch.int16)) and lora.stats()["fused"] == 0


# ---- end to end: whole decoder layers with LoRA and dropout on, against float64 --------------------------------------

class ReplayMasks(torch.nn.Module):
    """Dropout replaying recorded masks in order (the float64 reference)."""

    def __init__(self, masks, p):
        super().__init__()
        self.masks, self.p = masks, p

    def forward(self, x):
        m = self.masks.pop(0)
        return x * m.reshape(x.shape).to(x.device, x.dtype) / (1 - self.p)


@mps
@pytest.mark.parametrize("kind", ["linear", "full"])
@pytest.mark.parametrize("model,bt", [("0.8b", (4, 1024)), ("9b", (2, 1024))], ids=["0.8b-4x1024", "9b-2x1024"])
def test_decoder_layer_end_to_end(model, bt, kind):
    """The whole decoder layer, LoRA rank 32 with dropout 0.05 on (real nn.Dropout modules), bf16 weights and activations under
    bf16 autocast, the D4 parts and the rule as the readers run them: the fused adapter path against PEFT's (both drawing their
    masks from the same generator state, checked equal), both against a float64 CPU run of the same layer that replays the masks
    (transformers' code with its float32 casts kept at float64, transformers' own rule); the output, the input's gradient and
    every LoRA weight's gradient."""
    import test_layer as TL
    from peft import LoraConfig, inject_adapter_in_model
    M = TL.M
    cfg, base = TL.build_layer(model, kind)
    # TL.build_layer fixes the masks; the readers' layers draw them: real dropout modules back in
    for n, m in base.named_modules():
        if n.endswith("lora_dropout") and isinstance(m, torch.nn.ModuleDict):
            m["default"] = torch.nn.Dropout(P_DROP)
    B, T = bt
    x = TL.draw((B, T, cfg.hidden_size), torch.bfloat16, seed=20)
    gy = TL.draw((B, T, cfg.hidden_size), torch.bfloat16, scale=0.1, seed=21)
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

    res, masks = {}, {}
    for on in (False, True):
        lyr = copy.deepcopy(base)
        for n, p in lyr.named_parameters():
            if "lora_" not in n:
                p.data = p.data.to(torch.bfloat16)
        lyr = lyr.to("mps").train()
        with peft_masks(lyr) as pm, captured_masks() as fm, lora.configured(FUSED=on):
            torch.mps.manual_seed(2024)
            res[on] = go(lyr, "mps", torch.bfloat16, True)
        masks[on] = list(pm) if not on else list(fm)
        del lyr
        torch.mps.empty_cache()
    n_adapted = 6 if kind == "linear" else 7
    assert len(masks[False]) == len(masks[True]) == n_adapted
    for (kept, xzero), m in zip(masks[False], masks[True]):
        assert torch.equal(m.reshape(kept.shape)[~xzero], kept[~xzero])
    # the float64 reference replays the fused path's masks in the order they were drawn
    ref_layer = copy.deepcopy(base).double()
    replay = list(masks[True])
    order = [n for n, m in base.named_modules() if n.endswith("lora_dropout")]
    for n, m in ref_layer.named_modules():
        if n.endswith("lora_dropout") and isinstance(m, torch.nn.ModuleDict):
            m["default"] = ReplayMasks(replay, P_DROP)
    rule = macai.patch._saved.get(M.__name__, M.torch_chunk_gated_delta_rule)
    rule = getattr(rule, "__wrapped__", rule)
    saved_rule = M.torch_chunk_gated_delta_rule
    M.torch_chunk_gated_delta_rule = rule
    try:
        with layer.configured(**dict.fromkeys(layer.PARTS, False)), lora.configured(FUSED=False), TL.float64_everywhere():
            ry, rdx, rgr = go(ref_layer, "cpu", torch.float64, False)
    finally:
        M.torch_chunk_gated_delta_rule = saved_rule
    assert not replay and len(order) == n_adapted
    rows = {"y": (rel(res[False][0], ry), rel(res[True][0], ry)), "dx": (rel(res[False][1], rdx), rel(res[True][1], rdx))}
    for n in rgr:
        rows["d " + n.replace(".default.weight", "")] = (rel(res[False][2][n], rgr[n]), rel(res[True][2][n], rgr[n]))
    check(f"lora end to end {model} {kind} {B}x{T}", rows)
