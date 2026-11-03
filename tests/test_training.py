"""The training path (macai.training; series designs D8 and D9).

D8, the LM head and the loss at the labelled positions only: Qwen3.5 vision-language models at the real widths and the real
vocabulary of 0.8B and 9B (two decoder layers, one of each kind; tied embeddings; a one-block vision tower), LoRA rank 32 on
the record's targets (dropout 0.05 with fixed masks), bf16 weights under bf16 autocast on the Apple GPU, labels as the record's
collator sets them (the prompt and the padding ignored). The head at the labelled positions against the model's own path, both
against a float64 CPU run of the same model: the loss within relative 1e-5 of the model's own path, every gradient (to the
input embeddings, to the final hidden states, to every LoRA weight) within twice the model's own path's error against float64;
the logits at the labelled positions the same bits. Also: the reduction over num_items_in_batch, the arguments the loss function
receives, every condition under which the model's own path runs, a batch without labels.

D9, the frozen vision tower's output cached: on a small model, a served output equal to the computed one bit for bit and the
model's outputs unchanged; a picture with one pixel changed, or the same pixels at another grid, is not served; the cache only
in training with the tower frozen; a change of the tower's weights empties it; the tower draws no random numbers.

MACAI_PARITY_LOG=<file> appends D8's measured cases (errors of both paths) to that file as JSON lines.
"""
import contextlib
import copy
import functools
import json
import os

import pytest
import torch

import macai.patch
from macai import layer, training
from macai.training import _head, _qwen3_5 as TQ, _vision_cache as VC

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")
M = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
LOG = os.environ.get("MACAI_PARITY_LOG")
REPO = {"0.8b": "Qwen/Qwen3.5-0.8B", "9b": "Qwen/Qwen3.5-9B"}
TARGETS = ["in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# the real configurations' fields used here, for runs without the Hub cache
FALLBACK_CONFIG = {
    "0.8b": dict(text=dict(hidden_size=1024, intermediate_size=3584, num_attention_heads=8, num_key_value_heads=2, head_dim=256,
                           linear_num_value_heads=16, linear_num_key_heads=16, linear_key_head_dim=128,
                           linear_value_head_dim=128, linear_conv_kernel_dim=4, vocab_size=248320),
                 vision=dict(hidden_size=768, intermediate_size=3072, num_heads=12, out_hidden_size=1024)),
    "9b": dict(text=dict(hidden_size=4096, intermediate_size=12288, num_attention_heads=16, num_key_value_heads=4, head_dim=256,
                         linear_num_value_heads=32, linear_num_key_heads=16, linear_key_head_dim=128,
                         linear_value_head_dim=128, linear_conv_kernel_dim=4, vocab_size=248320),
               vision=dict(hidden_size=1152, intermediate_size=4304, num_heads=16, out_hidden_size=4096)),
}
TOKENS = dict(image_token_id=248056, video_token_id=248057, vision_start_token_id=248053, vision_end_token_id=248054)


@pytest.fixture(scope="module", autouse=True)
def patched():
    macai.patch.install()
    yield
    macai.patch.uninstall()


def rel(a: torch.Tensor, r: torch.Tensor) -> float:
    return ((a - r).abs().max() / r.abs().max().clamp_min(1e-300)).item()


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
    """transformers' code casts to float32 in places (x.float(), .to(torch.float32)): for the float64 reference those casts
    keep float64 tensors as they are (CPU only, inside this block)."""
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


def config(model, text_layers=("linear_attention", "full_attention"), vision_depth=1, small=False):
    """A Qwen3.5 vision-language configuration: the real one's widths and vocabulary with two decoder layers and a one-block
    tower (small: a narrow text model and tower, for the cache's tests)."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    try:
        from huggingface_hub import hf_hub_download
        real = json.load(open(hf_hub_download(REPO[model], "config.json")))
        tc, vc = dict(real["text_config"]), dict(real["vision_config"])
    except Exception:  # noqa: BLE001 - offline without the cache: the fields above
        tc, vc = dict(FALLBACK_CONFIG[model]["text"]), dict(FALLBACK_CONFIG[model]["vision"])
        vc.update(patch_size=16, temporal_patch_size=2, spatial_merge_size=2, in_channels=3, num_position_embeddings=2304,
                  hidden_act="gelu_pytorch_tanh")
    tc.update(num_hidden_layers=len(text_layers), layer_types=list(text_layers), tie_word_embeddings=True)
    vc.update(depth=vision_depth)
    if small:
        tc.update(hidden_size=128, intermediate_size=256, num_attention_heads=2, num_key_value_heads=1, head_dim=64,
                  linear_num_value_heads=2, linear_num_key_heads=2, linear_key_head_dim=64, linear_value_head_dim=64)
        vc.update(hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=128, depth=2)
    cfg = Qwen3_5Config(text_config=tc, vision_config=vc, tie_word_embeddings=True, **TOKENS)
    for c in (cfg, cfg.text_config, cfg.vision_config):
        c._attn_implementation = "sdpa"
    return cfg


def build(model, seed=0, small=False, lora=True):
    """The model in bf16 values (float32 storage until moved), LoRA rank 32 with fixed dropout masks, B drawn (not zero) so
    every adapter weight has a gradient."""
    from peft import LoraConfig, inject_adapter_in_model
    cfg = config(model, small=small)
    torch.manual_seed(seed)
    m = M.Qwen3_5ForConditionalGeneration(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "conv1d" in n:
                p.normal_(0, 0.3)
            elif "A_log" in n or "dt_bias" in n:
                pass
            elif p.dim() > 1:
                p.normal_(0, 0.02)
            elif "linear_attn.norm" in n:
                p.normal_(1.0, 0.1)
            elif "norm" in n and "visual" not in n:
                p.normal_(0, 0.1)
    if lora:
        regex = r".*language_model.*\.(" + "|".join(TARGETS) + ")$"
        m = inject_adapter_in_model(LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target_modules=regex), m)
        for i, (n, mod) in enumerate(sorted(m.named_modules())):
            if n.endswith("lora_dropout") and isinstance(mod, torch.nn.ModuleDict):
                mod["default"] = FixedDropout(0.05, 1000 + i)
    for n, p in m.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.02)
        if "lora_" not in n:
            p.data = p.data.to(torch.bfloat16).float()
        p.requires_grad_("lora_" in n)
    return cfg, m


def batch(cfg, B=2, T=97, seed=3):
    """input ids, attention mask and labels as the record's collator makes them: each row a prompt (ignored), an answer
    (labelled), the shorter row padded on the right (ignored, masked)."""
    gen = torch.Generator().manual_seed(seed)
    V = cfg.text_config.vocab_size
    ids = torch.randint(0, 240000, (B, T), generator=gen)
    am = torch.ones(B, T, dtype=torch.long)
    labels = ids.clone()
    prompts, ends = [40, 30][:B] + [20] * max(0, B - 2), [T, T - 27][:B] + [T] * max(0, B - 2)
    for i in range(B):
        labels[i, :prompts[i]] = -100
        labels[i, ends[i]:] = -100
        am[i, ends[i]:] = 0
    assert ids.max() < V
    return ids, am, labels


def run(m, cfg, ids, am, labels, device, dtype, autocast, head, num_items=None):
    """One forward and backward: (loss, logits, the input embeddings' gradient, the final hidden states' gradient, the LoRA
    weights' gradients)."""
    emb = m.get_input_embeddings().weight.detach()
    x = emb[ids.to(emb.device)].to(device, dtype).detach().clone().requires_grad_(True)
    seen = {}

    def keep_final(_mod, _inp, out):
        out.retain_grad()
        seen["h"] = out
    hook = m.model.language_model.norm.register_forward_hook(keep_final)
    kw = dict(inputs_embeds=x, attention_mask=am.to(device), labels=labels.to(device))
    if num_items is not None:
        kw["num_items_in_batch"] = torch.tensor(num_items, device=device)
    ctx = torch.autocast(device, dtype=torch.bfloat16) if autocast else contextlib.nullcontext()
    try:
        with training.configured(HEAD_AT_LABELS=head), ctx:
            out = m(**kw)
        out.loss.backward()
    finally:
        hook.remove()
    grads = {n: p.grad.detach().cpu().double() for n, p in m.named_parameters() if p.requires_grad}
    return (out.loss.detach().cpu().double(), out.logits.detach().cpu(), x.grad.detach().cpu().double(),
            seen["h"].grad.detach().cpu().double(), grads)


@mps
@pytest.mark.parametrize("model", ["0.8b", "9b"])
def test_head_at_labels_against_float64(model):
    cfg, base = build(model)
    ids, am, labels = batch(cfg)
    shift = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    keep = shift != -100
    # the float64 reference: the model's own path, transformers' own rule and code, on the CPU
    ref = copy.deepcopy(base).double()
    rule = macai.patch._saved.get(M.__name__, M.torch_chunk_gated_delta_rule)
    saved_rule = M.torch_chunk_gated_delta_rule
    M.torch_chunk_gated_delta_rule = getattr(rule, "__wrapped__", rule)
    try:
        with layer.configured(**dict.fromkeys(layer.PARTS, False)), float64_everywhere():
            r_loss, _, r_dx, r_dh, r_gr = run(ref, cfg, ids, am, labels, "cpu", torch.float64, False, head=False)
    finally:
        M.torch_chunk_gated_delta_rule = saved_rule
    del ref
    res = {}
    for head in (False, True):
        m = copy.deepcopy(base)
        for n, p in m.named_parameters():
            if "lora_" not in n:
                p.data = p.data.to(torch.bfloat16)
        m = m.to("mps").train()
        before = dict(_head.STATS)
        res[head] = run(m, cfg, ids, am, labels, "mps", torch.bfloat16, True, head=head)
        assert _head.STATS["taken"] - before["taken"] == (1 if head else 0)
        del m
        torch.mps.empty_cache()
    off, on = res[False], res[True]
    # the logits: every position on the model's own path, the labelled ones (flattened) on the head's: the same bits there
    assert off[1].shape == (*ids.shape, cfg.text_config.vocab_size) and on[1].shape == (int(keep.sum()), cfg.text_config.vocab_size)
    assert torch.equal(on[1], off[1][keep])
    loss_rel = abs(on[0].item() - off[0].item()) / abs(off[0].item())
    rows = {"loss": (rel(off[0], r_loss), rel(on[0], r_loss)), "d inputs_embeds": (rel(off[2], r_dx), rel(on[2], r_dx)),
            "d final hidden": (rel(off[3], r_dh), rel(on[3], r_dh))}
    for n in r_gr:
        rows["d " + n.replace(".default.weight", "")] = (rel(off[4][n], r_gr[n]), rel(on[4][n], r_gr[n]))
    # the positions without a label get no gradient on either path
    assert (off[3][~keep] == 0).all() and (on[3][~keep] == 0).all()
    case = f"head at labels {model} 2x97"
    if LOG:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"case": case, "loss_rel_to_model_path": loss_rel,
                                 "errors": {k: {"pytorch": v[0], "fused": v[1]} for k, v in rows.items()}}) + "\n")
    assert loss_rel <= 1e-5, f"{case}: the loss {on[0].item()} against the model's own {off[0].item()}"
    bad = {k: v for k, v in rows.items() if k != "loss" and not v[1] <= 2 * v[0]}
    assert not bad, f"{case}: error over twice the model's own path's (model path, head at labels): {bad}"


@mps
def test_head_at_labels_num_items_in_batch():
    """The loss reduced over num_items_in_batch (a sum, then the division the model's loss function makes) as on the model's
    own path."""
    cfg, base = build("0.8b", small=True)
    ids, am, labels = batch(cfg)
    n_valid = int((torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:] != -100).sum())
    m = base.to("mps", torch.bfloat16).train()
    out = {}
    for head in (False, True):
        for p in m.parameters():
            p.grad = None
        out[head] = run(m, cfg, ids, am, labels, "mps", torch.bfloat16, True, head=head, num_items=3 * n_valid)
    assert abs(out[True][0].item() - out[False][0].item()) <= 1e-5 * abs(out[False][0].item())
    # a third of the mean: the sum over this batch's labelled positions over three times their count
    plain = run(m, cfg, ids, am, labels, "mps", torch.bfloat16, True, head=True)[0].item()
    assert abs(out[True][0].item() * 3 - plain) <= 1e-5 * plain


@mps
def test_head_at_labels_loss_arguments():
    """The loss function receives the keyword arguments the model's own path gives it, with shift_labels added."""
    from transformers.loss import loss_utils
    cfg, base = build("0.8b", small=True)
    ids, am, labels = batch(cfg)
    m = base.to("mps", torch.bfloat16).train()
    seen = []

    @functools.wraps(loss_utils.ForCausalLMLoss)
    def capture(*a, **k):
        seen.append(dict(k))
        return loss_utils.ForCausalLMLoss(*a, **k)
    owner = next(k for k in type(m).__mro__ if "loss_function" in k.__dict__)
    saved = owner.__dict__["loss_function"]
    setattr(owner, "loss_function", property(lambda self: capture))
    try:
        for head in (False, True):
            with training.configured(HEAD_AT_LABELS=head), torch.autocast("mps", dtype=torch.bfloat16):
                m(input_ids=ids.to("mps"), attention_mask=am.to("mps"), labels=labels.to("mps"), num_items_in_batch=7,
                  output_attentions=None)
    finally:
        setattr(owner, "loss_function", saved)
    a, b = seen
    assert set(b) - set(a) == {"shift_labels"} and set(a) <= set(b)
    assert b["num_items_in_batch"] == a["num_items_in_batch"] == 7 and b["vocab_size"] == a["vocab_size"]
    shift = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    assert torch.equal(b["shift_labels"].cpu(), shift[shift != -100]) and b["logits"].shape[0] == int((shift != -100).sum())


@mps
def test_head_at_labels_conditions():
    """The model's own path runs, bit for bit, with the setting off, in evaluation, without labels, with logits_to_keep or
    shift_labels given, or with a tuple output asked."""
    cfg, base = build("0.8b", small=True)
    ids, am, labels = batch(cfg)
    m = base.to("mps", torch.bfloat16).train()
    kw = dict(input_ids=ids.to("mps"), attention_mask=am.to("mps"))
    lab = labels.to("mps")

    def logits(head, train=True, **extra):
        m.train(train)
        with training.configured(HEAD_AT_LABELS=head), torch.no_grad(), torch.autocast("mps", dtype=torch.bfloat16):
            out = m(**kw, **extra)
        return out
    full = logits(False, labels=lab)
    TQ.FALLBACKS.clear()
    assert torch.equal(logits(True, train=False, labels=lab).logits, full.logits)          # evaluation
    assert torch.equal(logits(True).logits, full.logits)                                   # no labels
    shift = torch.nn.functional.pad(lab, (0, 1), value=-100)[..., 1:].contiguous()
    got = logits(True, labels=lab, shift_labels=shift)                                     # shift_labels given
    assert torch.equal(got.logits, full.logits)
    tup = logits(True, labels=lab, return_dict=False)                                      # a tuple output
    assert isinstance(tup, tuple) and torch.equal(tup[1], full.logits)
    reasons = {r for (_, r) in TQ.FALLBACKS}
    assert reasons == {"shift_labels given", "tuple output asked"}, TQ.FALLBACKS
    # logits_to_keep given (with labels the model's own path then needs matching shift_labels): the model's own path;
    # logits_to_keep=0, the default, given explicitly: the head at the labelled positions
    assert TQ._head_blocker(m, (), {"logits_to_keep": 1}, lab) == "logits_to_keep given"
    assert TQ._head_blocker(m, (), {"logits_to_keep": torch.ones(2, 3, dtype=torch.bool)}, lab) == "logits_to_keep given"
    assert TQ._head_blocker(m, (), {"logits_to_keep": 0}, lab) is None
    assert TQ._head_blocker(m, (ids,), {}, lab) == "positional arguments"
    # with the setting on: the loss first in the output, the labelled rows' logits only
    on = logits(True, labels=lab)
    assert list(on.keys())[0] == "loss" and on.logits.shape[0] == int((shift != -100).sum())
    assert abs(on.loss.item() - full.loss.item()) <= 1e-5 * abs(full.loss.item())


@mps
def test_head_at_labels_without_labels_in_batch():
    """A batch whose labels are all ignored: the model's own loss (the mean over no position) and the head's agree (NaN)."""
    cfg, base = build("0.8b", small=True)
    ids, am, labels = batch(cfg)
    labels[:] = -100
    m = base.to("mps", torch.bfloat16).train()
    out = {}
    for head in (False, True):
        with training.configured(HEAD_AT_LABELS=head), torch.no_grad(), torch.autocast("mps", dtype=torch.bfloat16):
            out[head] = m(input_ids=ids.to("mps"), attention_mask=am.to("mps"), labels=labels.to("mps"))
    assert torch.isnan(out[False].loss) and torch.isnan(out[True].loss) and out[True].logits.shape[0] == 0


def test_training_patches_installed():
    """Every source hashes as in the version written against, and every patch is in force."""
    assert set(macai.patch.training_patches()) == set(TQ.PATCHES)
    macai.patch.uninstall()
    try:
        assert all(TQ.check_sources().values()), TQ.check_sources()
    finally:
        macai.patch.install()
    assert set(macai.patch.training_patches()) == set(TQ.PATCHES)


def test_training_source_mismatch_leaves_original(monkeypatch):
    import warnings
    macai.patch.uninstall()
    try:
        monkeypatch.setitem(TQ.SOURCES, (TQ.MODULE, "Qwen3_5Model.forward"), "0" * 16)
        before = M.Qwen3_5Model.__dict__["forward"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            macai.patch.install()
        assert M.Qwen3_5Model.__dict__["forward"] is before
        assert "vision_cache" not in macai.patch.training_patches() and "head_conditional" in macai.patch.training_patches()
        assert any("Qwen3_5Model.forward" in str(x.message) for x in w)
    finally:
        macai.patch.uninstall()
        monkeypatch.undo()
        macai.patch.install()
    assert "vision_cache" in macai.patch.training_patches()


# ---- D9: the vision tower's output cached --------------------------------------------------------------------------------

def pictures(cfg, grids=((1, 4, 6), (1, 6, 4)), seed=5):
    """Two rows, a picture each: input ids with each picture's placeholder tokens between its start and end tokens, the token
    types, the grids, the pixel rows (float32, as the processor gives them)."""
    gen = torch.Generator().manual_seed(seed)
    m2 = cfg.vision_config.spatial_merge_size ** 2
    width = cfg.vision_config.in_channels * cfg.vision_config.temporal_patch_size * cfg.vision_config.patch_size ** 2
    rows, types = [], []
    for t, h, w in grids:
        n = t * h * w // m2
        ids = [11, 12, 13, TOKENS["vision_start_token_id"]] + [TOKENS["image_token_id"]] * n + [TOKENS["vision_end_token_id"], 14, 15]
        rows.append(ids)
        types.append([1 if x == TOKENS["image_token_id"] else 0 for x in ids])
    T = max(len(r) for r in rows)
    ids = torch.full((len(rows), T), 0, dtype=torch.long)
    am = torch.zeros(len(rows), T, dtype=torch.long)
    tt = torch.zeros(len(rows), T, dtype=torch.int)
    for i, (r, ty) in enumerate(zip(rows, types)):
        ids[i, :len(r)] = torch.tensor(r)
        am[i, :len(r)] = 1
        tt[i, :len(r)] = torch.tensor(ty, dtype=torch.int)
    grid = torch.tensor(grids, dtype=torch.long)
    pv = torch.randn(int(grid.prod(-1).sum()), width, generator=gen)
    return dict(input_ids=ids, attention_mask=am, mm_token_type_ids=tt, image_grid_thw=grid, pixel_values=pv)


def small_vlm():
    cfg, m = build("0.8b", small=True, lora=False)
    return cfg, m.to("mps", torch.bfloat16).train()


def forward_inner(m, inputs, cache):
    """The inner model's forward (where the cache sits), under bf16 autocast as in training: its last hidden state."""
    with training.configured(VISION_CACHE=cache), torch.no_grad(), torch.autocast("mps", dtype=torch.bfloat16):
        return m.model(**{k: v.to("mps") for k, v in inputs.items()}).last_hidden_state.float().cpu()


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """The same bits (NaN included): compared as integers of the dtype's width."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    iv = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return torch.equal(a.contiguous().view(iv), b.contiguous().view(iv))


@mps
def test_vision_cache_serves_the_same_bits():
    cfg, m = small_vlm()
    inner = m.model
    training.clear_vision_cache()
    inputs = pictures(cfg)
    plain = forward_inner(m, inputs, cache=False)
    first = forward_inner(m, inputs, cache=True)                  # computed and stored
    second = forward_inner(m, inputs, cache=True)                 # served
    st = VC.cache_of(inner).summary()
    assert st["computed_calls"] == 1 and st["served_calls"] == 1 and st["pictures_served"] == 2 and st["entries"] == 2
    assert bits_equal(plain, first) and bits_equal(plain, second)
    # what get_image_features returns, in full, against what the cache serves
    with torch.no_grad(), torch.autocast("mps", dtype=torch.bfloat16):
        pv, grid = inputs["pixel_values"].to("mps"), inputs["image_grid_thw"].to("mps")
        computed = inner.get_image_features(pv, grid, return_dict=True)
        served = VC.cache_of(inner).image_outputs(inner, pv, grid, {})
    assert bits_equal(computed.last_hidden_state.cpu(), served.last_hidden_state.cpu())
    assert len(served.pooler_output) == 2
    for a, b in zip(computed.pooler_output, served.pooler_output):
        assert bits_equal(a.cpu(), b.cpu())
    # the stored bytes: the last hidden rows and the merged features of both pictures
    want = computed.last_hidden_state.numel() * 2 + sum(p.numel() for p in computed.pooler_output) * 2
    assert VC.cache_of(inner).summary()["bytes_stored"] == want


@mps
def test_vision_cache_changed_picture_not_served():
    cfg, m = small_vlm()
    inner = m.model
    training.clear_vision_cache()
    inputs = pictures(cfg)
    forward_inner(m, inputs, cache=True)
    c = VC.cache_of(inner)
    # one pixel value changed in the first picture: computed again, its output the uncached one of the changed pixels
    changed = dict(inputs, pixel_values=inputs["pixel_values"].clone())
    changed["pixel_values"][3, 100] += 0.25
    before = c.summary()
    got = forward_inner(m, changed, cache=True)
    after = c.summary()
    assert after["computed_calls"] == before["computed_calls"] + 1 and after["served_calls"] == before["served_calls"]
    assert after["pictures_new"] == before["pictures_new"] + 1 and after["pictures_again"] == before["pictures_again"] + 1
    assert bits_equal(got, forward_inner(m, changed, cache=False))
    assert not bits_equal(got, forward_inner(m, inputs, cache=False))
    # the same pixels at another grid (the first picture's 4 x 6 patches read as 6 x 4): another key
    other = dict(inputs, image_grid_thw=torch.tensor([[1, 6, 4], [1, 6, 4]]))
    before = c.summary()
    forward_inner(m, other, cache=True)
    after = c.summary()
    assert after["computed_calls"] == before["computed_calls"] + 1 and after["pictures_new"] == before["pictures_new"] + 1
    # the pixels in another dtype: another key
    half = dict(inputs, pixel_values=inputs["pixel_values"].to(torch.bfloat16))
    before = c.summary()
    forward_inner(m, half, cache=True)
    assert c.summary()["computed_calls"] == before["computed_calls"] + 1


@mps
def test_vision_cache_conditions():
    """Not in evaluation, not while a tower parameter trains, not with the setting off; a weight change empties it."""
    cfg, m = small_vlm()
    inner = m.model
    training.clear_vision_cache()
    TQ.FALLBACKS.clear()
    inputs = pictures(cfg)
    forward_inner(m, inputs, cache=True)
    c = VC.cache_of(inner)
    calls = c.summary()["calls"]
    m.eval()
    forward_inner(m, inputs, cache=True)
    m.train()
    inner.visual.merger.linear_fc2.weight.requires_grad_(True)
    forward_inner(m, inputs, cache=True)
    inner.visual.merger.linear_fc2.weight.requires_grad_(False)
    forward_inner(m, inputs, cache=False)
    assert c.summary()["calls"] == calls
    assert {r for (_, r) in TQ.FALLBACKS} == {"not training", "the tower trains"}
    # a tower weight changed in place: the cache empties, the new weights' output computed
    assert c.summary()["entries"] == 2
    with torch.no_grad():
        inner.visual.blocks[0].mlp.linear_fc1.weight.mul_(1.5)
    got = forward_inner(m, inputs, cache=True)
    st = c.summary()
    assert st["emptied"] == 1 and st["computed_calls"] == 2 and bits_equal(got, forward_inner(m, inputs, cache=False))


@mps
def test_vision_tower_draws_no_random_numbers():
    cfg, m = small_vlm()
    inputs = pictures(cfg)
    cpu, gpu = torch.get_rng_state(), torch.mps.get_rng_state()
    with torch.no_grad(), torch.autocast("mps", dtype=torch.bfloat16):
        m.model.get_image_features(inputs["pixel_values"].to("mps"), inputs["image_grid_thw"].to("mps"), return_dict=True)
    assert torch.equal(cpu, torch.get_rng_state()) and torch.equal(gpu, torch.mps.get_rng_state())


@mps
def test_vision_cache_same_picture_twice_in_a_batch():
    cfg, m = small_vlm()
    inner = m.model
    training.clear_vision_cache()
    inputs = pictures(cfg, grids=((1, 4, 6), (1, 4, 6)))
    inputs["pixel_values"][24:] = inputs["pixel_values"][:24]
    forward_inner(m, inputs, cache=True)
    st = VC.cache_of(inner).summary()
    assert st["pictures_new"] == 1 and st["pictures_again"] == 1 and st["entries"] == 1
    forward_inner(m, inputs, cache=True)
    assert VC.cache_of(inner).summary()["pictures_served"] == 2


def test_digest_exact():
    """The key's digest: any byte changed, or the bytes cut short, gives another digest; chunked and whole agree on content."""
    data = bytearray(os.urandom(3 * VC.CHUNK + 17))
    d0 = VC.digest(memoryview(bytes(data)))
    data[VC.CHUNK + 5] ^= 1
    assert VC.digest(memoryview(bytes(data))) != d0
    data[VC.CHUNK + 5] ^= 1
    assert VC.digest(memoryview(bytes(data))) == d0
    assert VC.digest(memoryview(bytes(data[:-1]))) != d0


@mps
def test_vision_cache_cap():
    """Past macai.training.VISION_CACHE_MAX_GB nothing more is stored: the pictures are computed every time."""
    cfg, m = small_vlm()
    inner = m.model
    inputs = pictures(cfg)
    old = training.VISION_CACHE_MAX_GB
    training.VISION_CACHE_MAX_GB = 1e-9
    try:
        a = forward_inner(m, inputs, cache=True)
        b = forward_inner(m, inputs, cache=True)
        st = VC.cache_of(inner).summary()
        assert st["entries"] == 0 and st["not_stored"] == 4 and st["served_calls"] == 0 and st["computed_calls"] == 2
        assert bits_equal(a, b)
    finally:
        training.VISION_CACHE_MAX_GB = old
