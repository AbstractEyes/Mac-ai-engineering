"""Series design D3 (the rule's chunking and tiling; docs/series_layer.md): the correctness harness's cases (tests/test_harness.py:
shapes on and off the chunk, batch > 1, key / value widths 32-256, starting states or none, the numerical stress cases, nearly
parallel keys, 2,048 tokens) for the chunked pair under each new setting, at the harness's float64 bars (2e-4; stress and
correlated keys 5e-4; 2,048 tokens and the 4,096-token case here 1e-3):

  fwd8       op.FWD_CHUNK = 8: a forward that keeps no states runs at chunk 8 (the backward recomputes at chunk 16)
  dbuf       _chunked.DBUF = 1: the state passes' double-buffered staging, the next chunk's inputs read a chunk ahead
  dbuf2      _chunked.DBUF = 2: the same reads ahead into a single staging buffer
  split2/4/8 op.VALUE_SPLIT: a head-row's value columns over 2 / 4 / 8 threadgroups in the state passes
  auto       op.VALUE_SPLIT = "auto": the measured rule (_chunked.AUTO_SPLIT; at the harness's few head-rows every pass splits)
  and the combinations measured (fwd8+auto, fwd8+dbuf, split2+dbuf, split4+dbuf2).

Beside the bars: the settings that change no arithmetic (dbuf, the splits) give the current kernels' bits (every output and
gradient, NaN-aware exact equality on CPU copies); fwd8 gives their gradients' bits (its forward output rounds by its own
chunking). MACAI_PARITY_LOG=<file> appends each case's worst errors (the variant's and the current kernels') as JSON lines."""
import json
import os

import pytest
import torch

from macai.gated_delta import _chunked, chunk_gated_delta_rule, l2norm, op, recurrent_reference
from test_harness import check, make
from test_memory import NAMES, identical

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")
LOG = os.environ.get("MACAI_PARITY_LOG")
BASE = {"FWD_CHUNK": 16, "VALUE_SPLIT": 1, "DBUF": 0}
VARIANTS = {"fwd8": {"FWD_CHUNK": 8}, "dbuf": {"DBUF": 1}, "dbuf2": {"DBUF": 2}, "split2": {"VALUE_SPLIT": 2},
            "split4": {"VALUE_SPLIT": 4}, "split8": {"VALUE_SPLIT": 8}, "auto": {"VALUE_SPLIT": "auto"},
            "fwd8+auto": {"FWD_CHUNK": 8, "VALUE_SPLIT": "auto"}, "fwd8+dbuf": {"FWD_CHUNK": 8, "DBUF": 1},
            "split2+dbuf": {"VALUE_SPLIT": 2, "DBUF": 1}, "split4+dbuf2": {"VALUE_SPLIT": 4, "DBUF": 2}}
SAME_BITS = [v for v in VARIANTS if "fwd8" not in v]


class settings:
    def __init__(self, **values):
        self.values = {**BASE, **values}

    def __enter__(self):
        self.old = {}
        for k, v in self.values.items():
            mod = _chunked if k == "DBUF" else op
            self.old[(mod, k)] = getattr(mod, k)
            setattr(mod, k, v)

    def __exit__(self, *exc):
        for (mod, k), v in self.old.items():
            setattr(mod, k, v)


def log(case, variant, got, base):
    if LOG:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"test": "test_chunking", "case": case, "variant": variant, "errors": got, "base_errors": base}) + "\n")


CASES = {f"shape{s}": (dict(shape=s), 2e-4) for s in [(1, 1, 1, 128, 128), (1, 2, 1, 128, 128), (2, 7, 3, 128, 128),
                                                     (1, 8, 2, 128, 128), (1, 9, 2, 128, 128), (3, 63, 2, 128, 128),
                                                     (1, 64, 1, 128, 128), (2, 65, 2, 128, 64), (1, 100, 5, 64, 64),
                                                     (1, 31, 2, 64, 32), (1, 33, 2, 256, 128)]}
CASES["no_starting_state"] = (dict(shape=(2, 40, 2, 128, 128), state=False), 2e-4)
for _b, _g in [(0.0, None), (1.0 - 1e-6, None), (None, 0.0), (None, -20.0), (1.0 - 1e-6, 0.0)]:
    CASES[f"stress_beta{_b}_g{_g}"] = (dict(shape=(1, 48, 2, 128, 128), beta=_b, g=_g), 5e-4)
for _b, _g in [(None, None), (1.0 - 1e-6, 0.0), (1.0 - 1e-6, None)]:
    CASES[f"correlated_beta{_b}_g{_g}"] = (dict(shape=(1, 96, 2, 128, 128), beta=_b, g=_g, corr=0.9, seed=6), 5e-4)
CASES["long_2048"] = (dict(shape=(1, 2048, 1, 128, 128), seed=3), 1e-3)


def inputs(case):
    c = dict(CASES[case][0])
    return make(*c.pop("shape"), **c)


def rule_bits(xs64, **values) -> list:
    """GatedDeltaRule under these settings (the chunked pair, the recomputed states): o, S_T and the six gradients, CPU copies."""
    with settings(**values):
        old = op.KERNEL, op.KERNEL_BWD, op.RECOMPUTE_STATES
        op.KERNEL, op.KERNEL_BWD, op.RECOMPUTE_STATES = "chunked", "chunked", True
        try:
            xs = [x.float().to("mps").requires_grad_(True) for x in xs64]
            o, sT = op.GatedDeltaRule.apply(*xs, True)
            gen = torch.Generator().manual_seed(9)
            ((o * torch.randn(o.shape, generator=gen).to("mps")).sum() + (sT * torch.randn(sT.shape, generator=gen).to("mps")).sum()).backward()
            return [t.detach().cpu() for t in (o, sT, *(x.grad for x in xs))]
        finally:
            op.KERNEL, op.KERNEL_BWD, op.RECOMPUTE_STATES = old


_BASE = {}      # the current kernels' errors a case (measured once)


@mps
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("case", list(CASES))
def test_harness_case(case, variant):
    """The chunked pair under the variant against the float64 step-by-step reference: outputs, final state, every gradient."""
    xs64 = inputs(case)
    rtol = CASES[case][1]
    if case not in _BASE:
        with settings():
            _BASE[case] = check(xs64, ("chunked", "chunked"), rtol=rtol)
    with settings(**VARIANTS[variant]):
        got = check(xs64, ("chunked", "chunked"), rtol=rtol)
    log(case, variant, got, _BASE[case])


@mps
@pytest.mark.parametrize("variant", SAME_BITS)
@pytest.mark.parametrize("case", [c for c in CASES if c != "long_2048"])
def test_same_bits(case, variant):
    """Double buffering and the splits change no arithmetic: every output and gradient is the current kernels' bits."""
    xs64 = inputs(case)
    got, want = rule_bits(xs64, **VARIANTS[variant]), rule_bits(xs64)
    bad = [n for n, a, b in zip(NAMES, got, want) if not identical(a, b)]
    assert not bad, f"{variant} {case}: not bit-identical: {bad}"


@mps
@pytest.mark.parametrize("variant", ["fwd8", "fwd8+dbuf"])
@pytest.mark.parametrize("case", [c for c in CASES if c != "long_2048"])
def test_fwd8_gradients_same_bits(case, variant):
    """Chunk 8 in the forward: the backward recomputes at chunk 16 from the same inputs, so every gradient is the current
    kernels' bits (the output and the final state are held to the harness's bars by test_harness_case)."""
    xs64 = inputs(case)
    got, want = rule_bits(xs64, **VARIANTS[variant]), rule_bits(xs64)
    bad = [n for n, a, b in zip(NAMES[2:], got[2:], want[2:]) if not identical(a, b)]
    assert not bad, f"{variant} {case}: gradients not bit-identical: {bad}"


@mps
@pytest.mark.parametrize("variant", ["fwd8+auto", "split2+dbuf", "split4"])
def test_long_4096(variant):
    """4,096 tokens, one row, two heads (the long sequences of D3 (b)), against float64 at 1e-3."""
    xs64 = make(1, 4096, 2, 128, 128, seed=11)
    with settings():
        base = check(xs64, ("chunked", "chunked"), rtol=1e-3)
    with settings(**VARIANTS[variant]):
        got = check(xs64, ("chunked", "chunked"), rtol=1e-3)
    log("long_4096", variant, got, base)


@mps
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dropin(dtype, variant):
    """The drop-in end to end (the layer's inputs: q, k in dtype, v, g, beta float32; l2 normalization in the call; a starting
    state; both outputs used), every input's gradient against float64 autograd through l2norm and the recurrence."""
    gen = torch.Generator().manual_seed(8)
    B, T, H, D = 2, 45, 3, 128
    raw = [torch.randn(B, T, H, D, generator=gen).to(dtype) for _ in range(2)] + [torch.randn(B, T, H, D, generator=gen)]
    g = -torch.rand(B, T, H, generator=gen)
    beta = torch.rand(B, T, H, generator=gen)
    s0 = torch.randn(B, H, D, D, generator=gen) * 0.1
    w_o = torch.randn(B, T, H, D, generator=gen, dtype=torch.float64)
    w_s = torch.randn(B, H, D, D, generator=gen, dtype=torch.float64)
    ref = [x.double().requires_grad_(True) for x in (*raw, g, beta, s0)]
    ro, rsT = recurrent_reference(l2norm(ref[0]) * D ** -0.5, l2norm(ref[1]), *ref[2:])
    ((ro * w_o).sum() + (rsT * w_s).sum()).backward()
    errs = {}
    for label, values in (("base", {}), (variant, VARIANTS[variant])):
        with settings(**values):
            xs = [x.to("mps").requires_grad_(True) for x in (*raw, g, beta, s0)]
            o, sT = chunk_gated_delta_rule(*xs[:5], initial_state=xs[5], output_final_state=True, use_qk_l2norm_in_kernel=True)
            ((o.float() * w_o.float().to("mps")).sum() + (sT * w_s.float().to("mps")).sum()).backward()
        e = {"o": ((o.detach().cpu().double() - ro).abs().max() / ro.abs().max()).item()}
        for name, a, b in zip(("dquery", "dkey", "dvalue", "dg", "dbeta", "dstate"), xs, ref):
            e[name] = ((a.grad.cpu().double() - b.grad).abs().max() / b.grad.abs().max()).item()
        errs[label] = e
    tol = {torch.float32: 2e-4, torch.bfloat16: 2e-2}[dtype]
    bad = {k: v for k, v in errs[variant].items() if not v < tol}
    log(f"dropin_{str(dtype).split('.')[-1]}", variant, errs[variant], errs["base"])
    assert not bad, (variant, bad)


@mps
def test_launch_rules():
    """The split's threadgroups, the measured split rule, the forward's chunk length by what the forward keeps."""
    assert _chunked.state_simdgroups(128) == 16
    assert [_chunked.state_simdgroups(128, s) for s in (1, 2, 4, 8, 16)] == [16, 8, 4, 2, 1]
    assert _chunked.state_simdgroups(64, 4) == 2 and _chunked.state_simdgroups(32, 8) == 1
    with settings(VALUE_SPLIT=4):
        g = op._state_grid(128, 16, 1, op.split_for(16, 4096, 128))
        assert g["group_size"][0] == 128 and g["threads"][0] == 4 * 128
    with settings(VALUE_SPLIT="auto"):
        # the measured rule: chunk-16 passes split at <= 16 head-rows, the chunk-8 forward at <= 64
        assert [op.split_for(r, 4096, 128, 16) for r in (1, 16, 17, 32, 64)] == [2, 2, 1, 1, 1]
        assert [op.split_for(r, 811, 128, 8) for r in (16, 32, 64, 65, 128)] == [2, 2, 2, 1, 1]
        assert op.split_for(32, 811, 128) == 1                                      # the default chunk is the backward's
        assert op.split_for(16, 4096, 32, 16) == 2 and op.split_for(16, 4096, 8, 16) == 1     # capped at the column blocks
    with settings(FWD_CHUNK=8):
        assert op.plan(1, 100, 2, 128, 128, save=False)["fwd_chunk"] == 8
        assert op.plan(1, 100, 2, 128, 128, save=True)["fwd_chunk"] == 8          # the recomputed states keep nothing
        old = op.RECOMPUTE_STATES
        op.RECOMPUTE_STATES = False
        try:
            assert op.plan(1, 100, 2, 128, 128, save=True)["fwd_chunk"] == 16     # the stored states are the backward's
        finally:
            op.RECOMPUTE_STATES = old
