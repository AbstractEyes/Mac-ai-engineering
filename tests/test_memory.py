"""The memory designs (docs/series_memory.md) against the stored path, bit for bit: size classes (op.SIZE_CLASSES), the chunk
states recomputed in the backward (op.RECOMPUTE_STATES) and both together must give every output and every gradient of the
stored path (both settings off, the chunked kernels) exactly, on the harness's shapes (tests/test_harness.py: lengths on and
off the chunk, batch > 1, key / value head widths 32-256, starting states or none, the numerical stress cases, 2,048 tokens)
and through the drop-in (float32, bfloat16 and float16 inputs, the fused and the PyTorch prologue, contiguous inputs and the
model's strided views). Compared on CPU copies: NaN-aware exact equality (NaN equals NaN at the same place) and the bits
themselves (signed zeros and NaN payloads included). A further set checks that each design is in force (the saved buffers'
storage sizes, what the forward keeps)."""
import contextlib

import pytest
import torch

from macai.gated_delta import _memory, chunk_gated_delta_rule, l2norm, op

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")
STORED = dict(KERNEL="chunked", KERNEL_BWD="chunked", FUSED_NORM=True, SIZE_CLASSES=False, RECOMPUTE_STATES=False)
DESIGNS = {"classes": dict(SIZE_CLASSES=True, RECOMPUTE_STATES=False),
           "recompute": dict(SIZE_CLASSES=False, RECOMPUTE_STATES=True),
           "classes+recompute": dict(SIZE_CLASSES=True, RECOMPUTE_STATES=True)}
_INT = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16}


@contextlib.contextmanager
def settings(**values):
    old = {k: getattr(op, k) for k in values}
    for k, v in values.items():
        setattr(op, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(op, k, v)


def identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    na, nb = torch.isnan(a), torch.isnan(b)
    if not (torch.equal(na, nb) and torch.equal(a[~na], b[~nb])):
        return False
    return torch.equal(a.view(_INT[a.dtype]), b.view(_INT[b.dtype]))


def make(B, T, H, DK, DV, seed=0, beta=None, g=None, state=True, corr=0.0):
    """The harness's inputs (tests/test_harness.py make), float64 on the CPU."""
    gen = torch.Generator().manual_seed(seed)
    q = l2norm(torch.randn(B, T, H, DK, generator=gen, dtype=torch.float64)) * DK ** -0.5
    k = torch.randn(B, T, H, DK, generator=gen, dtype=torch.float64)
    if corr:
        k = corr * torch.randn(B, 1, H, DK, generator=gen, dtype=torch.float64) + (1 - corr) * k
    k = l2norm(k)
    v = torch.randn(B, T, H, DV, generator=gen, dtype=torch.float64)
    gg = (-torch.rand(B, T, H, generator=gen, dtype=torch.float64)) if g is None else torch.full((B, T, H), float(g), dtype=torch.float64)
    bb = torch.rand(B, T, H, generator=gen, dtype=torch.float64) if beta is None else torch.full((B, T, H), float(beta), dtype=torch.float64)
    s0 = (torch.randn(B, H, DK, DV, generator=gen, dtype=torch.float64) * 0.1) if state else torch.zeros(B, H, DK, DV, dtype=torch.float64)
    return q, k, v, gg, bb, s0


def rule(xs64, grads: bool, **values) -> list:
    """GatedDeltaRule under these settings: o, S_T and (grads) the six input gradients, as CPU copies."""
    with settings(**{**STORED, **values}):
        xs = [x.float().to("mps").requires_grad_(grads) for x in xs64]
        if not grads:
            with torch.no_grad():
                return [t.cpu() for t in op.GatedDeltaRule.apply(*xs, False)]
        o, sT = op.GatedDeltaRule.apply(*xs, True)
        gen = torch.Generator().manual_seed(9)
        w_o = torch.randn(o.shape, generator=gen).to("mps")
        w_s = torch.randn(sT.shape, generator=gen).to("mps")
        ((o * w_o).sum() + (sT * w_s).sum()).backward()
        return [t.detach().cpu() for t in (o, sT, *(x.grad for x in xs))]


NAMES = ("o", "sT", "dq", "dk", "dv", "dg", "dbeta", "ds0")
CASES = {f"shape{s}": dict(shape=s) for s in [(1, 1, 1, 128, 128), (1, 2, 1, 128, 128), (2, 7, 3, 128, 128), (1, 8, 2, 128, 128),
                                             (1, 9, 2, 128, 128), (3, 63, 2, 128, 128), (1, 64, 1, 128, 128),
                                             (2, 65, 2, 128, 64), (1, 100, 5, 64, 64), (1, 31, 2, 64, 32),
                                             (1, 33, 2, 256, 128), (1, 2048, 1, 128, 128)]}
CASES["no_starting_state"] = dict(shape=(2, 40, 2, 128, 128), state=False)
for _b, _g in [(0.0, None), (1.0 - 1e-6, None), (None, 0.0), (None, -20.0), (1.0 - 1e-6, 0.0)]:
    CASES[f"stress_beta{_b}_g{_g}"] = dict(shape=(1, 48, 2, 128, 128), beta=_b, g=_g)
CASES["correlated_keys"] = dict(shape=(1, 96, 2, 128, 128), corr=0.9, seed=6)


def case_inputs(case: dict):
    c = dict(case)
    return make(*c.pop("shape"), **c)


def assert_identical(got, want, label):
    bad = [n for n, a, b in zip(NAMES, got, want) if not identical(a, b)]
    assert len(got) == len(want) and not bad, f"{label}: not bit-identical to the stored path: {bad}"


@mps
@pytest.mark.parametrize("design", list(DESIGNS))
@pytest.mark.parametrize("case", list(CASES))
def test_rule_bit_identical(case, design):
    xs64 = case_inputs(CASES[case])
    for grads in (True, False):
        assert_identical(rule(xs64, grads, **DESIGNS[design]), rule(xs64, grads), f"{design} {case} grads={grads}")


@mps
@pytest.mark.parametrize("design", list(DESIGNS))
@pytest.mark.parametrize("pair", [("auto", "chunked"), ("column", "chunked"), ("chunked", "head")])
@pytest.mark.parametrize("shape", [(2, 65, 2, 128, 64), (1, 31, 2, 64, 32), (3, 63, 2, 128, 128)])
def test_other_pairs_bit_identical(shape, pair, design):
    """The kernel pairs with a deterministic backward: size classes cover the chunked side of a mixed pair (the chunked
    backward's recomputation after a token-at-a-time forward, the chunked forward's 8-token states); the recomputed states
    apply to the chunked pair only, so here they must change nothing."""
    if pair[1] == "head" and not op.head_shape(shape[3], shape[4]):
        pytest.skip("the head backward does not take this shape")
    xs64 = make(*shape, seed=sum(shape))
    kernels = dict(KERNEL=pair[0], KERNEL_BWD=pair[1])
    assert_identical(rule(xs64, True, **kernels, **DESIGNS[design]), rule(xs64, True, **kernels), f"{design} {pair} {shape}")


@mps
@pytest.mark.parametrize("design", list(DESIGNS))
def test_column_backward_pair(design):
    """The chunked forward with the atomics-reduced column backward (not bitwise reproducible run to run, so the gradients
    are held to the harness's run-to-run bound): the forward bit-identical, the gradients within 1e-4."""
    xs64 = make(2, 64, 2, 128, 128, seed=5)
    kernels = dict(KERNEL="chunked", KERNEL_BWD="column")
    got, want = rule(xs64, True, **kernels, **DESIGNS[design]), rule(xs64, True, **kernels)
    assert identical(got[0], want[0]) and identical(got[1], want[1])
    assert max((a - b).abs().max().item() for a, b in zip(got[2:], want[2:])) < 1e-4


def model_layout(B, T, H, D, dtype, gen):
    """query, key, value as the Qwen3.5 layer passes them: views of a split of the transposed convolution output."""
    mixed = torch.randn(B, 3 * H * D, T, generator=gen).to("mps", dtype).transpose(1, 2)
    q, k, v = torch.split(mixed, [H * D, H * D, H * D], dim=-1)
    return [x.reshape(B, T, H, D) for x in (q, k, v)]


def dropin(dtype, layout, fused, **values) -> list:
    gen = torch.Generator().manual_seed(8)
    B, T, H, D = 2, 45, 3, 128
    with settings(**{**STORED, "FUSED_NORM": fused, **values}):
        if layout == "contiguous":
            leaves = [torch.randn(B, T, H, D, generator=gen).to("mps", dtype).requires_grad_(True) for _ in range(3)]
            q, k, v = leaves
            beta = torch.rand(B, T, H, generator=gen).to("mps").requires_grad_(True)
            s0 = (torch.randn(B, H, D, D, generator=gen) * 0.1).to("mps").requires_grad_(True)
            leaves += [beta, s0]
            kw = dict(initial_state=s0, output_final_state=True)
        else:
            raw = torch.randn(B, 3 * H * D, T, generator=gen).to("mps", dtype).requires_grad_(True)
            mixed = raw.transpose(1, 2)
            q, k, v = (x.reshape(B, T, H, D) for x in torch.split(mixed, [H * D] * 3, dim=-1))
            beta = torch.rand(B, T, H, generator=gen).to("mps", dtype).requires_grad_(True)   # the layer's b.sigmoid()
            leaves = [raw, beta]
            kw = dict(output_final_state=False)
        g = (-torch.rand(B, T, H, generator=gen)).to("mps").requires_grad_(True)
        leaves.append(g)
        o, sT = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True, **kw)
        w_o = torch.randn(o.shape, generator=gen).to("mps")
        loss = (o.float() * w_o).sum()
        if sT is not None:
            loss = loss + (sT * torch.randn(sT.shape, generator=gen).to("mps")).sum()
        loss.backward()
        return [t.detach().cpu() for t in (o, *([sT] if sT is not None else []), *(x.grad for x in leaves))]


@mps
@pytest.mark.parametrize("design", list(DESIGNS))
@pytest.mark.parametrize("layout", ["contiguous", "model"])
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dropin_bit_identical(dtype, fused, layout, design):
    got, want = dropin(dtype, layout, fused, **DESIGNS[design]), dropin(dtype, layout, fused)
    bad = [i for i, (a, b) in enumerate(zip(got, want)) if not identical(a, b)]
    assert len(got) == len(want) and not bad, f"{design} {dtype} fused={fused} {layout}: tensors {bad} differ"


def test_size_classes():
    """Four classes a doubling, whole pages, the smallest class that holds the size; at most 19% over from 10 MiB up."""
    import math
    import random
    prev = 0
    for j in range(80, 128):
        c = _memory.class_bytes(int(2 ** (j / 4)))
        assert c % _memory.PAGE == 0 and c >= prev
        prev = c
    assert len({_memory.class_bytes(int(2 ** (x / 64))) for x in range(64 * 27 + 1, 64 * 28 + 1)}) == 4    # (128, 256] MiB
    rng = random.Random(0)
    for _ in range(20000):
        n = int(math.exp(rng.uniform(math.log(10 * 2 ** 20), math.log(2 ** 31))))
        c = _memory.class_bytes(n)
        assert n <= c <= 1.19 * n
        assert _memory.class_bytes(c) == c and _memory.class_bytes(c + 1) > c


@mps
def test_size_classes_in_force():
    """With size classes the forward's output and every saved buffer that follows the length sit in storage of their class's
    size; without them, of their own size."""
    xs = [x.float().to("mps").requires_grad_(True) for x in make(2, 650, 4, 128, 128)]
    for classes in (False, True):
        with settings(**{**STORED, "SIZE_CLASSES": classes}):
            o, sT = op.GatedDeltaRule.apply(*xs, True)
            saved = o.grad_fn.saved_tensors      # q, k, beta (the caller's tensors), G, A, KG, M, the states, Vn, Xu
            assert len(saved) == 10
            for t in [o, *saved[3:]]:
                nbytes = t.numel() * t.element_size()
                want = _memory.class_bytes(nbytes) if classes else nbytes
                assert t.untyped_storage().nbytes() == want and t.is_contiguous() and t.storage_offset() == 0


@mps
def test_recompute_keeps_only_inputs():
    """With the states recomputed, the training forward keeps its six inputs and nothing a chunk."""
    xs = [x.float().to("mps").requires_grad_(True) for x in make(2, 650, 4, 128, 128)]
    with settings(**{**STORED, "RECOMPUTE_STATES": True}):
        o, sT = op.GatedDeltaRule.apply(*xs, True)
        saved = o.grad_fn.saved_tensors
        assert len(saved) == 6 and all(any(s is x or s.data_ptr() == x.data_ptr() for x in xs) for s in saved)
    with settings(**STORED):
        o, sT = op.GatedDeltaRule.apply(*xs, True)
        assert max(t.numel() for t in o.grad_fn.saved_tensors) == 2 * 4 * 41 * 128 * 128      # the chunk states
