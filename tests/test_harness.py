"""Phase 2: the correctness harness for the gated delta rule kernels. Every kernel pairing against a float64 step-by-step
reference on the CPU: shapes (lengths that are and are not multiples of the block and the chunk, batch, heads, head widths),
starting states, numerical stress (beta at 0 and near 1, no decay and very strong decay, keys nearly parallel, long sequences),
input dtypes through the drop-in, the packed-rows guard, the run-to-run spread of the atomics-reduced backward, and the
bitwise run-to-run identity of the chunked backward (no atomics)."""
import pytest
import torch

from macai.gated_delta import chunk_gated_delta_rule, l2norm, op, recurrent_reference

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")
# (forward, backward): the token-at-a-time kernels (auto = one threadgroup a head, column), the chunked kernels, and the mixed
# pairs (a token-at-a-time forward's checkpoints read by the chunked backward; the chunked forward's 8-token states read by
# the token-at-a-time backwards)
PAIRS = [("auto", "column"), ("auto", "head"), ("column", "column"), ("chunked", "chunked"), ("auto", "chunked"),
         ("column", "chunked"), ("chunked", "column"), ("chunked", "head")]


def make(B, T, H, DK, DV, seed=0, beta=None, g=None, state=True, corr=0.0):
    """corr > 0: every key is a shared direction plus (1 - corr) noise, so k_i . k_j -> 1 (the chunked inverse's hard case)."""
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


def check(xs64, pair, grads=True, rtol=2e-4):
    """The kernel (float32, on the GPU) against the float64 reference: outputs, final state and every gradient, each within
    rtol of the reference's largest magnitude."""
    old = op.KERNEL, op.KERNEL_BWD
    op.KERNEL, op.KERNEL_BWD = pair
    try:
        xs = [x.float().to("mps").requires_grad_(grads) for x in xs64]
        o, sT = op.GatedDeltaRule.apply(*xs, grads)
        ref = [x.clone().requires_grad_(grads) for x in xs64]
        ro, rsT = recurrent_reference(*ref)
        worst = {}
        for name, a, b in (("o", o, ro), ("sT", sT, rsT)):
            worst[name] = ((a.cpu().double() - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()
        if grads:
            gen = torch.Generator().manual_seed(9)
            w_o = torch.randn(ro.shape, generator=gen, dtype=torch.float64)
            w_s = torch.randn(rsT.shape, generator=gen, dtype=torch.float64)
            ((o * w_o.float().to("mps")).sum() + (sT * w_s.float().to("mps")).sum()).backward()
            ((ro * w_o).sum() + (rsT * w_s).sum()).backward()
            for name, a, b in zip(("dq", "dk", "dv", "dg", "dbeta", "ds0"), xs, ref):
                worst[name] = ((a.grad.cpu().double() - b.grad).abs().max() / b.grad.abs().max().clamp_min(1e-12)).item()
        bad = {k: v for k, v in worst.items() if not v <= rtol}
        assert not bad, f"{pair}: relative errors over {rtol}: {bad}"
        return worst
    finally:
        op.KERNEL, op.KERNEL_BWD = old


@mps
@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("shape", [(1, 1, 1, 128, 128), (1, 2, 1, 128, 128), (2, 7, 3, 128, 128), (1, 8, 2, 128, 128),
                                   (1, 9, 2, 128, 128), (3, 63, 2, 128, 128), (1, 64, 1, 128, 128), (2, 65, 2, 128, 64),
                                   (1, 100, 5, 64, 64), (1, 31, 2, 64, 32)])
def test_shapes(shape, pair):
    check(make(*shape), pair)


@mps
@pytest.mark.parametrize("pair", PAIRS)
def test_no_starting_state(pair):
    check(make(2, 40, 2, 128, 128, state=False), pair)


@mps
@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("beta,g", [(0.0, None), (1.0 - 1e-6, None), (None, 0.0), (None, -20.0), (1.0 - 1e-6, 0.0)])
def test_stress(beta, g, pair):
    check(make(1, 48, 2, 128, 128, beta=beta, g=g), pair, rtol=5e-4)


@mps
@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("beta,g", [(None, None), (1.0 - 1e-6, 0.0), (1.0 - 1e-6, None)])
def test_correlated_keys(beta, g, pair):
    """Keys nearly parallel (k_i . k_j about 0.99): the off-diagonal entries of the chunked inverse's matrix approach 1."""
    xs = make(1, 96, 2, 128, 128, beta=beta, g=g, corr=0.9, seed=6)
    assert (xs[1][0, 1:, 0] * xs[1][0, :-1, 0]).sum(-1).min() > 0.95
    check(xs, pair, rtol=5e-4)


@mps
@pytest.mark.parametrize("pair", PAIRS)
def test_long_sequence(pair):
    check(make(1, 2048, 1, 128, 128, seed=3), pair, rtol=1e-3)


@mps
@pytest.mark.parametrize("kernel", ["auto", "column", "chunked"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dropin_dtypes(dtype, kernel, monkeypatch):
    monkeypatch.setattr(op, "KERNEL", kernel)
    gen = torch.Generator().manual_seed(4)
    B, T, H, D = 2, 50, 3, 128
    q, k, v = (torch.randn(B, T, H, D, generator=gen).to("mps", dtype) for _ in range(3))
    g = (-torch.rand(B, T, H, generator=gen)).to("mps")
    beta = torch.rand(B, T, H, generator=gen).to("mps")
    o, sT = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
    qq = l2norm(q.cpu().double()) * D ** -0.5
    ro, rsT = recurrent_reference(qq, l2norm(k.cpu().double()), v.cpu().double(), g.cpu().double(), beta.cpu().double(),
                                  torch.zeros(B, H, D, D, dtype=torch.float64))
    assert o.dtype == dtype and sT.dtype == torch.float32
    tol = {torch.float32: 1e-4, torch.bfloat16: 2e-2, torch.float16: 5e-3}[dtype]
    assert (o.cpu().double() - ro).abs().max() / ro.abs().max() < tol


@mps
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_fused_norm(dtype):
    """The fused query / key prologue against float64 autograd through l2norm and the scaling: values and gradients."""
    from macai.gated_delta import _norm
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(3, 17, 5, 128, generator=gen).to(dtype)
    dy = torch.randn(3, 17, 5, 128, generator=gen, dtype=torch.float64)
    xm = x.to("mps").requires_grad_(True)
    y = _norm.L2NormScale.apply(xm, 128 ** -0.5)
    (y * dy.float().to("mps")).sum().backward()
    x64 = x.double().requires_grad_(True)
    y64 = l2norm(x64) * 128 ** -0.5
    (y64 * dy).sum().backward()
    assert y.dtype == torch.float32 and xm.grad.dtype == dtype
    assert ((y.cpu().double() - y64).abs().max() / y64.abs().max()).item() < 1e-6
    tol = {torch.float32: 1e-5, torch.bfloat16: 1e-2, torch.float16: 2e-3}[dtype]
    assert ((xm.grad.cpu().double() - x64.grad).abs().max() / x64.grad.abs().max()).item() < tol


@mps
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("pair", [("chunked", "chunked"), ("auto", "column")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dropin_gradients(dtype, pair, fused, monkeypatch):
    """The drop-in end to end (l2 normalization in the call, a starting state, both outputs used), every input's gradient
    against float64 autograd through l2norm and the step-by-step recurrence."""
    monkeypatch.setattr(op, "KERNEL", pair[0])
    monkeypatch.setattr(op, "KERNEL_BWD", pair[1])
    monkeypatch.setattr(op, "FUSED_NORM", fused)
    gen = torch.Generator().manual_seed(8)
    B, T, H, D = 2, 45, 3, 128
    raw = [torch.randn(B, T, H, D, generator=gen).to(dtype) for _ in range(3)]
    g = -torch.rand(B, T, H, generator=gen)
    beta = torch.rand(B, T, H, generator=gen)
    s0 = torch.randn(B, H, D, D, generator=gen) * 0.1
    w_o = torch.randn(B, T, H, D, generator=gen, dtype=torch.float64)
    w_s = torch.randn(B, H, D, D, generator=gen, dtype=torch.float64)
    xs = [x.to("mps").requires_grad_(True) for x in (*raw, g, beta, s0)]
    o, sT = chunk_gated_delta_rule(*xs[:5], initial_state=xs[5], output_final_state=True, use_qk_l2norm_in_kernel=True)
    ((o.float() * w_o.float().to("mps")).sum() + (sT * w_s.float().to("mps")).sum()).backward()
    ref = [x.double().requires_grad_(True) for x in (*raw, g, beta, s0)]
    ro, rsT = recurrent_reference(l2norm(ref[0]) * D ** -0.5, l2norm(ref[1]), *ref[2:])
    ((ro * w_o).sum() + (rsT * w_s).sum()).backward()
    tol = {torch.float32: 2e-4, torch.bfloat16: 2e-2}[dtype]
    for name, a, b in zip(("dquery", "dkey", "dvalue", "dg", "dbeta", "dstate"), xs, ref):
        err = ((a.grad.cpu().double() - b.grad).abs().max() / b.grad.abs().max()).item()
        assert err < tol, (name, err)


@mps
def test_packed_rows_refused():
    x = torch.zeros(1, 4, 1, 128, device="mps")
    with pytest.raises(NotImplementedError):
        chunk_gated_delta_rule(x, x, x, x[..., 0], x[..., 0], cu_seqlens=torch.tensor([0, 2, 4]))


@mps
def test_run_to_run_spread():
    """The column backward sums across threadgroups with float atomics (order not fixed): the spread between runs is reported
    and bounded."""
    xs64 = make(2, 64, 2, 128, 128, seed=5)
    old = op.KERNEL, op.KERNEL_BWD
    op.KERNEL, op.KERNEL_BWD = "auto", "column"
    try:
        grads = []
        for _ in range(3):
            xs = [x.float().to("mps").requires_grad_(True) for x in xs64]
            o, sT = op.GatedDeltaRule.apply(*xs, True)
            (o.sum() + sT.sum()).backward()
            grads.append([x.grad.clone() for x in xs])
        spread = max((a - b).abs().max().item() for i in (1, 2) for a, b in zip(grads[0], grads[i]))
        assert spread < 1e-4, spread
    finally:
        op.KERNEL, op.KERNEL_BWD = old


@mps
def test_chunked_backward_is_deterministic():
    """The chunked backward has one writer for every gradient element (no atomics): repeated runs are bitwise identical."""
    xs64 = make(2, 100, 2, 128, 128, seed=5)
    old = op.KERNEL, op.KERNEL_BWD
    op.KERNEL, op.KERNEL_BWD = "chunked", "chunked"
    try:
        grads = []
        for _ in range(3):
            xs = [x.float().to("mps").requires_grad_(True) for x in xs64]
            o, sT = op.GatedDeltaRule.apply(*xs, True)
            (o.sum() + sT.sum()).backward()
            grads.append([x.grad.clone() for x in xs])
        assert all(torch.equal(a, b) for i in (1, 2) for a, b in zip(grads[0], grads[i]))
    finally:
        op.KERNEL, op.KERNEL_BWD = old


@mps
@pytest.mark.parametrize("pair", PAIRS)
def test_wide_keys(pair):
    """DK 256: every forward runs; the chunked backward runs too; the token-at-a-time backwards are refused (their
    threadgroup sums would exceed 32 KB)."""
    xs64 = make(1, 33, 2, 256, 128)
    if pair[1] == "chunked":
        check(xs64, pair)
        return
    check(xs64, pair, grads=False)
    old = op.KERNEL, op.KERNEL_BWD
    op.KERNEL, op.KERNEL_BWD = pair
    try:
        with pytest.raises(NotImplementedError):
            op.GatedDeltaRule.apply(*[x.float().to("mps").requires_grad_(True) for x in xs64], True)
    finally:
        op.KERNEL, op.KERNEL_BWD = old
