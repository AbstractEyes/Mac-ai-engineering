"""Parity of the gated delta rule kernels against the step-by-step recurrence and against transformers' own implementation."""
import pytest
import torch

from macai.gated_delta import GatedDeltaRule, chunk_gated_delta_rule, l2norm, recurrent_reference

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")


def _inputs(B, T, H, DK, DV, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = l2norm(torch.randn(B, T, H, DK, generator=gen)) * DK ** -0.5
    k = l2norm(torch.randn(B, T, H, DK, generator=gen))
    v = torch.randn(B, T, H, DV, generator=gen)
    g = -torch.rand(B, T, H, generator=gen) * 0.5
    beta = torch.rand(B, T, H, generator=gen)
    s0 = torch.randn(B, H, DK, DV, generator=gen) * 0.1
    return [x.to("mps").requires_grad_(True) for x in (q, k, v, g, beta, s0)]


@mps
@pytest.mark.parametrize("shape", [(2, 37, 3, 128, 64), (1, 64, 2, 64, 32), (1, 8, 1, 128, 128)])
def test_matches_recurrence(shape):
    xs = _inputs(*shape)
    o, sT = GatedDeltaRule.apply(*xs, True)
    xr = [x.detach().clone().requires_grad_(True) for x in xs]
    ro, rsT = recurrent_reference(*xr)
    assert torch.allclose(o, ro, atol=1e-5, rtol=1e-5)
    assert torch.allclose(sT, rsT, atol=1e-5, rtol=1e-5)
    w_o, w_s = torch.randn_like(o), torch.randn_like(sT)
    ((o * w_o).sum() + (sT * w_s).sum()).backward()
    ((ro * w_o).sum() + (rsT * w_s).sum()).backward()
    for a, b in zip(xs, xr):
        assert torch.allclose(a.grad, b.grad, atol=1e-4, rtol=1e-4)


@mps
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_matches_transformers(dtype):
    mod = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    ref = getattr(mod.torch_chunk_gated_delta_rule, "__wrapped__", mod.torch_chunk_gated_delta_rule)
    gen = torch.Generator().manual_seed(1)
    B, T, H, DK, DV = 2, 100, 4, 128, 128
    q = torch.randn(B, T, H, DK, generator=gen).to("mps", dtype)
    k = torch.randn(B, T, H, DK, generator=gen).to("mps", dtype)
    v = torch.randn(B, T, H, DV, generator=gen).to("mps", dtype)
    g = (-torch.rand(B, T, H, generator=gen)).to("mps")
    beta = torch.rand(B, T, H, generator=gen).to("mps")
    s0 = (torch.randn(B, H, DK, DV, generator=gen) * 0.1).to("mps")
    args = dict(initial_state=s0, output_final_state=True, use_qk_l2norm_in_kernel=True)
    o, sT = chunk_gated_delta_rule(q, k, v, g, beta, **args)
    ro, rsT = ref(q, k, v, g, beta, **args)
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    assert o.dtype == ro.dtype
    assert torch.allclose(o.float(), ro.float(), atol=tol, rtol=tol)
    assert torch.allclose(sT, rsT.float(), atol=1e-4, rtol=1e-4)


@mps
@pytest.mark.parametrize("kernel,kernel_bwd", [("auto", "column"), ("auto", "head"), ("column", "column"), ("chunked", "chunked"),
                                               ("auto", "chunked"), ("chunked", "column")])
def test_every_kernel_pair_matches(kernel, kernel_bwd):
    from macai.gated_delta import op
    old = op.KERNEL, op.KERNEL_BWD
    op.KERNEL, op.KERNEL_BWD = kernel, kernel_bwd
    try:
        test_matches_recurrence((2, 37, 3, 128, 64))
        test_matches_recurrence((1, 19, 2, 128, 128))
    finally:
        op.KERNEL, op.KERNEL_BWD = old
