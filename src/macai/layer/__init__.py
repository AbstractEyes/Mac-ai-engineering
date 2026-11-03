"""The rest of Qwen3.5's decoder layers on the Mac's GPU (the kernel series' design D4, docs/layer.md): the parts around the
matrix products and the gated delta rule, each as one Metal pass forward and one backward in place of transformers' chains of
PyTorch operations. macai.patch.install() installs them into transformers' Qwen3.5 modeling code beside the rule; each part is a
setting of this module, read at every call (so a part can be switched on or off at any time):

    RMSNORM     the zero-centered RMSNorm (Qwen3_5RMSNorm: the input and post-attention norms of every layer, the full-attention
                layers' query / key norms, the final norm)
    CONV        the linear-attention layer's depthwise causal convolution and SiLU over the input projection's output in place,
                writing the rule's q, k (repeated over the value heads they serve) and v without transposes or copies
    GATED_NORM  the gated RMSNorm after the rule (Qwen3_5RMSNormGated)
    GATES       beta = sigmoid(b) and g = -exp(A_log) softplus(a + dt_bias) in one pass each way
    SWIGLU      the MLP's silu(gate) * up

A part runs only on the Apple GPU, for float32 / bfloat16 / float16 inputs and the shapes its kernels take; anything else
runs transformers' own code. The methods are replaced only while their source matches the transformers version they were
written against (_qwen3_5.SOURCES); otherwise transformers' code stays and a warning says so.
"""
import contextlib

from ._conv import ConvQKV, conv_qkv
from ._gated_norm import GatedRMSNorm, gated_rms_norm
from ._gates import Gates, gates
from ._rmsnorm import ZeroCenteredRMSNorm, rms_norm
from ._swiglu import SwiGLU, swiglu

# the parts in force: every part admitted by series design D4 (parity against float64 within twice the PyTorch path's error,
# >= 1.5x its PyTorch path at both bench shapes) and on by default since; docs/series_layer.md gives each part's measurements
RMSNORM = True
CONV = True
GATED_NORM = True
GATES = True
SWIGLU = True

PARTS = ("RMSNORM", "CONV", "GATED_NORM", "GATES", "SWIGLU")


def settings() -> dict:
    """The parts' settings now."""
    g = globals()
    return {p: bool(g[p]) for p in PARTS}


def configure(**values) -> dict:
    """Set parts by name (RMSNORM=True, ...); returns the previous settings."""
    g = globals()
    old = settings()
    for k, v in values.items():
        if k not in PARTS:
            raise KeyError(f"unknown part {k!r}; the parts are {PARTS}")
        g[k] = bool(v)
    return old


@contextlib.contextmanager
def configured(**values):
    """The parts set as given inside a block."""
    old = configure(**values)
    try:
        yield
    finally:
        configure(**old)


__all__ = ["PARTS", "settings", "configure", "configured", "ZeroCenteredRMSNorm", "rms_norm", "ConvQKV", "conv_qkv",
           "GatedRMSNorm", "gated_rms_norm", "Gates", "gates", "SwiGLU", "swiglu"]
