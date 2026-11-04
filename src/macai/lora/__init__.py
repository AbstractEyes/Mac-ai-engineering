"""The LoRA adapter path on the Mac's GPU (the kernel series' design D5, docs/lora.md): PEFT's vanilla LoRA layer
(peft.tuners.lora.layer.Linear, written against PEFT 0.21.2) with the passes around its rank-r products folded into two Metal
kernels. macai.patch.install() installs it beside the rule, the layer parts and the training path; the setting is read at
every call:

    FUSED   a LoRA linear layer's adapter path (the dropout, the two rank-r products, the scaled add; forward and backward)
            as one autograd function: the dropout's mask drawn by torch's own dropout call (the same masks as PEFT's from the
            same generator state), the reductions (x_d A^T, dY B, dY^T P, G^T x_d) by PyTorch's products, the expanding
            products by Metal kernels that add the base output (forward) or apply the mask and the scale (backward) as they
            write. The base layer stays PEFT's own module call.

Anything but vanilla LoRA in the form the kernels take runs PEFT's own forward (macai.lora._peft.blocker gives the conditions);
stats() counts the fused calls and the fallbacks by reason.
"""
import contextlib

# off until the series admits it (docs/series_layer.md, D5)
FUSED = False

PARTS = ("FUSED",)


def settings() -> dict:
    """The settings now."""
    g = globals()
    return {p: bool(g[p]) for p in PARTS}


def configure(**values) -> dict:
    """Set by name (FUSED=True); returns the previous settings."""
    g = globals()
    old = settings()
    for k, v in values.items():
        if k not in PARTS:
            raise KeyError(f"unknown setting {k!r}; the settings are {PARTS}")
        g[k] = bool(v)
    return old


@contextlib.contextmanager
def configured(**values):
    """The settings as given inside a block."""
    old = configure(**values)
    try:
        yield
    finally:
        configure(**old)


def stats() -> dict:
    """Since the process began (or reset_stats()): the fused calls, and the calls that ran PEFT's forward with FUSED on, by
    reason."""
    from . import _peft
    return {"fused": _peft.CALLS["fused"], "fallbacks": dict(_peft.FALLBACKS)}


def reset_stats() -> None:
    from . import _peft
    _peft.CALLS["fused"] = 0
    _peft.FALLBACKS.clear()


def adapter_path(x, result, a, b, p: float, s: float, cdt):
    """result + s * dropout(x, p) a^T b^T with the fused forward and backward (macai.lora._fused.AdapterPath)."""
    from ._fused import adapter_path as fn
    return fn(x, result, a, b, p, s, cdt)


__all__ = ["PARTS", "settings", "configure", "configured", "stats", "reset_stats", "adapter_path"]
