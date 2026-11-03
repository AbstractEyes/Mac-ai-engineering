"""Install the Metal gated delta rule, and the kernels for the rest of the layer, into transformers' Qwen3.5-family models, so
existing training and inference code runs them without changes.

- The rule: the models call the module-level function torch_chunk_gated_delta_rule; install() swaps that name in the Qwen3.5,
  Qwen3.5-MoE and Qwen3-Next modeling modules.
- The rest of the layer (macai.layer; docs/layer.md): install() also replaces four methods of transformers' Qwen3.5 classes
  (the zero-centered RMSNorm, the gated RMSNorm, the MLP, the linear-attention module's forward), each only while its source
  matches the transformers version it was written against. Which parts run is set by macai.layer's settings (RMSNORM, CONV,
  GATED_NORM, GATES, SWIGLU), read at every call; a part that is off runs transformers' own code.

    import macai.patch
    macai.patch.install()          # every loaded or later-loaded Qwen3.5 / Qwen3.5-MoE / Qwen3-Next model
    with macai.patch.installed():  # or only inside a block
        ...
"""
import contextlib
import importlib

from .gated_delta import chunk_gated_delta_rule

MODULES = ("transformers.models.qwen3_5.modeling_qwen3_5", "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
           "transformers.models.qwen3_next.modeling_qwen3_next")
NAME = "torch_chunk_gated_delta_rule"
_saved = {}


def install(layer: bool = True) -> list:
    """Swap the rule in every module that has it; returns the modules patched. With layer, also install the layer parts'
    methods (macai.layer; layer_patches() lists the ones in force)."""
    done = []
    for path in MODULES:
        try:
            mod = importlib.import_module(path)
        except ImportError:
            continue
        if hasattr(mod, NAME) and getattr(mod, NAME) is not chunk_gated_delta_rule:
            _saved[path] = getattr(mod, NAME)
            setattr(mod, NAME, chunk_gated_delta_rule)
            done.append(path)
    if layer:
        from .layer import _qwen3_5
        _qwen3_5.install()
    return done


def layer_patches() -> list:
    """The layer parts' methods installed (macai.layer._qwen3_5.PATCHES names)."""
    from .layer import _qwen3_5
    return _qwen3_5.installed()


def uninstall() -> None:
    for path, fn in list(_saved.items()):
        setattr(importlib.import_module(path), NAME, fn)
        del _saved[path]
    from .layer import _qwen3_5
    if _qwen3_5.installed():
        _qwen3_5.uninstall()


@contextlib.contextmanager
def installed(layer: bool = True):
    install(layer=layer)
    try:
        yield
    finally:
        uninstall()
