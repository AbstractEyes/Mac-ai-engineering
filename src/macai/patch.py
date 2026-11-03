"""Install the Metal gated delta rule into transformers' Qwen3.5-family models, so existing training and inference code runs it
without changes. The models call the module-level function torch_chunk_gated_delta_rule; this swaps that name.

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


def install() -> list:
    """Swap the function in every module that has it; returns the modules patched."""
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
    return done


def uninstall() -> None:
    for path, fn in list(_saved.items()):
        setattr(importlib.import_module(path), NAME, fn)
        del _saved[path]


@contextlib.contextmanager
def installed():
    install()
    try:
        yield
    finally:
        uninstall()
