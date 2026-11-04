"""The fused LoRA adapter path installed into PEFT (peft.tuners.lora.layer.Linear.forward), written against PEFT 0.21.2.

The method is replaced only while its source hashes as in PEFT 0.21.2; on a mismatch the original stays and a warning names it.
The replacement reads macai.lora.FUSED at every call and runs PEFT's own forward when it is off or when the call is anything
but vanilla LoRA in the form the fused path takes (each such call counted by its reason in FALLBACKS):

    one active adapter, present in lora_A; adapters enabled; nothing merged; no LoRA variant (DoRA and the others); no LoRA
    bias; input casting enabled; an nn.Linear base and fan_in_fan_out off; the dropout an nn.Dropout (not in place) or an
    nn.Identity; no arguments beside x (adapter_names and the variants' keyword arguments included); x and the adapters on the
    Apple GPU in float32 / bfloat16 / float16; the rank a multiple of 8 up to 64; both widths multiples of 8.

The fused call is the base layer's own module call (PEFT's `self.base_layer(x)`), then macai.lora._fused.adapter_path on its
output, looked up here at every call (an instrument can wrap it: bench/profile_step.py).
"""
import hashlib
import importlib
import inspect
import warnings

import torch

from . import _fused, _kernels

MODULE = "peft.tuners.lora.layer"
WRITTEN_AGAINST = "0.21.2"
# sha256 (first 16 hex digits) of inspect.getsource of the method replaced, in PEFT 0.21.2
SOURCES = {"Linear.forward": "ec0a80f5c5ce05f5"}
PATCHES = {"linear": ("Linear", "forward", ("Linear.forward",))}

adapter_path = _fused.adapter_path
FALLBACKS = {}            # reason -> calls that ran PEFT's forward although FUSED was on
CALLS = {"fused": 0}      # calls that ran the fused path

_orig = {}
_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _settings():
    from macai import lora
    return lora


def source_hash(obj) -> str:
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()[:16]


def check_sources() -> dict:
    """{name: True when PEFT's own method (the original, when the replacement is installed) hashes as in the version written
    against}."""
    try:
        mod = importlib.import_module(MODULE)
    except ImportError:
        return {k: False for k in SOURCES}
    originals = {f"{c}.{m}": _orig[p] for p, (c, m, _) in PATCHES.items() if p in _orig}
    out = {}
    for name, want in SOURCES.items():
        cls_name, meth = name.split(".")
        try:
            obj = originals.get(name) or getattr(mod, cls_name).__dict__[meth]
            out[name] = source_hash(obj) == want
        except (AttributeError, KeyError, OSError, TypeError):
            out[name] = False
    return out


def blocker(self, x, args, kwargs):
    """Why this call cannot take the fused path (None when it can)."""
    if args or kwargs:
        return "arguments beside x"
    if self.disable_adapters:
        return "adapters disabled"
    if self.merged:
        return "merged"
    active = self.active_adapters
    if len(active) != 1:
        return "not one active adapter"
    name = active[0]
    if name not in self.lora_A:
        return "active adapter not in lora_A"
    if name in self.lora_variant:
        return "a LoRA variant"
    if self.lora_bias.get(name, False) or self.lora_B[name].bias is not None:
        return "LoRA bias"
    if not getattr(self, "cast_input_dtype_enabled", True):
        return "input casting disabled"
    if type(self.base_layer) is not torch.nn.Linear or getattr(self, "fan_in_fan_out", False):
        return "base not nn.Linear"
    drop = self.lora_dropout[name]
    if not (type(drop) is torch.nn.Identity or (type(drop) is torch.nn.Dropout and not drop.inplace and 0 <= drop.p < 1)):
        return "dropout not nn.Dropout or nn.Identity"
    a, b = self.lora_A[name].weight, self.lora_B[name].weight
    if x.device.type != "mps" or a.device.type != "mps" or b.device.type != "mps":
        return "not on the Apple GPU"
    if x.dtype not in _DTYPES or a.dtype not in _DTYPES or b.dtype not in _DTYPES:
        return "dtype"
    if not _kernels.supported_rank(a.shape[0]):
        return "rank"
    if a.shape[1] % 8 or b.shape[0] % 8 or x.shape[-1] != a.shape[1]:
        return "width"
    if x.numel() == 0:
        return "empty input"
    return None


def _fallback(reason: str) -> None:
    FALLBACKS[reason] = FALLBACKS.get(reason, 0) + 1


def linear_forward(self, x, *args, **kwargs):
    orig = _orig["linear"]
    if not _settings().FUSED:
        return orig(self, x, *args, **kwargs)
    why = blocker(self, x, args, kwargs)
    if why is not None:
        _fallback(why)
        return orig(self, x, *args, **kwargs)
    name = self.active_adapters[0]
    drop = self.lora_dropout[name]
    p = float(drop.p) if (type(drop) is torch.nn.Dropout and drop.training) else 0.0
    a, b = self.lora_A[name].weight, self.lora_B[name].weight
    result = self.base_layer(x)
    dev = x.device.type
    cdt = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else a.dtype
    CALLS["fused"] += 1
    return adapter_path(x, result, a, b, p, float(self.scaling[name]), cdt)


_NEW = {"linear": linear_forward}


def install() -> list:
    """Replace the method if its source matches; returns the patches in force. A mismatch leaves the original, with a
    warning; without PEFT nothing is installed."""
    try:
        mod = importlib.import_module(MODULE)
    except ImportError:
        return []
    ok = check_sources()
    done = []
    for name, (cls_name, meth, deps) in PATCHES.items():
        if name in _orig:
            done.append(name)
            continue
        cls = getattr(mod, cls_name, None)
        bad = [d for d in deps if not ok.get(d)]
        if cls is None or bad:
            import peft
            warnings.warn(f"macai.lora: peft's lora.{cls_name}.{meth} left as PEFT's own: its source differs from PEFT "
                          f"{WRITTEN_AGAINST}, the version its replacement was written against (installed: {peft.__version__})",
                          stacklevel=2)
            continue
        _orig[name] = cls.__dict__[meth]
        setattr(cls, meth, _NEW[name])
        done.append(name)
    return done


def uninstall() -> None:
    if not _orig:
        return
    mod = importlib.import_module(MODULE)
    for name, orig in list(_orig.items()):
        cls_name, meth, _ = PATCHES[name]
        setattr(getattr(mod, cls_name), meth, orig)
        del _orig[name]


def installed() -> list:
    return list(_orig)
