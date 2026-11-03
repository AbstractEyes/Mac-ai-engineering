"""The training-path changes installed into transformers' Qwen3.5 modeling code (transformers.models.qwen3_5.modeling_qwen3_5).

Three methods are replaced, each only while the source of what it replaces (and of the functions its replacement relies on)
hashes as in transformers 5.22.1; on a mismatch the original stays and a warning names it. Each replacement calls the original
method itself (no line of transformers' code is copied): it reads its setting at every call and passes the call through
unchanged when the setting is off or a condition is unmet.

    Qwen3_5ForConditionalGeneration.forward  -> macai.training.HEAD_AT_LABELS: the original forward with labels=None and a boolean
    Qwen3_5ForCausalLM.forward                  logits_to_keep marking the positions whose shifted label is not ignored (the
                                                head runs on those rows only), then the model's own loss function on those
                                                logits with shift_labels and the keyword arguments the original passes it
    Qwen3_5Model.forward                     -> macai.training.VISION_CACHE: the pictures' get_image_features output from the
                                                cache (or computed and stored), handed to the original forward as its
                                                precomputed-encoder input mm_encoder_outputs (pixel_values then None)
"""
import dataclasses
import hashlib
import importlib
import inspect
import warnings

from . import _head, _vision_cache

MODULE = "transformers.models.qwen3_5.modeling_qwen3_5"
LOSS_MODULE = "transformers.loss.loss_utils"
WRITTEN_AGAINST = "5.22.1"
# sha256 (first 16 hex digits) of inspect.getsource of each method replaced and each function relied on, in transformers 5.22.1
SOURCES = {
    (MODULE, "Qwen3_5ForConditionalGeneration.forward"): "827db1ce8c9d0728",
    (MODULE, "Qwen3_5ForCausalLM.forward"): "24eff03538a54f00",
    (MODULE, "slice_logits_to_keep"): "4dbcb2dac1417cca",
    (LOSS_MODULE, "ForCausalLMLoss"): "c7f950f4eb7eba01",
    (LOSS_MODULE, "fixed_cross_entropy"): "a7d1a68de5ca0ce6",
    (MODULE, "Qwen3_5Model.forward"): "e15bc70bcfcc37e2",
    (MODULE, "Qwen3_5Model.get_image_features"): "f025aaaa71a69cd7",
    (MODULE, "Qwen3_5VisionModel.forward"): "3248428a30f8310d",
}
_HEAD_DEPS = ((MODULE, "slice_logits_to_keep"), (LOSS_MODULE, "ForCausalLMLoss"), (LOSS_MODULE, "fixed_cross_entropy"))
# each patch: the class, the method, the sources it depends on
PATCHES = {
    "head_conditional": ("Qwen3_5ForConditionalGeneration", "forward",
                         ((MODULE, "Qwen3_5ForConditionalGeneration.forward"),) + _HEAD_DEPS),
    "head_causal_lm": ("Qwen3_5ForCausalLM", "forward", ((MODULE, "Qwen3_5ForCausalLM.forward"),) + _HEAD_DEPS),
    "vision_cache": ("Qwen3_5Model", "forward", ((MODULE, "Qwen3_5Model.forward"), (MODULE, "Qwen3_5Model.get_image_features"),
                                                 (MODULE, "Qwen3_5VisionModel.forward"))),
}
# the training paths' functions, looked up at every call (an instrument can wrap them: bench/profile_step.py)
head_positions = _head.labelled_positions
head_targets = _head.targets_at
FALLBACKS = {}          # (patch, reason) -> calls that ran the original path although the setting was on

_orig = {}              # patch name -> the class attribute it replaced
_named = {}             # patch name -> the original's named parameters (its other keyword arguments reach **kwargs)


def _settings():
    from macai import training
    return training


def source_hash(obj) -> str:
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()[:16]


def _resolve(module: str, dotted: str):
    obj = importlib.import_module(module)
    for part in dotted.split("."):
        obj = obj.__dict__[part] if isinstance(obj, type) else getattr(obj, part)
    return obj


def check_sources() -> dict:
    """{'module:name': True when it hashes as in the version written against}."""
    out = {}
    for (module, name), want in SOURCES.items():
        try:
            out[f"{module}:{name}"] = source_hash(_resolve(module, name)) == want
        except (AttributeError, KeyError, ImportError, OSError, TypeError):
            out[f"{module}:{name}"] = False
    return out


def _fallback(patch: str, reason: str) -> None:
    FALLBACKS[(patch, reason)] = FALLBACKS.get((patch, reason), 0) + 1


# ---- D8: the head and the loss at the labelled positions ----------------------------------------------------------------

def _is_causal_lm_loss(fn) -> bool:
    loss_utils = importlib.import_module(LOSS_MODULE)
    return inspect.unwrap(fn) is loss_utils.ForCausalLMLoss


def _head_blocker(model, args, kwargs, labels):
    if args:
        return "positional arguments"
    if not hasattr(labels, "dim") or labels.dim() != 2 or labels.is_floating_point():
        return "labels not [batch, length]"
    ktk = kwargs.get("logits_to_keep", 0)
    if type(ktk) is not int or ktk != 0:
        return "logits_to_keep given"
    if kwargs.get("shift_labels") is not None:
        return "shift_labels given"
    rd = kwargs.get("return_dict")
    if rd is False or (rd is None and getattr(model.config, "return_dict", True) is False):
        return "tuple output asked"
    if not _is_causal_lm_loss(model.loss_function):
        return "another loss function"
    return None


def _head_forward(patch: str, vocab_of):
    def forward(self, *args, **kwargs):
        orig = _orig[patch]
        labels = kwargs.get("labels")
        if not (_settings().HEAD_AT_LABELS and self.training and labels is not None):
            return orig(self, *args, **kwargs)
        why = _head_blocker(self, args, kwargs, labels)
        if why is not None:
            _head.STATS["fallback"] += 1
            _fallback(patch, why)
            return orig(self, *args, **kwargs)
        keep, shift = head_positions(labels, kwargs.get("ignore_index", -100))
        # the original forward: the head on the kept rows only (slice_logits_to_keep flattens them to [N, hidden]), no loss
        out = orig(self, **dict(kwargs, labels=None, logits_to_keep=keep))
        targets = head_targets(shift, keep)
        # the keyword arguments the original passes its loss function: everything outside its named parameters (the
        # Trainer's num_items_in_batch when the model accepts it), return_dict taken by its output decorator
        extra = {k: v for k, v in kwargs.items() if k not in _named[patch] and k != "return_dict"}
        loss = self.loss_function(logits=out.logits, labels=labels, vocab_size=vocab_of(self), shift_labels=targets, **extra)
        _head.STATS["taken"] += 1
        _head.STATS["positions"] += labels.numel()
        _head.STATS["labelled"] += out.logits.shape[0]
        fields = {f.name: getattr(out, f.name) for f in dataclasses.fields(out)}
        fields["loss"] = loss
        return type(out)(**fields)
    return forward


# ---- D9: the frozen vision tower's output cached ------------------------------------------------------------------------

def model_forward(self, *args, **kwargs):
    orig = _orig["vision_cache"]
    if args or not _settings().VISION_CACHE or kwargs.get("pixel_values") is None:
        return orig(self, *args, **kwargs)
    # the keyword arguments the original passes get_image_features: everything outside its named parameters
    extra = {k: v for k, v in kwargs.items() if k not in _named["vision_cache"] and k != "return_dict"}
    why = _vision_cache.blocker(self, kwargs, extra)
    if why is not None:
        _fallback("vision_cache", why)
        return orig(self, *args, **kwargs)
    out = _vision_cache.cache_of(self).image_outputs(self, kwargs["pixel_values"], kwargs["image_grid_thw"], extra)
    return orig(self, **dict(kwargs, pixel_values=None, mm_encoder_outputs={"image": out}))


_NEW = {"head_conditional": lambda: _head_forward("head_conditional", lambda m: m.config.text_config.vocab_size),
        "head_causal_lm": lambda: _head_forward("head_causal_lm", lambda m: m.config.vocab_size),
        "vision_cache": lambda: model_forward}


def install() -> list:
    """Replace the methods whose sources match; returns the patches in force. A mismatch leaves the original, with a warning."""
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
        bad = [f"{m}:{d}" for m, d in deps if not ok.get(f"{m}:{d}")]
        if cls is None or bad:
            import transformers
            warnings.warn(f"macai.training: {cls_name}.{meth} left as transformers' own: the source of {bad or [cls_name]} "
                          f"differs from transformers {WRITTEN_AGAINST}, the version its replacement was written against "
                          f"(installed: {transformers.__version__})", stacklevel=2)
            continue
        orig = cls.__dict__[meth]
        _orig[name] = orig
        _named[name] = set(inspect.signature(inspect.unwrap(orig)).parameters) - {"self", "kwargs"}
        setattr(cls, meth, _NEW[name]())
        done.append(name)
    return done


def uninstall() -> None:
    mod = importlib.import_module(MODULE)
    for name, orig in list(_orig.items()):
        cls_name, meth, _ = PATCHES[name]
        setattr(getattr(mod, cls_name), meth, orig)
        del _orig[name]
        _named.pop(name, None)


def installed() -> list:
    return list(_orig)
