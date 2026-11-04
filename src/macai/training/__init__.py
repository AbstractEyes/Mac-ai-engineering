"""Changes to the training path around the decoder layers of Qwen3.5 (the kernel series' designs D8 and D9,
docs/training.md): no kernels, only less work for the same result. macai.patch.install() installs them into transformers'
Qwen3.5 modeling code beside the rule and the layer parts; each is a setting of this module, read at every call:

    HEAD_AT_LABELS  while a model trains with labels, the LM head and the loss run only at the positions whose next token
                    carries a label (transformers' boolean logits_to_keep and the loss's shift_labels): the logits of the
                    prompt, picture and padding positions, whose loss is masked, are never computed. The loss is the model's
                    own loss function, reduced as it reduces it; every other position's gradient is zero, as before.
    VISION_CACHE    while a model trains with its vision tower frozen, the tower's output for each picture is kept in CPU
                    memory on its first pass, keyed by the exact bytes of the picture's pixel rows, its grid and the tower's
                    settings, and served in later passes through the model's own precomputed-encoder input
                    (mm_encoder_outputs): the tower runs once a picture.

Each runs only where its conditions hold (training mode; for the cache, every tower parameter frozen); anywhere else the
model's own code runs. Both are on by default; configure() switches either off. stats() gives the counts of both (the head's
calls, the cache's stores, hits and its own time).
"""
import contextlib

# the settings in force: both admitted by series designs D8 and D9 (parity: the loss within float32 rounding of the model's own
# path, every gradient at its error against float64, the served tower output the same bits as a recompute; speed: the step
# 18% / 7% faster at 0.8B / 9B, and the later epochs 16% / 13% faster again) and on by default since; docs/series_layer.md
HEAD_AT_LABELS = True
VISION_CACHE = True
# the most a vision cache stores (GiB of CPU memory); past it, new pictures are computed every time
VISION_CACHE_MAX_GB = 16.0

PARTS = ("HEAD_AT_LABELS", "VISION_CACHE")


def settings() -> dict:
    """The settings now."""
    g = globals()
    return {p: bool(g[p]) for p in PARTS}


def configure(**values) -> dict:
    """Set by name (HEAD_AT_LABELS=True, ...); returns the previous settings."""
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
    """Since the process began (or reset_stats()): the head's calls (taken, fallen back, the positions seen and the rows
    computed), every vision cache's counts and times, and the calls that ran the original path with a setting on, by reason."""
    from . import _head, _qwen3_5, _vision_cache
    return {"head": dict(_head.STATS), "vision_cache": _vision_cache.all_stats(),
            "fallbacks": {f"{p}: {r}": n for (p, r), n in _qwen3_5.FALLBACKS.items()}}


def reset_stats() -> None:
    """The counts set to zero, and every vision cache emptied."""
    from . import _head, _qwen3_5, _vision_cache
    _head.reset()
    _vision_cache.reset_all()
    _qwen3_5.FALLBACKS.clear()


def clear_vision_cache(model=None) -> None:
    """Empty the vision cache of one model (the Qwen3_5Model holding the tower) or of every model. The cache empties itself
    when a tower tensor is replaced or changed in place through the tensor itself (its version counter moves); a change made
    through a tensor's .data bypasses that counter, so after one, empty the cache here."""
    from . import _vision_cache
    caches = [_vision_cache._caches.get(model)] if model is not None else list(_vision_cache._caches.values())
    for c in caches:
        if c is not None:
            c.entries.clear()
            c.pending.clear()
            c.fingerprint = None
            c.stats["bytes_stored"] = c.stats["bytes_merged"] = 0


__all__ = ["PARTS", "settings", "configure", "configured", "stats", "reset_stats", "clear_vision_cache"]
