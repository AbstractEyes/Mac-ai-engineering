"""The frozen vision tower's output cached per picture (series design D9).

While a vision-language model trains with its vision tower frozen, the tower computes the same output for the same picture in
every epoch. A cache, one per model, keeps what the model's get_image_features returns (the tower's last hidden states and the
merged features, in full) for each picture in CPU memory on the picture's first pass and serves it in later passes through the
model's own precomputed-encoder input (mm_encoder_outputs), so the tower runs once a picture.

The key of a picture is exact: the SHA-256 digest of the bytes of its rows of pixel_values (hashed in chunks of 1 MiB on a few
threads, then the chunks' digests and the length hashed together), the rows' dtype and width, its grid entry (t, h, w), and the
settings the tower's output depends on (the device, the tower's dtype, the autocast state and dtype, the attention
implementation). A picture is served only when every byte matches; one changed pixel, or the same pixels at another grid, is
another key. The cache belongs to one set of tower weights: every call reads each tower tensor's storage address and version
counter, and any change (a parameter loaded, modified in place or replaced) empties the cache.

A micro-batch is served from the cache only when every one of its pictures is there; otherwise the tower runs on the whole
micro-batch as the model runs it, and the pictures not seen before are stored. Their copies to CPU memory wait for the next call
(by then the tower's output is computed, and the call synchronizes anyway), so storing adds no synchronization of its own.
A cache stores at most macai.training.VISION_CACHE_MAX_GB of outputs; past it, new pictures are computed every time and
counted as not stored.
"""
import hashlib
import itertools
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import torch

CHUNK = 1 << 20          # bytes a hashing task
THREADS = 4              # hashing threads (hashlib releases the interpreter lock on large buffers)
# keyword arguments the model's forward passes on to get_image_features that do not change the tower's output
HARMLESS = {"num_items_in_batch", "use_cache"}

_pool = None
_pool_lock = threading.Lock()
_caches = weakref.WeakKeyDictionary()       # model -> its cache


def _executor():
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(THREADS, thread_name_prefix="macai-vision-cache")
    return _pool


def digest(buf: memoryview) -> bytes:
    """SHA-256 of each 1 MiB chunk of buf (on the pool), then of the length and the chunks' digests: equal only for equal
    bytes (short of a SHA-256 collision)."""
    n = len(buf)
    if n <= CHUNK:
        parts = [hashlib.sha256(buf).digest()]
    else:
        parts = list(_executor().map(lambda i: hashlib.sha256(buf[i:i + CHUNK]).digest(), range(0, n, CHUNK)))
    h = hashlib.sha256(n.to_bytes(8, "little"))
    for p in parts:
        h.update(p)
    return h.digest()


def _bytes_of(t: torch.Tensor) -> memoryview:
    """The bytes of a CPU tensor, row-major, without a copy when it is contiguous."""
    t = t.contiguous().reshape(-1)
    return memoryview(t.view(torch.uint8).numpy())


def _max_bytes() -> float:
    from macai import training
    return float(getattr(training, "VISION_CACHE_MAX_GB", 16.0)) * 2 ** 30


def _zero(stats: dict) -> None:
    for k in stats:
        stats[k] = 0.0 if isinstance(stats[k], float) else 0


class VisionCache:
    def __init__(self):
        self.entries = {}            # key -> (last hidden rows [P, D], merged features [P / m^2, D_out]), on the CPU
        self.pending = []            # (key, last hidden rows, merged features) still on the device, copied at the next call
        self.fingerprint = None
        # calls: get_image_features calls through the cache; served / computed: answered from it or by the tower; pictures:
        # served, new (stored at the next call), again (computed by the tower although already stored: a micro-batch with a
        # picture not yet stored); emptied: the weights changed; seconds: waiting for the GPU before the keys, the keys (the
        # pixel rows copied to the CPU and hashed), building a served output, the stores' copies to the CPU
        self.stats = {"calls": 0, "served_calls": 0, "computed_calls": 0, "pictures_served": 0, "pictures_new": 0,
                      "pictures_again": 0, "stored": 0, "not_stored": 0, "bytes_stored": 0, "bytes_merged": 0, "emptied": 0,
                      "wait_s": 0.0, "key_s": 0.0, "serve_s": 0.0, "store_s": 0.0}

    def check_weights(self, visual) -> None:
        fp = tuple((t.data_ptr(), t._version) for t in itertools.chain(visual.parameters(), visual.buffers()))
        if fp != self.fingerprint:
            if self.fingerprint is not None and (self.entries or self.pending):
                self.stats["emptied"] += 1
            self.entries.clear()
            self.pending.clear()
            self.stats["bytes_stored"] = self.stats["bytes_merged"] = 0
            self.fingerprint = fp

    def flush(self) -> None:
        """The pending stores copied to CPU memory."""
        if not self.pending:
            return
        t0 = time.perf_counter()
        for key, last, merged in self.pending:
            if key in self.entries:
                continue
            size = last.numel() * last.element_size() + merged.numel() * merged.element_size()
            if self.stats["bytes_stored"] + size > _max_bytes():
                self.stats["not_stored"] += 1
                continue
            a, b = last.detach().to("cpu"), merged.detach().to("cpu")
            self.entries[key] = (a, b)
            self.stats["stored"] += 1
            self.stats["bytes_stored"] += a.numel() * a.element_size() + b.numel() * b.element_size()
            self.stats["bytes_merged"] += b.numel() * b.element_size()
        self.pending.clear()
        self.stats["store_s"] += time.perf_counter() - t0

    def keys(self, model, pixel_values, grid):
        """(each picture's key, each picture's row count), or (None, None) when the rows do not add up to the grid."""
        visual = model.visual
        dev = pixel_values.device.type
        auto = torch.is_autocast_enabled(dev)
        ctx = (dev, str(visual.dtype), auto, str(torch.get_autocast_dtype(dev)) if auto else None,
               getattr(visual.config, "_attn_implementation", None))
        t0 = time.perf_counter()
        if dev == "mps":
            torch.mps.synchronize()           # the copies below would wait here anyway; timed apart from the keys
        t1 = time.perf_counter()
        self.stats["wait_s"] += t1 - t0
        thw = [tuple(int(x) for x in r) for r in grid.tolist()]
        rows = [t * h * w for t, h, w in thw]
        if pixel_values.dim() != 2 or sum(rows) != pixel_values.shape[0]:
            return None, None
        buf = _bytes_of(pixel_values.detach().to("cpu"))
        width = len(buf) // max(pixel_values.shape[0], 1)
        out, r0 = [], 0
        for g, n in zip(thw, rows):
            out.append((ctx, g, str(pixel_values.dtype), int(pixel_values.shape[1]),
                        digest(buf[r0 * width:(r0 + n) * width])))
            r0 += n
        self.stats["key_s"] += time.perf_counter() - t1
        return out, rows

    def image_outputs(self, model, pixel_values, grid, extra: dict):
        """What model.get_image_features(pixel_values, grid, return_dict=True, **extra) returns: served from the cache when
        every picture is in it, else computed by it (the pictures not yet stored are stored at the next call)."""
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        self.stats["calls"] += 1
        self.check_weights(model.visual)
        self.flush()
        keys, rows = self.keys(model, pixel_values, grid)
        if keys is not None and all(k in self.entries for k in keys):
            t0 = time.perf_counter()
            got = [self.entries[k] for k in keys]
            dev = pixel_values.device
            last = torch.cat([g[0] for g in got]).to(dev)
            merged = torch.cat([g[1] for g in got]).to(dev)
            pooled = torch.split(merged, [g[1].shape[0] for g in got])
            self.stats["served_calls"] += 1
            self.stats["pictures_served"] += len(keys)
            self.stats["serve_s"] += time.perf_counter() - t0
            return BaseModelOutputWithPooling(last_hidden_state=last, pooler_output=pooled)
        out = model.get_image_features(pixel_values, grid, return_dict=True, **extra)
        self.stats["computed_calls"] += 1
        if keys is None:
            return out
        whole = (getattr(out, "hidden_states", None) is None and getattr(out, "attentions", None) is None
                 and isinstance(out.pooler_output, (tuple, list)) and len(out.pooler_output) == len(keys)
                 and out.last_hidden_state is not None and out.last_hidden_state.shape[0] == sum(rows))
        if whole:
            r0, queued = 0, set()
            for k, n, merged in zip(keys, rows, out.pooler_output):
                if k in self.entries or k in queued:
                    self.stats["pictures_again"] += 1
                else:
                    self.pending.append((k, out.last_hidden_state[r0:r0 + n], merged))
                    queued.add(k)
                    self.stats["pictures_new"] += 1
                r0 += n
        return out

    def summary(self) -> dict:
        self.flush()
        return dict(self.stats, entries=len(self.entries))


def cache_of(model) -> VisionCache:
    c = _caches.get(model)
    if c is None:
        c = _caches[model] = VisionCache()
    return c


def all_stats() -> list:
    return [c.summary() for c in list(_caches.values())]


def reset_all() -> None:
    for c in list(_caches.values()):
        c.entries.clear()
        c.pending.clear()
        c.fingerprint = None
        _zero(c.stats)


def blocker(model, kwargs: dict, extra: dict):
    """None when a call may go through the cache, else the reason it may not."""
    if not model.training:
        return "not training"
    if kwargs.get("mm_encoder_outputs") is not None:
        return "precomputed outputs given"
    if kwargs.get("pixel_values_videos") is not None:
        return "videos"
    pv, grid = kwargs.get("pixel_values"), kwargs.get("image_grid_thw")
    if grid is None or not isinstance(pv, torch.Tensor) or pv.requires_grad:
        return "pixel values"
    if any(p.requires_grad for p in model.visual.parameters()):
        return "the tower trains"
    for k, v in extra.items():
        if k not in HARMLESS and v is not None and v is not False:
            return f"argument {k}"
    return None
