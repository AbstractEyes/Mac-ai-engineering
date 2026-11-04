"""Series design D3 (a2), the diagnostic of record for the double buffering's forms (docs/series_layer.md, D3): the state passes
and the per-chunk backward alone (bench/bench_d3.py kernel_parts: back-to-back Metal events, 5 x 3 calls) at the 0.8B bench shape,
a mean 0.8B micro-batch and 1 x 8,192 x 16 heads, under _chunked.DBUF 0 / 1 / 2 / 0 again, each without and with a split of 2.

    python bench/d3_dbuf_forms.py      # prints; the runs of record: bench/layer_logs/d3_dbuf_first_form.log (the first form,
                                       # DBUF 0 / 1 only), d3_dbuf_forms.log"""
import os
import sys
import time

print(f"[diag] pid {os.getpid()}", flush=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "bench"))
import torch  # noqa: E402
from macai.gated_delta import _chunked, op  # noqa: E402
from macai.instruments import warm_gpu  # noqa: E402
import bench_d3 as BD  # noqa: E402

warm_gpu(1.0)
t0 = time.time()
for shape in ((4, 1024, 16), (2, 811, 16), (1, 8192, 16)):
    for dbuf in (0, 1, 2, 0):
        for split in (1, 2):
            _chunked.DBUF, op.VALUE_SPLIT = dbuf, split
            t1 = time.time()
            r = BD.kernel_parts(*shape)
            print(f"[diag] {shape} dbuf {dbuf} split {split}: " + " ".join(f"{k} {v}" for k, v in r.items())
                  + f"  ({time.time() - t1:.1f} s)", flush=True)
_chunked.DBUF, op.VALUE_SPLIT = 0, 1
print(f"[diag] done {time.time() - t0:.1f} s", flush=True)
