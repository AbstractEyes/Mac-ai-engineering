"""Instruments for kernel work on the Apple GPU (series Phase 1).

- gpu_ms: GPU time of a callable from Metal events (warm-up, then the median and the best of several runs; inner > 1 for the
  back-to-back time of a short kernel); warm_gpu: bring the clocks up first
- pipeline: a compiled kernel's pipeline limits: the most threads a threadgroup may hold (it falls below 1,024 when the kernel
  needs many registers: the register-pressure sign, since this machine has no shader profiler), its static threadgroup memory,
  the execution width
- roofline: achieved FLOP/s and bytes a second against the machine's measured ceilings (bench/hw_ceilings.py)
"""
import statistics

import torch

# measured on the M3 Max (bench/results/hw_ceilings_*.json, 2026-11-02)
CEILINGS = {"mma_fp32_tflops": 13.78, "mma_fp16_tflops": 13.64, "mma_bf16_tflops": 11.96, "copy_gbps": 356.1}


def gpu_ms(fn, reps: int = 10, warm: int = 2, inner: int = 1) -> dict:
    """inner > 1 times inner calls back to back between the two events and reports the time a call: the steady state of
    a short kernel, without the GPU's idle-to-busy ramp after each synchronization."""
    for _ in range(warm):
        fn()
    torch.mps.synchronize()
    times = []
    for _ in range(reps):
        e0, e1 = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
        e0.record()
        for _ in range(inner):
            fn()
        e1.record()
        torch.mps.synchronize()
        times.append(e0.elapsed_time(e1) / inner)
    return {"median_ms": round(statistics.median(times), 3), "min_ms": round(min(times), 3), "reps": reps, "inner": inner}


def warm_gpu(seconds: float = 1.0) -> None:
    """Run matrix products for about this long, so the GPU's clocks are up before timing."""
    import time
    x = torch.randn(2048, 2048, device="mps")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(20):
            x = x @ x * 1e-3
        torch.mps.synchronize()


def pipeline(kernel) -> dict:
    return {"max_threads_per_threadgroup": int(kernel.max_threads_per_threadgroup),
            "static_threadgroup_memory_bytes": int(kernel.static_thread_group_memory_length),
            "thread_execution_width": int(kernel.thread_execution_width)}


def roofline(flops: float, nbytes: float, ms: float, peak_tflops: float = CEILINGS["mma_fp32_tflops"]) -> dict:
    tflops = flops / ms / 1e9
    gbps = nbytes / ms / 1e6
    return {"tflops": round(tflops, 3), "gbps": round(gbps, 1), "of_compute_peak": round(tflops / peak_tflops, 4),
            "of_bandwidth": round(gbps / CEILINGS["copy_gbps"], 4), "flops": flops, "bytes": nbytes}
