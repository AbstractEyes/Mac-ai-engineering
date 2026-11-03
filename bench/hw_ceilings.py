"""The GPU's own ceilings, measured with micro-kernels (series Phase 0, P0.3): what a kernel on this machine can reach at best.

- simdgroup-matrix multiply-accumulate peak (8x8 tiles in registers, no memory traffic), float32, half, bfloat16
- device-memory bandwidth (a float4 copy over 1 GiB)
- the cost of a threadgroup barrier (1,024 threads), of a simdgroup shuffle, of a simdgroup sum
- float atomic-add throughput, uncontended and with 4 / 32 / 1,024 threads on one address

    python bench/hw_ceilings.py      # writes bench/results/hw_ceilings_<stamp>.json
"""
import json
import os
import time

import torch

SRC = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define MMA_KERNEL(NAME, T)                                                                                      \
kernel void NAME(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],                          \
                 uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {               \
  simdgroup_matrix<T, 8, 8> a = simdgroup_matrix<T, 8, 8>(T(0.001f));                                            \
  simdgroup_matrix<T, 8, 8> b = simdgroup_matrix<T, 8, 8>(T(0.001f));                                            \
  simdgroup_matrix<T, 8, 8> c0 = simdgroup_matrix<T, 8, 8>(T(0)), c1 = c0, c2 = c0, c3 = c0;                    \
  for (uint i = 0; i < iters; ++i) {                                                                             \
    simdgroup_multiply_accumulate(c0, a, b, c0);                                                                 \
    simdgroup_multiply_accumulate(c1, b, a, c1);                                                                 \
    simdgroup_multiply_accumulate(c2, a, b, c2);                                                                 \
    simdgroup_multiply_accumulate(c3, b, a, c3);                                                                 \
  }                                                                                                              \
  threadgroup T tile[256];                                                                                       \
  simdgroup_store(c0, tile, 8);                                                                                  \
  simdgroup_store(c1, tile + 64, 8);                                                                             \
  simdgroup_store(c2, tile + 128, 8);                                                                            \
  simdgroup_store(c3, tile + 192, 8);                                                                            \
  if (lane == 0) out[gid / 32] = float(tile[0] + tile[64] + tile[128] + tile[192]);                              \
}
MMA_KERNEL(mma_f32, float)
MMA_KERNEL(mma_f16, half)
MMA_KERNEL(mma_bf16, bfloat)

kernel void copy4(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  y[i] = x[i];
}

kernel void barriers(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                     uint tid [[thread_index_in_threadgroup]], uint gid [[threadgroup_position_in_grid]]) {
  threadgroup float buf[1024];
  float acc = float(tid);
  for (uint i = 0; i < iters; ++i) {
    buf[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc += buf[(tid + 1) & 1023];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) out[gid] = acc;
}

kernel void shuffles(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  float x = float(gid);
  for (uint i = 0; i < iters; ++i) x += simd_shuffle_xor(x, (ushort)((i & 3) + 1));
  out[gid] = x;
}

kernel void simdsums(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  float x = float(gid & 7);
  for (uint i = 0; i < iters; ++i) x = simd_sum(x) * 0.03125f + 1.0f;
  out[gid] = x;
}

kernel void atomics(device atomic_float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                    constant uint& share [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  for (uint i = 0; i < iters; ++i) atomic_fetch_add_explicit(&out[gid / share], 1.0f, memory_order_relaxed);
}
"""


def gpu_ms(fn, reps=5):
    fn()
    torch.mps.synchronize()
    e0, e1 = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
    best = float("inf")
    for _ in range(reps):
        e0.record()
        fn()
        e1.record()
        torch.mps.synchronize()
        best = min(best, e0.elapsed_time(e1))
    return best


def main():
    lib = torch.mps.compile_shader(SRC)
    out = torch.zeros(1 << 22, device="mps")
    res = {"torch": torch.__version__}
    groups, tpg, iters = 40 * 64, 256, 4096                 # many simdgroups per core
    for name in ("mma_f32", "mma_f16", "mma_bf16"):
        kern = getattr(lib, name)
        ms = gpu_ms(lambda: kern(out, iters, threads=(groups * tpg, 1, 1), group_size=(tpg, 1, 1)))
        flops = groups * (tpg // 32) * iters * 4 * 2 * 8 * 8 * 8
        res[name] = {"tflops": round(flops / ms / 1e9, 2), "ms": round(ms, 3), "max_threads": kern.max_threads_per_threadgroup}
    n = (1 << 30) // 16                                     # 1 GiB of float4
    x, y = torch.empty(n * 4, device="mps"), torch.empty(n * 4, device="mps")
    ms = gpu_ms(lambda: lib.copy4(x, y, threads=(n, 1, 1), group_size=(256, 1, 1)))
    res["copy_gbps"] = round(2 * n * 16 / ms / 1e6, 1)
    del x, y
    for name, iters in (("barriers", 20000),):
        ms = gpu_ms(lambda: lib.barriers(out, iters, threads=(40 * 1024, 1, 1), group_size=(1024, 1, 1)))
        res["barrier_pair_ns"] = round(ms * 1e6 / iters, 1)
    ms = gpu_ms(lambda: lib.shuffles(out, 100000, threads=(40 * 1024, 1, 1), group_size=(1024, 1, 1)))
    res["shuffle_add_ns_per_thread_step"] = round(ms * 1e6 / 100000, 2)
    ms = gpu_ms(lambda: lib.simdsums(out, 100000, threads=(40 * 1024, 1, 1), group_size=(1024, 1, 1)))
    res["simd_sum_ns_per_step"] = round(ms * 1e6 / 100000, 2)
    res["atomics"] = {}
    threads, it = 40 * 4096, 256
    for share in (1, 4, 32, 1024):
        acc = torch.zeros(threads // share + 1, device="mps")
        ms = gpu_ms(lambda: lib.atomics(acc, it, share, threads=(threads, 1, 1), group_size=(256, 1, 1)))
        res["atomics"][f"{share}_threads_an_address_gops"] = round(threads * it / ms / 1e6, 2)
    print(json.dumps(res, indent=1))
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(d, exist_ok=True)
    json.dump(res, open(os.path.join(d, f"hw_ceilings_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
