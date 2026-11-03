"""Metal kernels for the gated delta rule (the linear attention of Qwen3.5 / Qwen3-Next), compiled through torch.mps.compile_shader.

The recurrence, per batch row b and head h, with the state S of shape [DK, DV]:

    a_t = exp(g_t)
    S~  = a_t * S_{t-1}
    u   = k_t^T S~                      (one value per column j)
    d   = beta_t * (v_t - u)
    S_t = S~ + k_t d^T
    o_t = S_t^T q_t

Columns of S are independent in the forward pass, so one simdgroup owns one column j (its 32 lanes hold DK / 32 rows each) and a
threadgroup of 32 simdgroups owns 32 columns. The whole forward runs in registers.

The backward pass walks blocks of BLK steps from the last to the first: it reloads the state saved at the block's start,
recomputes the block forward keeping each step's input state in registers, then steps back through it. Gradients of q, k, beta
and g sum over all DV columns: the 32 columns of a threadgroup are summed in threadgroup memory, then the DV / 32 threadgroups of a
head add into the output with float atomics.
"""
import functools

import torch

BLK = 8

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define DK __DK__
#define R (DK / 32)
#define BLK __BLK__

// q, k: [B, T, H, DK]; v, o, dout: [B, T, H, DV]; g, beta: [B, T, H]; s0, sT, ds0, dsT: [B, H, DK, DV]
// ckpt: [B, H, NB, DK, DV], NB = ceil(T / BLK): the state before each block's first step.

kernel void gdr_fwd(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                    device const float* v [[buffer(2)]], device const float* g [[buffer(3)]],
                    device const float* beta [[buffer(4)]], device const float* s0 [[buffer(5)]],
                    device float* o [[buffer(6)]], device float* sT [[buffer(7)]], device float* ckpt [[buffer(8)]],
                    constant uint& T [[buffer(9)]], constant uint& H [[buffer(10)]], constant uint& DV [[buffer(11)]],
                    constant uint& save [[buffer(12)]],
                    uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint b = tg.z, h = tg.y, j = tg.x * 32 + sg;
  const uint NB = (T + BLK - 1) / BLK;
  const ulong sbase = (ulong)(b * H + h) * DK * DV;
  float s[R];
  for (uint r = 0; r < R; ++r) s[r] = s0[sbase + (ulong)(lane + 32 * r) * DV + j];
  for (uint t = 0; t < T; ++t) {
    if (save != 0 && (t % BLK) == 0) {
      const ulong cb = ((ulong)(b * H + h) * NB + t / BLK) * DK * DV;
      for (uint r = 0; r < R; ++r) ckpt[cb + (ulong)(lane + 32 * r) * DV + j] = s[r];
    }
    const ulong bth = ((ulong)b * T + t) * H + h;
    const float a = exp(g[bth]);
    const float bt = beta[bth];
    const float vj = v[bth * DV + j];
    float kr[R];
    float u = 0.0f;
    for (uint r = 0; r < R; ++r) {
      kr[r] = k[bth * DK + lane + 32 * r];
      s[r] *= a;
      u = fma(s[r], kr[r], u);
    }
    u = simd_sum(u);
    const float d = bt * (vj - u);
    float ov = 0.0f;
    for (uint r = 0; r < R; ++r) {
      s[r] = fma(d, kr[r], s[r]);
      ov = fma(s[r], q[bth * DK + lane + 32 * r], ov);
    }
    ov = simd_sum(ov);
    if (lane == 0) o[bth * DV + j] = ov;
  }
  for (uint r = 0; r < R; ++r) sT[sbase + (ulong)(lane + 32 * r) * DV + j] = s[r];
}

kernel void gdr_bwd(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                    device const float* v [[buffer(2)]], device const float* g [[buffer(3)]],
                    device const float* beta [[buffer(4)]], device const float* ckpt [[buffer(5)]],
                    device const float* dout [[buffer(6)]], device const float* dsT [[buffer(7)]],
                    device atomic_float* dq [[buffer(8)]], device atomic_float* dk [[buffer(9)]],
                    device float* dv [[buffer(10)]], device atomic_float* dg [[buffer(11)]],
                    device atomic_float* dbeta [[buffer(12)]], device float* ds0 [[buffer(13)]],
                    constant uint& T [[buffer(14)]], constant uint& H [[buffer(15)]], constant uint& DV [[buffer(16)]],
                    uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[32 * DK];
  threadgroup float reds[64];
  const uint b = tg.z, h = tg.y, j = tg.x * 32 + sg;
  const uint NB = (T + BLK - 1) / BLK;
  const ulong sbase = (ulong)(b * H + h) * DK * DV;
  float dS[R];
  for (uint r = 0; r < R; ++r) dS[r] = dsT[sbase + (ulong)(lane + 32 * r) * DV + j];
  for (int blk = (int)NB - 1; blk >= 0; --blk) {
    const uint t0 = (uint)blk * BLK;
    const uint n = min((uint)BLK, T - t0);
    const ulong cb = ((ulong)(b * H + h) * NB + (uint)blk) * DK * DV;
    float hist[BLK][R];
    float s[R];
    for (uint r = 0; r < R; ++r) s[r] = ckpt[cb + (ulong)(lane + 32 * r) * DV + j];
    for (uint tt = 0; tt < BLK; ++tt) {
      if (tt < n) {
        for (uint r = 0; r < R; ++r) hist[tt][r] = s[r];
        const ulong bth = ((ulong)b * T + (t0 + tt)) * H + h;
        const float a = exp(g[bth]);
        const float bt = beta[bth];
        const float vj = v[bth * DV + j];
        float kr[R];
        float u = 0.0f;
        for (uint r = 0; r < R; ++r) {
          kr[r] = k[bth * DK + lane + 32 * r];
          s[r] *= a;
          u = fma(s[r], kr[r], u);
        }
        u = simd_sum(u);
        const float d = bt * (vj - u);
        for (uint r = 0; r < R; ++r) s[r] = fma(d, kr[r], s[r]);
      }
    }
    for (int tt = BLK - 1; tt >= 0; --tt) {
      if ((uint)tt < n) {
        const ulong bth = ((ulong)b * T + (t0 + (uint)tt)) * H + h;
        const float a = exp(g[bth]);
        const float bt = beta[bth];
        const float vj = v[bth * DV + j];
        float kr[R], qr[R], st[R];
        float u = 0.0f;
        for (uint r = 0; r < R; ++r) {
          kr[r] = k[bth * DK + lane + 32 * r];
          qr[r] = q[bth * DK + lane + 32 * r];
          st[r] = a * hist[tt][r];
          u = fma(st[r], kr[r], u);
        }
        u = simd_sum(u);
        const float d = bt * (vj - u);
        const float doj = dout[bth * DV + j];
        float dqp[R];
        float dd = 0.0f;
        for (uint r = 0; r < R; ++r) {
          dqp[r] = doj * fma(d, kr[r], st[r]);
          dS[r] = fma(doj, qr[r], dS[r]);
          dd = fma(dS[r], kr[r], dd);
        }
        dd = simd_sum(dd);
        if (lane == 0) dv[bth * DV + j] = dd * bt;
        const float du = -dd * bt;
        float dkp[R];
        float da = 0.0f;
        for (uint r = 0; r < R; ++r) {
          dkp[r] = fma(d, dS[r], du * st[r]);
          const float dst = fma(du, kr[r], dS[r]);
          da = fma(hist[tt][r], dst, da);
          dS[r] = a * dst;
        }
        da = simd_sum(da);
        for (uint r = 0; r < R; ++r) red[sg * DK + lane + 32 * r] = dqp[r];
        if (lane == 0) {
          reds[sg] = dd * (vj - u);
          reds[32 + sg] = da * a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < DK) {
          float acc = 0.0f;
          for (uint c = 0; c < 32; ++c) acc += red[c * DK + tid];
          atomic_fetch_add_explicit(&dq[bth * DK + tid], acc, memory_order_relaxed);
        } else if (tid == DK) {
          float acc = 0.0f;
          for (uint c = 0; c < 32; ++c) acc += reds[c];
          atomic_fetch_add_explicit(&dbeta[bth], acc, memory_order_relaxed);
        } else if (tid == DK + 1) {
          float acc = 0.0f;
          for (uint c = 0; c < 32; ++c) acc += reds[32 + c];
          atomic_fetch_add_explicit(&dg[bth], acc, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < R; ++r) red[sg * DK + lane + 32 * r] = dkp[r];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < DK) {
          float acc = 0.0f;
          for (uint c = 0; c < 32; ++c) acc += red[c * DK + tid];
          atomic_fetch_add_explicit(&dk[bth * DK + tid], acc, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }
  for (uint r = 0; r < R; ++r) ds0[sbase + (ulong)(lane + 32 * r) * DV + j] = dS[r];
}
"""


@functools.lru_cache(maxsize=None)
def library(dk: int):
    """The compiled kernels for one key head width (a multiple of 32; 128 in Qwen3.5)."""
    if dk % 32 != 0:
        raise ValueError(f"the key head width must be a multiple of 32, got {dk}")
    return torch.mps.compile_shader(_SOURCE.replace("__DK__", str(dk)).replace("__BLK__", str(BLK)))
