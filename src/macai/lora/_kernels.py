"""The two expanding products of a LoRA adapter (rank r, a multiple of 8 up to 64) as Metal kernels, each with its epilogue:

    up    Y[n, c] = round(float(R[n, c]) + s * sum_k P[n, k] B[c, k])          the forward: the base output R plus the scaled
                                                                               adapter term, Y written once (P = x_d A^T)
    down  D[n, c] = round(s * M[n, c] * sum_k G[n, k] A[k, c])                 the backward: the adapter's input gradient with
                                                                               the dropout mask M and scale s (G = s dY B)

The products accumulate in float32 on simdgroup matrices (8 x 8 fragments). A threadgroup of 4 simdgroups owns 64 columns
(16 a simdgroup) and walks a range of rows 32 at a time: the right operand's fragments for its columns stay in registers, the
left operand's 32 rows are staged in threadgroup memory as float32, the 32 x 64 accumulator tile goes through threadgroup memory
to an epilogue in which each thread reads and writes 16 contiguous elements (two 8-element vectors). Every output element is
written once, rounded once.
"""
import torch

from ..layer import _lib

_SOURCE = r"""
#include <metal_simdgroup_matrix>

#define RK __RANK__                  // the adapter's rank: a multiple of 8, at most 64
#define NKF (RK / 8)                 // 8-wide blocks of the rank
#define TM 32                        // rows a block
#define TN 64                        // columns a threadgroup (16 a simdgroup)
#define NT 128                       // threads a threadgroup

typedef simdgroup_matrix<float, 8, 8> mat;

// A lane holds two adjacent elements of one row of an 8 x 8 simdgroup matrix: row fm, columns fn and fn + 1.
inline ushort lane_row(uint lane) { return (ushort)(((lane >> 4) << 2) + ((lane >> 1) & 3u)); }
inline ushort lane_col(uint lane) { return (ushort)((((lane >> 3) & 1u) << 2) + ((lane & 1u) << 1)); }

// UP: L = P [N, RK], the right operand B [C, RK] (element (k, c) at W[c * RK + k]), the epilogue R + s * acc.
// DOWN: L = G [N, RK], the right operand A [RK, C] (element (k, c) at W[k * C + c]), the epilogue s * M * acc.
template <typename T, typename TL, bool UP>
inline void lora_expand(const device TL* L, const device TL* W, const device T* res, const device uchar* mask, device T* out,
                        uint N, uint C, uint ldres, float s, uint has_mask, uint rows_per_tg, uint2 tg, uint tid, uint lane,
                        uint sg, threadgroup float* Ls, threadgroup float* Os) {
  const uint c0 = tg.x * TN;
  const uint r_begin = tg.y * rows_per_tg;
  const uint r_end = min(N, r_begin + rows_per_tg);
  const ushort fm = lane_row(lane), fn = lane_col(lane);
  const uint csg = c0 + sg * 16;
  // the right operand's fragments for this simdgroup's 16 columns, every block of the rank, in registers
  mat Wf[NKF][2];
  for (uint kf = 0; kf < NKF; ++kf) {
    for (uint cf = 0; cf < 2; ++cf) {
      const uint k = kf * 8 + fm;
      const uint c = csg + cf * 8 + fn;
      float2 v = float2(0.0f);
      if (UP) {
        if (c < C) v.x = float(W[(ulong)c * RK + k]);
        if (c + 1 < C) v.y = float(W[(ulong)(c + 1) * RK + k]);
      } else {
        if (c < C) v.x = float(W[(ulong)k * C + c]);
        if (c + 1 < C) v.y = float(W[(ulong)k * C + c + 1]);
      }
      Wf[kf][cf].thread_elements()[0] = v.x;
      Wf[kf][cf].thread_elements()[1] = v.y;
    }
  }
  for (uint rb = r_begin; rb < r_end; rb += TM) {
    // the left operand's rows [rb, rb + TM) as float32 (rows at or past r_end: zeros)
    for (uint e = tid; e < TM * RK / 4; e += NT) {
      const uint r = e / (RK / 4), k4 = e % (RK / 4);
      float4 x = float4(0.0f);
      if (rb + r < r_end) x = ld4(L + (ulong)(rb + r) * RK + 4 * k4);
      *((threadgroup float4*)(Ls + r * RK + 4 * k4)) = x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (is_same<T, float>::value) {
      // float32 outputs: four fma chains over k (k mod 4) summed as a tree. One chain in k order, which is what the
      // simdgroup products give, accumulates a few float32 ulps looser than the platform's float32 product, which a float32
      // output shows (16-bit outputs round it away)
      const uint r = tid >> 2, q = tid & 3u;
      const uint n = rb + r;
      if (n < r_end) {
        for (uint j = 0; j < 16; ++j) {
          const uint c = c0 + q * 16 + j;
          if (c < C) {
            float4 part = float4(0.0f);
            for (uint k = 0; k < RK; k += 4) {
              float4 w;
              for (uint i = 0; i < 4; ++i) w[i] = UP ? float(W[(ulong)c * RK + k + i]) : float(W[(ulong)(k + i) * C + c]);
              part = fma(*((const threadgroup float4*)(Ls + r * RK + k)), w, part);
            }
            const float acc = (part.x + part.y) + (part.z + part.w);
            float o;
            if (UP) {
              o = float(res[(ulong)n * ldres + c]) + s * acc;
            } else {
              o = (has_mask == 0 || mask[(ulong)n * C + c] != 0) ? s * acc : 0.0f;
            }
            out[(ulong)n * C + c] = T(o);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      continue;
    }
    for (uint rf = 0; rf < TM / 8; ++rf) {
      mat acc0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
      mat acc1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
      for (uint kf = 0; kf < NKF; ++kf) {
        mat a;
        simdgroup_load(a, Ls + rf * 8 * RK + kf * 8, RK);
        simdgroup_multiply_accumulate(acc0, a, Wf[kf][0], acc0);
        simdgroup_multiply_accumulate(acc1, a, Wf[kf][1], acc1);
      }
      simdgroup_store(acc0, Os + rf * 8 * TN + sg * 16, TN);
      simdgroup_store(acc1, Os + rf * 8 * TN + sg * 16 + 8, TN);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the epilogue: thread tid takes row tid / 4 of the block and 16 of its columns, as two 8-element vectors
    const uint r = tid >> 2, q = tid & 3u;
    const uint n = rb + r;
    if (n < r_end) {
      for (uint v = 0; v < 2; ++v) {
        const uint cl = q * 16 + v * 8;
        const uint c = c0 + cl;
        if (c < C) {                                     // C % 8 == 0: the whole vector is in range
          float4 a0 = *((const threadgroup float4*)(Os + r * TN + cl));
          float4 a1 = *((const threadgroup float4*)(Os + r * TN + cl + 4));
          if (UP) {
            const device T* rp = res + (ulong)n * ldres + c;
            a0 = ld4(rp) + s * a0;
            a1 = ld4(rp + 4) + s * a1;
          } else {
            a0 *= s;
            a1 *= s;
            if (has_mask != 0) {
              const device uchar* mp = mask + (ulong)n * C + c;
              const uchar4 m0 = *((const device uchar4*)mp), m1 = *((const device uchar4*)(mp + 4));
              a0 = select(float4(0.0f), a0, m0 != uchar4(0));
              a1 = select(float4(0.0f), a1, m1 != uchar4(0));
            }
          }
          device T* op = out + (ulong)n * C + c;
          st4(op, a0);
          st4(op + 4, a1);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define LORA(T, TL, SUF)                                                                                               \
kernel void lora_up_##SUF(device const TL* L [[buffer(0)]], device const TL* W [[buffer(1)]], device const T* res [[buffer(2)]], \
                          device T* out [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& C [[buffer(5)]],   \
                          constant uint& ldres [[buffer(6)]], constant float& s [[buffer(7)]],                           \
                          constant uint& rows_per_tg [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],            \
                          uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],              \
                          uint sg [[simdgroup_index_in_threadgroup]]) {                                                  \
  threadgroup float Ls[TM * RK];                                                                                       \
  threadgroup float Os[TM * TN];                                                                                       \
  lora_expand<T, TL, true>(L, W, res, (const device uchar*)res, out, N, C, ldres, s, 0, rows_per_tg, tg, tid, lane, sg, \
                           Ls, Os);                                                                                    \
}                                                                                                                      \
kernel void lora_down_##SUF(device const TL* L [[buffer(0)]], device const TL* W [[buffer(1)]],                         \
                            device const uchar* mask [[buffer(2)]], device T* out [[buffer(3)]],                        \
                            constant uint& N [[buffer(4)]], constant uint& C [[buffer(5)]], constant float& s [[buffer(6)]], \
                            constant uint& has_mask [[buffer(7)]], constant uint& rows_per_tg [[buffer(8)]],             \
                            uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],          \
                            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {        \
  threadgroup float Ls[TM * RK];                                                                                       \
  threadgroup float Os[TM * TN];                                                                                       \
  lora_expand<T, TL, false>(L, W, (const device T*)out, mask, out, N, C, C, s, has_mask, rows_per_tg, tg, tid, lane, sg, \
                            Ls, Os);                                                                                   \
}

__INSTANCES__
"""

TM, TN, NT = 32, 64, 128
TARGET_GROUPS = 320          # threadgroups a call aims for (8 a GPU core on the M3 Max's 40), rows split to reach it
RANKS = (8, 16, 24, 32, 40, 48, 56, 64)


def library(rank: int):
    return _lib.library(_SOURCE.replace("__INSTANCES__", _lib.instances("LORA", True)), (("RANK", rank),))


def supported_rank(rank: int) -> bool:
    return rank in RANKS


def _grid(n: int, c: int):
    """(rows a threadgroup, threadgroups along the rows, threadgroups along the columns)."""
    gx = -(-c // TN)
    blocks = -(-n // TM)
    gy = max(1, min(blocks, -(-TARGET_GROUPS // gx)))
    per = -(-blocks // gy) * TM
    return per, -(-n // per), gx


def up(p: torch.Tensor, b: torch.Tensor, res: torch.Tensor, s: float) -> torch.Tensor:
    """res [N, C] (rows of any stride) + s * p [N, r] @ b[C, r]^T, written to a new [N, C] tensor in res's dtype."""
    n, c = res.shape
    rank = p.shape[1]
    rows, ld = _lib.as_rows(res, c)
    out = torch.empty((n, c), dtype=res.dtype, device=res.device)
    if n:
        per, gy, gx = _grid(n, c)
        _lib.kernel(library(rank), "lora_up", res.dtype, p.dtype)(
            _lib.aligned(p), _lib.aligned(b), rows, out, n, c, ld, float(s), per, threads=(gx * NT, gy, 1),
            group_size=(NT, 1, 1))
    return out


def down(g: torch.Tensor, a: torch.Tensor, mask, s: float, dtype) -> torch.Tensor:
    """s * mask * (g [N, r] @ a [r, C]) as a new [N, C] tensor of dtype (mask: [N, C] bool, or None for none)."""
    n, rank = g.shape
    c = a.shape[1]
    out = torch.empty((n, c), dtype=dtype, device=g.device)
    if n:
        per, gy, gx = _grid(n, c)
        if mask is None:
            m = torch.zeros(8, dtype=torch.uint8, device=g.device)          # not read (has_mask 0)
        else:
            m = mask.reshape(n, c)
            if not m.is_contiguous() or m.storage_offset() % 8:
                m = m.clone(memory_format=torch.contiguous_format)
        _lib.kernel(library(rank), "lora_down", dtype, g.dtype)(
            _lib.aligned(g), _lib.aligned(a), m, out, n, c, float(s), int(mask is not None), per, threads=(gx * NT, gy, 1),
            group_size=(NT, 1, 1))
    return out
