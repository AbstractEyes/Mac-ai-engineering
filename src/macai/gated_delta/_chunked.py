"""Chunked (WY / UT) Metal kernels for the gated delta rule: float32 throughout, every product on 8 x 8 simdgroup matrices.

The sequence is cut into chunks of C tokens. Per chunk c of one (batch row, head), with the state S_c entering the chunk
(docs/prior_art_gated_delta.md s.1.4-1.5; tools/check_chunk_equations.py checks these equations in float64):

    G_r  = sum_{s<=r} g_s (the in-chunk cumulative log decay); gamma_r = exp(G_r); e_r = exp(G_last - G_r)
    KG   = (K K^T) * exp(G_i - G_j), strictly lower          M = (Q K^T) * exp(G_i - G_j), lower with the diagonal
    A    = (I + diag(beta) KG)^-1                             (unit lower triangular)
    Xu   = V - diag(gamma) K S_c                              Vn = A diag(beta) Xu    (row r = the token's correction d_t)
    O    = diag(gamma) Q S_c + M Vn
    S_c+1 = exp(G_last) S_c + K^T diag(e) Vn

Decays are only ever exp of differences of in-chunk log sums, masked before the exponent, so they never exceed 1 and never
overflow; nothing is clamped. The inverse A is found by substitution on its 8 x 8 diagonal blocks, then the blocks below the
diagonal one distance at a time by products (A_ij = -A_ii sum_m L_im A_mj).

Kernels (one GPU pass each):
- gdc_prep: one threadgroup a (chunk, head, batch row), all independent: G, KG, M, A.
- gdc_fwd: the state pass, one threadgroup a (32 value columns, head, batch row); each simdgroup owns 8 * NCB value columns of
  S for all key rows, in registers, and walks the chunks in order. Columns of S are independent, so simdgroups never wait on
  each other. When training it stores the state entering each chunk, Vn and Xu for the backward; or (op.RECOMPUTE_STATES)
  the training forward stores nothing and the backward runs this pass again without its outputs to rebuild them.
- gdc_bwd_state: the reverse state pass on the same grid, carrying dS = dLoss/dS from the last chunk to the first:
      dVn = M^T dO + diag(e) K dS_{c+1}
      dS_c = exp(G_last) dS_{c+1} + Q^T diag(gamma) dO - K^T diag(beta gamma) A^T dVn
  It stores dS_{c+1} and dVn per chunk.
- gdc_bwd_chunk: everything else, one threadgroup a (chunk, head, batch row), all independent; each gradient tile has one
  writer, so there are no atomics:
      dM = exp(G_i - G_j) (dO Vn^T) (lower), dA = (dVn Xu^T) diag(beta) (strictly lower), dL = -A^T dA A^T (strictly lower),
      P = dL * exp(G_i - G_j)
      dQ = diag(gamma) dO S_c^T + dM K
      dK = dM^T Q + diag(e) Vn dS_{c+1}^T - diag(beta gamma) A^T dVn S_c^T + diag(beta) P K + P^T diag(beta) K
      dV = diag(beta) A^T dVn
      dbeta, dg from row sums of the same products, then dg's reverse cumulative sum inside the chunk.
"""
import functools

import torch

C = 16       # chunk length (tokens); the kernels map one lane to one row of a chunk, so C <= 32 and C % 8 == 0.
             # Measured on the M3 Max (docs/series_phase3.md): 16 gives the fastest forward with states + backward
NCB = 1      # 8-column blocks of the state a simdgroup owns in the state passes
NSG = 16     # most simdgroups a threadgroup in the state passes (a threadgroup owns 8 * NCB * NSG value columns: one head
             # at DV = 128, so a chunk's staged K and Q serve all its value columns)
STAGE = 1    # the state passes stage each chunk's K and Q in threadgroup memory (2 C DK floats: needs C * DK <= 4,096)
HWLOAD = 0   # whole device tiles through simdgroup_load instead of per-lane loads
NSGB = 8     # simdgroups a threadgroup in the per-chunk backward
KGRP = 1     # 8-column key blocks a simdgroup of the per-chunk backward accumulates at once
DBUF = 0     # series design D3 (a2): the state passes read the next chunk's keys and queries into registers before this chunk's
             # products and each per-lane input of the next chunk (decay sums, beta, value / output-gradient tiles, A, M) right
             # after its last use in this chunk. 1: the keys and queries staged into one of two threadgroup buffers, one barrier a
             # chunk (needs C * DK <= 2,048: four C x DK buffers in 32 KB); 2: one buffer, written after a barrier that retires
             # its reads (two barriers a chunk, 16 KB). 0 = Phase 3's staging (each chunk's loads after its first barrier)

_SOURCE = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define DK __DK__
#define C __C__
#define NCB __NCB__
#define STAGE __STAGE__              // 1: the state passes stage each chunk's K and Q in threadgroup memory
#define HWLOAD __HWLOAD__            // 1: whole device tiles through simdgroup_load
#define NSGB __NSGB__                // simdgroups a threadgroup in gdc_bwd_chunk
#define KGRP __KGRP__                // 8-column key blocks a simdgroup of gdc_bwd_chunk accumulates at once
#define DBUF __DBUF__                // the state passes read the next chunk's inputs ahead: 1 into two staging buffers, 2 into one
#define NBUF ((DBUF) == 1 ? 2 : 1)   // staging buffers with DBUF
#define NSGT __NSGT__                // with DBUF: the state passes' simdgroups a threadgroup (the launch's width)
#define NRB (C / 8)                  // 8-row blocks in a chunk
#define NKB (DK / 8)                 // 8-row blocks of the state (key rows)
#define NLB (NRB * (NRB + 1) / 2)    // lower 8 x 8 blocks of a C x C matrix
#define NSG1 4                       // simdgroups a threadgroup in gdc_prep and gdc_bwd_chunk

typedef simdgroup_matrix<float, 8, 8> mat;
#define EL(M, I) ((M).thread_elements()[I])

// A lane holds two adjacent elements of one row of an 8 x 8 simdgroup matrix: row fm, columns fn and fn + 1.
inline ushort lane_row(uint lane) { return (ushort)(((lane >> 4) << 2) + ((lane >> 1) & 3u)); }
inline ushort lane_col(uint lane) { return (ushort)((((lane >> 3) & 1u) << 2) + ((lane & 1u) << 1)); }

inline mat mzero() { return make_filled_simdgroup_matrix<float, 8>(0.0f); }

inline mat scaled(mat m, float s) {
  mat r;
  EL(r, 0) = EL(m, 0) * s;
  EL(r, 1) = EL(m, 1) * s;
  return r;
}

// the 8 x 8 tile (rows row0.., columns col0..) of a row-major array with ld elements a row; rows at or past nrows read 0
// (with HWLOAD a whole tile in range takes the hardware load; the branch is uniform across the simdgroup)
inline mat ld_tile(const device float* p, uint ld, uint row0, uint col0, uint nrows, ushort fm, ushort fn) {
  mat m;
#if HWLOAD
  if (row0 + 8 <= nrows) {
    simdgroup_load(m, p + (ulong)row0 * ld + col0, ld);
    return m;
  }
#endif
  const uint r = row0 + fm;
  float2 x = float2(0.0f);
  if (r < nrows) x = *((const device float2*)(p + (ulong)r * ld + col0 + fn));
  EL(m, 0) = x.x;
  EL(m, 1) = x.y;
  return m;
}

// the transpose of that tile: element (i, j) = p[row0 + j][col0 + i]
inline mat ld_tile_t(const device float* p, uint ld, uint row0, uint col0, uint nrows, ushort fm, ushort fn) {
  mat m;
#if HWLOAD
  if (row0 + 8 <= nrows) {
    simdgroup_load(m, p + (ulong)row0 * ld + col0, ld, ulong2(0, 0), true);
    return m;
  }
#endif
  const uint r = row0 + fn;
  EL(m, 0) = (r < nrows) ? p[(ulong)r * ld + col0 + fm] : 0.0f;
  EL(m, 1) = (r + 1 < nrows) ? p[(ulong)(r + 1) * ld + col0 + fm] : 0.0f;
  return m;
}

inline void st_tile(mat m, device float* p, uint ld, uint row0, uint col0, uint nrows, ushort fm, ushort fn) {
  if (row0 + 8 <= nrows) {
    simdgroup_store(m, p + (ulong)row0 * ld + col0, ld);
    return;
  }
  const uint r = row0 + fm;
  if (r < nrows) *((device float2*)(p + (ulong)r * ld + col0 + fn)) = float2(EL(m, 0), EL(m, 1));
}

// a chunk's rows [0, n) (row stride ld floats, DK columns) copied into threadgroup memory as 8 x 8 tiles, tile (rb, kk) at
// (rb * NKB + kk) * 64, so each tile is 64 contiguous floats; rows past n become zeros. Every thread of the threadgroup
// takes part with 16-byte loads.
inline void stage_chunk(threadgroup float* dst, const device float* src, uint ld, uint n, uint tid, uint nthreads) {
  for (uint e = tid; e < C * (DK / 4); e += nthreads) {
    const uint r = e / (DK / 4), c4 = e % (DK / 4);
    float4 x = float4(0.0f);
    if (r < n) x = *((const device float4*)(src + (ulong)r * ld + 4 * c4));
    *((threadgroup float4*)(dst + ((r / 8) * NKB + c4 / 2) * 64 + (r % 8) * 8 + (c4 % 2) * 4)) = x;
  }
}

inline mat tg_tile(const threadgroup float* src, uint rb, uint kk) {
  mat m;
  simdgroup_load(m, src + (rb * NKB + kk) * 64, 8);
  return m;
}

inline mat tg_tile_t(const threadgroup float* src, uint rb, uint kk) {
  mat m;
  simdgroup_load(m, src + (rb * NKB + kk) * 64, 8, ulong2(0, 0), true);
  return m;
}

#if DBUF
// a thread's share of a chunk's rows [0, n) (row stride ld floats, DK columns) read into registers (rows past n read as zeros),
// and written to threadgroup memory as stage_chunk lays it out; the launch has exactly 32 * NSGT threads
#define PFN ((C * (DK / 4) + 32 * NSGT - 1) / (32 * NSGT))
inline void kq_load(thread float4* x, const device float* src, uint ld, uint n, uint tid) {
  for (uint i = 0; i < PFN; ++i) {
    const uint e = tid + i * (32 * NSGT);
    const uint r = e / (DK / 4), c4 = e % (DK / 4);
    x[i] = float4(0.0f);
    if (e < C * (DK / 4) && r < n) x[i] = *((const device float4*)(src + (ulong)r * ld + 4 * c4));
  }
}

inline void kq_store(threadgroup float* dst, thread const float4* x, uint tid) {
  for (uint i = 0; i < PFN; ++i) {
    const uint e = tid + i * (32 * NSGT);
    if (e < C * (DK / 4)) {
      const uint r = e / (DK / 4), c4 = e % (DK / 4);
      *((threadgroup float4*)(dst + ((r / 8) * NKB + c4 / 2) * 64 + (r % 8) * 8 + (c4 % 2) * 4)) = x[i];
    }
  }
}
#define KSRC Kb
#define QSRC Qb
#else
#define KSRC Ks
#define QSRC Qs
#endif

// the state passes' K and Q tiles: tile (rb, kk) of the chunk, and the transpose of tile (rb, kk)
#if STAGE
#define KTILE(rb, kk) tg_tile(KSRC, rb, kk)
#define KTILE_T(rb, kk) tg_tile_t(KSRC, rb, kk)
#define QTILE(rb, kk) tg_tile(QSRC, rb, kk)
#define QTILE_T(rb, kk) tg_tile_t(QSRC, rb, kk)
#else
#define KTILE(rb, kk) ld_tile(kc, ld, 8 * (rb), 8 * (kk), n, fm, fn)
#define KTILE_T(rb, kk) ld_tile_t(kc, ld, 8 * (rb), 8 * (kk), n, fm, fn)
#define QTILE(rb, kk) ld_tile(qc, ld, 8 * (rb), 8 * (kk), n, fm, fn)
#define QTILE_T(rb, kk) ld_tile_t(qc, ld, 8 * (rb), 8 * (kk), n, fm, fn)
#endif

// lower block index -> (block row, block column), idx = bi (bi + 1) / 2 + bj
inline void decode_lower(uint idx, thread uint& bi, thread uint& bj) {
  uint r = 0;
  while ((r + 1) * (r + 2) / 2 <= idx) ++r;
  bi = r;
  bj = idx - r * (r + 1) / 2;
}

// the sum over the four lanes that hold one tile row
inline float row_reduce(float x) {
  x += simd_shuffle_xor(x, (ushort)1);
  x += simd_shuffle_xor(x, (ushort)8);
  return x;
}

inline float row_dot(mat a, mat b) { return EL(a, 0) * EL(b, 0) + EL(a, 1) * EL(b, 1); }

// q, k: [B, T, H, DK]; v, o, dout, dv: [B, T, H, DV]; g, beta, dg, dbeta: [B, T, H]; s0, sT, dsT, ds0: [B, H, DK, DV]
// per chunk (NT = ceil(T / C) chunks a head): Gc [B, H, NT, C]; Am, Mm, KGm [B, H, NT, C, C]; Sc, dSn [B, H, NT, DK, DV];
// Vn, Xu, dVn [B, H, NT * C, DV] (rows past T hold zeros)

kernel void gdc_prep(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                     device const float* g [[buffer(2)]], device const float* beta [[buffer(3)]],
                     device float* Gc [[buffer(4)]], device float* Am [[buffer(5)]],
                     device float* Mm [[buffer(6)]], device float* KGm [[buffer(7)]],
                     constant uint& T [[buffer(8)]], constant uint& H [[buffer(9)]],
                     uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float Lt[C * C];
  threadgroup float At[C * C];
  const uint c = tg.x, h = tg.y, b = tg.z;
  const uint NT = (T + C - 1) / C;
  const uint t0 = c * C, n = min((uint)C, T - t0);
  const ulong ch = ((ulong)b * H + h) * NT + c;
  const ushort fm = lane_row(lane), fn = lane_col(lane);
  const ulong row0 = ((ulong)b * T + t0) * H + h;
  const float gl = (lane < n) ? g[row0 + (ulong)lane * H] : 0.0f;
  const float Gl = simd_prefix_inclusive_sum(gl);
  const float bl = (lane < n) ? beta[row0 + (ulong)lane * H] : 0.0f;
  if (sg == 0 && lane < C) Gc[ch * C + lane] = Gl;
  const device float* kc = k + row0 * DK;
  const device float* qc = q + row0 * DK;
  const uint ld = H * DK;
  device float* A_ = Am + ch * (C * C);
  device float* M_ = Mm + ch * (C * C);
  device float* KG_ = KGm + ch * (C * C);
  // the lower blocks of K K^T and Q K^T, shared over the simdgroups
  for (uint w = sg; w < 2 * NLB; w += NSG1) {
    const bool isq = w >= NLB;
    uint bi, bj;
    decode_lower(isq ? w - NLB : w, bi, bj);
    mat acc = mzero();
    for (uint kk = 0; kk < NKB; ++kk) {
      const mat a = ld_tile(isq ? qc : kc, ld, 8 * bi, 8 * kk, n, fm, fn);
      const mat bt = ld_tile_t(kc, ld, 8 * bj, 8 * kk, n, fm, fn);
      simdgroup_multiply_accumulate(acc, a, bt, acc);
    }
    const uint i = 8 * bi + fm;
    const float Gi = simd_shuffle(Gl, (ushort)i);
    const float bi_ = simd_shuffle(bl, (ushort)i);
    mat dec, lm;
    for (uint e = 0; e < 2; ++e) {
      const uint j = 8 * bj + fn + e;
      const float Gj = simd_shuffle(Gl, (ushort)j);
      const bool keep = isq ? (i >= j) : (i > j);
      const float d = keep ? exp(keep ? Gi - Gj : 0.0f) : 0.0f;
      EL(dec, e) = EL(acc, e) * d;
      EL(lm, e) = bi_ * EL(dec, e);
    }
    if (isq) {
      simdgroup_store(dec, M_ + (8 * bi) * C + 8 * bj, C);
    } else {
      simdgroup_store(dec, KG_ + (8 * bi) * C + 8 * bj, C);
      simdgroup_store(lm, Lt + (8 * bi) * C + 8 * bj, C);
    }
  }
  for (uint e = tid; e < C * C; e += 32 * NSG1) {
    if ((e % C) / 8 > (e / C) / 8) {
      M_[e] = 0.0f;
      KG_[e] = 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // the diagonal blocks of A by substitution: one lane a (block, column)
  if (sg == 0 && lane < 8 * NRB) {
    const uint bi = lane >> 3, cc = lane & 7u;
    float x[8];
    for (uint r = 0; r < 8; ++r) {
      float acc = (r == cc) ? 1.0f : 0.0f;
      for (uint j = 0; j < r; ++j) acc = fma(-Lt[(8 * bi + r) * C + 8 * bi + j], x[j], acc);
      x[r] = acc;
    }
    for (uint r = 0; r < 8; ++r) At[(8 * bi + r) * C + 8 * bi + cc] = x[r];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // the blocks below the diagonal, one block distance at a time: A_ij = -A_ii sum_{m=j}^{i-1} L_im A_mj
  for (uint d = 1; d < NRB; ++d) {
    for (uint i = d + sg; i < NRB; i += NSG1) {
      const uint j = i - d;
      mat acc = mzero();
      for (uint m = j; m < i; ++m) {
        mat lt, at;
        simdgroup_load(lt, Lt + (8 * i) * C + 8 * m, C);
        simdgroup_load(at, At + (8 * m) * C + 8 * j, C);
        simdgroup_multiply_accumulate(acc, lt, at, acc);
      }
      mat aii, r;
      simdgroup_load(aii, At + (8 * i) * C + 8 * i, C);
      simdgroup_multiply(r, aii, acc);
      simdgroup_store(scaled(r, -1.0f), At + (8 * i) * C + 8 * j, C);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint e = tid; e < C * C; e += 32 * NSG1) A_[e] = ((e % C) / 8 <= (e / C) / 8) ? At[e] : 0.0f;
}

kernel void gdc_fwd(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                    device const float* v [[buffer(2)]], device const float* beta [[buffer(3)]],
                    device const float* s0 [[buffer(4)]], device const float* Gc [[buffer(5)]],
                    device const float* Am [[buffer(6)]], device const float* Mm [[buffer(7)]],
                    device float* o [[buffer(8)]], device float* sT [[buffer(9)]],
                    device float* Sc [[buffer(10)]], device float* Vn [[buffer(11)]],
                    device float* Xu [[buffer(12)]], device float* ck8 [[buffer(13)]],
                    constant uint& T [[buffer(14)]], constant uint& H [[buffer(15)]],
                    constant uint& DV [[buffer(16)]], constant uint& flags [[buffer(17)]],
                    constant uint& sstride [[buffer(18)]], constant uint& nstates [[buffer(19)]],
                    uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint nsg [[simdgroups_per_threadgroup]]) {
#if DBUF
  threadgroup float Ks[NBUF * C * DK];
  threadgroup float Qs[NBUF * C * DK];
#elif STAGE
  threadgroup float Ks[C * DK];
  threadgroup float Qs[C * DK];
#endif
  // flags: 1 = store the chunk states (Sc), Vn and Xu for the chunked backward; 2 = store the state every 8 tokens (ck8,
  // [B, H, ceil(T / 8), DK, DV], the token-at-a-time backward's checkpoints); 4 = recompute Vn and Xu of the single chunk
  // tg.z = b * NT + c from stored states (Sc read at state c * sstride of nstates a head), without outputs; 8 = no outputs
  // (o and sT are not written, q is not read): with 1, the backward's recomputation of the chunk states, Vn and Xu
  const uint NT = (T + C - 1) / C;
  const bool redo = (flags & 4u) != 0;
  const bool outs = (flags & 12u) == 0;
  const uint h = tg.y, b = redo ? tg.z / NT : tg.z;
  const uint cbeg = redo ? tg.z % NT : 0, cend = redo ? cbeg + 1 : NT;
  const uint j0 = (tg.x * nsg + sg) * (8 * NCB);
  const ulong bh = (ulong)b * H + h;
  const ushort fm = lane_row(lane), fn = lane_col(lane);
  const uint ld = H * DK, ldv = H * DV;
  const ulong SS = (ulong)DK * DV;
  const uint NB8 = (T + 7) / 8;
  const device float* sinit = redo ? ((const device float*)Sc + (bh * nstates + (ulong)cbeg * sstride) * SS) : (s0 + bh * SS);
  mat S[NKB][NCB];
  for (uint kk = 0; kk < NKB; ++kk)
    for (uint cb = 0; cb < NCB; ++cb) simdgroup_load(S[kk][cb], sinit + (ulong)(8 * kk) * DV + j0 + 8 * cb, DV);
#if DBUF
  // chunk c's keys and queries sit in buffer (c - cbeg) & 1: chunk c + 1's are read into registers (pk, pq) one chunk ahead and
  // written to the other buffer after chunk c's products; the barrier closing each chunk publishes them and retires every read
  // of the buffer written next. Each per-lane input of chunk c + 1 (decay sums and beta, the value tiles, A, M) is read into
  // its own registers right after its last use in chunk c, so its latency runs under the rest of chunk c's products (one set
  // of registers, as the single-buffered pass holds).
  float4 pk[PFN], pq[PFN];
  mat Vr[NRB][NCB], Ar[NLB], Mr[NLB];
  float Gn, bn;
#define KQ_LOAD(cn)                                                                                                           \
  {                                                                                                                           \
    const uint t0n = (cn) * C, nn = min((uint)C, T - t0n);                                                                    \
    const ulong r0n = ((ulong)b * T + t0n) * H + h;                                                                           \
    kq_load(pk, k + r0n * DK, ld, nn, tid);                                                                                   \
    if (outs) kq_load(pq, q + r0n * DK, ld, nn, tid);                                                                         \
  }
#define LANE_G(cn)                                                                                                            \
  {                                                                                                                           \
    const uint t0n = (cn) * C, nn = min((uint)C, T - t0n);                                                                    \
    const ulong r0n = ((ulong)b * T + t0n) * H + h;                                                                           \
    Gn = Gc[(bh * NT + (cn)) * C + min(lane, (uint)C - 1)];                                                                   \
    bn = (lane < nn) ? beta[r0n + (ulong)lane * H] : 0.0f;                                                                    \
  }
#define LANE_V(cn)                                                                                                            \
  {                                                                                                                           \
    const uint t0n = (cn) * C, nn = min((uint)C, T - t0n);                                                                    \
    const device float* vn_ = v + (((ulong)b * T + t0n) * H + h) * DV;                                                        \
    for (uint rb = 0; rb < NRB; ++rb)                                                                                         \
      for (uint cb = 0; cb < NCB; ++cb) Vr[rb][cb] = ld_tile(vn_, ldv, 8 * rb, j0 + 8 * cb, nn, fm, fn);                      \
  }
#define LANE_A(cn)                                                                                                            \
  {                                                                                                                           \
    const device float* an_ = Am + (bh * NT + (cn)) * (C * C);                                                                \
    for (uint bi = 0; bi < NRB; ++bi)                                                                                         \
      for (uint bj = 0; bj <= bi; ++bj) Ar[bi * (bi + 1) / 2 + bj] = ld_tile(an_, C, 8 * bi, 8 * bj, C, fm, fn);              \
  }
#define LANE_M(cn)                                                                                                            \
  {                                                                                                                           \
    const device float* mn_ = Mm + (bh * NT + (cn)) * (C * C);                                                                \
    for (uint bi = 0; bi < NRB; ++bi)                                                                                         \
      for (uint bj = 0; bj <= bi; ++bj) Mr[bi * (bi + 1) / 2 + bj] = ld_tile(mn_, C, 8 * bi, 8 * bj, C, fm, fn);              \
  }
  KQ_LOAD(cbeg);
  LANE_G(cbeg);
  LANE_V(cbeg);
  LANE_A(cbeg);
  if (outs) LANE_M(cbeg);
  kq_store(Ks, pk, tid);
  if (outs) kq_store(Qs, pq, tid);
  if (cbeg + 1 < cend) KQ_LOAD(cbeg + 1);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = cbeg; c < cend; ++c) {
    const uint cur = NBUF == 2 ? (c - cbeg) & 1u : 0u;
    const threadgroup float* Kb = Ks + cur * (C * DK);
    const threadgroup float* Qb = Qs + cur * (C * DK);
    const bool more = c + 1 < cend;
    const uint t0 = c * C, n = min((uint)C, T - t0);
    const ulong ch = bh * NT + c;
    const ulong row0 = ((ulong)b * T + t0) * H + h;
    const float Gl = Gn;
    const float bl = bn;
    if (more) LANE_G(c + 1);
    const float Glast = simd_shuffle(Gl, (ushort)(C - 1));
    float gam[NRB], ee[NRB], bb[NRB];
    for (uint rb = 0; rb < NRB; ++rb) {
      const float Gi = simd_shuffle(Gl, (ushort)(8 * rb + fm));
      gam[rb] = exp(Gi);
      ee[rb] = exp(Glast - Gi);
      bb[rb] = simd_shuffle(bl, (ushort)(8 * rb + fm));
    }
    if (!redo && (flags & 1u)) {
      device float* sc = Sc + ch * SS;
      for (uint kk = 0; kk < NKB; ++kk)
        for (uint cb = 0; cb < NCB; ++cb) simdgroup_store(S[kk][cb], sc + (ulong)(8 * kk) * DV + j0 + 8 * cb, DV);
    }
    if (flags & 2u) {
      device float* sc = ck8 + (bh * NB8 + (ulong)c * (C / 8)) * SS;
      for (uint kk = 0; kk < NKB; ++kk)
        for (uint cb = 0; cb < NCB; ++cb) simdgroup_store(S[kk][cb], sc + (ulong)(8 * kk) * DV + j0 + 8 * cb, DV);
    }
    // K S and Q S
    mat KS[NRB][NCB], QS[NRB][NCB];
    for (uint rb = 0; rb < NRB; ++rb)
      for (uint cb = 0; cb < NCB; ++cb) {
        KS[rb][cb] = mzero();
        QS[rb][cb] = mzero();
      }
    for (uint kk = 0; kk < NKB; ++kk) {
      for (uint rb = 0; rb < NRB; ++rb) {
        const mat kt = KTILE(rb, kk);
        for (uint cb = 0; cb < NCB; ++cb) simdgroup_multiply_accumulate(KS[rb][cb], kt, S[kk][cb], KS[rb][cb]);
        if (outs) {
          const mat qt = QTILE(rb, kk);
          for (uint cb = 0; cb < NCB; ++cb) simdgroup_multiply_accumulate(QS[rb][cb], qt, S[kk][cb], QS[rb][cb]);
        }
      }
    }
    // Xu = V - diag(gamma) K S; the products' input diag(beta) Xu replaces K S
    for (uint rb = 0; rb < NRB; ++rb)
      for (uint cb = 0; cb < NCB; ++cb) {
        mat xu;
        for (uint e = 0; e < 2; ++e) {
          EL(xu, e) = EL(Vr[rb][cb], e) - gam[rb] * EL(KS[rb][cb], e);
          EL(KS[rb][cb], e) = bb[rb] * EL(xu, e);
        }
        if (flags & 5u) st_tile(xu, Xu + ch * (C * DV), DV, 8 * rb, j0 + 8 * cb, C, fm, fn);
      }
    if (more) LANE_V(c + 1);
    // Vn = A diag(beta) Xu
    mat VN[NRB][NCB];
    for (uint rb = 0; rb < NRB; ++rb) {
      for (uint cb = 0; cb < NCB; ++cb) VN[rb][cb] = mzero();
      for (uint m = 0; m <= rb; ++m)
        for (uint cb = 0; cb < NCB; ++cb) simdgroup_multiply_accumulate(VN[rb][cb], Ar[rb * (rb + 1) / 2 + m], KS[m][cb], VN[rb][cb]);
      if (flags & 5u)
        for (uint cb = 0; cb < NCB; ++cb) st_tile(VN[rb][cb], Vn + ch * (C * DV), DV, 8 * rb, j0 + 8 * cb, C, fm, fn);
    }
    if (more) LANE_A(c + 1);
    if (!redo) {
      // O = diag(gamma) Q S + M Vn
      if (outs) {
        for (uint rb = 0; rb < NRB; ++rb) {
          mat ot[NCB];
          for (uint cb = 0; cb < NCB; ++cb) ot[cb] = scaled(QS[rb][cb], gam[rb]);
          for (uint m = 0; m <= rb; ++m)
            for (uint cb = 0; cb < NCB; ++cb) simdgroup_multiply_accumulate(ot[cb], Mr[rb * (rb + 1) / 2 + m], VN[m][cb], ot[cb]);
          for (uint cb = 0; cb < NCB; ++cb) st_tile(ot[cb], o + row0 * DV, ldv, 8 * rb, j0 + 8 * cb, n, fm, fn);
        }
        if (more) LANE_M(c + 1);
      }
      // the states every 8 tokens inside the chunk (only for the token-at-a-time backward)
      if (flags & 2u) {
        for (uint m = 1; m < NRB; ++m) {
          if (t0 + 8 * m >= T) break;
          const float Gm = simd_shuffle(Gl, (ushort)(8 * m - 1));
          mat W[NRB][NCB];
          for (uint rb = 0; rb < m; ++rb) {
            const float f = exp(Gm - simd_shuffle(Gl, (ushort)(8 * rb + fm)));
            for (uint cb = 0; cb < NCB; ++cb) W[rb][cb] = scaled(VN[rb][cb], f);
          }
          const float gm = exp(Gm);
          device float* sc = ck8 + (bh * NB8 + (ulong)c * (C / 8) + m) * SS;
          for (uint kk = 0; kk < NKB; ++kk) {
            mat acc[NCB];
            for (uint cb = 0; cb < NCB; ++cb) acc[cb] = scaled(S[kk][cb], gm);
            for (uint rb = 0; rb < m; ++rb) {
              const mat ktt = KTILE_T(rb, kk);
              for (uint cb = 0; cb < NCB; ++cb) simdgroup_multiply_accumulate(acc[cb], ktt, W[rb][cb], acc[cb]);
            }
            for (uint cb = 0; cb < NCB; ++cb) simdgroup_store(acc[cb], sc + (ulong)(8 * kk) * DV + j0 + 8 * cb, DV);
          }
        }
      }
      // S <- exp(G_last) S + K^T diag(e) Vn
      const float glast = exp(Glast);
      for (uint rb = 0; rb < NRB; ++rb)
        for (uint cb = 0; cb < NCB; ++cb) VN[rb][cb] = scaled(VN[rb][cb], ee[rb]);
      for (uint kk = 0; kk < NKB; ++kk) {
        for (uint cb = 0; cb < NCB; ++cb) S[kk][cb] = scaled(S[kk][cb], glast);
        for (uint rb = 0; rb < NRB; ++rb) {
          const mat ktt = KTILE_T(rb, kk);
          for (uint cb = 0; cb < NCB; ++cb) simdgroup_multiply_accumulate(S[kk][cb], ktt, VN[rb][cb], S[kk][cb]);
        }
      }
    }
    if (more) {
      if (NBUF == 1) threadgroup_barrier(mem_flags::mem_threadgroup);
      kq_store(Ks + (cur ^ (NBUF - 1u)) * (C * DK), pk, tid);
      if (outs) kq_store(Qs + (cur ^ (NBUF - 1u)) * (C * DK), pq, tid);
      if (c + 2 < cend) KQ_LOAD(c + 2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#undef KQ_LOAD
#undef LANE_G
#undef LANE_V
#undef LANE_A
#undef LANE_M
#else
  for (uint c = cbeg; c < cend; ++c) {
    const uint t0 = c * C, n = min((uint)C, T - t0);
    const ulong ch = bh * NT + c;
    const ulong row0 = ((ulong)b * T + t0) * H + h;
    const device float* kc = k + row0 * DK;
    const device float* qc = q + row0 * DK;
    const device float* vc = v + row0 * DV;
#if STAGE
    threadgroup_barrier(mem_flags::mem_threadgroup);
    stage_chunk(Ks, kc, ld, n, tid, 32 * nsg);
    if (outs) stage_chunk(Qs, qc, ld, n, tid, 32 * nsg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    const float Gl = Gc[ch * C + min(lane, (uint)C - 1)];
    const float bl = (lane < n) ? beta[row0 + (ulong)lane * H] : 0.0f;
    const float Glast = simd_shuffle(Gl, (ushort)(C - 1));
    float gam[NRB], ee[NRB], bb[NRB];
    for (uint rb = 0; rb < NRB; ++rb) {
      const float Gi = simd_shuffle(Gl, (ushort)(8 * rb + fm));
      gam[rb] = exp(Gi);
      ee[rb] = exp(Glast - Gi);
      bb[rb] = simd_shuffle(bl, (ushort)(8 * rb + fm));
    }
    if (!redo && (flags & 1u)) {
      device float* sc = Sc + ch * SS;
      for (uint kk = 0; kk < NKB; ++kk)
        for (uint cb = 0; cb < NCB; ++cb) simdgroup_store(S[kk][cb], sc + (ulong)(8 * kk) * DV + j0 + 8 * cb, DV);
    }
    if (flags & 2u) {
      device float* sc = ck8 + (bh * NB8 + (ulong)c * (C / 8)) * SS;
      for (uint kk = 0; kk < NKB; ++kk)
        for (uint cb = 0; cb < NCB; ++cb) simdgroup_store(S[kk][cb], sc + (ulong)(8 * kk) * DV + j0 + 8 * cb, DV);
    }
    // K S and Q S
    mat KS[NRB][NCB], QS[NRB][NCB];
    for (uint rb = 0; rb < NRB; ++rb)
      for (uint cb = 0; cb < NCB; ++cb) {
        KS[rb][cb] = mzero();
        QS[rb][cb] = mzero();
      }
    for (uint kk = 0; kk < NKB; ++kk) {
      for (uint rb = 0; rb < NRB; ++rb) {
        const mat kt = KTILE(rb, kk);
        for (uint cb = 0; cb < NCB; ++cb) simdgroup_multiply_accumulate(KS[rb][cb], kt, S[kk][cb], KS[rb][cb]);
        if (outs) {
          const mat qt = QTILE(rb, kk);
          for (uint cb = 0; cb < NCB; ++cb) simdgroup_multiply_accumulate(QS[rb][cb], qt, S[kk][cb], QS[rb][cb]);
        }
      }
    }
    // Xu = V - diag(gamma) K S; the products' input diag(beta) Xu replaces K S
    for (uint rb = 0; rb < NRB; ++rb)
      for (uint cb = 0; cb < NCB; ++cb) {
        const mat vt = ld_tile(vc, ldv, 8 * rb, j0 + 8 * cb, n, fm, fn);
        mat xu;
        for (uint e = 0; e < 2; ++e) {
          EL(xu, e) = EL(vt, e) - gam[rb] * EL(KS[rb][cb], e);
          EL(KS[rb][cb], e) = bb[rb] * EL(xu, e);
        }
        if (flags & 5u) st_tile(xu, Xu + ch * (C * DV), DV, 8 * rb, j0 + 8 * cb, C, fm, fn);
      }
    // Vn = A diag(beta) Xu
    const device float* A_ = Am + ch * (C * C);
    mat VN[NRB][NCB];
    for (uint rb = 0; rb < NRB; ++rb) {
      for (uint cb = 0; cb < NCB; ++cb) VN[rb][cb] = mzero();
      for (uint m = 0; m <= rb; ++m) {
        const mat at = ld_tile(A_, C, 8 * rb, 8 * m, C, fm, fn);
        for (uint cb = 0; cb < NCB; ++cb) simdgroup_multiply_accumulate(VN[rb][cb], at, KS[m][cb], VN[rb][cb]);
      }
      if (flags & 5u)
        for (uint cb = 0; cb < NCB; ++cb) st_tile(VN[rb][cb], Vn + ch * (C * DV), DV, 8 * rb, j0 + 8 * cb, C, fm, fn);
    }
    if (redo) continue;
    // O = diag(gamma) Q S + M Vn
    if (outs) {
      const device float* M_ = Mm + ch * (C * C);
      for (uint rb = 0; rb < NRB; ++rb) {
        mat ot[NCB];
        for (uint cb = 0; cb < NCB; ++cb) ot[cb] = scaled(QS[rb][cb], gam[rb]);
        for (uint m = 0; m <= rb; ++m) {
          const mat mt = ld_tile(M_, C, 8 * rb, 8 * m, C, fm, fn);
          for (uint cb = 0; cb < NCB; ++cb) simdgroup_multiply_accumulate(ot[cb], mt, VN[m][cb], ot[cb]);
        }
        for (uint cb = 0; cb < NCB; ++cb) st_tile(ot[cb], o + row0 * DV, ldv, 8 * rb, j0 + 8 * cb, n, fm, fn);
      }
    }
    // the states every 8 tokens inside the chunk (only for the token-at-a-time backward)
    if (flags & 2u) {
      for (uint m = 1; m < NRB; ++m) {
        if (t0 + 8 * m >= T) break;
        const float Gm = simd_shuffle(Gl, (ushort)(8 * m - 1));
        mat W[NRB][NCB];
        for (uint rb = 0; rb < m; ++rb) {
          const float f = exp(Gm - simd_shuffle(Gl, (ushort)(8 * rb + fm)));
          for (uint cb = 0; cb < NCB; ++cb) W[rb][cb] = scaled(VN[rb][cb], f);
        }
        const float gm = exp(Gm);
        device float* sc = ck8 + (bh * NB8 + (ulong)c * (C / 8) + m) * SS;
        for (uint kk = 0; kk < NKB; ++kk) {
          mat acc[NCB];
          for (uint cb = 0; cb < NCB; ++cb) acc[cb] = scaled(S[kk][cb], gm);
          for (uint rb = 0; rb < m; ++rb) {
            const mat ktt = KTILE_T(rb, kk);
            for (uint cb = 0; cb < NCB; ++cb) simdgroup_multiply_accumulate(acc[cb], ktt, W[rb][cb], acc[cb]);
          }
          for (uint cb = 0; cb < NCB; ++cb) simdgroup_store(acc[cb], sc + (ulong)(8 * kk) * DV + j0 + 8 * cb, DV);
        }
      }
    }
    // S <- exp(G_last) S + K^T diag(e) Vn
    const float glast = exp(Glast);
    for (uint rb = 0; rb < NRB; ++rb)
      for (uint cb = 0; cb < NCB; ++cb) VN[rb][cb] = scaled(VN[rb][cb], ee[rb]);
    for (uint kk = 0; kk < NKB; ++kk) {
      for (uint cb = 0; cb < NCB; ++cb) S[kk][cb] = scaled(S[kk][cb], glast);
      for (uint rb = 0; rb < NRB; ++rb) {
        const mat ktt = KTILE_T(rb, kk);
        for (uint cb = 0; cb < NCB; ++cb) simdgroup_multiply_accumulate(S[kk][cb], ktt, VN[rb][cb], S[kk][cb]);
      }
    }
  }
#endif
  if (outs)
    for (uint kk = 0; kk < NKB; ++kk)
      for (uint cb = 0; cb < NCB; ++cb) simdgroup_store(S[kk][cb], sT + bh * SS + (ulong)(8 * kk) * DV + j0 + 8 * cb, DV);
}

kernel void gdc_bwd_state(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                          device const float* beta [[buffer(2)]], device const float* dout [[buffer(3)]],
                          device const float* dsT [[buffer(4)]], device const float* Gc [[buffer(5)]],
                          device const float* Am [[buffer(6)]], device const float* Mm [[buffer(7)]],
                          device float* dSn [[buffer(8)]], device float* dVn [[buffer(9)]],
                          device float* ds0 [[buffer(10)]],
                          constant uint& T [[buffer(11)]], constant uint& H [[buffer(12)]],
                          constant uint& DV [[buffer(13)]],
                          uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]]) {
#if DBUF
  threadgroup float Ks[NBUF * C * DK];
  threadgroup float Qs[NBUF * C * DK];
#elif STAGE
  threadgroup float Ks[C * DK];
  threadgroup float Qs[C * DK];
#endif
  const uint NT = (T + C - 1) / C;
  const uint h = tg.y, b = tg.z;
  const uint j0 = (tg.x * nsg + sg) * (8 * NCB);
  const ulong bh = (ulong)b * H + h;
  const ushort fm = lane_row(lane), fn = lane_col(lane);
  const uint ld = H * DK, ldv = H * DV;
  const ulong SS = (ulong)DK * DV;
  mat dS[NKB][NCB];
  for (uint kk = 0; kk < NKB; ++kk)
    for (uint cb = 0; cb < NCB; ++cb) simdgroup_load(dS[kk][cb], dsT + bh * SS + (ulong)(8 * kk) * DV + j0 + 8 * cb, DV);
#if DBUF
  // as gdc_fwd's double buffering, the chunks in reverse: chunk c's keys and queries in buffer (NT - 1 - c) & 1, chunk c - 1's
  // read into registers a chunk ahead; each per-lane input of chunk c - 1 (decay sums and beta, M^T, A^T, the output-gradient
  // tiles) read right after its last use in chunk c
  float4 pk[PFN], pq[PFN];
  mat DOr[NRB][NCB], MTr[NLB], ATr[NLB];
  float Gn, bn;
#define KQ_LOAD(cn)                                                                                                           \
  {                                                                                                                           \
    const uint t0n = (cn) * C, nn = min((uint)C, T - t0n);                                                                    \
    const ulong r0n = ((ulong)b * T + t0n) * H + h;                                                                           \
    kq_load(pk, k + r0n * DK, ld, nn, tid);                                                                                   \
    kq_load(pq, q + r0n * DK, ld, nn, tid);                                                                                   \
  }
#define LANE_G(cn)                                                                                                            \
  {                                                                                                                           \
    const uint t0n = (cn) * C, nn = min((uint)C, T - t0n);                                                                    \
    const ulong r0n = ((ulong)b * T + t0n) * H + h;                                                                           \
    Gn = Gc[(bh * NT + (cn)) * C + min(lane, (uint)C - 1)];                                                                   \
    bn = (lane < nn) ? beta[r0n + (ulong)lane * H] : 0.0f;                                                                    \
  }
#define LANE_DO(cn)                                                                                                           \
  {                                                                                                                           \
    const uint t0n = (cn) * C, nn = min((uint)C, T - t0n);                                                                    \
    const device float* don_ = dout + (((ulong)b * T + t0n) * H + h) * DV;                                                    \
    for (uint rb = 0; rb < NRB; ++rb)                                                                                         \
      for (uint cb = 0; cb < NCB; ++cb) DOr[rb][cb] = ld_tile(don_, ldv, 8 * rb, j0 + 8 * cb, nn, fm, fn);                    \
  }
#define LANE_MT(cn)                                                                                                           \
  {                                                                                                                           \
    const device float* mn_ = Mm + (bh * NT + (cn)) * (C * C);                                                                \
    for (uint bi = 0; bi < NRB; ++bi)                                                                                         \
      for (uint bj = 0; bj <= bi; ++bj) MTr[bi * (bi + 1) / 2 + bj] = ld_tile_t(mn_, C, 8 * bi, 8 * bj, C, fm, fn);           \
  }
#define LANE_AT(cn)                                                                                                           \
  {                                                                                                                           \
    const device float* an_ = Am + (bh * NT + (cn)) * (C * C);                                                                \
    for (uint bi = 0; bi < NRB; ++bi)                                                                                         \
      for (uint bj = 0; bj <= bi; ++bj) ATr[bi * (bi + 1) / 2 + bj] = ld_tile_t(an_, C, 8 * bi, 8 * bj, C, fm, fn);           \
  }
  KQ_LOAD(NT - 1);
  LANE_G(NT - 1);
  LANE_DO(NT - 1);
  LANE_MT(NT - 1);
  LANE_AT(NT - 1);
  kq_store(Ks, pk, tid);
  kq_store(Qs, pq, tid);
  if (NT >= 2) KQ_LOAD(NT - 2);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int ci = (int)NT - 1; ci >= 0; --ci) {
    const uint c = (uint)ci;
    const uint cur = NBUF == 2 ? (NT - 1 - c) & 1u : 0u;
    const threadgroup float* Kb = Ks + cur * (C * DK);
    const threadgroup float* Qb = Qs + cur * (C * DK);
    const bool more = ci >= 1;
    const ulong ch = bh * NT + c;
    const float Gl = Gn;
    const float bl = bn;
    if (more) LANE_G(c - 1);
    const float Glast = simd_shuffle(Gl, (ushort)(C - 1));
    float gam[NRB], ee[NRB], bb[NRB];
    for (uint rb = 0; rb < NRB; ++rb) {
      const float Gi = simd_shuffle(Gl, (ushort)(8 * rb + fm));
      gam[rb] = exp(Gi);
      ee[rb] = exp(Glast - Gi);
      bb[rb] = simd_shuffle(bl, (ushort)(8 * rb + fm));
    }
    {
      device float* sc = dSn + ch * SS;
      for (uint kk = 0; kk < NKB; ++kk)
        for (uint cb = 0; cb < NCB; ++cb) simdgroup_store(dS[kk][cb], sc + (ulong)(8 * kk) * DV + j0 + 8 * cb, DV);
    }
    // K dS
    mat KD[NRB][NCB];
    for (uint rb = 0; rb < NRB; ++rb)
      for (uint cb = 0; cb < NCB; ++cb) KD[rb][cb] = mzero();
    for (uint kk = 0; kk < NKB; ++kk)
      for (uint rb = 0; rb < NRB; ++rb) {
        const mat kt = KTILE(rb, kk);
        for (uint cb = 0; cb < NCB; ++cb) simdgroup_multiply_accumulate(KD[rb][cb], kt, dS[kk][cb], KD[rb][cb]);
      }
    mat DO[NRB][NCB];
    for (uint rb = 0; rb < NRB; ++rb)
      for (uint cb = 0; cb < NCB; ++cb) DO[rb][cb] = DOr[rb][cb];
    if (more) LANE_DO(c - 1);
    // dVn = M^T dO + diag(e) K dS
    mat DVN[NRB][NCB];
    for (uint rb = 0; rb < NRB; ++rb) {
      for (uint cb = 0; cb < NCB; ++cb) DVN[rb][cb] = scaled(KD[rb][cb], ee[rb]);
      for (uint m = rb; m < NRB; ++m)
        for (uint cb = 0; cb < NCB; ++cb) simdgroup_multiply_accumulate(DVN[rb][cb], MTr[m * (m + 1) / 2 + rb], DO[m][cb], DVN[rb][cb]);
      for (uint cb = 0; cb < NCB; ++cb) st_tile(DVN[rb][cb], dVn + ch * (C * DV), DV, 8 * rb, j0 + 8 * cb, C, fm, fn);
    }
    if (more) LANE_MT(c - 1);
    // Z = -diag(beta gamma) A^T dVn (into KD); dO scaled by gamma
    for (uint rb = 0; rb < NRB; ++rb) {
      for (uint cb = 0; cb < NCB; ++cb) KD[rb][cb] = mzero();
      for (uint m = rb; m < NRB; ++m)
        for (uint cb = 0; cb < NCB; ++cb) simdgroup_multiply_accumulate(KD[rb][cb], ATr[m * (m + 1) / 2 + rb], DVN[m][cb], KD[rb][cb]);
      for (uint cb = 0; cb < NCB; ++cb) {
        KD[rb][cb] = scaled(KD[rb][cb], -bb[rb] * gam[rb]);
        DO[rb][cb] = scaled(DO[rb][cb], gam[rb]);
      }
    }
    if (more) LANE_AT(c - 1);
    // dS <- exp(G_last) dS + Q^T diag(gamma) dO - K^T diag(beta gamma) A^T dVn
    const float glast = exp(Glast);
    for (uint kk = 0; kk < NKB; ++kk) {
      for (uint cb = 0; cb < NCB; ++cb) dS[kk][cb] = scaled(dS[kk][cb], glast);
      for (uint rb = 0; rb < NRB; ++rb) {
        const mat qtt = QTILE_T(rb, kk);
        const mat ktt = KTILE_T(rb, kk);
        for (uint cb = 0; cb < NCB; ++cb) {
          simdgroup_multiply_accumulate(dS[kk][cb], qtt, DO[rb][cb], dS[kk][cb]);
          simdgroup_multiply_accumulate(dS[kk][cb], ktt, KD[rb][cb], dS[kk][cb]);
        }
      }
    }
    if (more) {
      if (NBUF == 1) threadgroup_barrier(mem_flags::mem_threadgroup);
      kq_store(Ks + (cur ^ (NBUF - 1u)) * (C * DK), pk, tid);
      kq_store(Qs + (cur ^ (NBUF - 1u)) * (C * DK), pq, tid);
      if (ci >= 2) KQ_LOAD(c - 2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#undef KQ_LOAD
#undef LANE_G
#undef LANE_DO
#undef LANE_MT
#undef LANE_AT
#else
  for (int ci = (int)NT - 1; ci >= 0; --ci) {
    const uint c = (uint)ci;
    const uint t0 = c * C, n = min((uint)C, T - t0);
    const ulong ch = bh * NT + c;
    const ulong row0 = ((ulong)b * T + t0) * H + h;
    const device float* kc = k + row0 * DK;
    const device float* qc = q + row0 * DK;
    const device float* doc = dout + row0 * DV;
#if STAGE
    threadgroup_barrier(mem_flags::mem_threadgroup);
    stage_chunk(Ks, kc, ld, n, tid, 32 * nsg);
    stage_chunk(Qs, qc, ld, n, tid, 32 * nsg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    const float Gl = Gc[ch * C + min(lane, (uint)C - 1)];
    const float bl = (lane < n) ? beta[row0 + (ulong)lane * H] : 0.0f;
    const float Glast = simd_shuffle(Gl, (ushort)(C - 1));
    float gam[NRB], ee[NRB], bb[NRB];
    for (uint rb = 0; rb < NRB; ++rb) {
      const float Gi = simd_shuffle(Gl, (ushort)(8 * rb + fm));
      gam[rb] = exp(Gi);
      ee[rb] = exp(Glast - Gi);
      bb[rb] = simd_shuffle(bl, (ushort)(8 * rb + fm));
    }
    {
      device float* sc = dSn + ch * SS;
      for (uint kk = 0; kk < NKB; ++kk)
        for (uint cb = 0; cb < NCB; ++cb) simdgroup_store(dS[kk][cb], sc + (ulong)(8 * kk) * DV + j0 + 8 * cb, DV);
    }
    // K dS
    mat KD[NRB][NCB];
    for (uint rb = 0; rb < NRB; ++rb)
      for (uint cb = 0; cb < NCB; ++cb) KD[rb][cb] = mzero();
    for (uint kk = 0; kk < NKB; ++kk)
      for (uint rb = 0; rb < NRB; ++rb) {
        const mat kt = KTILE(rb, kk);
        for (uint cb = 0; cb < NCB; ++cb) simdgroup_multiply_accumulate(KD[rb][cb], kt, dS[kk][cb], KD[rb][cb]);
      }
    mat DO[NRB][NCB];
    for (uint rb = 0; rb < NRB; ++rb)
      for (uint cb = 0; cb < NCB; ++cb) DO[rb][cb] = ld_tile(doc, ldv, 8 * rb, j0 + 8 * cb, n, fm, fn);
    // dVn = M^T dO + diag(e) K dS
    const device float* M_ = Mm + ch * (C * C);
    mat DVN[NRB][NCB];
    for (uint rb = 0; rb < NRB; ++rb) {
      for (uint cb = 0; cb < NCB; ++cb) DVN[rb][cb] = scaled(KD[rb][cb], ee[rb]);
      for (uint m = rb; m < NRB; ++m) {
        const mat mtt = ld_tile_t(M_, C, 8 * m, 8 * rb, C, fm, fn);
        for (uint cb = 0; cb < NCB; ++cb) simdgroup_multiply_accumulate(DVN[rb][cb], mtt, DO[m][cb], DVN[rb][cb]);
      }
      for (uint cb = 0; cb < NCB; ++cb) st_tile(DVN[rb][cb], dVn + ch * (C * DV), DV, 8 * rb, j0 + 8 * cb, C, fm, fn);
    }
    // Z = -diag(beta gamma) A^T dVn (into KD); dO scaled by gamma
    const device float* A_ = Am + ch * (C * C);
    for (uint rb = 0; rb < NRB; ++rb) {
      for (uint cb = 0; cb < NCB; ++cb) KD[rb][cb] = mzero();
      for (uint m = rb; m < NRB; ++m) {
        const mat att = ld_tile_t(A_, C, 8 * m, 8 * rb, C, fm, fn);
        for (uint cb = 0; cb < NCB; ++cb) simdgroup_multiply_accumulate(KD[rb][cb], att, DVN[m][cb], KD[rb][cb]);
      }
      for (uint cb = 0; cb < NCB; ++cb) {
        KD[rb][cb] = scaled(KD[rb][cb], -bb[rb] * gam[rb]);
        DO[rb][cb] = scaled(DO[rb][cb], gam[rb]);
      }
    }
    // dS <- exp(G_last) dS + Q^T diag(gamma) dO - K^T diag(beta gamma) A^T dVn
    const float glast = exp(Glast);
    for (uint kk = 0; kk < NKB; ++kk) {
      for (uint cb = 0; cb < NCB; ++cb) dS[kk][cb] = scaled(dS[kk][cb], glast);
      for (uint rb = 0; rb < NRB; ++rb) {
        const mat qtt = QTILE_T(rb, kk);
        const mat ktt = KTILE_T(rb, kk);
        for (uint cb = 0; cb < NCB; ++cb) {
          simdgroup_multiply_accumulate(dS[kk][cb], qtt, DO[rb][cb], dS[kk][cb]);
          simdgroup_multiply_accumulate(dS[kk][cb], ktt, KD[rb][cb], dS[kk][cb]);
        }
      }
    }
  }
#endif
  for (uint kk = 0; kk < NKB; ++kk)
    for (uint cb = 0; cb < NCB; ++cb) simdgroup_store(dS[kk][cb], ds0 + bh * SS + (ulong)(8 * kk) * DV + j0 + 8 * cb, DV);
}

kernel void gdc_bwd_chunk(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                          device const float* beta [[buffer(2)]], device const float* dout [[buffer(3)]],
                          device const float* Gc [[buffer(4)]], device const float* Am [[buffer(5)]],
                          device const float* KGm [[buffer(6)]], device const float* Mm [[buffer(7)]],
                          device const float* Sc [[buffer(8)]], device const float* dSn [[buffer(9)]],
                          device const float* Vn [[buffer(10)]], device const float* Xu [[buffer(11)]],
                          device const float* dVn [[buffer(12)]],
                          device float* dq [[buffer(13)]], device float* dk [[buffer(14)]],
                          device float* dv [[buffer(15)]], device float* dg [[buffer(16)]],
                          device float* dbeta [[buffer(17)]],
                          constant uint& T [[buffer(18)]], constant uint& H [[buffer(19)]],
                          constant uint& DV [[buffer(20)]], constant uint& sstride [[buffer(21)]],
                          constant uint& nstates [[buffer(22)]],
                          uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float dMt[C * C];       // exp(G_i - G_j) (dO Vn^T), lower with the diagonal
  threadgroup float Pt[C * C];        // dA, then P = dL * exp(G_i - G_j), strictly lower
  threadgroup float T1t[C * C];       // dA A^T
  threadgroup float KGt[C * C];       // dL * KG, strictly lower
  threadgroup float Et[C * C];        // (dO Vn^T) * M, strictly lower: M's share of dG
  threadgroup float red[4][NSGB][C];  // per-simdgroup row sums
  threadgroup float reds[NSGB];
  const uint c = tg.x, h = tg.y, b = tg.z;
  const uint NT = (T + C - 1) / C;
  const uint t0 = c * C, n = min((uint)C, T - t0);
  const ulong bh = (ulong)b * H + h, ch = bh * NT + c;
  const ushort fm = lane_row(lane), fn = lane_col(lane);
  const ulong row0 = ((ulong)b * T + t0) * H + h;
  const uint ld = H * DK, ldv = H * DV;
  const ulong SS = (ulong)DK * DV;
  const device float* kc = k + row0 * DK;
  const device float* qc = q + row0 * DK;
  const device float* doc = dout + row0 * DV;
  const device float* A_ = Am + ch * (C * C);
  const device float* KG_ = KGm + ch * (C * C);
  const device float* M_ = Mm + ch * (C * C);
  const device float* S_ = Sc + (bh * nstates + (ulong)c * sstride) * SS;
  const device float* dS_ = dSn + ch * SS;
  const device float* Vn_ = Vn + ch * (C * DV);
  const device float* Xu_ = Xu + ch * (C * DV);
  const device float* dVn_ = dVn + ch * (C * DV);
  const float Gl = Gc[ch * C + min(lane, (uint)C - 1)];
  const float bl = (lane < n) ? beta[row0 + (ulong)lane * H] : 0.0f;
  const float Glast = simd_shuffle(Gl, (ushort)(C - 1));
  float gam[NRB], ee[NRB], bb[NRB];
  for (uint rb = 0; rb < NRB; ++rb) {
    const float Gi = simd_shuffle(Gl, (ushort)(8 * rb + fm));
    gam[rb] = exp(Gi);
    ee[rb] = exp(Glast - Gi);
    bb[rb] = simd_shuffle(bl, (ushort)(8 * rb + fm));
  }
  // 1: dM = exp(G_i - G_j) (dO Vn^T) and dA = (dVn Xu^T) diag(beta), contracting over the value columns
  for (uint w = sg; w < 2 * NLB; w += NSGB) {
    const bool isa = w >= NLB;
    uint bi, bj;
    decode_lower(isa ? w - NLB : w, bi, bj);
    mat acc = mzero();
    for (uint vv = 0; vv < DV / 8; ++vv) {
      const mat a = isa ? ld_tile(dVn_, DV, 8 * bi, 8 * vv, C, fm, fn) : ld_tile(doc, ldv, 8 * bi, 8 * vv, n, fm, fn);
      const mat bt = ld_tile_t(isa ? Xu_ : Vn_, DV, 8 * bj, 8 * vv, C, fm, fn);
      simdgroup_multiply_accumulate(acc, a, bt, acc);
    }
    const uint i = 8 * bi + fm;
    const float Gi = simd_shuffle(Gl, (ushort)i);
    const mat mt = ld_tile(M_, C, 8 * bi, 8 * bj, C, fm, fn);
    mat r, et;
    for (uint e = 0; e < 2; ++e) {
      const uint j = 8 * bj + fn + e;
      const float Gj = simd_shuffle(Gl, (ushort)j);
      const float bj_ = simd_shuffle(bl, (ushort)j);
      if (isa) {
        EL(r, e) = (i > j) ? EL(acc, e) * bj_ : 0.0f;
      } else {
        EL(r, e) = (i >= j) ? EL(acc, e) * exp((i >= j) ? Gi - Gj : 0.0f) : 0.0f;
        // the diagonal's decay is exactly 1 and cancels in dG; leaving it out keeps dg exact under strong decay
        EL(et, e) = (i > j) ? EL(acc, e) * EL(mt, e) : 0.0f;
      }
    }
    simdgroup_store(r, (isa ? Pt : dMt) + (8 * bi) * C + 8 * bj, C);
    if (!isa) simdgroup_store(et, Et + (8 * bi) * C + 8 * bj, C);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // 2a: T1 = dA A^T, lower blocks: T1(bi, bj) = sum_{bm <= bj} dA(bi, bm) A(bj, bm)^T
  for (uint idx = sg; idx < NLB; idx += NSGB) {
    uint bi, bj;
    decode_lower(idx, bi, bj);
    mat acc = mzero();
    for (uint bm = 0; bm <= bj; ++bm) {
      mat a;
      simdgroup_load(a, Pt + (8 * bi) * C + 8 * bm, C);
      const mat at = ld_tile_t(A_, C, 8 * bj, 8 * bm, C, fm, fn);
      simdgroup_multiply_accumulate(acc, a, at, acc);
    }
    simdgroup_store(acc, T1t + (8 * bi) * C + 8 * bj, C);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // 2b: dL = -A^T T1 (strictly lower); P = dL * exp(G_i - G_j); dL * KG
  for (uint idx = sg; idx < NLB; idx += NSGB) {
    uint bi, bj;
    decode_lower(idx, bi, bj);
    mat acc = mzero();
    for (uint bm = bi; bm < NRB; ++bm) {
      const mat att = ld_tile_t(A_, C, 8 * bm, 8 * bi, C, fm, fn);
      mat t1;
      simdgroup_load(t1, T1t + (8 * bm) * C + 8 * bj, C);
      simdgroup_multiply_accumulate(acc, att, t1, acc);
    }
    const mat kg = ld_tile(KG_, C, 8 * bi, 8 * bj, C, fm, fn);
    const uint i = 8 * bi + fm;
    const float Gi = simd_shuffle(Gl, (ushort)i);
    mat p, kgd;
    for (uint e = 0; e < 2; ++e) {
      const uint j = 8 * bj + fn + e;
      const float Gj = simd_shuffle(Gl, (ushort)j);
      const float dl = (i > j) ? -EL(acc, e) : 0.0f;
      EL(p, e) = (i > j) ? dl * exp((i > j) ? Gi - Gj : 0.0f) : 0.0f;
      EL(kgd, e) = dl * EL(kg, e);
    }
    simdgroup_store(p, Pt + (8 * bi) * C + 8 * bj, C);
    simdgroup_store(kgd, KGt + (8 * bi) * C + 8 * bj, C);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // 3: dQ and dK, each simdgroup KGRP blocks of 8 key columns at a time; dV on blocks of 8 value columns
  float pq[NRB], px[NRB], pkv[NRB], py[NRB];
  for (uint rb = 0; rb < NRB; ++rb) pq[rb] = px[rb] = pkv[rb] = py[rb] = 0.0f;
  float sdot = 0.0f;
  for (uint kb0 = sg * KGRP; kb0 < NKB; kb0 += NSGB * KGRP) {
    // dO S_c^T, dVn S_c^T and Vn dS_{c+1}^T on these key columns, contracting over the value columns
    mat dOS[NRB][KGRP], dVS[NRB][KGRP], VdS[NRB][KGRP];
    for (uint rb = 0; rb < NRB; ++rb)
      for (uint j = 0; j < KGRP; ++j) {
        dOS[rb][j] = mzero();
        dVS[rb][j] = mzero();
        VdS[rb][j] = mzero();
      }
    for (uint vv = 0; vv < DV / 8; ++vv) {
      mat st[KGRP], dst[KGRP];
      for (uint j = 0; j < KGRP; ++j) {
        st[j] = ld_tile_t(S_, DV, 8 * (kb0 + j), 8 * vv, DK, fm, fn);      // tile (vv, kb) of S_c^T
        dst[j] = ld_tile_t(dS_, DV, 8 * (kb0 + j), 8 * vv, DK, fm, fn);    // tile (vv, kb) of dS_{c+1}^T
        sdot += row_dot(st[j], dst[j]);
      }
      for (uint rb = 0; rb < NRB; ++rb) {
        const mat a = ld_tile(doc, ldv, 8 * rb, 8 * vv, n, fm, fn);
        const mat d = ld_tile(dVn_, DV, 8 * rb, 8 * vv, C, fm, fn);
        const mat w = ld_tile(Vn_, DV, 8 * rb, 8 * vv, C, fm, fn);
        for (uint j = 0; j < KGRP; ++j) {
          simdgroup_multiply_accumulate(dOS[rb][j], a, st[j], dOS[rb][j]);
          simdgroup_multiply_accumulate(dVS[rb][j], d, st[j], dVS[rb][j]);
          simdgroup_multiply_accumulate(VdS[rb][j], w, dst[j], VdS[rb][j]);
        }
      }
    }
    for (uint j = 0; j < KGRP; ++j) {
      const uint kb = kb0 + j;
      mat KT[NRB], QT[NRB], BK[NRB];
      for (uint rb = 0; rb < NRB; ++rb) {
        KT[rb] = ld_tile(kc, ld, 8 * rb, 8 * kb, n, fm, fn);
        QT[rb] = ld_tile(qc, ld, 8 * rb, 8 * kb, n, fm, fn);
        BK[rb] = scaled(KT[rb], bb[rb]);
      }
      for (uint rb = 0; rb < NRB; ++rb) {
        // dQ = diag(gamma) dO S_c^T + dM K
        pq[rb] += row_dot(QT[rb], dOS[rb][j]);
        mat dqt = scaled(dOS[rb][j], gam[rb]);
        for (uint m = 0; m <= rb; ++m) {
          mat dm;
          simdgroup_load(dm, dMt + (8 * rb) * C + 8 * m, C);
          simdgroup_multiply_accumulate(dqt, dm, KT[m], dqt);
        }
        st_tile(dqt, dq + row0 * DK, ld, 8 * rb, 8 * kb, n, fm, fn);
        // dK, its part through M and the state: dM^T Q + diag(e) Vn dS_{c+1}^T
        mat dk3 = scaled(VdS[rb][j], ee[rb]);
        for (uint m = rb; m < NRB; ++m) {
          mat dmt;
          simdgroup_load(dmt, dMt + (8 * m) * C + 8 * rb, C, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(dk3, dmt, QT[m], dk3);
        }
        pkv[rb] += row_dot(KT[rb], VdS[rb][j]);
        // X = A^T dW with dW = -dVn S_c^T
        mat x = mzero();
        for (uint m = rb; m < NRB; ++m) {
          const mat att = ld_tile_t(A_, C, 8 * m, 8 * rb, C, fm, fn);
          simdgroup_multiply_accumulate(x, att, dVS[m][j], x);
        }
        x = scaled(x, -1.0f);
        px[rb] += row_dot(KT[rb], x);
        // diag(beta) P K + P^T diag(beta) K
        mat pk = mzero(), ptk = mzero();
        for (uint m = 0; m <= rb; ++m) {
          mat pm;
          simdgroup_load(pm, Pt + (8 * rb) * C + 8 * m, C);
          simdgroup_multiply_accumulate(pk, pm, KT[m], pk);
        }
        for (uint m = rb; m < NRB; ++m) {
          mat pmt;
          simdgroup_load(pmt, Pt + (8 * m) * C + 8 * rb, C, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(ptk, pmt, BK[m], ptk);
        }
        mat dkt;
        const float bg = bb[rb] * gam[rb];
        for (uint e = 0; e < 2; ++e) EL(dkt, e) = EL(dk3, e) + bg * EL(x, e) + bb[rb] * EL(pk, e) + EL(ptk, e);
        st_tile(dkt, dk + row0 * DK, ld, 8 * rb, 8 * kb, n, fm, fn);
      }
    }
  }
  // dV = diag(beta) A^T dVn
  for (uint vb = sg; vb < DV / 8; vb += NSGB) {
    mat DVt[NRB];
    for (uint m = 0; m < NRB; ++m) DVt[m] = ld_tile(dVn_, DV, 8 * m, 8 * vb, C, fm, fn);
    for (uint rb = 0; rb < NRB; ++rb) {
      mat y = mzero();
      for (uint m = rb; m < NRB; ++m) {
        const mat att = ld_tile_t(A_, C, 8 * m, 8 * rb, C, fm, fn);
        simdgroup_multiply_accumulate(y, att, DVt[m], y);
      }
      st_tile(scaled(y, bb[rb]), dv + row0 * DV, ldv, 8 * rb, 8 * vb, n, fm, fn);
      py[rb] += row_dot(y, ld_tile(Xu_, DV, 8 * rb, 8 * vb, C, fm, fn));
    }
  }
  for (uint rb = 0; rb < NRB; ++rb) {
    const float a0 = row_reduce(pq[rb]), a1 = row_reduce(px[rb]), a2 = row_reduce(pkv[rb]), a3 = row_reduce(py[rb]);
    if ((lane & 9u) == 0) {
      const uint r = 8 * rb + fm;
      red[0][sg][r] = a0;
      red[1][sg][r] = a1;
      red[2][sg][r] = a2;
      red[3][sg][r] = a3;
    }
  }
  sdot = simd_sum(sdot);
  if (lane == 0) reds[sg] = sdot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // 4: one lane a row. G_r's gradient dG_r gathers the terms through gamma_r (Q's state term, W), through the decays of M
  // and L (row sums minus column sums of their strictly lower parts) and through exp(G_last) (on the last row); then
  // dg_t = sum_{s >= t} dG_s. The decays e_r = exp(G_last - G_r) depend on g_t for t > r only: they enter as the prefix sum
  // sum_{r < t} e_r k_r . (Vn dS^T)_r. Written this way no term cancels another exactly, so dg keeps its relative precision
  // when the decay is strong and every dg is tiny.
  if (sg == 0) {
    const uint r = lane;
    const bool row = r < C;
    float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float rsK = 0.0f, csR = 0.0f, rsE = 0.0f, csE = 0.0f, sd = 0.0f;
    for (uint s_ = 0; s_ < NSGB; ++s_) {
      sd += reds[s_];
      if (row)
        for (uint qn = 0; qn < 4; ++qn) s[qn] += red[qn][s_][r];
    }
    for (uint j = 0; j < C; ++j) {
      const float bj_ = simd_shuffle(bl, (ushort)j);
      if (row && j < r) {
        rsK += KGt[r * C + j];
        rsE += Et[r * C + j];
      }
      if (row && j > r) {
        csR += bj_ * KGt[j * C + r];
        csE += Et[j * C + r];
      }
    }
    const float gr = exp(Gl);
    float dG = gr * s[0] + (rsE - csE) + bl * gr * s[1] + (bl * rsK - csR);
    if (r == C - 1) dG += exp(Glast) * sd;
    if (!row) dG = 0.0f;
    const float pre = row ? exp(Glast - Gl) * s[2] : 0.0f;
    const ushort src = row ? (ushort)(C - 1 - r) : (ushort)r;
    const float cum = simd_prefix_inclusive_sum(simd_shuffle(dG, src));
    const float dgr = simd_shuffle(cum, src) + simd_prefix_exclusive_sum(pre);
    if (r < n) {
      dg[row0 + (ulong)r * H] = dgr;
      dbeta[row0 + (ulong)r * H] = s[3] + rsK;
    }
  }
}
"""


def library(dk: int, c: int | None = None, nsg: int | None = None):
    """The compiled chunked kernels for one key head width (a multiple of 32) at chunk length c (default C) and the module's
    tiling. With DBUF (where the chunk fits: C * DK <= 2,048), the state passes are compiled for threadgroups of nsg simdgroups
    (default NSG): launch them only that wide."""
    c = c or C
    stage = int(STAGE and c * dk <= 4096)
    dbuf = dbuf_for(dk, c)
    return _library(dk, c, NCB, stage, HWLOAD, NSGB, min(KGRP, dk // 8 // NSGB) or 1, dbuf, (nsg or NSG) if dbuf else NSG)


def dbuf_for(dk: int, c: int) -> int:
    """The state passes' DBUF form at this key width and chunk length: DBUF 1 needs four C x DK buffers in 32 KB (C * DK <=
    2,048), DBUF 2 the two of single-buffered staging (C * DK <= 4,096); 0 where they do not fit."""
    if not (DBUF and STAGE):
        return 0
    return int(DBUF) if c * dk <= (2048 if DBUF == 1 else 4096) else 0


def state_simdgroups(dv: int, split: int = 1) -> int:
    """Simdgroups a threadgroup of the state passes for value width dv: as many 8 * NCB column slices as fit, up to NSG; with
    split > 1, at most NSG / split, so a head-row's columns go to at least `split` threadgroups (series design D3 (b))."""
    per = 8 * NCB
    if dv % per != 0:
        raise ValueError(f"the chunked kernels need the value head width to be a multiple of {per}, got {dv}")
    n = max(1, min(NSG, dv // per) // max(1, split))
    while (dv // per) % n:
        n -= 1
    return n


# VALUE_SPLIT = "auto" (op.split_for): the split for a state pass of chunk length c over a call's head-rows (B x H), measured on
# the M3 Max (docs/series_layer.md, D3, bench/layer_logs/d3_bench*.json): a pass at chunk 16 gains from 2 threadgroups a head-row
# only while the head-rows leave most cores idle (16 head-rows: 1.10-1.14x; 32: 0.96-0.99x; 64: 0.94-0.97x); the chunk-8 forward,
# with half the products a chunk, gains at up to 64 (32 head-rows: 1.10x over chunk 8 unsplit; 64: 1.05x). 4 and 8 threadgroups a
# head-row measured slower than 2 at every shape. ((chunk length, max head-rows, split), ...): the first match wins; else 1
AUTO_SPLIT = ((8, 64, 2), (16, 16, 2))


def auto_split(rows: int, T: int, c: int = C) -> int:
    for chunk, max_rows, s in AUTO_SPLIT:
        if c == chunk and rows <= max_rows:
            return s
    return 1


@functools.lru_cache(maxsize=None)
def _library(dk: int, c: int, ncb: int, stage: int, hwload: int, nsgb: int, kgrp: int, dbuf: int = 0, nsgt: int = NSG):
    if dk % 32 != 0:
        raise ValueError(f"the key head width must be a multiple of 32, got {dk}")
    if c % 8 != 0 or not 8 <= c <= 32:
        raise ValueError(f"the chunk length must be 8, 16, 24 or 32, got {c}")
    src = _SOURCE
    for key, val in (("__DK__", dk), ("__C__", c), ("__NCB__", ncb), ("__STAGE__", stage),
                     ("__HWLOAD__", hwload), ("__NSGB__", nsgb), ("__KGRP__", kgrp), ("__DBUF__", dbuf), ("__NSGT__", nsgt)):
        src = src.replace(key, str(val))
    return torch.mps.compile_shader(src)
