# Series phase 3: the chunked kernels (D1 forward, D2 backward), a design log

Phase 3 of the kernel series: the gated delta rule's chunked (WY / UT) algorithm as Metal kernels on 8 x 8 simdgroup matrices,
float32 throughout, on one machine (an M3 Max, 40-core GPU, 128 GB, macOS 26, PyTorch 2.15). The algorithm and its equations
are in [prior_art_gated_delta.md](prior_art_gated_delta.md) (s.1.4-1.5; checked in float64 by `tools/check_chunk_equations.py`);
the kernels are in `src/macai/gated_delta/_chunked.py`. Every design tried is logged below with its prediction, its measured
times and its verdict, failed ones included. Times are Metal-event medians on the GPU. "Back-to-back" means several calls between
two events, which removes the GPU's idle-to-busy ramp after each synchronization (`macai.instruments.gpu_ms(inner=)`); for
kernels under 1 ms the single-call figure overstates by up to 3x. The training shapes are Qwen3.5-0.8B (4 rows x 1,024 tokens x
16 heads) and Qwen3.5-9B (2 x 1,024 x 32 heads), key and value head widths 128: 65,536 token-heads either way.

## The registered predictions and where they stand

Registered before any code (series journal, 2026-11-02 19:3x PST). Baselines: the 0.2.0 kernels (D0), profiled in phase 1.

| prediction | registered | measured now (0.8B / 9B) | verdict |
|---|---|---|---|
| D1, the forward with its states stored | 2.5-4.0 ms, and >= 1.2x D0's 4.6 ms, else not adopted | 3.20 / 3.26 ms; D0 4.50 / 4.59 ms: 1.41x | met |
| D2, the backward | 6-10 ms, and >= 2x D0's 26.6 ms | 4.55 / 4.57 ms; D0 25.9 ms: 5.7x | met, faster than the range |
| the rule's forward + backward through the drop-in | 15-25 ms (0.2.0: about 40; the reference about 66) | 15.6 / 15.6 ms; 0.2.0 39.3 / 38.4; reference 68.3 / 69.0 | met |
| a whole linear-attention layer, forward + backward | >= 1.4x the reference (0.2.0: 1.22-1.23x) | 0.8B 1.72x (116.8 -> 67.8 ms); 9B 1.37-1.40x across runs (251.3-258.4 -> 184.0-184.8 ms) | met at 0.8B; at the bar at 9B |
| parity | the phase 2 harness at its bars (2e-4, stress 5e-4, 2,048 tokens 1e-3) | 192 tests pass; worst relative error of the chunked pairs 1.1e-6 to 6.9e-6 | met |

The defaults are `op.KERNEL = "chunked"`, `op.KERNEL_BWD = "chunked"` (the fastest pair that passes the whole harness).

## The design as built

Per chunk of C tokens (C = 16 by measurement, below), with S_c the state entering the chunk:

```
G_r = sum_{s<=r} g_s   gamma_r = exp(G_r)   e_r = exp(G_last - G_r)          (in-chunk log decay sums; padding rows g = 0)
KG  = (K K^T) * exp(G_i - G_j) strictly lower      M = (Q K^T) * exp(G_i - G_j) lower with the diagonal
A   = (I + diag(beta) KG)^-1                       (unit lower triangular)
Xu  = V - diag(gamma) K S_c                        Vn = A diag(beta) Xu
O   = diag(gamma) Q S_c + M Vn                     S_c+1 = exp(G_last) S_c + K^T diag(e) Vn
```

| kernel | one threadgroup owns | what it does | stores |
|---|---|---|---|
| `gdc_prep` | one (chunk, head, batch row); 4 simdgroups | G by one prefix sum (a lane a row); the lower 8 x 8 blocks of K K^T and Q K^T; decays and masks on each lane's two elements; A by substitution on the 8 x 8 diagonal blocks (a lane a (block, column)), then the blocks below, one block distance at a time: A_ij = -A_ii sum_m L_im A_mj | G, A, M, KG |
| `gdc_fwd` | one head (up to 16 simdgroups, 8 value columns each, S's columns in registers) | per chunk: K and Q copied once into threadgroup memory as 8 x 8 tiles; K S, Q S, Xu, Vn = A (beta Xu), O, the state update | when training: S_c per chunk, Vn, Xu |
| `gdc_bwd_state` | one head, as above, dS's columns in registers | chunks in reverse: dVn = M^T dO + diag(e) K dS_c+1; dS_c = exp(G_last) dS_c+1 + Q^T diag(gamma) dO - K^T diag(beta gamma) A^T dVn | dS_c+1 per chunk, dVn |
| `gdc_bwd_chunk` | one (chunk, head, batch row); 8 simdgroups | dM = exp(G_i - G_j)(dO Vn^T); dA = (dVn Xu^T) diag(beta); dL = -A^T dA A^T; P = dL * exp(G_i - G_j); dQ, dK (one 8-column key block a simdgroup at a time), dV, dbeta, dg | the gradients |

- Every gradient element has exactly one writer: no atomics; the backward is bitwise identical run to run (a harness test).
- Decays are only exp of in-chunk log-sum differences, masked before the exponent; nothing is clamped.
- dg is written so that no term cancels another exactly: M's share of dG comes from the strictly lower part of
  E = (dO Vn^T) * M (row sums minus column sums), and the key decays e_r enter dg_t as the prefix sum
  sum_{r < t} e_r k_r . (Vn dS_c+1^T)_r. Under strong decay every dg is tiny, and the direct form (q . dQ - k . dK) loses it
  to cancellation of O(1) terms (found by the harness, entry 1).
- Any forward pairs with any backward: after a token-at-a-time forward the chunked backward reads its 8-token checkpoints
  (every C / 8-th is a chunk state) and recomputes the per-chunk matrices, Vn and Xu, every chunk in parallel; the chunked
  forward can store the 8-token states for the token-at-a-time backwards.
- The chunked backward takes key heads of 256 (the token-at-a-time backwards refuse them); packed rows stay refused.

## The log

### 1. 2026-11-02 19:4x PST: D1.0 + D2.0, the first build (commit 45d46d6)
Design: C = 32. `gdc_prep` as above. The state passes one threadgroup a (32 value columns, head, batch row): 4 simdgroups x 8
columns, every K and Q tile read from device memory by each simdgroup (per-lane loads, a row guard on each). `gdc_bwd_chunk` with
4 simdgroups, each owning 32 key columns (4 blocks, one at a time).

Parity: every pairing matched the float64 reference at 1e-7 to 2e-6 on the first run. The harness found one fault: at g = -20
the relative error of dg was 37 (the cancellation above); rewritten, it passed. 192 tests; worst relative errors 4e-7 to 2.6e-6.

Measured (single calls; 0.8B / 9B):

| part | ms |
|---|---|
| `gdc_prep` | 0.40 / 0.87 (0.28 back-to-back: the 0.87 is the clock ramp) |
| `gdc_fwd` without / with states | 4.32 / 4.31; 5.22 / 5.22 |
| forward with states | 5.53 / 5.51 |
| `gdc_bwd_state`; `gdc_bwd_chunk` | 4.21 / 4.20; 4.93 / 4.94 |
| backward | 9.07 / 9.02 |

Verdict: D1.0 missed (5.5 ms against 2.5-4.0, and slower than D0's 4.6); D2.0 within its range (9.0 in 6-10).

### 2. 20:0x: what limits the state passes (experiments on D1.0; some variants compute wrong values on purpose)

| experiment (forward state pass, 0.8B, no states stored) | ms |
|---|---|
| as built | 4.29 |
| every chunk's loads from chunk 0 (cache-hot) | 3.46 |
| the transposed K loads of the state update made plain loads | 4.18 |
| whole tiles through `simdgroup_load` from device memory | 3.81 |
| no row guards on the loads | 3.64 |
| no Q S | 3.51 |
| 16 value columns a simdgroup (4 / 2 simdgroups a threadgroup) | 6.86 / 6.10 |
| 32 value columns a simdgroup | 13.0 |

| heads (state-pass threadgroups x 4) | 1 | 4 | 10 | 20 | 40 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|
| forward state pass, ms | 1.58 | 1.59 | 1.68 | 1.93 | 2.67 | 4.57 | 8.49 | 16.3 |

The chain: one head alone takes 1.58 ms, 49 us a chunk for about 212 products, so a simdgroup's chunk is a chain of about 280
device loads issued one after another (each behind a guard branch, each followed by the product that consumes it) -> nothing
overlaps the load latency -> the time is flat until the GPU is full, then linear from 64 heads -> the training shapes sit at
the knee, with no single dominant cost. More columns a simdgroup only adds register pressure.

### 3. 20:0x-20:2x: D1.1, the chunk's K and Q staged in threadgroup memory, one threadgroup a head (commit 1683874)
Design: every thread of the threadgroup copies part of the chunk's K and Q with 16-byte loads into threadgroup memory, stored
as 8 x 8 tiles (each tile 64 contiguous floats: conflict-free `simdgroup_load` without padding; 2 C DK floats, 32 KB at C = 32);
products read tiles from there. Prediction: one bulk latency a chunk instead of ~280 serial ones.

| state passes, C = 32, single calls, 0.8B | forward with states | backward state pass |
|---|---|---|
| D1.0 (unstaged, 4 simdgroups) | 5.53 | 4.21 |
| staged, 4 simdgroups (32 columns a threadgroup) | 5.54 | 3.91 |
| staged, 8 simdgroups (64 columns) | 3.42 | 2.29 |
| staged, 16 simdgroups (one head) | 3.24 | 2.20 |

Staging pays only when the staged chunk serves many simdgroups: with 4 it gained nothing. With one threadgroup a head, one head
alone takes 1.10 ms and two heads on one core take about twice that, so a core is saturated by one threadgroup's instruction
stream (about 29% of the core's matrix peak), and 64 heads on 40 cores run in about two rounds. Verdict: adopted; D1 met its
prediction from here on.

### 4. 20:0x-20:2x: D2.1, the per-chunk backward reorganized (commit 1683874)

| `gdc_bwd_chunk`, C = 32, 0.8B | ms |
|---|---|
| D2.0 (4 simdgroups, 4 key blocks each in order), whole tiles through `simdgroup_load` | 5.50 |
| the same, per-lane loads | 5.02 |
| 4 simdgroups, key blocks interleaved over simdgroups | 4.32 |
| 8 simdgroups, interleaved, one key block at a time | 3.65 |
| 16 simdgroups | 3.65 |
| 8 simdgroups, two key blocks at once (24 accumulator tiles) | 9.50 |
| 8 simdgroups, whole tiles through `simdgroup_load` | 4.03 |

Limiter experiments on D2.0: cache-hot loads 4.73 ms (memory about 14%); without its value-contracting products 2.98 (they cost
about 2.5 ms); without the C x C phases 4.31. Verdict: 8 simdgroups, one block at a time, per-lane loads adopted.

### 5. 20:1x: the chunk length (commit b8c7afa)
Back-to-back times; forward = `gdc_prep` + `gdc_fwd`, backward = `gdc_bwd_state` + `gdc_bwd_chunk`; 0.8B / 9B, ms.

| C | forward with states | forward without states | backward | chunk states stored a layer call |
|---|---|---|---|---|
| 8 | 3.29 / 3.31 | 2.17 / 2.18 | 6.45 / 6.17 | 537 MB |
| 16 | 3.10 / 3.22 | 2.40 / 2.49 | 4.48 / 4.55 | 268 MB |
| 24 | 3.30 / 3.14 | 2.60 / 2.48 | 5.59 / 5.54 | 179 MB |
| 32 | 3.18 / 3.25 | 2.59 / 2.64 | 6.01 / 5.98 | 134 MB |

The prior art expected 32-64 from the bytes (fewer stored states). Measured, 16 wins the backward by 25% while storing twice the
states: a token's C x C work halves, and the per-chunk backward's threadgroup memory drops from 24.6 KB to 7.2 KB, so more of
them run on a core. 64 does not fit (a chunk's staged K and Q alone would be 64 KB). Chunk 8 has the fastest forward without
states (inference), not adopted here. At C = 16, the simdgroup counts were re-checked (per-chunk backward 4 / 8 / 16: 2.82 /
2.53-2.67 / 2.66-2.79 ms; state passes 8 instead of 16 simdgroups: slower; 16 columns a simdgroup: forward with states 4.6-4.75).
Verdict: C = 16 adopted.

### 6. 20:1x: the profile and the benchmark at C = 16 (profile_gated_delta_20261102_201216, gated_delta_20261102_201233)

Pipeline limits (no kernel is flagged for register pressure: all keep 1,024 threads a threadgroup):

| kernel | max threads a threadgroup | threadgroup memory | launched with |
|---|---|---|---|
| `gdc_prep` | 1,024 | 2,048 B | 128 threads, one (chunk, head, row) |
| `gdc_fwd` | 1,024 | 16,384 B | 512 threads, one (head, row) |
| `gdc_bwd_state` | 1,024 | 16,384 B | 512 threads, one (head, row) |
| `gdc_bwd_chunk` | 1,024 | 7,200 B | 256 threads, one (chunk, head, row) |

Rooflines (back-to-back; FLOPs = the 8 x 8 x 8 products each kernel runs; bytes = every input read once and every output written
once; ceilings 13.78 TFLOP/s and 356 GB/s; 0.8B, the 9B shape within 3%):

| kernel | ms | GFLOP | TFLOP/s (of peak) | MB | GB/s (of ceiling) |
|---|---|---|---|---|---|
| `gdc_prep` | 0.33 | 0.41 | 1.26 (9%) | 80 | 247 (69%) |
| `gdc_fwd`, no states | 2.20 | 6.85 | 3.11 (23%) | 143 | 65 (18%) |
| `gdc_fwd`, states | 3.00 | 6.85 | 2.28 (17%) | 479 | 160 (45%) |
| `gdc_bwd_state` | 1.98 | 6.85 | 3.45 (25%) | 412 | 208 (58%) |
| `gdc_bwd_chunk` | 2.68 | 8.09 | 3.02 (22%) | 852 | 319 (89%) |
| forward with states | 3.20 | 7.26 | 2.27 (16%) | 559 | 175 (49%) |
| backward | 4.55 | 14.93 | 3.28 (24%) | 1,264 | 278 (78%) |

| 0.8B / 9B, ms | reference | D0 (0.2.0) | chunked |
|---|---|---|---|
| the rule (`GatedDeltaRule`, float32 prepared inputs) forward | | 4.20 / 3.96 | 2.49 / 2.51 |
| the rule forward + backward | | 31.3 / 31.3 | 9.04 / 8.70 |
| the drop-in (bf16 inputs, l2 norm in the call) forward + backward | 68.3 / 69.0 | 39.3 / 38.4 | 15.6 / 15.6 |
| the layer (bench, wall clock) forward + backward | 116.8 / 258.4 | 92.4 / 208.0 | 67.8 / 184.8 |
| the layer, against the reference | | 1.26x / 1.24x | 1.72x / 1.40x |

Where the time goes now: the per-chunk backward runs at 89% of the bandwidth ceiling (its 852 MB are dominated by the chunk
states and their gradients, 2 x 268 MB); the state passes are bound by one head's chain of chunks (two rounds of 64 heads on 40
cores); the drop-in adds about 6.6 ms around the 9 ms of kernels (the casts, the l2 normalization and its backward as separate
PyTorch operations).
