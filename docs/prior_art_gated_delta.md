# Prior art: the chunked gated delta rule, and kernels for Apple GPUs

Phase 0 of the kernel series: P0.1 (the algorithm) and P0.2 (kernels and facts for Apple GPUs), read on 2026-11-02 from
primary sources only: the two papers, the projects' own source code, Apple's own documents. Every statement carries its source.
Statements marked **[I]** are this document's inference (derivations, arithmetic, design proposals), not a source's claim.

| source | version read | what was read |
|---|---|---|
| S. Yang, B. Wang, Y. Zhang, Y. Shen, Y. Kim, "Parallelizing Linear Transformers with the Delta Rule over Sequence Length", arXiv 2406.06484 | v6, 2025-01-15 | §2.1-3.3, App. B, App. D |
| S. Yang, J. Kautz, A. Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule", arXiv 2412.06464 | v3, 2025-03-06 (ICLR 2025) | §2.1, §3.1, §3.3 |
| flash-linear-attention, github fla-org/flash-linear-attention | main `cf9f520` (2026-11-03) | `fla/ops/gated_delta_rule/*`, `fla/ops/common/{chunk_delta_h,chunk_o,chunk_scaled_dot_kkt}.py`, `fla/ops/utils/{solve_tril,cumsum,index,constant}.py` |
| transformers | 5.22.1 (installed in the bench venv) | `models/qwen3_5/modeling_qwen3_5.py` |
| MLX, github ml-explore/mlx | 0.32.4 installed (its shipped headers `mlx/include/mlx/backend/metal/kernels/gated_delta_*.h` are identical to tag `v0.32.4`); main `c823b01` (2026-11-02) | `mlx/backend/metal/{gated_delta_update,device,matmul,scaled_dot_product_attention}.cpp`, `kernels/gated_delta_*.h`, `kernels/steel/{gemm,attn}/*`, `python/tests/test_fast_gated_delta.py` |
| mlx-vlm | 0.7.11 (installed) | `mlx_vlm/models/qwen3_5/{gated_delta,language}.py` |
| mlx-lm, github ml-explore/mlx-lm | main `736ab0d` | `mlx_lm/models/gated_delta.py` |
| metal-flash-attention, github philipturner/metal-flash-attention | main `8671cdd` (2024-09-22) | `README.md`, `Sources/FlashAttention/{GEMM,Attention}/**` |
| Apple, Metal Feature Set Tables | edition dated 2026-10-23 | pp. 2, 4, 5, 7 |
| Apple tech talks (transcripts) | 111375 "Explore GPU advancements in M3 and A17 Pro"; 111374 "Discover new Metal profiling tools for M3 and A17 Pro"; 111431 "Boost your graphics performance with the M5 and A19 GPUs"; 111432 "Accelerate your machine learning workloads with the M5 and A19 GPUs" | developer.apple.com/videos/play/tech-talks/<id> |
| this repo | `bench/results/hw_ceilings_20261102_183256.json`, `bench/results/profile_gated_delta_20261102_183452.json` | the M3 Max's measured ceilings; the baseline kernels' profile |

Not read: the Metal Shading Language Specification 4.1 (2026-06-04). Its PDF is larger than the fetch tool's 10 MB limit, so the
simdgroup-matrix API in §3.6 is taken from code that compiles on this machine (MLX, metal-flash-attention, `bench/hw_ceilings.py`)
and from the Feature Set Tables.

Contents: 0 findings that change the design · 1 the algorithm · 2 the reference kernels · 3 Apple GPUs · 4 implications for the
M3 Max.

---

## 0. Findings that change the design

1. **The chunked form leaves one short sequential pass and makes the backward atomic-free.**
   The chunk split (§1.4-1.5) → per chunk only three products touch the carried state (W·S, Kᵀ·Vn, Q·S), everything C×C is
   independent across chunks → in the backward every gradient except dS has a single owner per (chunk, head, key tile)
   (flash-linear-attention's `chunk_bwd_dqkwg` writes its own [C, BK] blocks, no atomics) → the token-serial chains and the
   cross-threadgroup float atomics of the baseline disappear. At the training shapes the measured ceilings put the backward's
   floor at 3.0-4.7 ms against the baseline's measured 26.6 ms, the forward's at 1.7-2.8 ms (staged) against 3.7 ms **[I, §4.1]**.
   The backward is where the series gains; the forward gains only if its stages are fused.
2. **No Apple-GPU chunked backward exists outside M5 hardware.** MLX's only chunked backward runs on the M5's neural
   accelerators (MetalPerformancePrimitives tensor ops), gated on GPU generation ≥ 17 and macOS ≥ 26.2. On the M3 MLX trains through
   a sequential save pass that stores every step's state ([B, Hv, T, Dv, Dk] in fp32: 4.3 GB at 4 × 16 heads × 1,024 tokens) and
   a sequential backward that adds dq and dk with per-element float atomics from all 128 value columns (§3.1-3.2). Its chunk-8
   simdgroup forward is inference-only. The closest structural template for our chunked backward is MLX's M5 one (one threadgroup
   per head, per-chunk cached KKᵀ / inverse / QKᵀ / delta, state checkpoints every 16 chunks with a cheap replay).
3. **On the M3, fp32 matrix math is free and every staged kernel is bandwidth-bound.** Measured: simdgroup-matrix
   multiply-accumulate 13.78 TFLOP/s in fp32, 13.64 in fp16, 11.96 in bf16; copy bandwidth 356 GB/s; balance ≈ 39 FLOP/byte. The
   flash-linear-attention stages at C ≤ 64 have arithmetic intensities of 4-34 FLOP/byte (§4.1) → each one run as a separate kernel
   is bandwidth-bound → fuse the stages that hand C×D intermediates to each other (W, U, Vn, dW), keep C×C matrices on chip, keep
   fp32 operands (precision costs bandwidth only).
4. **The M3 favors small per-simdgroup register tiles, little threadgroup memory, and more than one threadgroup per head.**
   Dynamic caching allocates registers and threadgroup memory from L1 at run time, and the occupancy manager lowers occupancy
   instead of spilling; heavy threadgroup memory stalls thread launches (Apple talks 111375, 111374). The limit is 32 KB of
   threadgroup memory per threadgroup (Feature Set Tables p.7): a C = 64 fp32 [64 × 128] key tile alone fills it. On Apple9
   metal-flash-attention computes a 32 × 32 GEMM tile per single simdgroup with K-step 8 and direct device loads, because its
   async copies slow Apple9 and later (`GEMMKernelDescriptor.swift`), and it uses row blocks of 16 for attention. Float atomics are
   emulated (metal-flash-attention README) and collapse under contention (measured 155 → 3 G ops/s from 1 to 1,024 threads an
   address). The training shapes have B·Hv = 64 heads for 40 cores: one threadgroup per head leaves a 1.6-wave tail **[I]**.
   → Design: C = 32 as the fp32 fit, BV = 16-32 value columns per simdgroup, ≥ 128 threadgroups.
5. **Numerics: decay only as differences of in-chunk log sums; invert by substitution, not by repeated squaring.** The paper writes
   decay ratios as γ_i/γ_j; every implementation computes exp(G_i − G_j) of in-chunk cumulative logs and masks before the exponent
   (transformers, flash-linear-attention; MLX additionally clamps the decay at 1e-6, which the transformers reference does not, so
   parity forbids it). For the triangular inverse, flash-linear-attention uses fp32 row substitution on 16 × 16 diagonal blocks plus
   block merges by matrix products. MLX uses a log-depth product (I − N)(I + N²)(I + N⁴) at C = 8 and a 15-step fixed point at
   C = 16, on the ungated matrix with the decays applied afterwards. mlx-vlm's own source records that the repeated-squaring
   inverse fails when keys are correlated (off-diagonals near 1). Store instead of recompute: flash-linear-attention keeps only the
   inverse and recomputes w, u, the chunk states and Vn in the backward (a second sequential pass); storing the per-chunk states
   costs 67-134 MB a layer at the training shapes, a quarter or less of the baseline's 537 MB of 8-token checkpoints **[I]**.

---

## 1. The algorithm (P0.1)

### 1.1 The recurrence and notation
Per batch row and value head, with the state S of shape [DK, DV] (this repo's and transformers' orientation):

```
a_t = exp(g_t)             g_t ≤ 0 is the log decay;  β_t ∈ (0, 1) is sigmoid(b_t)
S~  = a_t S_{t-1}
d_t = β_t (v_t − S~ᵀ k_t)                        [DV]   the delta-rule correction
S_t = S~ + k_t d_tᵀ
o_t = S_tᵀ q_t                                   [DV]   q_t, k_t l2-normalized; q_t scaled by DK^-1/2
```

The papers write the transposed state ([d_v, d_k], o_t = S_t q_t): the delta rule as S_t = S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
(2406.06484 §3.1) and the gated rule as S_t = S_{t−1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ with α_t ∈ (0, 1) (2412.06464 §3.1,
Eq. 10). With α_t = a_t these are the recurrence above, transposed.

Qwen3.5's shapes (the models' configs; transformers' layer code): DK = DV = 128; 16 key heads and 16 value heads (0.8B, 2B) or 16
key heads and 32 value heads (4B, 9B). transformers repeats q and k across value heads (`repeat_interleave`) before calling the
rule, computes g = −exp(A_log)·softplus(a + dt_bias) and β = sigmoid(b), and asks for the l2 normalization inside the call
(`use_qk_l2norm_in_kernel=True`).

### 1.2 The delta rule's chunkwise form (Yang et al. 2024, arXiv 2406.06484)
- **WY representation** (§3.2, Eqs. 5-7; proofs App. B.1): inside a chunk the product of the transition matrices is written as
  P^r = I − Σ_{i≤r} w^i k^{iᵀ} and the accumulated writes as H^r = Σ_{i≤r} u^i k^{iᵀ}, so the state after r steps is
  S^r = S^0 P^r + H^r. The vectors obey w^r = β^r(k^r − Σ_{i<r} w^i (k^{iᵀ}k^r)) and u^r = β^r(v^r − Σ_{i<r} u^i (k^{iᵀ}k^r)),
  which is still a recurrence and cannot use matrix units as written.
- **UT transform** (§3.2, Eqs. 10-11; derivation App. B.2): T = (I + tril(diag(β) K Kᵀ, −1))⁻¹ diag(β), W = T K, U = T V. The paper
  names forward substitution as the efficient way to invert the lower-triangular matrix.
- **Chunk update and output** (Eqs. 8-9, paper's orientation): S_[t+1] = S_[t] + (U − W S_[t]ᵀ)ᵀ K and
  O = Q S_[t]ᵀ + (Q Kᵀ ⊙ M)(U − W S_[t]ᵀ), M the causal mask.
- **Cost** (§2.1, for chunked linear attention in general): O(LCd + Ld²) work in O(L/C) sequential steps; C is typically 64 or
  128.
- **Backward** (§3.2): the authors adapt FlashLinearAttention and recompute the hidden states in the backward to save GPU memory.
  The paper's appendix pseudocode (App. D) is the forward only.

### 1.3 The gate (Yang, Kautz, Hatamizadeh 2024, arXiv 2412.06464)
- §2.1 defines the in-chunk cumulative decay γ^r (the product of the gates from the chunk's first position), the decay matrix
  Γ_ij = γ_i/γ_j for i ≥ j (0 otherwise), and arrows: a query decayed to the chunk's first position (γ^r q^r), a key decayed to the
  chunk's last position ((γ^C/γ^r) k^r), the state decayed over the chunk (γ^C S).
- §3.3: because the gates are scalars they factor out of the transition product: the gated product is γ^r times the ungated one.
  The chunk update becomes S_[t+1] = γ^C S_[t] + (Ũ − W← S_[t]ᵀ)ᵀ K→ with
  Ũ = [I + strictLower(diag(β)(Γ ⊙ K Kᵀ))]⁻¹ diag(β) V (the gate inside the inverse) and W← = diag(γ) W, W the **ungated** WY factor.
  The paper gives no backward and no numerical discussion; it writes the decay ratios as divisions.

**[I] The two published conventions are the same matrix.** Let D = diag(γ), L_u = strictLower(diag(β) K Kᵀ) and
L = strictLower(diag(β)(Γ ⊙ K Kᵀ)) = D L_u D⁻¹. Then (I + L)⁻¹ = D (I + L_u)⁻¹ D⁻¹, so

```
W_fla = (I + L)⁻¹ diag(β ⊙ γ) K = D (I + L_u)⁻¹ D⁻¹ diag(β) D K = D (I + L_u)⁻¹ diag(β) K = W←_paper
```

flash-linear-attention and transformers invert the gated matrix once and use it for both W and U. The paper and MLX invert the
ungated matrix and apply the decays afterwards (MLX: row scale by γ for W, ⊙ exp(G_i − G_j) for U's inverse). The two forms are
equal in exact arithmetic; only the gated form never builds the undamped entries.

### 1.4 The chunked forward, stage by stage
Chunk c covers tokens t = cC … cC + C − 1; rows r = 0 … C − 1; a tail chunk has n < C valid rows (padding rows carry g = 0, β = 0,
q = k = v = 0). **[I]** The equations below are written in this repo's orientation. They are checked term by term against
`fla/ops/gated_delta_rule/naive.py`, flash-linear-attention's kernels and transformers' `torch_chunk_gated_delta_rule`.

```
Inputs of chunk c:  Q, K ∈ R^{C×DK}   V ∈ R^{C×DV}   β, g ∈ R^C   S_c ∈ R^{DK×DV} (state entering the chunk)

F1  G_r = Σ_{s≤r} g_s        γ_r = exp(G_r)        G_last = G_{n−1}        e_r = exp(G_last − G_r)
F2  Kg   = K Kᵀ                                                                     [C, C]
    L_ij = β_i · Kg_ij · exp(G_i − G_j)        for i > j;   0 otherwise              [C, C]
    A    = (I + L)⁻¹                           unit lower triangular                 [C, C]
F3  W    = A · diag(β ⊙ γ) · K                                                       [C, DK]
    U    = A · diag(β) · V                                                           [C, DV]
F4  Vn   = U − W · S_c                         row r = d_t, the per-token correction  [C, DV]
    S_{c+1} = γ_last · S_c + Kᵀ · diag(e) · Vn                                       [DK, DV]
F5  M_ij = (q_i · k_j) · exp(G_i − G_j)        for i ≥ j;   0 otherwise              [C, C]
    O    = diag(γ) · Q · S_c + M · Vn                                                [C, DV]
```

Why row r of Vn is d_t **[I]**: inside the chunk, S after r steps = γ_r S_c + Σ_{j≤r} e^{G_r−G_j} k_j d_jᵀ. Substituting into
d_r = β_r(v_r − a_r S_{r−1}ᵀ k_r) gives d_r + Σ_{j<r} β_r e^{G_r−G_j}(k_r·k_j) d_j = β_r v_r − β_r γ_r S_cᵀ k_r, that is
(I + L) D_c = diag(β) V − diag(β ⊙ γ) K S_c, so D_c = U − W S_c. Then o_r = S_rᵀ q_r gives F5.

Names in the code: transformers calls U `new_values`, W `k_cumdecay`, Vn `v_new`, M `intra_chunk_attn`, the decay matrix
`pairwise_decay`, and builds `ut_system` = diag(β)·Kg ⊙ decay with the diagonal included but ignored by the unit-triangular solve.
flash-linear-attention calls A `A` (the solved inverse), W `w`, U `u`, S_c `h[c]`, Vn `v_new`, and keeps G in base 2 (§2.1).

| stage | output | FLOPs per chunk and head (dense; a multiply-add = 2) | needs the previous chunk? |
|---|---|---|---|
| F1 | G [C] | C | no |
| F2 | Kg, L, A [C, C] | 2C²·DK + C³/3 (substitution) | no |
| F3 | W [C, DK], U [C, DV] | 2C²·DK + 2C²·DV | no |
| F4 | Vn [C, DV], S_{c+1} [DK, DV] | 4C·DK·DV | yes: the only sequential stage |
| F5 | M [C, C], O [C, DV] | 2C²·DK + 2C²·DV + 2C·DK·DV | needs S_c only |

Kg, M and the products with A are triangular; exploiting it halves those terms. flash-linear-attention computes only the 10 lower
16 × 16 blocks of K Kᵀ at C = 64.

### 1.5 The chunked backward, stage by stage
Given dO ∈ R^{C×DV} for every chunk and the gradient of the final state (zeros if unused), walk the chunks from the last to the
first carrying dSn = ∂L/∂S_{c+1}. **[I]** Derived from F1-F5 and checked term by term against flash-linear-attention's
`chunk_gated_delta_rule_bwd_dhu`, `chunk_bwd_dv_local`, `chunk_bwd_dqkwg`, `prepare_wy_repr_bwd` and the comments of MLX's
`gated_delta_vjp_fused_nax`.

```
B1  dVn = Mᵀ · dO + diag(e) · K · dSn                                              [C, DV]
B2  dSc = γ_last · dSn + (diag(γ) Q)ᵀ · dO − Wᵀ · dVn          (= ∂L/∂S_c; carry it)  [DK, DV]
B3  dM  = Γ ⊙ (dO · Vnᵀ)            Γ_ij = exp(G_i − G_j) for i ≥ j, 0 above          [C, C]
    dQ  = diag(γ) · dO · S_cᵀ + dM · K                                               [C, DK]
    dK  = dMᵀ · Q + diag(e) · Vn · dSnᵀ                        (WY terms added in B5)  [C, DK]
    dW  = − dVn · S_cᵀ                                                               [C, DK]
    dU  = dVn                                                                        [C, DV]
B4  dG_r  = q_r · dQ_r − k_r · dK_r                            (dQ, dK of B3)
    dG_last += γ_last · ⟨S_c, dSn⟩_F + Σ_r k_r · (diag(e) · Vn · dSnᵀ)_r
B5  X = Aᵀ · dW        Y = Aᵀ · dU                                                   [C, DK], [C, DV]
    dK += diag(β ⊙ γ) · X          dβ_r += γ_r (k_r · X_r)          dG_r += β_r γ_r (k_r · X_r)
    dV  = diag(β) · Y              dβ_r += v_r · Y_r
    dA  = strictLower( dW · (diag(β ⊙ γ) K)ᵀ + dU · (diag(β) V)ᵀ )                   [C, C]
    dL  = strictLower( − Aᵀ · dA · Aᵀ )
    P   = dL ⊙ Γ                    P_ij = dL_ij · exp(G_i − G_j),  i > j
    dK += diag(β) · P · K + Pᵀ · diag(β) · K
    dβ_r += Σ_j P_rj (k_r · k_j)
    R   = dL ⊙ L                    (= P ⊙ diag(β) Kg)
    dG_r += Σ_j R_rj − Σ_i R_ir     (row sum minus column sum)
B6  dg_t = Σ_{s ≥ r(t), s < n} dG_s                            (reverse cumsum inside the chunk)
end: dS_0 = the carried dS. With shared key heads (Hk < Hv) dQ and dK are summed over the value heads of a key head;
     dV, dβ, dg are per value head.
```

**Checked numerically.** F1-F5 and B1-B6 were implemented literally in float64 on the CPU and compared with autograd through
the token recurrence of §1.1. The case was T = 21 (two full chunks and a 5-row tail), C = 8, DK = 6, DV = 5, with an initial
state and a final-state gradient. Outputs, the final state, dq, dk, dv, dg, dβ and dS_0 all agree to ≤ 1.3e-15.

Masking dA to its strictly lower part before the two products is exact: a perturbation of L (strictly lower) moves A only in its
strictly lower part, because A's diagonal is fixed at 1. flash-linear-attention masks before and after.

| stage | FLOPs per chunk and head (dense) | where flash-linear-attention does it |
|---|---|---|
| B1 | 2C²·DV + 2C·DK·DV (+ 2C²·DK to rebuild M) | `chunk_bwd_dv_local` (Mᵀ dO) + `chunk_gated_delta_rule_bwd_dhu` (diag(e) K dSn) |
| B2 | 4C·DK·DV | `chunk_gated_delta_rule_bwd_dhu` |
| B3, B4 | 2C²·DV + 4C²·DK + 6C·DK·DV | `chunk_bwd_dqkwg` |
| B5 | 8C²·DK + 4C²·DV + 4C³ (+ 2C²·DK to rebuild Kg) | `prepare_wy_repr_bwd` |
| B6 | C | `chunk_local_cumsum(reverse=True)` |
| recompute (flash-linear-attention only) | W, U: 2C²·DK + 2C²·DV; states and Vn: 4C·DK·DV | `recompute_w_u_fwd`, `chunk_gated_delta_rule_fwd_h` |

### 1.6 Work per token, and at the training shapes
**[I]** With D = DK = DV, per token and head (dense counts; the recurrence counted as 8D² forward, as this repo's profile does):

```
recurrence forward                  8 D²
chunked forward                     6 D² + 10 C·D + C²/3
chunked backward, stored states    12 D² + 24 C·D + 4 C²       (flash-linear-attention's split without its recompute)
chunked backward, recomputed       16 D² + 28 C·D + 4 C²       (flash-linear-attention as written)
```

| D = 128 | C = 16 | C = 32 | C = 64 |
|---|---|---|---|
| forward, FLOPs per token and head | 118,869 | 139,605 | 181,589 |
| backward, stored states | 246,784 | 299,008 | 409,600 |
| backward, recomputed | 320,512 | 380,928 | 507,904 |
| sequential steps per head at T = 1,024 | 64 | 32 | 16 |

Both training shapes (0.8B: 4 rows × 16 heads; 9B: 2 rows × 32 value heads; 1,024 tokens) have 65,536 token-heads. At C = 32 that
is 9.2 GFLOP forward and 19.6 GFLOP backward with stored states. The measured fp32 peak (§3.8) would run them in 0.66 and 1.42 ms;
§4.1 shows bandwidth sets the floor instead.

### 1.7 Numerical concerns
- **Decay ratios.** The paper writes γ_i/γ_j. A literal division gives 0/0 when both cumulative products underflow (fp32 reaches
  zero between exp(−87) and exp(−103), depending on denormal handling). All three implementations instead take exp of
  differences of the in-chunk log cumsum:
  - transformers masks the upper triangle to −inf before `exp`, so the positive differences never overflow;
  - flash-linear-attention masks with both a causal and a boundary mask, the latter so that out-of-range rows loaded as g = 0 do
    not produce exp2(0 − G) = inf and then 0·inf = NaN (`chunk_fwd.py`, the `m_tc` masks);
  - MLX clamps the linear decay at 1e-6 before taking its log, so −13.8 is the smallest per-step log decay
    (`gated_delta_update.h`, `gated_delta_update_nax.h`). mlx-vlm clips g to [1e-6, 1] for the same reason (log(0) = −inf would
    poison the ratios).
  **[I]** Decays that are ≤ 1 (exp(G_i − G_j) for i ≥ j, γ_r, e_r) can only underflow, which is benign. Never form exp(+G)
  alone. The clamp changes results for very negative g, so a kernel held to transformers' parity must not clamp.
- **The triangular inverse.** With l2-normalized keys and β ∈ (0, 1), |L_ij| ≤ 1. Three methods appear in the sources:
  - forward substitution: transformers (`torch.linalg.solve_triangular(..., unitriangular=True)`, two solves, fp32);
    flash-linear-attention (row substitution on 16 × 16 diagonal blocks in fp32, then block merges such as
    A_10 = −A_11 L_10 A_00 by matrix products); mlx-vlm (row by row);
  - MLX at C = 8: a log-depth product on the ungated strictly-lower N, T = (I − N)(I + N²)(I + N⁴), exact because N⁸ = 0;
  - MLX at C = 16: 15 iterations of T ← I − N·T, exact because N¹⁶ = 0.

  mlx-vlm's `_invert_unit_lower` docstring states that the faster log-depth repeated-squaring inverse breaks down (they say
  overflows) when N has entries of order one, which correlated keys produce, and that this is why mlx-vlm substitutes row by row.
  **[I]** For C ≥ 16 use substitution on small diagonal blocks plus product merges. The 8 × 8 product form is in production use
  in MLX and is a candidate for the diagonal blocks, but only after the harness's correlated-key stress test passes.
- **Precision.**
  - transformers: fp32 throughout.
  - flash-linear-attention: fp32 accumulation and an fp32 state in registers. Matrix-unit operands are in the input dtype (bf16
    in bf16 training), so the stored inverse A and the per-chunk states h are in the input dtype, and the new values are rounded
    to it before the Kᵀ·Vn product. The block merges of the inverse run in tf32 where the GPU supports it.
  - MLX: fp32 state and accumulators. Its neural-accelerator path treats fp32 matrix products as TF32 (`MLX_ENABLE_TF32`,
    default on), and its tests accept atol 1e-1 for that forward and 1e-2 for the chunked backward, against 1e-4 / 1e-5 for the
    simdgroup and sequential kernels (`python/tests/test_fast_gated_delta.py`).
  **[I]** On the M3, fp32 simdgroup products run at the fp16 rate (§3.8), so the chunked kernels can stay fp32 end to end and
  keep this repo's 1e-7-scale parity.
- **Tails and padding.** Padding rows with g = 0 and β = 0 leave the state untouched (decay 1, no write); the reverse cumsum and
  G_last must use the last valid row (flash-linear-attention's `last_idx = min((i_t+1)·BT, T) − 1`).

---

## 2. The reference kernels

### 2.1 flash-linear-attention: `chunk_gated_delta_rule` (`fla/ops/gated_delta_rule/chunk.py`)
Entry `chunk_gated_delta_rule` → `ChunkGatedDeltaRuleFunction` → `chunk_gated_delta_rule_fwd` / `_bwd`. Layout [B, T, H, D] with
q, k on H key heads and v, g, β on HV value heads (HV a multiple of H; key head = hv // (HV/H)). Chunk sizes 16, 32 or 64 (default 64).
G is kept in base 2: the cumsum is scaled by 1/ln 2 (`RCP_LN2`) and every decay is `exp2`, while derivatives are taken with
respect to the natural log.

**Forward**

| step | file : function | one program owns | tiling, launch | reads → writes | precision |
|---|---|---|---|---|---|
| 1 cumsum | `utils/cumsum.py : chunk_local_cumsum` (scale = 1/ln 2) | one chunk of one head | BT = C | g → G | fp32 output |
| 2 KKᵀ + inverse (C = 64, not on Intel) | `gated_delta_rule/chunk_fwd.py : chunk_gated_delta_rule_fwd_kkt_solve_kernel` | (chunk, b·hv) | BC = 16: the ten lower 16 × 16 blocks of KKᵀ held in registers; BK ∈ {32, 64}, 1-4 warps (autotuned) | k, G, β → A [B, T, HV, C] | fp32 math; diagonal blocks by row substitution; merges by dots in tf32 where supported; A stored in the input dtype |
| 2′ (C = 16, 32; Intel) | `common/chunk_scaled_dot_kkt.py : chunk_scaled_dot_kkt_fwd` + `utils/solve_tril.py : solve_tril` | (chunk, b·hv) | a 16 × 16 substitution kernel, then 32/64 merge kernels | → L (fp32) → A | A in the input dtype |
| 3 W, U | `gated_delta_rule/wy_fast.py : recompute_w_u_fwd` | (chunk, b·hv) | BK = BV = 64 loops over the K and V tiles; 2-8 warps | A, k, v, β, G → w [B, T, HV, K], u [B, T, HV, V] | u without tf32; outputs in the input dtype |
| 4 state pass | `common/chunk_delta_h.py : chunk_gated_delta_rule_fwd_h` (kernel `..._fwd_kernel_h_blockdim64`) | (value tile, n·hv); walks all chunks in order | state [K, BV] fp32 in registers as 64-row blocks (K ≤ 256); BV ∈ {32, 64}; 2-4 warps | k, w, u, G, h0 → h [B, NT, HV, K, V] (the state entering each chunk), v_new, final state | h in the input dtype; v_new rounded to it before the Kᵀ·v_new product; final state fp32 |
| 5 output | `common/chunk_o.py : chunk_fwd_o` | (value tile × chunk, b·hv) | (BK, BV, warps) ∈ {(128, 128, 8), (64, 64, 4), (32, 32, 2)} | q, k, v_new, h, G → o | fp32 accumulation |

The state pass decays the new values instead of the keys: it applies exp(G_last − G_r) to the rows of v_new, scales the state by
exp(G_last), then accumulates kᵀ·v_new. That is F4 with diag(e) moved onto Vn.

**What the forward keeps** (`ChunkGatedDeltaRuleFunction.forward`): q, k, v, the cumsummed G, β, A, the initial state, plus the
l2 reciprocal norms and raw β / gate inputs when those are fused. It does not keep w, u, h or v_new; the backward recomputes them.

**Backward** (`chunk_gated_delta_rule_bwd`)

| step | function | one program owns | computes | notes |
|---|---|---|---|---|
| 1 | `recompute_w_u_fwd` | (chunk, b·hv) | w, u from A | recompute |
| 2 | `chunk_gated_delta_rule_fwd_h` | (value tile, n·hv), in order | h, v_new again | a second sequential pass |
| 3 | `chunk_bwd_dv_local` | (chunk, b·hv) | dv_local = Mᵀ dO | rebuilds QKᵀ |
| 4 | `chunk_gated_delta_rule_bwd_dhu` (kernel `..._bwd_kernel_dhu_blockdim64`) | (value tile, n·hv), in reverse | stores dh[c] = ∂L/∂S_{c+1}; dv2 = the full dVn (B1); dh0; carries B2 | dS [K, BV] fp32 in registers; BV ∈ {32, 64}; dh stored in the input dtype |
| 5 | `chunk_bwd_dqkwg` | (key tile × chunk, b·hv); loops over all value tiles inside | dQ, dK (B3), dW (stored negated), dG partial per key tile (B4) | single writer for every block, no atomics; the dG partials [NK, …] are summed afterwards; shared key heads reduced by a view-sum over the value-head groups |
| 6 | `prepare_wy_repr_bwd` | (chunk, b·hv) | B5: dK += WY terms, dV, dβ, dG += | the C × C chain Aᵀ dA Aᵀ; rebuilds KKᵀ |
| 7 | `chunk_local_cumsum(reverse=True)` | (chunk, b·hv) | B6: dg | |

**Details that matter for a port**
- Variable-length rows (packed sequences, `cu_seqlens`): `utils/index.py : prepare_chunk_indices` lists (sequence, chunk) pairs,
  so no chunk straddles two sequences. `chunk_offsets` index each sequence's per-chunk states, and initial and final states are
  per sequence.
- Fusions offered: `use_qk_l2norm_in_kernel`; `use_gate_in_kernel` (−exp(A_log)·softplus(a + dt_bias) fused with the chunk cumsum,
  `gate.py`); `use_beta_sigmoid_in_kernel`; `state_v_first` ([V, K] state layout).
- Register pressure: the comment in `chunk_gated_delta_rule_fwd_intra` says the fused kernel's ten fp32 16 × 16 accumulators fit
  NVIDIA's register file but spill on Intel GPUs, where the unfused two-kernel path measured 2.3-3.0x faster. The fused path is
  therefore disabled there.

### 2.2 transformers' `torch_chunk_gated_delta_rule` (the parity reference)
`modeling_qwen3_5.py`, lines 287-420 in 5.22.1. It runs fp32 throughout and lays tensors out as [B, H, T, D]. Its steps:
1. optional l2 norm, then the query scale DK^-1/2;
2. pad T to a multiple of C (default 64) with zeros;
3. form vβ and kβ, the in-chunk cumsum of g, and the pairwise decay exp(G_i − G_j) with the upper triangle masked to −inf;
4. two unit-triangular solves for U and W;
5. decays applied once: Q·exp(G), K·exp(G_last − G), exp(G_last);
6. a Python loop over chunks for F4-F5;
7. drop the padding and cast back to the input dtype.

It is decorated with `use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")`, so it is the fallback a Mac gets.
This repo's profile of it (`profile_gated_delta_20261102_183452.json`, 0.8B shape, forward stage medians) is listed below. The
stages were timed separately, so they sum to more than the 11.1 ms whole forward. Its forward + backward is 66.3 ms, through
4,015 aten ops.

| stage | ms |
|---|---|
| prologue (transpose, cast, l2 norm, scale, pad) | 3.58 |
| β products, chunk cumsum, pairwise decay | 2.45 |
| KKᵀ and QKᵀ | 1.24 |
| the two triangular solves | 0.58 |
| decay rescaling | 1.08 |
| the sequential pass over chunks | 4.45 |
| epilogue | 0.70 |

---

## 3. Kernels and facts for Apple GPUs (P0.2)

### 3.1 MLX's gated delta kernels (`mx.fast.gated_delta_update`)
Dispatch: `mlx/backend/metal/gated_delta_update.cpp`. Kernels: `mlx/backend/metal/kernels/gated_delta_update{,_vjp,_nax,_nax_vjp}.h`.
The math is the same at v0.32.4 and at main; main adds per-key gates and the TF32 condition.

| kernel | used when | grid / threadgroup | what a simdgroup owns | saves for the backward |
|---|---|---|---|---|
| `gated_delta_seq` | C = 1 (T ≤ 8, per-key gates); the save pass of every non-NAX backward | (32, Dv, B·Hv) / (32, 4, 1) | one value column; each lane holds Dk/32 = 4 state rows | the state every Ckpt steps; Ckpt = 1 unless the chunk is 16 |
| `gated_delta_fused_chunk`, C = 8 | forward, T > 8, no NAX (the M3) | (32, Dv/8, B·Hv) / (32, 4, 1) | 8 value columns × all 128 key rows as 16 `simdgroup_float8x8` tiles (32 floats a lane). Each simdgroup recomputes the 8 × 8 KKᵀ, its inverse and QKᵀ, and reads q, k, v directly from device memory | nothing: inference only |
| `gated_delta_fused_nax`, C = 16 | NAX GPUs | (32, Dv/16, B·Hv) / (32, 4, 1) | 16 value columns (a 16 × 128 tile, 64 floats a lane); products through MPP `matmul2d` on 16 × 16 / 16 × 32 fragments | when saving: the state every 16 chunks; per chunk the raw KKᵀ, the pre-decay inverse, the raw QKᵀ (3 × 16 × 16, written only by the first value slice) and delta = Vn [C, Dv] |
| `gated_delta_vjp_seq` | every backward without NAX | (32, Dv, B·Hv) / (32, 4, 1) | one column; replays each Ckpt segment from its saved state, keeping the segment's states in registers | dq, dk, dg, dβ by per-element float atomics, so the Dv = 128 column simdgroups of a head add into the same dq and dk rows |
| `gated_delta_vjp_fused_nax`, C = 16 | NAX | (32, Dv/16, B·Hv) / (32, Dv/16, 1): one threadgroup per (b, hv), 8 simdgroups | a 16-column slice of S and dS; dq, dk reduced over Dv in threadgroup memory (≈ 17.5 KB); atomics only across shared key heads. Each 16-chunk segment is replayed with only S ← exp(G_last)·S + K_decᵀ·delta, which its comment puts at about a tenth of a chunk's full work | — |
| `gated_delta_dgamma_to_dg` | after the NAX backward | (B·Hv, chunks) | reverse in-chunk cumsum, then divides by the linear decay (the chain rule for MLX's linear-space g) | — |

Facts of the dispatch:
- Only Dk = Dv = 128 with (Hk, Hv) ∈ {(16, 16), (16, 32), (16, 48), (16, 64), (24, 24), (32, 32)} run on the GPU kernels. Other
  shapes and any mask fall back to plain ops.
- The state layout is [B, Hv, Dv, Dk] (transposed against transformers'), and g is the linear decay rate, not its log.
- Under autograd the forward runs twice: the primal forward saves nothing, and the backward re-runs a save pass.
- Without NAX the save pass stores every step's state, `state_cache` [B, Hv, T, Dv, Dk] fp32 (4.3 GB at 4 × 16 heads × 1,024
  tokens).
- The NAX backward's equations, written in its comments, are B1-B5 of §1.5 in MLX's ungated-inverse form:
  - dODT = (dO·deltaᵀ) ⊙ D;
  - ddelta = Mᵀ dO + K_dec dSᵀ;
  - dq = γ ⊙ (dO S) + dODT K;
  - dv = β ⊙ (T_Uᵀ ddelta);
  - dA = −Tᵀ dTinv Tᵀ;
  - dβ = rowsum(V ⊙ dV_β) + rowsum(K ⊙ dK_β) + rowsum(tril(dA) ⊙ KKᵀ);
  - dS ← γ_C dS + dOᵀ(γ ⊙ Q) − ddeltaᵀ W.

### 3.2 Why MLX's chunked form needs an M5
- `is_nax_available()` (`device.cpp`, line 1046) is true only on macOS ≥ 26.2 and GPU architecture generation ≥ 17 (≥ 18 for phone
  parts). At main the gated delta path additionally requires TF32 to be enabled or a non-fp32 input. MLX's attention code treats
  generation 15 as the M3 (`before_m3 = gen < 15`, `scaled_dot_product_attention.cpp`). The Feature Set Tables (p.2) list M3 and
  M4 as Apple9 and M5 as Apple10.
- NAX means the neural accelerators that the M5 / A19 put in each GPU core: dedicated matrix-multiply hardware, reached through
  Metal 4 tensors and the TensorOps of MetalPerformancePrimitives (talk 111432). Apple states the same TensorOps code runs from M1
  to M5, falling back to shader implementations on GPUs without the accelerators (talk 111432). Metal 4 tensors are listed from
  Apple7 (Feature Set Tables p.5).
- So "needs an M5" is MLX's dispatch choice, not an API limit. **[I]** Presumably the fallback offers nothing over MLX's own
  simdgroup kernels on older GPUs; MLX publishes no such measurement in these sources. The consequence for this machine is that
  MLX has no chunked backward on the M3, and its chunk-8 forward never runs under autograd.

### 3.3 mlx-vlm 0.7.11 and mlx-lm
- `mlx_vlm/models/qwen3_5/gated_delta.py` holds Metal sources in Python strings (`mx.fast.metal_kernel`):
  - the generic recurrent kernel: one simdgroup per value row, Dk/32 state elements a lane;
  - a packed Dk = 128 prefill kernel: 4 lanes a value row, 8 rows a simdgroup, 32 contiguous state elements a lane, sums by an
    explicit shuffle butterfly made bit-identical to `simd_sum`. Its docstring says it was synced from mlx-lm PR 1559;
  - a variant that returns intermediate states.
- In training, `language.py` passes `use_kernel=not self.training`. A training forward therefore runs `gated_delta_chunked`
  (C = 64, fp32, plain MLX ops under MLX's autograd, row-substitution inverse, a Python loop over chunks), the MLX twin of
  transformers' reference. On this machine a whole Qwen3.5 linear-attention layer through it ran at 0.96-1.15x PyTorch's
  reference path (forward + backward).
- mlx-lm main has the recurrent kernels only (generic, the packed one, an explicit-tree comparator), no chunked form.

### 3.4 metal-flash-attention on the M3 (Apple9)

| finding | where |
|---|---|
| Apple GPUs lack native fp32 atomics: `atomic<float>` is emulated. MFA therefore splits attention's backward into a dQ kernel and a dK/dV kernel (7 GEMMs instead of 5), each with a single writer, giving full parallelism along both dimensions of the attention matrix | README; `AttentionKernelType.swift` |
| Register pressure: at large head dimensions not even the accumulator fits, so blocks run along D too, spills are intentional and arranged to cost little bandwidth, and block aspect ratios are skewed (16-32 along the parallel dimension, 80-128 along the traversal) | README |
| Apple9 GEMM: block 32 × 32 × 8 (48 × 48 × 24-32 on Apple7/8); one simdgroup per threadgroup (splits 1 × 1; 2 × 2 on Apple7/8) | `GEMMKernelDescriptor.swift` (blockDimensions, splits); `GEMMDescriptor.swift` (the Apple9 branch) |
| Async copies help Apple7/8 and harm Apple9 and later; on Apple9 they are kept only where needed for edge correctness | `GEMMKernelDescriptor.swift` (preferAsyncLoad); `AttentionDescriptor.swift` (Apple9: async for caching, not for loads) |
| Apple9 casts bf16 to fp32 natively inside the FMA, so bf16 operands stay bf16 in registers there (fp32 on M1) | `AttentionDescriptor+Precisions.swift` (hasNativeBF16Casting) |
| Attention block tables per family: Apple9 uses parallelization blocks of 16 rows (64 threads, 2 simdgroups), traversal 32-128, head blocks 8-32. At D = 128 in fp32: forward 16/64/16 caching O; dQ 16/64/32 caching Q, dQ; dK/dV 16/128/16 caching dV, dK | `AttentionDescriptor+Parameters.swift`; threadgroup size = 32·(rows/8) in `AttentionKernel.swift` |
| Occupancy proxy: among candidate pipelines, keep the ones with the highest `maxTotalThreadsPerThreadgroup` (it drops when registers per thread rise) | `GEMMDescriptor+PipelineCache.swift` |
| Reported utilization on the M3 architecture (fp16): forward 94 / 91 / 82 % and forward + backward 71 / 69 / 61 % of ALU at D = 64 / 128 / 256; the README notes its model breaks down on the M3 at small head dimensions because different units issue in parallel | README |
| The 8 × 8 simdgroup-matrix lane layout: lane ℓ holds two adjacent elements of one row, row = 4·⌊ℓ/16⌋ + ⌊ℓ/2⌋ mod 4, column = 4·(⌊ℓ/8⌋ mod 2) + 2·(ℓ mod 2) (Morton order) | `GEMMHeaders.swift` (`morton_order`); the same mapping in MLX's `gated_delta_update.h` (fm, fn) |

### 3.5 MLX's GEMM and attention tiling (main)
- GEMM (`matmul.cpp`, `GEMM_TPARAM_MACRO`): MLX's device classes are 'g' base/pro, 's' Max, 'd' Ultra (`device.cpp`). For 's'
  (the "medium device" branch) the tile is BM = BN = 64, BK = 16 with 2 × 2 simdgroups, each owning 32 × 32 of the output as 4 × 4
  interleaved 8 × 8 fragments (`steel/gemm/mma.h`: TM = BM/(8·WM)). A and B tiles are staged in threadgroup memory with rows padded
  by 16 bytes against bank conflicts (`steel/gemm/gemm.h`: `tgp_padding = 16/sizeof(T)`).
- Attention forward (`sdpa_full_self_attention_metal`): BQ = 32 rows over 4 simdgroups (8 rows each), BK = 16 keys a step at
  D = 128, Q and K/V tiles in threadgroup memory with the same 16-byte padding. The NAX variant uses BQ = 64, BK = 32. The split-key
  decode path targets 160 threadgroups (80 on base GPUs).

### 3.6 simdgroup_matrix and the M3's limits
- Availability: SIMD-scoped matrix multiply from Apple7 (Feature Set Tables p.4); the M3 is Apple9 (p.2).
- Types used in code that compiles here:
  - `simdgroup_matrix<T, 8, 8>` for T = float, half, bfloat (`bench/hw_ceilings.py`);
  - the aliases `simdgroup_float8x8` etc. (MLX).
- Calls used (MLX `gated_delta_update.h`):
  - `make_filled_simdgroup_matrix<float, 8>(v)`;
  - `simdgroup_load(m, ptr, elements_per_row, ulong2 origin, bool transpose)`;
  - `simdgroup_store(m, ptr, elements_per_row, origin, transpose)`;
  - `simdgroup_multiply(d, a, b)`;
  - `simdgroup_multiply_accumulate(d, a, b, c)`.
- A thread's two elements are reached with `thread_elements()` in the layout of §3.4. MLX builds its masks, row scalings and the
  identity directly on those elements, without memory traffic.
- Whether these calls must be made in simdgroup-uniform control flow is a specification question this document could not
  check. MLX and metal-flash-attention only ever call them uniformly across the simdgroup.
- Limits (Feature Set Tables p.7, Apple4-Apple11):
  - at most 1,024 threads per threadgroup (the real maximum is `maxTotalThreadsPerThreadgroup` of the pipeline);
  - at most 32 KB of threadgroup memory per threadgroup, allocated in 16-byte multiples;
  - 31 buffer arguments; `setBytes` up to 4 KB.
- Feature rows (p.5): floating-point atomics from Apple7 (the API exists; MFA reports it emulated); Metal 4 tensors and
  machine-learning encoding from Apple7.

### 3.7 Dynamic caching and occupancy on the M3 family
- **Before Apple9.** A simdgroup could not start until it had reserved registers equal to its program's peak register use, held
  for its whole life (talk 111375).
- **Apple9.**
  - Registers are allocated and freed during execution according to what each part of the program uses. The register file has
    become a cache.
  - Threadgroup, tile, stack and buffer data are cached on chip in the same way, in fewer, larger on-chip memories that serve all
    these types; a kernel that uses no threadgroup memory leaves that space to the others (111375).
  - These memories share the L1 cache, backed by the GPU's last-level cache and device memory (111374).
- **Occupancy.**
  - It is the number of simdgroups resident on a core.
  - Peak register use no longer dictates it. When the on-chip working set would exceed L1, the core's occupancy manager lowers
    occupancy instead of spilling (111375, 111374). Occupancy is thus set by how a kernel uses threadgroup, tile, stack and buffer
    memory as well as its dynamic register use.
  - A compute dispatch with large threadgroup memory stalls new thread launches.
  - Xcode exposes Occupancy, the Occupancy Manager Target, L1 eviction rate and L1 residency, plus limiters for the shader launch,
    last-level cache and MMU (111374).
- **ALU.** Apple9 cores issue FP16, FP32 and integer instructions in parallel to a greater degree than before, up to 2x prior
  ALU throughput, which needs instructions from several simdgroups, that is, occupancy (111375).
- **M5.** It adds a second generation of dynamic caching with faster register access and an occupancy unit that watches more
  signals (111431). These are not on this machine.

### 3.8 The M3 Max measured (this repo)
From `bench/results/hw_ceilings_20261102_183256.json`, micro-kernels through `torch.mps.compile_shader`:

| ceiling | measured |
|---|---|
| simdgroup MMA, fp32 / fp16 / bf16 (8 × 8 tiles in registers, 4 independent accumulators a simdgroup) | 13.78 / 13.64 / 11.96 TFLOP/s |
| device copy bandwidth (1 GiB float4) | 356 GB/s → balance 13.78 T / 356 G ≈ 39 FLOP/byte **[I]** |
| threadgroup barrier pair, 1,024 threads | 179 ns |
| dependent shuffle-add / `simd_sum` step | 45 / 75 ns |
| float atomic add: 1 / 4 / 32 / 1,024 threads an address | 155 / 133 / 12.8 / 3.0 G ops/s |

The baseline kernels at the 0.8B shape, from `profile_gated_delta_20261102_183452.json`:

| kernel | time | achieved |
|---|---|---|
| `gdr_fwd8` (one threadgroup a head), without / with saving | 3.70 / 4.63 ms | 2.3 TFLOP/s, 17 % of peak |
| `gdr_bwd` (the column backward) | 26.6 ms | 1.1 TFLOP/s, 8 % of peak |

The baseline stores the state every 8 tokens: [B, H, T/8, DK, DV] fp32, 537 MB a layer call at the training shapes.

---

## 4. Implications for a Metal design on the M3 Max
Everything in this section is **[I]** unless a source is named. Here D1 is the series' chunked forward and D2 its chunked
backward.

### 4.1 Roofline of the stage split at the training shapes
Per chunk and head, all fp32: FLOPs (dense) and device bytes, if every stage of §2.1 is its own kernel with its intermediates in
device memory. Times are lower bounds at the measured ceilings for the 65,536 token-heads of either training shape. Arithmetic
intensity is in FLOP/byte; anything under ≈ 39 is bandwidth-bound.

| stage | C = 32: MFLOP / KB / intensity | C = 64: MFLOP / KB / intensity |
|---|---|---|
| KKᵀ + inverse → A | 0.27 / 20 / 13 | 1.14 / 49 / 23 |
| W, U | 0.52 / 68 / 8 | 2.10 / 145 / 14 |
| state pass (h, Vn) | 2.10 / 128 / 16 | 4.19 / 192 / 21 |
| output | 1.57 / 128 / 12 | 4.19 / 192 / 21 |
| local dVn (Mᵀ dO) | 0.52 / 64 / 8 | 2.10 / 128 / 16 |
| dS pass (dh, dVn) | 3.15 / 160 / 19 | 6.29 / 256 / 24 |
| dQ, dK, dW, dG | 3.93 / 256 / 15 | 9.44 / 384 / 24 |
| WY backward | 1.97 / 116 / 17 | 8.39 / 240 / 34 |

| totals (65,536 token-heads) | C = 16 | C = 32 | C = 64 |
|---|---|---|---|
| forward, staged: compute / bytes bound | 0.57 / 2.76 ms | 0.66 / 2.03 ms | 0.86 / 1.70 ms |
| backward without recompute: compute / bytes bound | 1.17 / 4.63 ms | 1.42 / 3.51 ms | 1.95 / 2.97 ms |
| backward with flash-linear-attention's recompute | 1.52 / 6.15 ms | 1.81 / 4.67 ms | 2.42 / 3.96 ms |
| baseline (measured): forward / backward | 3.70 / 26.6 ms | | |

The chain:
- every staged kernel is bandwidth-bound, so a staged design is bounded at 1.7-2.8 ms forward and 3-4.7 ms backward;
- the backward bound is 6-9x under the baseline's 26.6 ms, so the backward is the prize, while the forward bound is under 2x
  better than the baseline's 3.7 ms, so the forward only pays if fused;
- smaller C cuts FLOPs but multiplies the per-chunk state traffic (64 KB each chunk-head), which is why C = 16 has the worst
  bytes. C = 32 and 64 are the candidates for the size sweep.

### 4.2 The forward (D1): one fused kernel per (head, value slice)
- **Structure.** One threadgroup per (b, hv, value half or quarter), not per head. 64 heads over 40 cores is 1.6 waves, so
  splitting the value dimension into 2-4 threadgroups gives 128-256 threadgroups. MLX makes the same choice with 2-4 threadgroups
  a head. The price is that each threadgroup redoes the head's C × C work: KKᵀ, QKᵀ, A.
- **Each simdgroup** owns BV = 16 (or 32) value columns × all 128 key rows of S in registers as 8 × 8 tiles: 64 (or 128) floats a
  lane. This is MLX's and flash-linear-attention's ownership: the columns of S are independent in F4-F5, so no simdgroup needs
  another's state.
- **Per chunk (C = 32)**, on chip:
  1. G by `simd_prefix_inclusive_sum` (one lane a row, as MLX does), β.
  2. Kg and M (32 × 32 each) split across the threadgroup's simdgroups, from q and k tiles loaded directly from device memory,
     per metal-flash-attention's Apple9 rule. The decay and mask are applied on `thread_elements()`, then the results are written
     to threadgroup memory (4 KB each).
  3. A = (I + L)⁻¹: four 8 × 8 diagonal blocks in parallel (substitution, or MLX's 4-product form once validated), then the block
     merges (fla's A_10 = −A_11 L_10 A_00 pattern) into threadgroup memory (4 KB). A barrier.
  4. Per simdgroup:
     - P = (diag(β ⊙ γ) K)·S_slice [C × BV], contracting over DK;
     - W·S_slice = A·P, which never forms W [C × DK];
     - U_slice = A·(diag(β) V_slice);
     - Vn = U − A·P;
     - O_slice = diag(γ) Q S_slice + M·Vn;
     - then S ← γ_last S + (diag(e) K)ᵀ Vn.
     Rewriting W·S as A·(K·S) cuts the per-slice C × DK work to the slice's width.
  5. When training, store S_c's slice (the per-chunk state, 64 KB a chunk-head) and Vn's slice. Do not run the forward twice
     (MLX does).
- **Threadgroup memory**: about 12 KB (A, Kg or M, G, β), under 32 KB. No staged K: on Apple9 direct device loads beat staging
  (§3.4), and large threadgroup memory throttles launches (§3.7).
- **Predicted limiter**: the per-chunk dependent chain in each simdgroup (about 480 8 × 8 products at BV = 16, plus the load
  latency of the q/k/v tiles that every simdgroup of a head re-reads from L1) and the 2-3 barriers a chunk around the shared
  C × C phase.
- **Forecast**: about 10-12.5 GFLOP (including each threadgroup's C × C rework) and 0.3-0.36 GB of traffic when training (the
  per-chunk states dominate). That is a floor near 1 ms from either limit, and 1.5-3 ms at 30-50 % efficiency, against 3.7 ms. A
  modest forward gain, consistent with §4.1.

### 4.3 The backward (D2): one reverse pass plus parallel per-chunk work, no atomics
- **D2a, the dS pass.** Per (b, hv, value slice), walking chunks in reverse with dS [128 × BV] in registers:
  - B1: dVn_slice = Mᵀ dO_slice + diag(e) K dSn_slice;
  - B2: dS ← γ_last dS + (diag(γ) Q)ᵀ dO_slice − Wᵀ dVn_slice.

  Columns stay independent here too (flash-linear-attention's `bwd_dhu` grid), so there are no cross-slice reductions. It needs
  M and W per chunk: store them in the forward (M and A are 8-17 MB a layer; W is 34 MB) or rebuild them from A, K and Q on
  chip. It stores dSn per chunk (64 KB a chunk-head) and dVn.
- **D2b, the per-chunk gradients.** Per (b, hv, chunk), all independent, one threadgroup owning all of DV:
  - B3-B4 (dQ, dK, dW, dG), looping over value tiles exactly as `chunk_bwd_dqkwg` does, so every contraction over DV happens
    inside matrix products;
  - B5 (the WY backward: X, Y, dA, Aᵀ dA Aᵀ, P, R);
  - B6 (the reverse cumsum) fused at the end, since a chunk lives in one threadgroup.

  Every output has one writer, so the baseline's float atomics, and MLX's 128-way ones, are gone. Shared key heads (9B: 16 key
  heads, 32 value heads) need dQ and dK summed over 2 value heads: either let the program own both value heads of a key head, or
  keep transformers' `repeat_interleave` outside, where autograd sums them. The second is what the drop-in receives today.
- **Store instead of recompute.** Keep A (or W), the per-chunk states S_c and Vn from the forward: 151-210 MB a layer at
  C = 64-32 in fp32, against the baseline's 537 MB of checkpoints. That removes flash-linear-attention's second sequential pass
  (its `fwd_h` recompute) and about a fifth of its backward FLOPs. The repo already runs training with checkpointing off (128 GB).
- **Predicted limiters**: D2a is a sequential latency chain over T/C chunks (16-32 steps), as the state pass is. D2b is
  bandwidth-bound at intensity 15-24 (§4.1).
- **Forecast**: a 3.5 ms floor at C = 32 (3.0 ms at C = 64; 4.7 ms if flash-linear-attention's recompute is kept); 6-10 ms at a
  realistic 40-60 % of it, against 26.6 ms.
- **The alternative (MLX's M5 structure).** One threadgroup per head walks all chunks in reverse with S and dS slices in
  registers, reducing dQ and dK over value slices in threadgroup memory. It moves fewer bytes, but has 64 threadgroups (the
  1.6-wave tail), heavier threadgroup memory, and more barriers. It is a second registered entry if D2a/D2b leaves bandwidth
  on the table.

### 4.4 Where simdgroup matrices apply, and where they do not
- **Apply (8 × 8 tiles, fp32):**
  - every product above: KKᵀ, QKᵀ (C × C × 128); A·X (C × C × BV or 128); K·S and Q·S (C × 128 × BV); Kᵀ·Vn (128 × C × BV);
  - the backward's dO·Vnᵀ, dM·K, dMᵀ·Q, dO·Sᵀ, Vn·dSᵀ, dVn·Sᵀ, Aᵀ·dW, Aᵀ·dU;
  - the C × C × C chain Aᵀ dA Aᵀ;
  - the inverse's block merges.
- **Do not apply** (scalar or shuffle work on `thread_elements()` or lanes): the in-chunk cumsum (`simd_prefix_inclusive_sum`),
  masks and decays, the 8 × 8 diagonal substitutions (or the 4-product form), the row and column sums for dβ and dG, the reverse
  cumsum.
- **Tile shapes.** metal-flash-attention's Apple9 GEMM (32 × 32 per simdgroup, K-step 8) suggests each simdgroup should hold 4 × 4
  accumulator tiles where the output allows (e.g. K·S at C = 32, BV = 32). The measured peak was reached with 4 independent
  accumulators per simdgroup.

### 4.5 Numerics to carry into the correctness harness
- Decays only as exp(G_i − G_j), exp(G_r) and exp(G_last − G_r) from the in-chunk log cumsum; masks before the exponent; no clamp.
- The inverse: compare substitution, the 8 × 8 product form, and MLX's ungated-inverse-then-damp form at C = 32 and 64 under
  correlated keys (k_i·k_j → 1), β → 1, and g → 0 (no damping). Measure the error of A·(I + L) − I and of the outputs against
  float64.
- fp32 storage of A, S_c and Vn. A bf16 h, as flash-linear-attention stores it, saves bandwidth but costs parity.
- Tails: G_last and the reverse cumsum on the last valid row; zero rows beyond.
- Packed rows: chunk indices per sequence, as flash-linear-attention builds them, so no chunk crosses a sequence boundary.

### 4.6 Open questions the sources do not settle
- Whether MetalPerformancePrimitives' TensorOps fallback on the M3 matches hand-written simdgroup products. Apple says it runs; no
  source gives its speed. A one-kernel micro-benchmark beside `bench/hw_ceilings.py` would settle whether an MPP port would carry
  to a later M5 at no cost here.
- L1 capacity and the occupancy manager's response to 64-128 live floats a lane plus 12-16 KB of threadgroup memory. Measure with
  the pipeline's `maxTotalThreadsPerThreadgroup` (already in `macai.instruments`) and the time per chunk across BV ∈ {16, 32}.
- C (16 / 32 / 64) and BV are the series' tiling sweep. §4.1 predicts C = 32-64 for bytes and C = 32 for the fp32
  threadgroup-memory fit.
