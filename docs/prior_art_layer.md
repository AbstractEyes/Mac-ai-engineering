# Prior art: the rest of the linear-attention layer, the adapters, the loss head, products and long sequences on Apple GPUs

The prior art for the D3-D5 phase of the kernel series: the layer's surroundings (D4), the adapters, products and loss head
(D5), chunking and long sequences (D3). Read on 2026-11-03 from primary sources only: the libraries' own source files at named
commits and the papers. Every statement carries its source as `file: lines` at the version in the table. Statements marked
**[I]** are this document's inference (derivations, arithmetic, design proposals), not a source's claim. The rule itself (the
chunked gated delta rule, D1/D2) is covered in `docs/prior_art_gated_delta.md`, cited below as "the previous document".

| source | version read | what was read |
|---|---|---|
| causal-conv1d, github Dao-AILab/causal-conv1d | tag `v1.7.0` = `cd81f04` (2026-08-20) | `csrc/causal_conv1d_fwd.cu`, `csrc/causal_conv1d_bwd.cu`, `csrc/causal_conv1d.cpp`, `causal_conv1d/causal_conv1d_interface.py`, `causal_conv1d/cpp_functions.py` |
| flash-linear-attention, github fla-org/flash-linear-attention | main `cf9f520` (2026-11-03), the commit the previous document read | `fla/modules/causal_conv1d/ops.py`, `fla/modules/norm/{fused_norm_gate,layernorm}.py`, `fla/layers/gated_deltanet.py`, `fla/ops/gated_delta_rule/{chunk,gate}.py`, `fla/ops/utils/softplus.py`, `fla/ops/common/{chunk_delta_h,intracard_cp}.py`, `fla/ops/common/backends/intracard.py`, `fla/ops/cp/{chunk_fwd,chunk_bwd,chunk_merge}.py`, `fla/ops/cp/README.md` |
| Liger-Kernel, github linkedin/Liger-Kernel | tag `v0.8.5` = `1ab779f` (2026-10-21) | `src/liger_kernel/ops/{rms_norm,swiglu,fused_linear_cross_entropy,cross_entropy}.py`, `src/liger_kernel/transformers/swiglu.py`, `src/liger_kernel/utils.py` |
| unsloth, github unslothai/unsloth | main `81dc73a` (2026-11-03) | `unsloth/kernels/{fast_lora,utils,swiglu}.py` |
| Cut Cross-Entropy, github apple-aiml-research/ml-cross-entropy (the repository moved from apple/ml-cross-entropy) | main `3de376c` (2026-09-11) | `cut_cross_entropy/{cce,cce_lse_forward,cce_backward,linear_cross_entropy,torch_compile,utils}.py` |
| E. Wijmans, B. Huval, A. Hertzberg, V. Koltun, P. Krähenbühl, "Cut Your Losses in Large-Vocabulary Language Models", arXiv 2411.09009 | v2, 2025-03-10 | abstract, §4.1-4.3 (with gradient filtering) |
| P.-L. Hsu et al., "Liger Kernel: Efficient Triton Kernels for LLM Training", arXiv 2410.10989 | v3, 2025-01-24 | abstract; the code is the source |
| M. Milakov, N. Gimelshein, "Online normalizer calculation for softmax", arXiv 1805.02867 | v2 | abstract (the online max and sum used by Liger, MLX and Cut Cross-Entropy) |
| B. Zhang, R. Sennrich, "Root Mean Square Layer Normalization", arXiv 1910.07467; N. Shazeer, "GLU Variants Improve Transformer", arXiv 2002.05202; E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", arXiv 2106.09685 | v1, v1, v2 | abstracts (definitions only) |
| MLX, github ml-explore/mlx | tag `v0.32.4` = `36969c0` (installed 0.32.4; its shipped headers `cross_entropy.h`, `gemv.h`, `steel/gemm/{gemm,mma,loader,transforms}.h`, `steel/gemm/kernels/steel_gemm_{fused,splitk}.h` are byte-identical to the tag) | `mlx/backend/metal/{matmul,conv,cross_entropy,normalization,gated_delta_update}.cpp`, `mlx/backend/metal/kernels/{conv.metal,rms_norm.metal,cross_entropy.h,gemv.h,defines.h}`, `kernels/steel/gemm/*`, `mlx/{fast,primitives,compile}.cpp`, `python/mlx/nn/losses.py`, `python/mlx/nn/layers/activations.py` |
| mlx-lm, github ml-explore/mlx-lm | main `736ab0d` (2026-10-30) | `mlx_lm/models/{qwen3_5,qwen3_next,activations,gated_delta}.py`, `mlx_lm/tuner/{lora,trainer}.py` |
| metal-flash-attention, github philipturner/metal-flash-attention | main `8671cdd` (2024-09-22) | `README.md`, `Sources/FlashAttention/GEMM/GEMMKernelDescriptor.swift` |
| PyTorch, github pytorch/pytorch | tag `v2.15.0` = `31a77fc` | `aten/src/ATen/native/mps/operations/{Linear,LinearAlgebra,Convolution}.mm`, `aten/src/ATen/native/mps/kernels/Convolution.{metal,h}`, `torch/_inductor/codegen/mps.py` |
| transformers | 5.22.1 installed in the bench venv; tag `v5.22.1` = `446a570` (2026-10-30) for files outside the model | installed `models/qwen3_5/modeling_qwen3_5.py`; at the tag `src/transformers/modeling_utils.py` (`slice_logits_to_keep`), `src/transformers/loss/loss_utils.py` |
| PEFT | 0.21.2 installed; tag `v0.21.2` = `eaa35c7` (2026-10-01); the installed `tuners/lora/layer.py` is byte-identical to the tag | `tuners/lora/layer.py`, `tuners/tuners_utils.py`, `peft_model.py` (defaults) |
| this series | the M3 Max's measured ceilings (previous document §3.8) and large-product rates | simdgroup MMA 13.78 fp32 / 13.64 fp16 / 11.96 bf16 TFLOP/s; device copy 356 GB/s; 32 KB threadgroup memory; 40 GPU cores; bf16 large products 12.8-12.9 TFLOP/s (PyTorch) and 12.7-13.1 (MLX) |

Not read: Apple's MPSGraph and `MPSNDArrayMatrixMultiplication` are closed source; PyTorch hands its bf16 products and its
convolution backward to them (§2.2, §8.3), so their tiling is unknown here. The blog post cited in unsloth's `LoRA_MLP`
docstring (`fast_lora.py: 456, 477`) was not read; the formulas below are taken from the code. Every repository and paper in
the table was reached.

The shapes used throughout: Qwen3.5-0.8B hidden H = 1,024, MLP width I = 3,584, linear attention 16 key and 16 value heads of
128 (convolution channels C = 6,144); Qwen3.5-9B H = 4,096, I = 12,288, 16 key and 32 value heads of 128 (C = 8,192);
vocabulary V = 248,320. Bench shapes 4 × 1,024 (0.8B) and 2 × 1,024 (9B) tokens; training micro-batches of 1-2 sequences of
~650-1,282 tokens. The training recipe: bf16 autocast, frozen bf16 base weights, LoRA rank r = 32 (alpha 64, so the scale
s = 2; dropout p = 0.05) on in_proj_qkv, in_proj_z, out_proj, q/k/v/o_proj and gate/up/down_proj.

Contents: 0 findings that change the designs · 1 the layer as transformers computes it (the parity reference) · 2 the short
convolution · 3 the norms · 4 SwiGLU · 5 beta and g · 6 the LoRA path · 7 the loss head · 8 matrix products on Apple GPUs ·
9 long-sequence state passes · 10 design → what the prior art says to do on the M3 Max.

---

## 0. Findings that change the designs

1. **The frozen weights remove every cross-program reduction from D4.**
   The recipe trains adapters on the projections only, so the convolution weight, both norm weights, A_log and dt_bias are
   frozen → the backward of each D4 part needs only input gradients (dx, dz, dgate, dup) → the machinery the libraries build
   for weight gradients is unneeded: causal-conv1d's per-tile float atomics or its fp32 [B, chunks, C, K] workspace summed in a
   second pass (`causal_conv1d_bwd.cu: 495-520`), flash-linear-attention's fp32 partials per program summed afterwards
   (`causal_conv1d/ops.py: 758-799`; `fused_norm_gate.py: 563-624`), MLX's 512-group partials (`normalization.cpp: 138-212`).
   Liger already skips its weight gradient when the weight does not require one (`rms_norm.py: 697`); MLX's RMSNorm backward
   computes the weight partials whenever a weight exists (`normalization.cpp: 119, 148-153`). → Every D4 backward is a
   single-pass kernel with one writer per output **[I]**; a trained-weight path stays selectable, with fp32 partials and a
   second pass (Apple GPUs emulate float atomics, metal-flash-attention `README.md: 11`).
2. **Save inputs, recompute activations; the bandwidth floor is about 2 ms a layer.**
   Every reference saves the inputs plus at most an fp32 statistic per row and recomputes the activation in the backward:
   causal-conv1d recomputes the convolution's pre-activation in registers (`causal_conv1d_bwd.cu: 461-481`), Liger and unsloth
   recompute silu (Liger `swiglu.py: 97-102`; unsloth `swiglu.py: 64-104`), MLX recomputes the RMS normalizer
   (`rms_norm.metal: 207-259`). At the bench shapes the four D4 parts have a combined forward + backward floor of 1.99 ms
   (0.8B) and 2.45 ms (9B) at 356 GB/s **[I, §2.4, §3.4, §4]**. The series' registration estimates the surroundings at
   ~17-28 ms a layer by subtraction (bench layer minus products minus the rule: about 20 ms at 0.8B, 27 ms at 9B) → about
   8-11x headroom; D4's "each part within 3x of its floor" bar sums to 6-7.4 ms a layer.
3. **The convolution belongs in channel-last order, where the projection already wrote it.**
   transformers transposes the projection output [B, T, C] to a [B, C, T] view (`modeling_qwen3_5.py: 555`), whose
   stride(1) = 1 is exactly causal-conv1d's channel-last condition (`causal_conv1d.cpp: 176, 296`): CUDA reads the buffer in
   place; flash-linear-attention is [B, T, D] natively (`causal_conv1d/ops.py: 621-627`) and so is MLX (`qwen3_5.py: 160-169`).
   PyTorch 2.15 on the Apple GPU instead makes the view contiguous (a transposing copy) before its depthwise kernel
   (`Convolution.mm: 594-598`) and runs the backward in MPSGraph (`Convolution.mm: 1036-1048`). → D4's convolution reads
   [B, T, C] in place, accumulates in fp32, applies SiLU to the fp32 pre-activation, and writes q, k, v in the rule kernels'
   layout, one 128-channel head per tile so the q/k l2 norm can fold in **[I]**.
4. **The loss head: logits only where labels exist; chunk by bandwidth, not by memory parity.**
   (a) transformers 5.22.1 slices the hidden states by a boolean [B, T] mask before the head (`modeling_utils.py: 5008-5031`)
   and its loss accepts pre-shifted labels (`loss_utils.py: 57, 63-66`) → the head and the loss can run at supervised
   positions only, with no kernel. (b) Liger's fused linear cross-entropy sizes its token chunks so transient logits stay near
   N × H elements (`fused_linear_cross_entropy.py: 92-106`); with V/H = 243 (0.8B) and 61 (9B) that is 16- and 32-token
   chunks, each re-reading the 0.51 or 2.03 GB head weight twice: about 165 GB, 0.46 s per micro-batch **[I, §7.6]**.
   (c) Cut Cross-Entropy's kernels refuse macOS and fall back to a torch.compile path over the valid rows
   (`linear_cross_entropy.py: 31-39, 163-167`; `torch_compile.py: 73-80`). (d) MLX 0.32.4 ships a fused one-pass
   cross-entropy whose gradient overwrites the logits (`cross_entropy.h: 3-133`). → the loss head, a candidate entry outside
   D3-D5 (proposed, not part of this phase's builds): supervised rows only,
   chunks of at least 256 rows, a fused cross-entropy per chunk, no head-weight gradient **[I]**.
5. **Adapters: the rank-32 products are cheap; the passes around them are not.**
   The adapter products of a linear-attention layer are 21 GFLOP (0.8B) and 31 GFLOP (9B) at the bench shapes, about 1.7 and
   2.4 ms, 6% and 1.7% of the frozen products' FLOPs **[I, §6.5]**. PEFT wraps them in an fp32 copy of the input (adapters are
   upcast to fp32 by default, `tuners_utils.py: 628-641`; the cast, `layer.py: 1100`), a dropout pass per projection, a scaled
   copy and an add (`layer.py: 1102`): about 3.6 GB (0.8B) and 5.5 GB (9B) of extra traffic a layer, 10 and 15 ms, if each
   step is its own pass **[I]**. unsloth removes those passes: the scaled add happens inside the GEMM's accumulator
   (`utils.py: 1875-1883`), the base GEMM accumulates the input gradient onto the adapter's term (`fast_lora.py: 1361-1369`),
   masks come from torch's own dropout call and are replayed (`fast_lora.py: 285-307`). → D5(b) fuses the adapter path and
   takes its masks from torch, which keeps parity with dropout on **[I]**.
6. **On the M3, bf16 products should run on fp32 fragments.**
   MLX's steel GEMM loads bf16 tiles into `simdgroup_matrix<float>` fragments (`mma.h: 46, 64, 481-483`), so its bf16 GEMM runs
   on the fp32 multiply-accumulate path; this machine measured that path faster than bf16 fragments (13.78 against 11.96
   TFLOP/s), and MLX's bf16 rate (12.7-13.1) sits above the bf16-fragment ceiling **[I]**. On the M3 Max (MLX's 's' class) MLX
   uses one tile (64 × 64 × 16, 2 × 2 simdgroups) for every large shape (`matmul.cpp: 166-172`) and sends small-output,
   long-reduction products (the adapters' weight gradients) to split-K with fp32 partials (`matmul.cpp: 957-992`). PyTorch 2.15
   runs its bf16 products in Apple's closed MPS kernels (`Linear.mm: 172-199`; `LinearAlgebra.mm: 1179-1208`), at the same
   rate. → No custom large GEMM; any custom small product uses fp32 fragments, and split-K when the reduction runs over tokens
   **[I]**.
7. **Long sequences: value slicing first; affine segment scans only for very long sequences.**
   flash-linear-attention splits the state pass over value tiles, ⌈V/BV⌉ programs per head with BV ∈ {32, 64}
   (`chunk_delta_h.py: 50-53, 726`); MLX uses 2-4 threadgroups per head (`gated_delta_update.cpp: 155-180`). For long
   sequences flash-linear-attention adds intra-card splitting: each subsequence is summarized as an affine map
   S_out = M·S_in + h with M a 128 × 128 fp32 product chain (`cp/chunk_fwd.py: 59-63, 247-286`), inference only
   (`backends/intracard.py: 60-61`), applied by its own rule only from 3 × 128 chunks (24,576 tokens at C = 64) at 16 heads on
   40 cores (`intracard_cp.py: 132-158, 181-198`) **[I, applying the rule]**. The summaries cost 0.5-1.2x a whole forward's
   FLOPs **[I, §9.4]**. → D3(b): 4-8 threadgroups per head at 1 row × 16 heads; segment scans as a later entry.

---

## 1. The layer as transformers computes it (the parity reference)
The linear-attention decoder layer of `modeling_qwen3_5.py` (transformers 5.22.1), in order:

```
res = h;  x1 = RMSNormZC(h)                                   input_layernorm, zero-centered        848-901
qkv = x1·Wqkvᵀ (+ adapter)   z = x1·Wzᵀ (+ adapter)   b = x1·Wbᵀ   a = x1·Waᵀ                          554-561
u = depthwise_causal_conv(qkv, K = 4);  y = silu(u);  q, k, v = split(y)                             555, 578-603
β = sigmoid(b);  g = −exp(A_log)·softplus(a + dt_bias)                                               605-607
9B: q, k repeated over the 2 value heads of each key head                                            608-610
o = rule(q, k, v, g, β) with the q/k l2 norm inside                                                  626-637
o' = RMSNormGated(o, z), per head of 128;  h = res + o'·Woutᵀ (+ adapter)                           643-650
res = h;  x2 = RMSNormZC(h);  h = res + down( silu(gate(x2)) ⊙ up(x2) )  (+ adapters)               811-824
```

Rounding points with a bf16 input and bf16 weights:

| part | transformers' arithmetic | roundings to bf16 | lines |
|---|---|---|---|
| zero-centered RMSNorm | fp32: x̂ = x·rsqrt(mean(x²) + ε), y = x̂·(1 + w) | once, at the end | 827-845 |
| convolution + SiLU | `F.conv1d` on the [B, C, T] view with padding K − 1, sliced to T; SiLU on the bf16 result | u, then y | 255-275 |
| β, g | sigmoid in bf16; g in fp32 (`A_log.float()`, `a.float()`) | β | 605-607 |
| gated RMSNorm | fp32 x̂; `weight * x̂.to(bf16)` in bf16; × silu(z.float()) in fp32 | x̂, x̂·w, the output | 203-220 |
| SwiGLU | `act_fn(gate) * up` | silu(e), the product | 822-824 |
| q/k l2 norm | x·rsqrt(Σx² + 1e-6) | in the rule | 280-283 |

transformers notes that flash-linear-attention's gated norm does not round x̂ to the input dtype (`:202`) and that its l2 norm
is x / sqrt(Σx² + ε) (`:278-279`). The full-attention layers add q/k norms (zero-centered RMSNorm over the head dimension,
`:760-761, 779-780`) and an output gate `attn_output * sigmoid(gate)` (`:805`).

---

## 2. The short causal convolution with SiLU

### 2.1 The arithmetic
Per channel c with K taps (Qwen3.5: K = 4, no bias), zero history before the sequence start:

```
u[t,c]  = Σ_{j=0..K−1} w[c,j] · x[t−(K−1)+j, c]
y[t,c]  = u · σ(u)
du      = dy · σ(u) · (1 + u·(1 − σ(u)))
dx[t,c] = Σ_{j=0..K−1} w[c,j] · du[t+(K−1)−j, c]        (du = 0 past the sequence end)
dw[c,j] = Σ_{b,t} du[t,c] · x[t−(K−1)+j, c]              (only when the weight trains)
```

**[I]** Checked term by term against causal-conv1d (`causal_conv1d_fwd.cu: 327-333`; `causal_conv1d_bwd.cu: 478-479, 490,
531-532`) and flash-linear-attention (`causal_conv1d/ops.py: 238-257`).

### 2.2 The implementations

| | layout and tiling | reductions | saved / recomputed | numerics |
|---|---|---|---|---|
| causal-conv1d, channel-last kernels | input [B, C, T] with stride(1) = 1, detected as `x.stride(1) == 1 && x.stride(2) > 1` (`causal_conv1d.cpp: 176, 296`). Forward grid (B, ⌈T/64⌉, ⌈C/Cc⌉), 128 threads (`causal_conv1d_fwd.cu: 354-385`); Cc = 128 bytes of channels: 64 bf16 or 32 fp32 (`:187-195`). A shared tile of (K − 1 + 64) time steps × (Cc + 8) channels, filled with 16-byte vector loads along channels, plus K − 1 halo rows (`:221, 245-266`). Each thread computes a run of one channel (32 outputs for bf16) with the K weights in registers (`:278-335`), then stores through the shared tile with vector stores (`:337-350`). Backward: tiles of 64 (T ≤ 128) or 128 time steps (`causal_conv1d_bwd.cu: 607-611`), shared dout and x tiles with K − 1 halos on both sides (`:346-347`) | dx: none (a gather, one writer). dw, db: per-thread sums, a shuffle all-reduce over the threads sharing a channel (`:495, 512`), then `atomicAdd` into fp32 (`:504, 520`), or under deterministic mode a store into an fp32 workspace [B, ⌈T/L⌉, C, K] (`:497-502`; `causal_conv1d.cpp: 412-429`) summed on the host (`:450-462`) | saves x, weight, bias (`causal_conv1d_interface.py: 56`); the backward recomputes u in registers from x, loading K − 1 extra rows on the right (`causal_conv1d_bwd.cu: 406-414, 461-481`) | fp32 accumulation; SiLU as u / (1 + exp(−u)) on the fp32 u (`causal_conv1d_fwd.cu: 316-335`); one rounding at the store; dw allocated in fp32 and cast at the end (`cpp_functions.py: 134-160`) |
| flash-linear-attention, Triton | [B, T, D], strides passed so a view works (`causal_conv1d/ops.py: 621-627`). Grid (⌈D/BD⌉, ⌈T/64⌉, B), BD autotuned over {16, 32, 64, 128} (`:34-48, 636-638`); each program accumulates K shifted [64, BD] tiles (`:100-136`); an optional residual add is fused (`:141-145`) | dw, db: fp32 partials per (batch, time chunk), [B·NT, D, K] (`:758-759`), summed by torch (`:796-799`); no atomics | saves x, weight, bias, residual, initial state (`:957`); recomputes u with a whole forward call written to memory (`:743-756`), read back by the backward kernel (`:244-247`) | fp32; y·sigmoid(y); stores rounded to nearest even (`:348`) |
| flash-linear-attention, the layer | one convolution over the concatenated q, k, v channels when no cache is used (`fla/layers/gated_deltanet.py: 200-268`), the same structure as Qwen3.5's single convolution | | | |
| MLX (mlx-lm's Qwen3.5) | NLC arrays: [B, T, C] read as is. mlx-lm prepends a zero state of K − 1 rows and runs `nn.Conv1d` without padding, then `nn.silu` (`qwen3_5.py: 150-169`); weights stored [C, K, 1] (`:367-368`). Forward: `depthwise_conv_1D_gpu` when groups = C = O, stride 1, no padding, no flip (`conv.cpp: 1561-1565`): one thread per output, grid (C, T_out, B) (`:1533-1536`; kernel `conv.metal: 303-321`) | backward by the generic rule: the input gradient is a convolution with flipped, padded weights (`primitives.cpp: 1496-1537`), not the fast path's case, so it runs in the grouped implicit-GEMM path (`conv.cpp: 1570-1596`); the weight gradient is a grouped convolution whose kernel spans the sequence (`primitives.cpp: 1571-1612`) | autograd keeps the convolution's input | fp32 accumulation in the forward kernel; SiLU is a compiled elementwise function (MLX `python/mlx/nn/layers/activations.py: 138-139`); MLX's compiler fuses only unary, binary, ternary and broadcast ops (`compile.cpp: 78-80`) |
| PyTorch 2.15, Apple GPU (today's path) | a depthwise conv1d selects `Conv1dKernel::Depthwise` (`Convolution.mm: 528-531`), which first calls `input_t.contiguous()` (`:594-598`): the [B, C, T] view is copied into channel-first order. One thread per 8 consecutive outputs of a channel (`kernels/Convolution.h: 6`; `kernels/Convolution.metal: 313-342`), 256-thread groups (`Convolution.mm: 679-683`); transformers pads both ends and slices | backward in MPSGraph, whose depthwise data-gradient op is channel-first only (`Convolution.mm: 1036-1048`) | autograd | fp32 accumulation, one rounding of u; SiLU is a separate op on the rounded u |

### 2.3 Numerics
**[I]** The fused kernels round once (fp32 u → SiLU → bf16). transformers rounds u to bf16 before SiLU (§1). A fused kernel's
error against float64 should therefore be below the PyTorch path's, inside the "at most twice the PyTorch path's own error"
parity bar. Packed sequences need the window cut at sequence starts: causal-conv1d's `seq_idx`
(`causal_conv1d_fwd.cu: 304-333`), flash-linear-attention's `cu_seqlens` (`causal_conv1d/ops.py: 79-83`). This only matters if
micro-batches are packed.

### 2.4 For the M3 Max [I]
- **Floor.** With a frozen weight the forward moves 4 bytes per element (read x, write y) and the backward 6 (read x and dy,
  write dx). That is 251.7 MB → 0.71 ms at 0.8B (C = 6,144, 4,096 tokens) and 167.8 MB → 0.47 ms at 9B (C = 8,192, 2,048
  tokens).
- **Layout.** Read the projection's [B, T, C] buffer in place, as causal-conv1d's channel-last kernels, flash-linear-attention
  and MLX do. Never make the transposing copy that PyTorch's path makes.
- **Tiling without threadgroup memory.** causal-conv1d's shared tiles would fit: (3 + 64) × 72 bf16 = 9.6 KB forward and
  (67 + 70) × 72 × 2 B = 19.7 KB backward. But Apple9 favors little threadgroup memory (previous document §3.7). A
  register-held sliding window works instead:
  - a lane owns 4-8 adjacent channels: one 8-16-byte load per time step, contiguous across the simdgroup;
  - it walks a run of 32-64 time steps, keeping the last K − 1 = 3 inputs in registers;
  - each x is read once, plus a 3-row halo per run;
  - no barriers.
- **Backward.** The same walk over x and dy with a 3-step look-ahead on dy. u and σ(u) are recomputed in registers
  (causal-conv1d's choice), not written to memory first (flash-linear-attention's choice). dx has one writer.
- **Epilogue.** 128 channels are one head of q, k or v. A tile aligned to a head can:
  - apply transformers' l2 norm (x·rsqrt(Σx² + 1e-6)) and the DK^-1/2 query scale with one `simd_sum` over the head;
  - write q, k, v directly in the layout the rule kernels read.

  At 9B, q and k can be written once per key head and the value-head repeat left to the rule's indexing, as
  flash-linear-attention does (previous document §2.1).
- **When the weight trains.** Per-threadgroup fp32 partials [tiles, C, K] and a second-pass sum (flash-linear-attention's
  form), not float atomics.

---

## 3. RMSNorm and the gated RMSNorm

### 3.1 The arithmetic

```
r  = (mean_i x_i² + ε)^(−1/2)          x̂ = r·x
RMSNorm:  y = x̂ ⊙ w'                   w' = w (standard),  w' = 1 + w (zero-centered, Qwen3.5's Qwen3_5RMSNorm)
gated:    o = (x̂ ⊙ w) ⊙ silu(z)        (Qwen3.5's Qwen3_5RMSNormGated, rows of 128 = one head)

backward, RMSNorm:  dx = r · ( w' ⊙ dy − x̂ · ⟨x̂, w' ⊙ dy⟩ / D )          dw = Σ_rows dy ⊙ x̂
backward, gated:    dz = do ⊙ (x̂ ⊙ w) ⊙ σ(z)·(1 + z·(1 − σ(z)))
                    dy = do ⊙ silu(z)   → then the RMSNorm backward with w' = w
```

**[I]** The dx formula is the same in all three sources: flash-linear-attention computes (w ⊙ dy − x̂·c1)·r with
c1 = ⟨x̂, w ⊙ dy⟩/D (`fused_norm_gate.py: 289-299`); Liger computes r·m − r³·⟨m, x⟩·x / N with m = dy ⊙ w
(`rms_norm.py: 230-232`); MLX computes g·w·r − x·mean(x·w·g)·r³ (`rms_norm.metal: 265-267`).

### 3.2 The implementations

| | grid | reductions | saved / recomputed | numerics |
|---|---|---|---|---|
| flash-linear-attention `FusedRMSNormGated` (`fla/modules/norm/fused_norm_gate.py`) | rows in blocks of BT ∈ {16, 32, 64} with 4-16 warps when D ≤ 512 (`:31, 482-489`), one row per program above (`:124-188`) | row sum in-program. Backward: NS = min(#SMs, T) programs over contiguous row blocks (`:563-569`); dw, db as fp32 partials [NS, D] (`:567-568`) summed afterwards (`:623-624`) | saves x (or the residual sum), g, w, b, mean, rstd (fp32 per row) (`:476, 667`). The norm-gate-linear fusion recomputes y in the backward rather than saving it, to feed the linear's weight gradient (`:761-762, 787-802`) | fp32 throughout, one rounding (`:93-107`). A residual add can be fused: the sum is stored as the next residual (`:69-74`) |
| flash-linear-attention `RMSNorm` (`layernorm.py`) | the same pattern (`:193-293`) | NS programs (`:507-508`), partials summed (`:570`) | saves the residual sum, w, b, mean, rstd (`:614`); the norm-linear fusion recomputes y (`:827-828`) | fp32; no offset (zero-centered) option |
| Liger `rms_norm` (`src/liger_kernel/ops/rms_norm.py`) | one program per row forward (`:93-150`); backward on `sm_count` programs, each a block of rows (`:552-553`). `sm_count` is read only for cuda, xpu and npu devices (`:532-538`) | dW: fp32 partials [sm_count, D] (`:544`) summed afterwards (`:622`); skipped when the weight needs no gradient (`needs_dw`, `:697`); dX written over dY by default (`:555-556, 646-648`) | saves X, W, RSTD (fp32 for the llama and gemma modes) (`:465-466, 674`) | y = x̂·(offset + w) (`:85, 143`); offset = 1 is the zero-centered weight. Casting modes (`:62-64, 106-148`): 'llama' rounds x̂ to the input dtype before × w (`:139-140`); 'gemma' computes in fp32 and rounds once (`:111-114, 147-148`); 'none' stays in the input dtype |
| MLX `rms_norm` | one threadgroup per row, 4 elements a thread (`RMS_N_READS`, `defines.h: 15`); simd_sum, then a threadgroup pass over 32 partials, then `precise::rsqrt` (`rms_norm.metal: 12-64`); a looped kernel for rows above 4,096 (`defines.h: 16`) | VJP: per-row dx; dw as partials over at most 512 row groups (`normalization.cpp: 138-144`), stored in the input dtype (`rms_norm.metal: 286-296`) and reduced by a strided-reduce pass (`normalization.cpp: 207-212`). One primitive yields both outputs (`fast.cpp: 166-170`); argnums only pick which are returned (`:172-175`), and the weight partials are computed whenever a weight exists (`normalization.cpp: 119, 148-153`) | saves only x and w: the VJP recomputes the normalizer (`rms_norm.metal: 207-259`) | fp32 sums; the output is w·T(x·r), x̂ rounded before the weight multiply (`rms_norm.metal: 66-80`) |
| mlx-lm's Qwen3.5 | zero-centered weights converted to w + 1 at load (`qwen3_5.py: 359-371`), then `nn.RMSNorm`. Gated norm: `mx.fast.rms_norm`, then `precise_swiglu` = (silu(z_f32) · y_f32) → input dtype (mlx-lm `qwen3_next.py: 58-71`, `mlx_lm/models/activations.py: 14-18`) | | | |

### 3.3 Numerics against transformers [I, from the lines in §1 and §3.2]

| norm | transformers | reproduced by | not reproduced by |
|---|---|---|---|
| zero-centered RMSNorm | fp32, one rounding | Liger 'gemma' mode with offset 1.0 | MLX `rms_norm` with w + 1 (rounds x̂ first); Liger 'llama' |
| gated RMSNorm | x̂ rounded, × w in bf16, × silu(z) in fp32, rounded | MLX `rms_norm` + `precise_swiglu` | flash-linear-attention (one rounding; transformers' own note, `modeling_qwen3_5.py: 202`) |

### 3.4 For the M3 Max [I]
- **Floors.**
  - Gated norm: forward reads x and z and writes o (6 bytes per element); backward reads x, z, do and writes dx, dz (10 bytes).
    That is 134.2 MB → 0.38 ms at both sizes (2,048 or 4,096 value channels per token).
  - The two zero-centered norms: 4 + 6 bytes per element each. That is 83.9 MB → 0.24 ms (0.8B) and 167.8 MB → 0.47 ms (9B).
- **Gated norm.** A row of 128 is 32 lanes × 4 elements, MLX's read width. So:
  - one row per simdgroup, a `simd_sum` only, no threadgroup memory and no barriers;
  - 4-8 rows per threadgroup;
  - 65,536 rows at either bench shape.
- **Hidden-size norms (1,024 / 4,096).** Two candidates to measure:
  - one threadgroup per row with MLX's two-level sum (256 / 1,024 threads at 4 elements a thread);
  - one simdgroup per row with 32-128 elements a lane.
- **What to keep.** Save only the inputs and recompute r in the backward (MLX), since x is read anyway. A saved fp32 r per
  row (flash-linear-attention, Liger) costs 0.1% more writes; either choice sits at the floor.
- **Residual add.** Fold each residual add into the following norm, as flash-linear-attention's `residual`/`residual_out`
  does. The norm then reads residual and branch, writes the sum (the next residual) and the normed output: one full pass
  saved per add.
- **Weight gradient.** Skipped when frozen (Liger's `needs_dw`). When trained: fp32 partials and a second pass, not
  input-dtype partials (MLX).
- **Parity.** Reproduce transformers' rounding points per §3.3.
- **A no-kernel baseline.** PyTorch 2.15's Inductor has a Metal code generator with threadgroup reductions
  (`torch/_inductor/codegen/mps.py: 612-618, 719-890`). torch.compile of the norm and gate chains is a baseline to measure
  D4 against; it was not measured here.

---

## 4. SwiGLU

```
h  = silu(e) ⊙ u                  e = gate(x),  u = up(x)
du = dh ⊙ silu(e)
de = dh ⊙ u ⊙ σ(e)·(1 + e·(1 − σ(e)))
```

| | forward | backward | saved / recomputed | numerics |
|---|---|---|---|---|
| Liger (`src/liger_kernel/ops/swiglu.py`) | one program per row (`:57-75`, launch `:185-199`); a Blackwell-only column-tiled variant (`:20-41`) | recomputes σ(e) and silu(e) (`:97-102`); writes de and du over e and u in place (`:104-105`) | saves e, u (`:272`) | silu in fp32, cast to u's dtype, then × u (`:72-74`): transformers' rounding |
| unsloth (`unsloth/kernels/swiglu.py`) | the same rounding, marked "Exact copy from HF" (`:37-43`) | one kernel recomputes h and writes h, du and de over the three input buffers (`:64-104`) | h, the down projection's input, is recomputed for that projection's adapter gradient rather than saved (`fast_lora.py: 651-661`) | de computed in fp32 and rounded (`:98-100`) |
| MLX / mlx-lm | `swiglu` and `precise_swiglu` are `mx.compile`d elementwise functions (`mlx_lm/models/activations.py: 9-18`); the whole training step is compiled (`trainer.py: 247`) | MLX's autograd | | `precise_swiglu` computes in fp32 |

**For the M3 Max [I].**
- **Floor.** Forward reads e and u and writes h (6 bytes per element); backward reads dh, e, u and writes de, du (10 bytes).
  That is 234.9 MB → 0.66 ms (0.8B, I = 3,584) and 402.7 MB → 1.13 ms (9B, I = 12,288).
- **Kernel.** Elementwise, 8 bf16 per lane in vector loads; backward in place (Liger).
- **h for the adapter.** When the down projection carries an adapter, write h in the backward (unsloth) instead of keeping it
  from the forward.
- **No GEMM fusion.** Folding SwiGLU into the down projection's operand load would save the h write and read: 4 bytes per
  element, 0.16 / 0.28 ms. It needs a custom GEMM, so it is not worth it while the large products tie (§8.4).

---

## 5. beta and g

```
β = σ(b)                                     db = dβ · β·(1 − β)
g = −exp(A_log) · softplus(a + dt_bias)      da = dg · (−exp(A_log)) · σ(a + dt_bias)
                                             dA_log = Σ dg·g,  ddt_bias = Σ da   (only when trained)
```

- **flash-linear-attention fuses both into the rule.**
  - The gate is computed together with the in-chunk cumulative sum (`fla/ops/gated_delta_rule/gate.py: 62-100`), with a
    fused backward (`:121-159`).
  - dA_log and ddt_bias are per-chunk partials summed afterwards (`:213-230`).
  - sigmoid(β) is applied inside the rule (`chunk.py: 287-288`) with its backward (`:386-387`).
  - The layer passes raw a and b with `use_gate_in_kernel` and `use_beta_sigmoid_in_kernel` (`gated_deltanet.py: 302-318`).
- **Softplus numerics.** flash-linear-attention's softplus keeps torch's threshold of 20 but uses approximate exp2/log2
  instructions (`fla/ops/utils/softplus.py: 17-35`). **[I]** For transformers parity: log1p(exp(x)) in fp32 with the same
  threshold.
- **MLX's form.** mlx-lm computes the decay in linear space, exp(−exp(A_log)·softplus(a + dt_bias)) (`gated_delta.py: 19-20`);
  the previous document covers the consequences for MLX's kernels.

**For the M3 Max [I].** Read raw a and b ([N, Hv] bf16) in the rule's prologue and compute β and g in fp32. Fold dβ and dg back
to db and da in D2's epilogue. Skip dA_log and ddt_bias when frozen. Bytes are negligible (about 2 KB per token at 9B). The
gain is about six launches and two [N, Hv] fp32 arrays per layer.

---

## 6. The LoRA adapter path

### 6.1 The arithmetic
PEFT's orientation: A = `lora_A.weight` ∈ R^{r × d_in}, B = `lora_B.weight` ∈ R^{d_out × r}, s = α / r (`layer.py: 278-281`),
mask M ∈ {0, 1}^{N × d_in}, X ∈ R^{N × d_in}:

```
X_d = (M ⊙ X) / (1 − p)
P   = X_d · Aᵀ                       [N × r]
Y   = X · Wᵀ + s · P · Bᵀ
G   = dY · B                         [N × r]
dB  = s · dYᵀ · P                    [d_out × r]
dA  = s · Gᵀ · X_d                   [r × d_in]
dX  = dY · W + s · (M ⊙ (G · A)) / (1 − p)
FLOPs: forward 2·N·r·(d_in + d_out); backward 4·N·r·(d_in + d_out)
```

### 6.2 PEFT 0.21.2 (installed)
- **Modules.** One `nn.Dropout(p)` per adapted module (`layer.py: 255-260`); A and B are `nn.Linear` layers (`:263-264`).
- **Forward.** `lora.Linear.forward`:
  1. computes the base result;
  2. casts x to the adapter's dtype (`:1100`);
  3. adds `lora_B(lora_A(dropout(x))) * scaling` (`:1102`);
  4. casts back to the result's dtype (`:1113`).
- **The cast.** `_cast_input_dtype` copies x whenever the dtypes differ (`tuners_utils.py: 2220-2235`). Adapters are upcast
  from bf16 to fp32 by default (`autocast_adapter_dtype = True`, `tuners_utils.py: 628-641`; `peft_model.py: 157, 425`).
- **[I] With a bf16 base and default settings,** each adapted module therefore runs:
  - an fp32 copy of x;
  - dropout in fp32;
  - autocast's cast back to bf16 for the two products;
  - a scaled copy and an add;

  each as its own kernel in eager mode. Autograd keeps the cast and dropped copies and the masks.

### 6.3 unsloth `fast_lora`
- **Formulas.** The docstrings of `LoRA_MLP` (`fast_lora.py: 442-478`), `LoRA_QKV` (`:920-948`) and `LoRA_W` (`:1264-1289`).
- **Forward without dropout or adapter bias.** `matmul_lora` writes s·(X·Aᵀ)·Bᵀ into a fresh buffer with beta 0 and lets
  the base GEMM accumulate onto it (`unsloth/kernels/utils.py: 1875-1883, 2123-2137`): no output-sized temporary and no separate scaled add.
  Projections that read the same X share one product over the concatenated A matrices (`fast_lora.py: 338-345, 516-518, 997`).
- **With dropout.**
  - The masks are drawn with `torch.native_dropout`, the call `nn.Dropout` makes, so the RNG draws and rounding match
    (`:285-294`).
  - They are drawn per projection in PEFT's order (`:542-551`).
  - The adapter product reads the dropped X; its scaled add is `addmm_(out, XA, Bᵀ, alpha = s)` onto the base result
    (`:310-327`).
  - The backward replays the masks with `native_dropout_backward` (`:297-307, 330-335`).
- **Saved and recomputed.** It saves A, B, X, e, g and the masks (`:567`). h is recomputed in the SwiGLU backward
  (`:651-652`).
- **Backward.**
  - dA and dB come from `addmm_(…, alpha = s, beta = 0)` (`:656-661, 1346-1354`).
  - dX starts as s·(dY·B)·A and the base GEMM accumulates onto it (`:1361-1369`), or it is accumulated by `addmm_`
    (`:333`).
  - With a shared X, one product serves every A and the LoRA terms of dX (`:348-378`).
  - dX can be written over X's storage (`:737`).
- **Numerics [I].** The scaled add happens in the GEMM's fp32 accumulator with one rounding. PEFT rounds the B product, the
  scaled copy and the sum.
- **Kernels.** The products are torch matmuls; only the SwiGLU kernels are Triton.

### 6.4 mlx-lm `LoRALinear`
- **Unfused.** y = linear(x); z = (dropout(x) @ A) @ B; the output is y + (s·z).astype(x.dtype) (`lora.py: 95-98`).
- **fp32 products.** A is created in fp32 (`:88-92`), so the adapter products promote to fp32.
- **Compilation.** The compiled training step fuses the elementwise parts (`trainer.py: 247`).
- **The Metal epilogue.** MLX's GEMM epilogue computes α·A·B + β·C in fp32 before the store (`steel_gemm_fused.h: 11-12,
  165-199`; `transforms.h: 39-54`). That is the Metal form of unsloth's accumulate.

### 6.5 For the M3 Max [I]
Per linear-attention layer at the bench shapes; adapted projections in_proj_qkv, in_proj_z, out_proj, gate, up, down.

| | 0.8B, N = 4,096 | 9B, N = 2,048 |
|---|---|---|
| Σ d_in, Σ d_out | 9,728; 17,408 | 32,768; 45,056 |
| frozen products, 4·N·Σ(d_in·d_out) (forward + input gradient) | 352 GFLOP → 27.5 ms at 12.8 TFLOP/s | 1,787 GFLOP → 140 ms |
| adapter products, 6·N·r·(Σ d_in + Σ d_out) | 21.3 GFLOP → 1.7 ms | 30.6 GFLOP → 2.4 ms |
| PEFT's extra traffic, each step a pass: forward 23 B per input + 12 B per output element, backward 31 + 8 | 3.6 GB → 10.0 ms | 5.5 GB → 15.4 ms |
| fused adapter traffic: about 11 B per input + 8 B per output element (X and mask reads, read-modify-write of Y and dX) | 1.0 GB → 2.8 ms | 1.5 GB → 4.1 ms |

The 23 + 12 bytes count, per element: the fp32 cast of x (6), dropout in fp32 with a byte mask (9), autocast's cast back to
bf16 (6), the A product's read (2); then the B product's write (2), the scaled copy (4) and the add (6) per output element.
The frozen-product FLOPs match the registration's counts (353 and 1,789 GFLOP).

**The kernel this implies.**
- **Keep the base GEMMs in PyTorch.** Their rate ties MLX's (§8.4).
- **Forward:** one kernel computing P = X_d·Aᵀ with the mask applied as X is loaded; Y += s·P·Bᵀ as a read-modify-write of
  the base output.
- **Backward:** G = dY·B and dB = s·dYᵀ·P from one pass over dY tiles; dA = s·Gᵀ·X_d, re-applying the saved mask on load;
  dX += s·M ⊙ (G·A)/(1 − p), accumulated into the base dX.
- **Arithmetic.** fp32 fragments, P saved ([N × r]), split-K over tokens for dA and dB (§8.4).
- **Masks.** From torch's own dropout call (unsloth's choice), so a fused path and PEFT draw identical masks and every gradient
  can be held to parity with dropout on.
- **Shared X.** PEFT draws a separate mask per module, so the shared-X product only applies across modules whose masks are
  shared, which PEFT never does.

---

## 7. The large-vocabulary loss

### 7.1 The arithmetic

```
Z    = E·Cᵀ                    E [N × H] final hidden states,  C [V × H] head weight
m_n  = max_v Z_nv;   lse_n = m_n + log Σ_v exp(Z_nv − m_n)
ℓ_n  = lse_n − Z_{n,y_n};       L = Σ_{n valid} ℓ_n / n_valid
dZ   = (exp(Z − lse) − onehot(y)) · γ_n          γ_n = 1/n_valid for valid rows, 0 otherwise
dE   = dZ · C;                  dC = dZᵀ · E   (not needed for a frozen head)
online max/sum:  m' = max(m, z),  d' = d·exp(m − m') + exp(z − m')     (Milakov & Gimelshein, arXiv 1805.02867)
```

### 7.2 transformers 5.22.1
- **Logits at every position by default.** The head is applied to every position unless `logits_to_keep` is set
  (`modeling_qwen3_5.py: 1808, 1867-1868`).
- **The boolean mask.** `slice_logits_to_keep` accepts a boolean [B, T] tensor and returns `hidden_states[mask]`, flattened
  to [num_true, H] (`modeling_utils.py: 5008-5031`).
- **The loss.** `ForCausalLMLoss`:
  - upcasts all logits to fp32 (`loss_utils.py: 60-61`);
  - shifts the labels unless `shift_labels` is passed (`:63-66`);
  - calls `F.cross_entropy` (`:34-48, 72`).
- **[I] The no-kernel saving.** A mask of the positions whose next label is supervised, plus the matching `shift_labels`,
  computes the head and the loss at those rows only.

### 7.3 Liger fused linear cross-entropy (`src/liger_kernel/ops/fused_linear_cross_entropy.py`)
- **Chunk size.** Tokens are chunked so transient logits stay near N × H elements:
  inc_factor = ⌈V / (c·H)⌉ with c = 1, chunk = next_pow2(⌈N / inc_factor⌉) (`:24-27, 92-106`).
- **Per chunk.**
  - the logits product (`:156-170`);
  - a Triton cross-entropy that writes the gradient over the logits in place (`cross_entropy.py: 138-284`: an online
    max/sum pass, then a gradient pass);
  - grad_input = dZ·C written immediately (`:319-322`);
  - grad_weight accumulated in the parameter dtype unless fp32 accumulation is requested (`:111-123, 324-376`), skipped for a
    frozen head.
- **The gradient is computed during the forward.** The backward only rescales it when the upstream gradient is not 1
  (`:418-464`).
- **Devices.** cuda, npu, xpu, else cpu (`utils.py: 43-56`); there is no Apple GPU path.

### 7.4 Cut Cross-Entropy
- **The paper.** The loss splits into an indexed dot product (the correct-class logit) and a linear log-sum-exp
  (arXiv 2411.09009 §4.1-4.2). Blocks of the logit matrix are computed in on-chip memory and reduced into the LSE without
  global memory: Gemma 2 (2B)'s loss memory falls from 24 GB to 1 MB (abstract).
  - The backward recomputes the logits blockwise and needs no normalization constant, since softmax = exp(Z − LSE) (§4.3).
  - Gradient filtering skips blocks whose softmax entries all fall below ε = 2^-12, the smallest bf16 value not truncated;
    under 0.02% of entries are non-zero in the models measured, giving a 3.5x speedup without precision loss (§4.3).
  - Vocabulary sorting makes those blocks denser (§4.3).
- **The code.**
  - Forward: a grid over (token blocks × vocabulary blocks) (`cce_lse_forward.py: 209-210`); each tile accumulates
    E·Cᵀ in fp32 (`:69-86`) and takes its local LSE (`:123-124`).
  - The global fp32 LSE is merged under a spin lock taken with `atomic_cas` (`:130-139, 193`). The correct logit is written
    where target = column (`:112-119`).
  - Saved: e, c, bias, lse, targets, valids (`cce.py: 120`).
  - Backward: recomputes each logit tile (`cce_backward.py: 175`); forms exp(Z − lse) − onehot (`:197-208`); skips filtered
    blocks (`:212-217`).
  - dE and dC accumulate across tiles through lock-protected adds, optionally with Kahan summation (`:20-72, 259-312`);
    fp32 accumulation buffers are optional (`:391-395`).
  - The default ε = finfo(dtype).eps / 32 (`utils.py: 19-27`) is 2^-12 for bf16.
- **On macOS.** The default implementation is `torch_compile` (`linear_cross_entropy.py: 31-39`), and the CCE kernels raise
  "CCE does not support MacOS" (`:163-167`). The fallback keeps only the rows whose labels are not ignored
  (`torch_compile.py: 73-80`; `utils.py: 30-51`), then materializes their logits under `torch.compile` and runs
  `F.cross_entropy` in fp32 (`torch_compile.py: 18-49`).

### 7.5 MLX's fused cross-entropy (new in this version)
- **Where it is used.** `nn.losses.cross_entropy` calls `mx.fast.cross_entropy` on the GPU for class-index targets, the last
  axis and no label smoothing (`losses.py: 98-108`).
- **Forward** (`cross_entropy.h: 3-84`):
  - one threadgroup per row; each thread reads 4 consecutive logits a step (`defines.h: 17`);
  - online max and rescaled sum in fp32 (`:31-53`), then simd and threadgroup reductions (`:55-76`);
  - the loss is log(sum) + (max − z_y): the two same-magnitude logits are subtracted first (`:78-82`);
  - the threadgroup grows with the vocabulary up to the pipeline's maximum (`cross_entropy.cpp: 16-30`).
- **Backward** (`cross_entropy.h: 86-133`):
  - p = exp((z − z_y) − ℓ) is rebuilt from the saved loss instead of an LSE;
  - g·(p − onehot) is written over the logits when the input can be donated (`cross_entropy.cpp: 97-106`), after a device
    barrier so every thread has read z_y first (`cross_entropy.h: 110-112`).
- **The logits are still materialized.** The head is a separate product.
- **mlx-lm's trainer** computes logits at every position and masks the loss (`trainer.py: 88-101`).

### 7.6 For the M3 Max [I]
At the micro-batch shapes, every row supervised. Every line scales with the supervised fraction f once `logits_to_keep`
selects the rows.

| | 0.8B, 2 × 1,282 = 2,564 rows | 9B, 1 × 1,282 rows |
|---|---|---|
| head forward 2·N·H·V | 1.30 TFLOP → 102 ms at 12.8 TFLOP/s | 2.61 TFLOP → 204 ms |
| dE = dZ·C (frozen head: no dC) | 1.30 TFLOP → 102 ms | 2.61 TFLOP → 204 ms |
| logits, bf16 | 1.27 GB | 0.64 GB |
| transformers' loss traffic (fp32 upcast, log-softmax, backward, cast back) | ≈ 20 GB → ≈ 56 ms | ≈ 10 GB → ≈ 28 ms |
| a fused cross-entropy as MLX's (write, read, read + write in place, read by dE) | 6.4 GB → 18 ms | 3.2 GB → 9 ms |
| head weight, bf16 | 0.51 GB | 2.03 GB |
| Liger's default chunk | 16 rows, 161 chunks; the weight read twice a chunk: 164 GB → 0.46 s | 32 rows, 41 chunks: 167 GB → 0.47 s |
| chunks of 256 rows | 11 chunks: 11 GB of weight reads, under the 204 ms of products | 6 chunks: 24 GB |

**The chain.**
- The head's products are ~15% (0.8B) and ~13% (9B) of a micro-batch at the series' measured step times (10.98 s over 8
  micro-batches ≈ 1,370 ms; 50.9 s over 16 ≈ 3,180 ms). That matches the registration's 12-20% and 8-14%.
- A 16-32-row chunk has an arithmetic intensity of 16-32 FLOP/byte, under the M3 Max's balance of 39 (previous document
  §3.8), so Liger's rule turns the head into a bandwidth-bound loop.
- Chunks of ≥ 256 rows keep it compute-bound and bound the transient fp32 logits to 254 MB a chunk.
- Cut Cross-Entropy's form trades memory for up to one more product: the backward recomputes the logits (102 / 204 ms) and
  filtering trims the dE product. On a 128 GB machine the 1.27 GB of logits does not bind, so it is the later option.

The order:
1. supervised rows only (`logits_to_keep` + `shift_labels`);
2. chunks of ≥ 256 rows;
3. per chunk, the platform product, then a fused cross-entropy in MLX's form (online max/sum in fp32, gradient written over the
   chunk), then dE = dZ·C by the platform product;
4. no dC.

---

## 8. Matrix products on Apple GPUs

### 8.1 MLX's steel GEMM (v0.32.4)

| item | MLX | lines |
|---|---|---|
| device classes | 'g'/'p' small, 'd' Ultra, everything else (the 's' Max class, previous document §3.5) medium | `matmul.cpp: 91-172` |
| tile on the medium class | BM = BN = 64, BK = 16, WM = WN = 2 (4 simdgroups, 128 threads), for every dtype and shape | `matmul.cpp: 166-172, 388-392, 480` |
| fragments | 8 × 8 `simdgroup_matrix` in the accumulator type (float for bf16 and fp16): A and B are converted to fp32 when loaded from threadgroup memory; each simdgroup owns 32 × 32 (TM = TN = 4) | `mma.h: 46, 64, 453-483`; `transforms.h: 57-59` |
| staging | A and B tiles in threadgroup memory, rows padded by 16 bytes; one buffer: barrier, load, barrier, multiply per BK step | `gemm.h: 37-61, 97-116` |
| swizzle | off on this path (`swizzle_log = 0`) | `matmul.cpp: 452-457` |
| epilogue | D = α·A·B + β·C in fp32, by function constants | `steel_gemm_fused.h: 11-12, 165-199` |
| split-K | if not NAX, no batch, ⌈M/16⌉·⌈N/16⌉ ≤ 2,048 (Max and Ultra; 1,024 otherwise), K/16 ≥ 8 and K ≥ max(M, N); tiles 16 or 32 (by M, N < 40) × 16; partitions min(max(2, next_pow2(⌊K/16⌋ / (⌈M/32⌉·⌈N/32⌉))), 32); fp32 partials [P, M, N] plus an accumulate kernel | `matmul.cpp: 519-670, 957-992` |
| gemv | M = N = 1: a dot product; 2 ≤ M ≤ 15 on GPU generation ≥ 15 (M3 and later): a "wide" gemv that streams the matrix once per ≤ 5 vectors ("pre-M3 generations are limited by load issue rate rather than bandwidth"); min(M, N) = 1: gemv | `matmul.cpp: 1401-1456, 1606-1666` |

### 8.2 metal-flash-attention's GEMM notes (Apple9)
- **Blocks.** 32 × 32 × 8 with a single simdgroup per threadgroup on Apple9 and later, against 48 × 48 × 24-32 and 2 × 2 on
  Apple7/8 (`GEMMKernelDescriptor.swift: 22-36, 54-62, 168-173`).
- **Async copies** help Apple7/8 but "harm performance on Apple9 and later" (`:131-139`).
- **Registers.** BF16 operands are kept in BF16 registers from M3 on and widened to FP32 before it (`:147-164`).
- **Atomics.** Apple GPUs lack native fp32 atomics (`README.md: 11`).

### 8.3 What PyTorch 2.15 calls on the Apple GPU
- **`F.linear` forward** (every projection):
  - a single row, or Apple7/8 overflow shapes, go to `addmm`/`mm` (`Linear.mm: 157-170`);
  - contiguous operands go to `MPSNDArrayMatrixMultiplication`, with the weight described as transposed and no copy
    (`:38-70, 172-199`).
- **`F.linear` backward:** grad input = `at::mm(grad_output_2d, weight)` (`:279-302`); grad weight =
  `at::mm(grad_output_2dᵀ, input_2d.contiguous())` (`:304-339`).
- **`mm`:**
  - rank-1 shapes go to PyTorch's own gemv kernels (`LinearAlgebra.mm: 293-323, 1166-1169`);
  - a short list of cases go to a naive 16 × 16-thread Metal kernel (`:325-355`; conditions `:622-663`): integer types,
    non-contiguous outputs before macOS 26, complex inner sizes over 2,048, an fp16 skinny-shape fault in MPSGraph's
    "LORADOWN GEMV" kernel, Apple7/8 reductions over 2^15;
  - everything else goes to MPSGraph's `matrixMultiplicationWithPrimaryTensor` (`:1179-1208`).
- **`addmm`:** an MPSGraph graph of the product, × α and + β·C (`:1423-1481`).
- **[I]** On the M3 under macOS 26, every bf16 product of the training step runs in Apple's closed kernels.

### 8.4 For the M3 Max [I]
- **The rates tie.** Large bf16 products run at 12.8-12.9 TFLOP/s in PyTorch and 12.7-13.1 in MLX on this machine, so D5(c)
  builds no large GEMM.
- **fp32 fragments.** The measured simdgroup ceilings are 13.78 TFLOP/s (fp32 fragments) and 11.96 (bf16 fragments). MLX's
  bf16 GEMM reaches 12.7-13.1 because it multiplies in fp32 fragments (§8.1). A custom bf16 product on the M3 should
  therefore convert at load, as MLX does, rather than use bf16 fragments as metal-flash-attention's descriptor recommends for
  M3.
- **MLX's routing applied to the adapter shapes** (0.8B in_proj_qkv, d_in = 1,024, d_out = 6,144, N = 4,096 tokens, r = 32):

| product | M × N × K | MLX route by §8.1's rules |
|---|---|---|
| P = X_d·Aᵀ | 4,096 × 32 × 1,024 | regular 64 × 64 × 16 tile (K < max(M, N)); the 32-wide output half-fills the tile |
| Y += s·P·Bᵀ | 4,096 × 6,144 × 32 | regular, two BK steps |
| G = dY·B | 4,096 × 32 × 6,144 | split-K, 4 partitions |
| dB = s·dYᵀ·P | 6,144 × 32 × 4,096 | regular (K < M) |
| dA = s·Gᵀ·X_d | 32 × 1,024 × 4,096 | split-K, 8 partitions, 16 × 32 × 16 tiles |
| dX += s·G·A | 4,096 × 1,024 × 32 | regular, two BK steps |

- **What step 0(c) must measure.** PyTorch's routes for these shapes are hidden in MPSGraph. The registration's prediction is
  ≤ 30% of the large-product rate for the rank-32 products.
- **The design either way.** A fused adapter kernel (§6.5) uses split-K over tokens for dA (and dB when K ≥ M), fp32
  partials, and single-buffered loads without async copies (metal-flash-attention, MLX).

---

## 9. Long-sequence state passes

### 9.1 flash-linear-attention's state passes (`fla/ops/common/chunk_delta_h.py`)
- **Forward grid.** ⌈V/BV⌉·N·HV programs (`:726`); one program per value tile of a head walks all chunks in order.
- **The state tile.** The state is held as ⌈K/64⌉ fp32 blocks of [64, BV] in registers (`:101-115`).
- **Tuning.** BV ∈ {32, 64}, 64 only on Ada-class shared memory; 2-16 warps by GPU; 1-4 stages (`:25-37, 50-53`).
- **Storage.** Per-chunk states h [B, NT, HV, K, V] are stored in the input dtype (`:719-722`).
- **The backward dh pass.** The same grid (`:798`); BV ∈ {32, 64} with 2-4 warps (`:361-366`); dh per chunk in the input
  dtype, dh0 in fp32 (`:785-796`).
- **At 1 row × 16 heads, V = 128** **[I]**: 64 programs at BV = 32, 32 at BV = 64.

### 9.2 MLX's grids
Threadgroups of (32, 4, 1) — 4 simdgroups — in every case (`gated_delta_update.cpp: 155-157, 178-180, 207-209`). The chunk
is 8 when T > 8 on GPUs without the M5's neural accelerators (`:18-19`).

| kernel | grid | threadgroups per head |
|---|---|---|
| NAX, chunk 16 | (32, Dv/16, B·Hv) | 2 |
| chunk 8 | (32, Dv/8, B·Hv) | 4 |
| sequential | (32, Dv, B·Hv) | 32 |

### 9.3 flash-linear-attention's intra-card splitting (inference)
- **The summary.** Each subsequence is summarized as h_out = M·h_in + h_local, stored as [h_local | M] of shape [K, V + K]
  per value head (`fla/ops/cp/chunk_fwd.py: 59-63`).
  - M starts as the identity (`:247-248`).
  - Each chunk left-composes M ← (γ_last·I − K̃ᵀ·W)·M in fp32, "repeated bf16 casts accumulate error along the chain"
    (`:264-286`); K̃ is the keys decayed to the chunk end.
  - **[I]** This is F4 of the previous document rearranged:
    S_{c+1} = γ_last·S_c + K̃ᵀ·(U − W·S_c) = (γ_last·I − K̃ᵀ·W)·S_c + K̃ᵀ·U.
- **Merge.** A sequential composition gives each subsequence its entering state (`cp/chunk_merge.py: 30-34, 110-115`); then
  each subsequence runs the ordinary state pass (`intracard_cp.py: 8-13`).
- **Backward summary.** dh_in = dM·dh_out + dh_local, scanned in reverse with transposed chunk transforms
  (`cp/chunk_bwd.py: 58-62, 270-277`), used by the multi-device context parallelism for training (`cp/README.md`).
- **When it splits** (`intracard_cp.py: 132-158, 181-198`):
  - never under 8 chunks;
  - target splits = max(4, SMs // (2·heads));
  - subsequences of at least 128 chunks;
  - only sequences of at least 3 subsequence lengths are split;
  - at most 32 subsequences, "to control precision loss" (`backends/intracard.py: 25-26`);
  - only under `torch.inference_mode()` with `cu_seqlens` (`:60-64`).
- **Precision.** The affine chains run in TF32x3 on NVIDIA and IEEE fp32 on Ascend; other devices raise an error
  (`cp/README.md`).

### 9.4 For the M3 Max [I]
- **Value slicing.** At 1 row × 16 heads, value slicing gives 64 threadgroups at BV = 32 (MLX's chunk-8 count) and 128 at
  BV = 16, that is 1.6-3.2 per core.
  - Each extra threadgroup per head redoes the chunk's C × C work (KKᵀ, QKᵀ, the inverse). At C = 32 that is 0.54 MFLOP of
    the 4.47 MFLOP per chunk-head of the previous document's forward (§1.6): +36% FLOPs at 4 slices, +84% at 8.
  - At 16 heads the pass is latency-bound, so the extra FLOPs can be free.
  - Otherwise the C × C matrices can be computed once by a parallel kernel and read back (A and the masked QKᵀ, 8 KB per
    chunk-head at C = 32: 134 MB at 32,768 tokens).
- **Segment scans.**
  - A segment summary costs 2·C·K² + 2·K³ per chunk-head: 5.2 MFLOP at C = 32 (1.2x the forward) and 6.3 MFLOP at C = 64
    (0.54x of the 11.6 MFLOP forward).
  - With S segments the sequential chain falls from T/C chunk steps to about T/(S·C) + S, for 1.5-2.2x the total work.
  - flash-linear-attention's own rule on this machine (40 cores, 16 heads) gives target splits = max(4, 40 // 32) = 4 and
    would split only from 24,576 tokens at C = 64. So for the registered 4,096-16,384-token range, slicing is the lever and
    segment scans a later entry.
- **Precision.** The chains in fp32, which costs nothing on the M3: fp32 fragments are the fastest measured.
- **Double buffering (D3(a)).** The sources lean against threadgroup double buffers on Apple9: metal-flash-attention reports
  async copies harmful there, and MLX's GEMM loop is single-buffered (§8.1-8.2). Prefetching the next chunk's q, k, v tiles
  into registers while the current chunk computes is the form consistent with them.
- **State checkpoints (D3(a)).** MLX's M5 path keeps the state every 16 chunks and replays (previous document §3.1);
  flash-linear-attention stores one state per chunk or recomputes them.

---

## 10. Design → what the prior art says to do on the M3 Max

All entries are **[I]**, drawn from the sections named.

| design | part | what to do | from |
|---|---|---|---|
| D4 | convolution + SiLU | one kernel over the projection's [B, T, C] buffer in place, no transposes or copies; a lane holds 4-8 channels and slides over 32-64 time steps with the 3 previous inputs in registers (no threadgroup memory); fp32 accumulation, SiLU on the fp32 pre-activation, one rounding; head-aligned 128-channel tiles so the q/k l2 norm and the query scale fold into the epilogue, and q, k, v land in the rule's layout (9B: q and k once per key head). Backward: u recomputed in registers from x, dx gathered with one writer. dw only when trained, as fp32 per-tile partials and a second pass. Floor 0.71 / 0.47 ms forward + backward at the bench shapes | §2 (causal-conv1d channel-last; flash-linear-attention; MLX; PyTorch 2.15) |
| D4 | gated RMSNorm (128 per head) | one row per simdgroup (4 elements a lane, `simd_sum` only), 4-8 rows per threadgroup; fp32 math with transformers' rounding points (x̂ rounded, × w in bf16, × silu(z) in fp32, rounded), which MLX's composition reproduces; save only the inputs, recompute r; backward writes dx and dz in one pass; dw skipped when frozen. Floor 0.38 ms at both sizes | §3 (flash-linear-attention, Liger, MLX) |
| D4 | zero-centered RMSNorm (two a layer; also the full-attention q/k norms) | fp32 throughout with one rounding (Liger's gemma mode with offset 1, which is transformers' form); MLX's two-level row sum; fold the preceding residual add into the norm (store the sum as the next residual), as flash-linear-attention's residual path does; recompute r in the backward; dw skipped when frozen. Floor 0.24 / 0.47 ms for the two | §3 |
| D4 | SwiGLU | elementwise, 8 bf16 per lane; silu in fp32 rounded before the product (transformers, Liger, unsloth); backward recomputes silu and writes de, du in place (Liger), and writes h when the down projection carries an adapter (unsloth). Floor 0.66 / 1.13 ms | §4 |
| D4 | beta and g | no separate kernels: raw a and b read in the rule's prologue, β and g in fp32 (softplus as log1p(exp) with torch's threshold 20), gradients folded back in D2's epilogue (flash-linear-attention's `use_gate_in_kernel`, `use_beta_sigmoid_in_kernel`); dA_log, ddt_bias skipped when frozen | §5 |
| D5 | the LoRA adapter path | base GEMMs stay in PyTorch; one fused adapter kernel forward (mask applied as X is loaded, P = X_d·Aᵀ kept, Y += s·P·Bᵀ as one read-modify-write) and one backward (G and dB from one pass over dY, dA with the mask re-applied on load, dX += s·mask ⊙ (G·A)/(1 − p)); masks drawn by torch's own dropout for exact parity (unsloth); fp32 fragments; split-K over tokens for the weight gradients. Predicted traffic 2.8 / 4.1 ms a layer against about 10 / 15 ms for PEFT's passes | §6, §8.4 |
| D5 | the products | no custom large GEMM (PyTorch and MLX tie at 12.7-13.1 TFLOP/s); a custom product only if step 0(c) finds a shape carrying ≥ 5% of the step below 80% of MLX, then MLX's medium-class tile (64 × 64 × 16, 2 × 2 simdgroups), fp32 fragments for bf16 data, single-buffered loads, split-K when M·N is small and K ≥ max(M, N) | §8 |
| candidate (outside D3-D5, proposed only) | the loss head | (1) head and loss at supervised rows only (`logits_to_keep` as a boolean mask + `shift_labels`), no kernel; (2) row chunks of ≥ 256 (never Liger's 16-32-row rule at this vocabulary); (3) a fused cross-entropy per chunk in MLX's form: online max and sum in fp32, loss as log(sum) + (max − z_y), gradient written over the chunk's logits; dE by the platform product; (4) no dC for the frozen head; Cut Cross-Entropy's recompute only if memory binds. Predicted: transformers' ≈ 56 / 28 ms of loss traffic down to ≈ 18 / 9 ms, then everything × the supervised fraction | §7 |
| D3 | the training shapes | the previous document's chunk sizes and state storage; register prefetch of the next chunk's tiles instead of threadgroup double buffers, which metal-flash-attention reports harmful on Apple9 and MLX does not use | §9.4, §8.1-8.2 |
| D3 | long sequences (1 row, 4,096-32,768 tokens) | split each head's value columns over 4-8 threadgroups (BV = 32-16: 64-128 threadgroups for 16 heads on 40 cores); either redo the chunk's C × C work per threadgroup (+36-84% FLOPs on a latency-bound pass) or compute it once in a parallel kernel and read it back; keep every chain in fp32; affine segment summaries (S_out = M·S_in + h, M = Π(γ_last·I − K̃ᵀ·W)) only beyond about 16-24k tokens or when the profile shows the GPU idle, with the backward's transposed summaries for training | §9 |
