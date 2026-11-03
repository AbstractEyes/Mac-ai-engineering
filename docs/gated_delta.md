# The gated delta rule on Apple GPUs

## Why
Qwen3.5 (and Qwen3-Next) mix full attention with a linear attention, the gated delta rule, in three of every four layers. On
NVIDIA GPUs it runs on fused kernels (flash-linear-attention, causal-conv1d); those are CUDA-only, so on a Mac transformers
falls back to its reference PyTorch implementation, a chunked algorithm written as many small tensor operations.

Measured on an M3 Max (PyTorch 2.15, bf16 model, LoRA-style training: the layer's weights frozen, the gradient flowing to its
input), one linear-attention decoder layer of Qwen3.5-0.8B at 4 sequences of 1,024 tokens, forward + backward:

| part | ms |
|---|---|
| the gated delta rule (every input's gradient) | 64 |
| the input projections | 13 |
| the short convolution | 5 |
| the gated output norm | 6 |
| the output projection | 3 |
| the whole layer (with the MLP and norms) | 113 |

So the rule itself is about 60-70% of the layer, and three of every four layers are of this kind.

## The kernel (0.1.0)
`macai.gated_delta.chunk_gated_delta_rule` has transformers' signature; `macai.patch.install()` swaps it into the Qwen3.5
modeling modules. It runs the recurrence one token at a time:

    a_t = exp(g_t);  S~ = a_t S_{t-1};  u = k_t^T S~;  d = beta_t (v_t - u);  S_t = S~ + k_t d^T;  o_t = S_t^T q_t

with the state S [DK, DV] split by columns: one simdgroup owns one column (32 lanes x DK/32 rows), a threadgroup of 32
simdgroups owns 32 columns, the whole forward runs in registers. The backward walks blocks of 8 tokens from the end: it reloads
the state saved at the block's start, recomputes the block keeping each step's input state in registers, and steps back through
it; the gradients of q, k, g and beta (sums over the DV columns) are reduced in threadgroup memory and added across threadgroups
with float atomics. All arithmetic is float32.

Parity: the step-by-step recurrence to 1e-8 (outputs) and ~1e-6 (gradients); transformers' reference within 1e-4 (float32) and
2e-2 (bf16) (tests/test_gated_delta.py).

Speed (bench/bench_gated_delta.py, M3 Max):

| shape | rule fwd / fwd+bwd, reference | rule, Metal 0.1.0 | layer fwd+bwd, reference | layer, Metal 0.1.0 |
|---|---|---|---|---|
| Qwen3.5-0.8B, 4 x 1,024 | 10.8 / 64.1 ms | 11.2 / 57.0 ms | 112.7 ms | 99.3 ms (1.13x) |
| Qwen3.5-9B, 2 x 1,024 | 10.7 / 64.4 ms | 10.9 / 46.9 ms | 252.4 ms | 213.7 ms (1.18x) |

## 0.2.0: one threadgroup a head for the forward
The forward now runs a whole head in one threadgroup (8 lanes a column, DK / 8 rows each, cross-lane sums by three shuffle rounds,
B x H threadgroups): 1.8x the reference's forward. The backward stays 0.1.0's column kernel (a one-threadgroup-a-head backward,
which recomputes each block's states from its saved start, measured slower on the M3 Max and is kept behind KERNEL_BWD = "head");
both forwards save the same state layout, so either backward reads it. Parity: 8 tests pass (every forward/backward pairing).

| shape | rule fwd / fwd+bwd, reference | rule, Metal 0.2.0 | layer fwd+bwd, reference | layer, Metal 0.2.0 |
|---|---|---|---|---|
| Qwen3.5-0.8B, 4 x 1,024 | 10.7 / 64.2 ms | 6.1 / 40.4 ms (1.77x / 1.59x) | 112.9 ms | 91.7 ms (1.23x) |
| Qwen3.5-9B, 2 x 1,024 | 10.9 / 64.5 ms | 6.0 / 39.6 ms (1.81x / 1.63x) | 250.5 ms | 204.9 ms (1.22x) |

## Series phase 3: the chunked kernels (now the default)
The chunked (WY / UT) algorithm on 8 x 8 simdgroup matrices, float32 throughout, chunks of 16 tokens: a per-chunk kernel for the
decayed Gram matrices and the triangular inverse, a forward state pass and a backward state pass (one threadgroup a head, each
chunk's keys and queries staged in threadgroup memory), and a per-chunk backward with one writer for every gradient (no atomics;
bitwise identical run to run). The drop-in's query / key normalization runs as one fused pass each way. Design, every variant
tried and the measurements: [series_phase3.md](series_phase3.md). Parity: the phase 2 harness, 203 tests.

| 0.8B (4 x 1,024) / 9B (2 x 1,024), ms | reference | 0.2.0 | chunked |
|---|---|---|---|
| the rule's forward with states (kernels) | | 4.54 / 4.66 | 3.09 / 3.12 |
| the rule's backward (kernels) | | 26.2 / 26.3 | 4.40 / 4.37 |
| the rule forward + backward through the drop-in | 65.3 / 65.1 | 37.6 / 37.8 | 9.86 / 9.72 |
| a whole linear-attention layer, forward + backward (wall clock) | 113.7 / 254.6 | 93.1 / 206.8 | 58.6 / 177.7 (1.94x / 1.43x) |

## The kernels' memory (the memory series: the chunk states recomputed, now the default)
Two settings in `macai.gated_delta.op` choose how the chunked kernels hold their buffers in training. Every output and gradient is
the same bits under each choice (`tests/test_memory.py`); only the memory and the time differ. Design, tests and the training
runs: [series_memory.md](series_memory.md).

| setting | default | what it does | measured (M3 Max) |
|---|---|---|---|
| `RECOMPUTE_STATES` | `True` | the training forward keeps only its inputs (q, k, v, g, beta, S_0); the backward rebuilds G, A, M, KG, the chunk states, Vn and Xu with the forward's own kernels (without the output's work), then runs the backward passes | one forward keeps 116.8 MiB instead of 413 (4 x 1,024 tokens x 16 heads); the rule's forward + backward 9.94 -> 11.71 ms (+17.6%); in LoRA training of Qwen3.5 no measurable cost a step, and the process's peak footprint 0.8B 46.3 GB (stored states 57.7, the reference implementation 48.65), 9B 64.4 GB over the first 25 steps (stored 79.5, the reference 67.8) |
| `SIZE_CLASSES` | `False` | every buffer whose size follows the sequence length is cut from a buffer of the smallest of 4 sizes a doubling that holds it | +0.6% on the rule; in training the allocator's pool as with exact sizes (no memory gained) |

Both off is Phase 3's stored path: the forward keeps every chunk's state (82-164 MiB a layer at 650-1,300 tokens) for the
backward; spread over the Metal allocator's shared 1 GiB heaps, those buffers keep the heaps from being released between the
forward and the backward, and the process held 17-19% more than with the reference implementation although its live tensors
were fewer.

## Next
- The per-chunk backward is bandwidth-bound on the chunk states (now recomputed in the backward), and double-buffered staging
  in the state passes: series_phase3.md, "Next".
- The layer's other parts on this GPU: the input projections, the gated norm, the short convolution; at 9B the rule is now
  about 11 of the layer's 178 ms.
- Packed rows (cu_seqlens), refused until supported.
