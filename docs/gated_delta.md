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

## Next
- The backward (~34 ms of the rule's 40): a chunked (WY-form) algorithm on simdgroup matrices, the shape GPUs run best, in place
  of one token at a time.
- The layer's other parts on this GPU: the input projections (13 ms), the gated norm (6 ms), the short convolution (5 ms).
