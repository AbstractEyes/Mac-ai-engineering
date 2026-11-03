# The rest of the layer on Apple GPUs (macai.layer)

## Why
A Qwen3.5 decoder layer is more than its matrix products and its gated delta rule: around them sit two zero-centered RMSNorms,
the linear-attention layer's depthwise convolution with its SiLU, beta and g, the gated RMSNorm, the SwiGLU, and in the
full-attention layers the query / key norms. transformers runs each of these as a chain of small PyTorch operations, each a
full pass over memory, forward and backward. On an M3 Max the step profile of a LoRA training step (series step 0,
[series_layer.md](series_layer.md)) measured them at 14.9% of a micro-batch at 0.8B and 9.0% at 9B, every norm at 17-20x the
time its least memory traffic needs, the convolution path at 8.5-12.5x.

## What it does
Each part is one Metal kernel forward and one backward, compiled through `torch.mps.compile_shader`, reading its inputs where
the layer left them and writing its outputs where the next part reads them:

| setting (`macai.layer`) | replaces | forward | backward |
|---|---|---|---|
| `RMSNORM` | `Qwen3_5RMSNorm.forward` (the input and post-attention norms, the full-attention q / k norms, the final norm) | y = x rsqrt(mean(x^2) + eps) (1 + w) in float32, one rounding; any row stride (the query norm reads every other half of the projection's rows in place) | dx = r ((1 + w) dy - x r^2 <x, (1 + w) dy> / D) from the saved x (r recomputed); dw only when the weight trains |
| `CONV` | the convolution path inside `Qwen3_5GatedDeltaNet.forward` | the depthwise causal convolution (kernel 4) and SiLU over the input projection's [B, T, C] output in place, float32 accumulation, SiLU on the float32 pre-activation; q and k written as [B, T, H_v, 128] (each key head once for every value head it serves: the 9B repeat without a copy), v as float32 (the rule computes in float32) | the pre-activation recomputed in registers from the saved projection output; dx written in the projection's own [B, T, C] layout, one writer an element; dw only when the weight trains |
| `GATED_NORM` | `Qwen3_5RMSNormGated.forward` | one simdgroup a 128-wide row, transformers' rounding points (x_hat rounded, times w rounded, times silu(z) in float32, rounded) | dx and dz in one pass, float32, r recomputed; dw only when the weight trains |
| `GATES` | beta and g inside `Qwen3_5GatedDeltaNet.forward` | beta = sigmoid(b), g = -exp(A_log) softplus(a + dt_bias), both float32 (softplus with PyTorch's threshold of 20) | db, da in one pass; A_log's and dt_bias's gradients only when they train |
| `SWIGLU` | `Qwen3_5MLP.forward` (the projections stay module calls) | silu(gate) rounded, times up, rounded (transformers' rounding) | dgate, dup in one pass, silu recomputed |

Weight gradients, when a weight trains, are summed as float32 partials and a second pass: there are no float atomics anywhere.
The projections stay PEFT's module calls in transformers' order, so LoRA (its dropout draws included) is unchanged; the rule is
called through the modeling module's own name for it, so `macai.patch`'s rule (or any other) runs.

## Install
    import macai.patch
    macai.patch.install()       # the rule and the layer methods (each method only while its source matches)
    from macai import layer
    layer.configure(SWIGLU=False)                     # every part is on by default; switch any off at any time,
                                                      # or layer.configured(...) inside a block

A replaced method reads its part's setting at every call; a part that is off, an input on another device or in another dtype,
a shape the kernels do not take, a cache already holding the layer's states (decoding) or packed rows: transformers' own code
runs. The methods are replaced only while the source of what they replace hashes as in transformers 5.22.1
(`macai.layer._qwen3_5.SOURCES`); on a mismatch the original stays and a warning says which. Qwen3.5-MoE and Qwen3-Next are not
touched by these parts (the rule is).

## Parity
`tests/test_layer.py`: each part's forward and every gradient against float64 on the CPU, beside transformers' own path against
the same float64 reference, at the real head counts and widths of Qwen3.5-0.8B and 9B, odd lengths, more than one row, float32,
bfloat16 and float16 inputs: the part's error at most twice the PyTorch path's; the installation (bits unchanged with every
part off, the submodules and dropout draws in the same order with every part on, a source mismatch leaves the original); and
whole decoder layers of both kinds with LoRA at the bench shapes against a float64 run of the same layer.

## Measured
Every part is on by default since series design D4 admitted it ([series_layer.md](series_layer.md), D4). On an M3 Max (PyTorch
2.15, bf16 under autocast, frozen layers at the bench shapes 4 x 1,024 / 2 x 1,024 tokens, forward + backward against each part's
PyTorch path, Metal events, interleaved rounds):

| part | 0.8B | 9B | kernel against its least memory traffic |
|---|---|---|---|
| zero-centered RMSNorm (input norm; q_norm on its strided rows; k_norm) | 16.1x; 17.1x; 7.2x | 18.9x; 18.0x; 8.4x | 1.02-1.21x (k_norm 2.2-2.7x) |
| convolution path (to the rule's inputs as its kernels read them) | 9.3x | 9.9x | 1.38-1.76x |
| gated RMSNorm | 13.7x | 13.1x | 1.30-1.37x |
| beta and g | 1.60x | 1.60x | launch-bound (~0.08 ms) |
| SwiGLU | 1.66x | 1.69x | 1.08-1.13x |
| a whole linear-attention layer | 61.9 -> 42.2 ms | 183.2 -> 159.6 ms | |
| a whole full-attention layer | 63.4 -> 53.0 ms | 182.4 -> 166.1 ms | |

In LoRA training of the Qwen3.5 picture readers (rank 32, the Phase 4 recipe, the rule's kernels with the recomputed states):
a step 10.98 -> 9.18 s at 0.8B (2 x 8 sequences) and 51.1 -> 46.2 s at 9B (1 x 16); the process's peak footprint 46.3 -> 41.6 GB
and 64.4 -> 60.4 GB; the loss like the reference implementation's. The convolution path in the projection's own layout also
takes away a slow LoRA backward of the input projection on the transposed gradient transformers' path hands it.

Training forwards carry a fresh cache (transformers' Trainer sets the model's `use_cache` from its own arguments); the replaced
linear-attention forward takes the parts there and fills the cache as transformers does.
