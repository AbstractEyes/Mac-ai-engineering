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
    layer.configure(RMSNORM=True, CONV=True)          # or switch parts at any time; layer.configured(...) inside a block

A replaced method reads its part's setting at every call; a part that is off, an input on another device or in another dtype,
a shape the kernels do not take, a cache (generation) or packed rows: transformers' own code runs. The methods are replaced only
while the source of what they replace hashes as in transformers 5.22.1 (`macai.layer._qwen3_5.SOURCES`); on a mismatch the
original stays and a warning says which. Qwen3.5-MoE and Qwen3-Next are not touched by these parts (the rule is).

## Parity
`tests/test_layer.py`: each part's forward and every gradient against float64 on the CPU, beside transformers' own path against
the same float64 reference, at the real head counts and widths of Qwen3.5-0.8B and 9B, odd lengths, more than one row, float32,
bfloat16 and float16 inputs: the part's error at most twice the PyTorch path's; the installation (bits unchanged with every
part off, the submodules and dropout draws in the same order with every part on, a source mismatch leaves the original); and
whole decoder layers of both kinds with LoRA at the bench shapes against a float64 run of the same layer.

## Measured
See [series_layer.md](series_layer.md), design D4.
