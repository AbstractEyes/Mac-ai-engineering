# The LoRA adapter path (macai.lora)

## Why
PEFT's LoRA layer adds `lora_B(lora_A(dropout(x))) * scaling` to its base layer's output. In the readers' configuration (a bf16
base, the adapters in float32 as PEFT keeps them by default, bf16 autocast) each adapted projection then runs, besides its two
rank-32 products: a float32 copy of x, dropout on that copy, autocast's cast of the dropped copy back to bf16, the B product's
output written and read again, a scaled copy and an add; and in the backward the same chain reversed (the scaled gradient, the
input gradient's casts to float32 and back, dropout's backward in float32, the add to the base's input gradient). Each is a pass
over a token-wide tensor; the products themselves are small. The step profile ([series_layer.md](series_layer.md)) measured this
work at 154.8 ms of an 839.4 ms Qwen3.5-0.8B micro-batch (18.4%) after the series' earlier designs.

## What it does

| setting (`macai.lora`) | replaces | what runs |
|---|---|---|
| `FUSED` | PEFT's `peft.tuners.lora.layer.Linear.forward` (vanilla LoRA) | the base layer's own module call, then one autograd function for the adapter path: the dropout drawn by torch's own dropout call; P = x_d A^T by PyTorch's product; Y = R + s P B^T by a Metal kernel (the base output R read and Y written once); in the backward G = s dY B, dB = s dY^T P and dA = G^T x_d by PyTorch's products (the scale applied to their small results), the adapter's input gradient M (G A) / (1 - p) by a Metal kernel with the mask and the scale in its epilogue |

The masks: torch's own dropout call (`torch.native_dropout`, the call `nn.Dropout` makes on the Apple GPU), so the fused path and
PEFT draw the same mask from the same generator state, and the training's random stream is the one it was. Under autocast with
float32 adapters and x in the compute dtype the dropout runs on x itself: the same mask as on PEFT's float32 copy, and the
dropped bf16 / fp16 values bit for bit PEFT's dropped copy after autocast's cast (tested). Otherwise x is cast to the adapters'
dtype for the dropout, as PEFT casts it.

The compute dtype is autocast's when autocast is on for the device, else the adapters' own (PEFT's two cases). The kernels'
products of rank r accumulate in float32 on simdgroup matrices and round once: the output where PEFT rounds the B product, the
scaled copy and the sum (three times under autocast), the input gradient where PEFT rounds the product and the cast back. With a
power-of-two scale (alpha / r = 2 in the readers' recipe) the products PyTorch runs see the same bits as PEFT's, so the adapters'
weight gradients of a layer are PEFT's bit for bit. Float32 outputs (x in float32 without autocast) take four fma chains summed as
a tree in the kernels, which keeps them within the platform product's float32 error.

## Install
    import macai.patch
    macai.patch.install()          # the rule, the layer parts, the training path and the LoRA adapter path
    from macai import lora
    lora.configure(FUSED=False)                      # switch it off at any time, or lora.configured(...) in a block
    lora.stats()                                     # the fused calls; PEFT's own forward by reason

The replaced method reads the setting at every call and runs PEFT's own forward when it is off, and for anything but vanilla LoRA
in the form the kernels take: more or fewer than one active adapter, an adapter not in lora_A, adapters disabled, merged
weights, a LoRA variant (DoRA and the others), a LoRA bias, input casting disabled, a base other than nn.Linear (or
fan_in_fan_out), a dropout other than nn.Dropout (not in place) or nn.Identity, arguments beside x (adapter_names, the variants'
keyword arguments), tensors off the Apple GPU or in another dtype than float32 / bfloat16 / float16, a rank that is not a
multiple of 8 up to 64, widths that are not multiples of 8, an empty input. Each such call is counted by its reason. The method
is replaced only while its source hashes as in PEFT 0.21.2 (`macai.lora._peft.SOURCES`); otherwise PEFT's stays and a warning
says so. The replacement applies to every model PEFT adapts, not only Qwen3.5. The properties PEFT's forward reads from its base
tuner layer (the active adapters, disabled, merged, the input cast) are read through PEFT's own attributes; their behaviour is
checked by the tests against PEFT's own forward, not by a source hash.

## Parity
`tests/test_lora.py`: one PEFT LoRA layer at the real widths of Qwen3.5-0.8B's and 9B's adapted projections, rows and lengths
(1, 1), (2, 7), (3, 33), (2, 129), dropout on: x in bf16 and fp16 under autocast, in float32 and bf16 without; the output and the
gradients to x, A and B, the fused path and PEFT's both against float64 on the CPU applying the same mask (the bar: the fused
path's error at most twice PEFT's); with the base weight zero, the adapter term alone at every width; the masks from the same
generator state equal; the weight gradients bit for bit PEFT's under bf16 autocast; every fallback condition counted and giving
PEFT's bits; whole decoder layers of both kinds with LoRA and dropout on, both paths against a float64 run replaying the same
masks. `bench/lora_probe.py` repeats the checks inside a real training run of each model.

## Measured
On by default since series design D5 admitted it ([series_layer.md](series_layer.md), D5). On an M3 Max (PyTorch 2.15,
transformers 5.22.1, PEFT 0.21.2), LoRA training of the Qwen3.5 picture readers (rank 32, alpha 64, dropout 0.05, the adapters in
float32, bf16 autocast) with the rest of macai on:

| | Qwen3.5-0.8B, 2 x 8 sequences | Qwen3.5-9B, 1 x 16 |
|---|---|---|
| the adapter path alone, every adapted projection at the micro-batch's token count, forward + backward a micro-batch | 125.0 -> 58.7 ms (2.13x) | 231.2 -> 104.9 ms (2.20x) |
| the adapter path in a training step (the step profile) | 154.8 -> 83.3 ms a micro-batch (1.86x) | 256.5 -> 128.7 ms (1.99x) |
| the step, against the same configuration without it | 6.52 -> 6.00 s (-8.0%) | 43.1 -> 40.9 s (-5.1%) |
| the process's peak footprint | 29.2 -> 31.3 GB | 50.8 -> 49.3 GB (25 steps) |
| the loss | like the reference by Phase 4's readings (100 steps); steps 1-2 the same bits | steps 1-2 the same bits; over 25 steps .0015 from the reference run |

The kernels: the up kernel at its widest outputs within 1.2-1.4x its memory traffic; the narrow outputs and the down kernel take
0.04-0.07 ms a call, bound by the launch and the per-threadgroup setup. The reductions left to PyTorch's products run at about a
third of the large-product rate at these shapes.
