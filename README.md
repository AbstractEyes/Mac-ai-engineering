# Mac AI engineering

Metal kernels and settings for training and running machine-learning models on Apple Silicon, built and measured on one
machine: a MacBook Pro with an M3 Max (40-core GPU, 128 GB unified memory), macOS 26, PyTorch 2.15.

**Status: prototype.** 0.1.0 and 0.2.0 are a first recurrent kernel, the baseline of an engineering series: prior art, GPU
instruments, a full correctness harness and the chunked kernels on simdgroup matrices (phases 0-3) are done, and phase 4 ran
them in real LoRA training of Qwen3.5 0.8B and 9B: the same loss as the reference, steps 1.375x / 1.22x faster
([docs/series_phase4.md](docs/series_phase4.md)). Their larger allocator footprint is fixed: the chunk states are now
recomputed in the backward (the same bits), and the process's footprint sits under the reference's at the same speed
([docs/series_memory.md](docs/series_memory.md)). The rest of the layer around the rule (the RMSNorms, the short convolution,
beta and g, the gated norm, SwiGLU) now runs as Metal kernels too, on by default (`macai.layer`): a LoRA training step of
Qwen3.5 10.98 -> 9.18 s at 0.8B and 51.1 -> 46.2 s at 9B against the kernels before, with a smaller footprint and the
reference's loss ([docs/series_layer.md](docs/series_layer.md)). The training path around the layers is lighter too, on by
default (`macai.training`): the LM head and the loss run only at the positions that carry a label, and a frozen vision tower's
output is kept per picture after the first epoch: a step 9.18 -> 7.51 s at 0.8B (46.2 -> 42.9 s at 9B), and 6.29 s (37.5 s) from
the second epoch on, the same loss within float32 rounding. PEFT's LoRA layers run their adapter path fused, on by default
(`macai.lora`): the dropout drawn by torch's own call (PEFT's masks), the passes around the rank-32 products folded into two Metal
kernels: the adapter path about twice as fast, a step 6.52 -> 6.00 s at 0.8B (43.1 -> 40.9 s at 9B) with the reference's loss.
The rule's own chunking is tuned too, on by default (series design D3): forwards that keep no states run at chunks of 8 tokens, and
the state passes split a head's value columns over two threadgroups where the batch's heads leave GPU cores idle: the rule's
training forward 1.21x at the readers' micro-batches, a long single-row prefill 1.23-1.25x at 16 heads (1.17-1.18x at 32), the
gradients' bits unchanged. Treat the numbers below as measurements of a prototype.

## What is here
- `macai.gated_delta`: the gated delta rule (the linear attention in three of every four layers of Qwen3.5 / Qwen3-Next) as
  Metal kernels with a hand-written backward, a drop-in for transformers' reference implementation, which is all a Mac gets
  (the fast kernels upstream are CUDA-only). The default is the chunked algorithm on simdgroup matrices (series phase 3): on
  an M3 Max the rule's forward + backward 5.9-6.4x the reference through the drop-in, a whole Qwen3.5 linear-attention layer
  1.94x (0.8B) / 1.43x (9B), forward + backward at 4 / 2 x 1,024 tokens, wall clock (0.2.0's recurrent kernels: 1.6x and
  1.22-1.23x).
  Design and measurements: [docs/gated_delta.md](docs/gated_delta.md), [docs/series_phase3.md](docs/series_phase3.md).
- `macai.layer`: the rest of Qwen3.5's decoder layers as Metal kernels (series design D4): the zero-centered RMSNorm, the
  linear-attention layer's short convolution with its SiLU (reading the projection's output in place and writing the rule's
  inputs), beta and g, the gated RMSNorm and SwiGLU, each one pass forward and one backward, at their parity bars against float64:
  7-19x their PyTorch paths (beta / g and SwiGLU 1.6-1.7x); a whole Qwen3.5 linear-attention layer 1.47x (0.8B) / 1.15x (9B)
  faster again. Each part is a setting. Design and measurements: [docs/layer.md](docs/layer.md),
  [docs/series_layer.md](docs/series_layer.md).
- `macai.training`: the training path around the layers (series designs D8, D9), no kernels: while a Qwen3.5 model trains with
  labels, the LM head and the loss run only at the positions whose next token carries a label (transformers' boolean
  `logits_to_keep` and the loss's `shift_labels`; the model's own loss and reduction); while its vision tower is frozen, each
  picture's tower output is kept in CPU memory, keyed by the exact bytes of its pixels, and served from the second epoch on
  through the model's own precomputed-encoder input (the same bits as a recompute). Each is a setting. Design and measurements:
  [docs/training.md](docs/training.md), [docs/series_layer.md](docs/series_layer.md).
- `macai.lora`: PEFT's vanilla LoRA layer with its adapter path fused (series design D5): the mask from torch's own dropout call
  (the same masks as PEFT's from the same generator state), PyTorch's products for the rank-32 reductions, two Metal kernels for
  the expanding products with the base output's add (forward) and the mask and the dropout scale (backward) as they write; the
  forward and every gradient at or under twice PEFT's own error against float64, the weight gradients PEFT's bits under bf16
  autocast; the adapter path 2.1x (0.8B) / 2.2x (9B) faster. A setting; anything but vanilla LoRA runs PEFT's own forward. Design
  and measurements: [docs/lora.md](docs/lora.md), [docs/series_layer.md](docs/series_layer.md).
- `macai.patch`: installs the rule, the layer parts and the training path into transformers' Qwen3.5-family modeling code, and
  the fused adapter path into PEFT (each replaced method only while its source matches the version it was written against), so
  existing training code runs them.
- `macai.configure`: the settings measured for long PyTorch training on the Apple GPU (allocator watermarks, thread counts).

## Install
    pip install -e .            # or: pip install "mac-ai-engineering @ git+https://github.com/AbstractEyes/Mac-ai-engineering@v0.2.0"

    import macai.patch
    macai.patch.install()       # before building the model

## Measured on the M3 Max
- PyTorch's attention backward on the Apple GPU keeps the whole sequence-by-sequence score matrix, and its allocator keeps the
  buffers of every new batch shape: long LoRA runs of Qwen3.5 grew past 96 GB until the allocator's watermarks were set
  (`macai.configure`).
- Gradient checkpointing costs 1.27x in step time for Qwen3.5 0.8B and 9B LoRA training; with 128 GB it can be off for short
  sequences. With sequences up to 1,282 tokens and the watermarks of `macai.configure`, LoRA training with it off fits only at 2
  sequences a micro-batch (0.8B) or 1 (9B), with the allocator's cache emptied each micro-batch (`torch_empty_cache_steps=1`):
  the allocator's pool, fragmented by changing lengths, outgrows the high watermark although the live tensors fit
  ([docs/series_phase4.md](docs/series_phase4.md)).
- MLX and PyTorch train these models within 10% of each other on this chip; MLX's fused gated-delta kernel needs a newer GPU
  generation for its chunked form.
- PyTorch's `PYTORCH_MPS_PREFER_METAL` and `PYTORCH_MPS_FAST_MATH` made a small byte-level model's training 12-18% slower.

## Tests and benchmarks
    pytest                                  # parity on small shapes (needs an Apple GPU)
    python bench/bench_gated_delta.py       # the kernel against the reference at Qwen3.5 training shapes
    python bench/bench_layer.py             # the layer parts against their PyTorch paths, and whole layers, interleaved
