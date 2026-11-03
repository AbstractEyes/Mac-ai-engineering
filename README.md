# Mac AI engineering

Metal kernels and settings for training and running machine-learning models on Apple Silicon, built and measured on one
machine: a MacBook Pro with an M3 Max (40-core GPU, 128 GB unified memory), macOS 26, PyTorch 2.15.

**Status: prototype.** 0.1.0 and 0.2.0 are a first recurrent kernel, the baseline of an engineering series: prior art, GPU
instruments, a full correctness harness and the chunked kernels on simdgroup matrices (phases 0-3) are done, and phase 4 ran
them in real LoRA training of Qwen3.5 0.8B and 9B: the same loss as the reference, steps 1.375x / 1.22x faster
([docs/series_phase4.md](docs/series_phase4.md)). Their larger allocator footprint is fixed: the chunk states are now
recomputed in the backward (the same bits), and the process's footprint sits under the reference's at the same speed
([docs/series_memory.md](docs/series_memory.md)). The rest of the design series is not done. Treat the numbers below as
measurements of a prototype.

## What is here
- `macai.gated_delta`: the gated delta rule (the linear attention in three of every four layers of Qwen3.5 / Qwen3-Next) as
  Metal kernels with a hand-written backward, a drop-in for transformers' reference implementation, which is all a Mac gets
  (the fast kernels upstream are CUDA-only). The default is the chunked algorithm on simdgroup matrices (series phase 3): on
  an M3 Max the rule's forward + backward 5.9-6.4x the reference through the drop-in, a whole Qwen3.5 linear-attention layer
  1.94x (0.8B) / 1.43x (9B), forward + backward at 4 / 2 x 1,024 tokens, wall clock (0.2.0's recurrent kernels: 1.6x and
  1.22-1.23x).
  Design and measurements: [docs/gated_delta.md](docs/gated_delta.md), [docs/series_phase3.md](docs/series_phase3.md).
- `macai.patch`: installs that kernel into transformers' Qwen3.5-family modeling code, so existing training code runs it.
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
