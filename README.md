# Mac AI engineering

Metal kernels and settings for training and running machine-learning models on Apple Silicon, built and measured on one
machine: a MacBook Pro with an M3 Max (40-core GPU, 128 GB unified memory), macOS 26, PyTorch 2.15.

## What is here
- `macai.gated_delta`: the gated delta rule (the linear attention in three of every four layers of Qwen3.5 / Qwen3-Next) as
  Metal kernels with a hand-written backward, a drop-in for transformers' reference implementation, which is all a Mac gets
  (the fast kernels upstream are CUDA-only): the rule 1.6x the reference forward + backward, a whole Qwen3.5 linear-attention
  layer 1.22x, on an M3 Max. Design and measurements: [docs/gated_delta.md](docs/gated_delta.md).
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
- Gradient checkpointing costs 1.27x in step time for Qwen3.5 0.8B and 9B LoRA training; with 128 GB it can be off.
- MLX and PyTorch train these models within 10% of each other on this chip; MLX's fused gated-delta kernel needs a newer GPU
  generation for its chunked form.
- PyTorch's `PYTORCH_MPS_PREFER_METAL` and `PYTORCH_MPS_FAST_MATH` made a small byte-level model's training 12-18% slower.

## Tests and benchmarks
    pytest                                  # parity on small shapes (needs an Apple GPU)
    python bench/bench_gated_delta.py       # the kernel against the reference at Qwen3.5 training shapes
