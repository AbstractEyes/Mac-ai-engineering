# Series phase 4: the kernels in real training, a dated log

Phase 4 of the kernel series: the chunked gated delta rule (phase 3, [series_phase3.md](series_phase3.md)) installed by
`macai.patch.install()` into LoRA training of two Qwen3.5 picture readers, against transformers' reference implementation, on one
machine (an M3 Max, 40-core GPU, 128 GB, macOS 26, PyTorch 2.15, transformers 5.22.1, PEFT 0.21.2). The trainer is
qwen-symbolic-thinking's own (`qwen_symbolic_thinking.train.run`, 0.1.35), unchanged except for the runner's documented patches.
The tests were registered before any run (series journal, 2026-11-02 20:xx PST); this log gives each test's setup, prediction,
measurement and verdict.

## Setup (both arms identical except the one call)

| | |
|---|---|
| trainer | `qwen_symbolic_thinking.train.run`: LoRA rank 32, alpha 64, dropout 0.05 on the language model's projections; plain Adam (`torch.optim.Adam`, no weight decay) at 2e-4 under a cosine schedule with 3% warm-up; bf16; sequences up to 2,048 tokens; seed 20261030; nothing shipped |
| data | a sample of the readers' four training sets (every 55th item): 382 items (372 with a picture), 400 examples (18 items carry two prompt forms); the pictures read from disk, nothing downloaded |
| arms | reference: transformers' `torch_chunk_gated_delta_rule` (the only path on a Mac: the fast kernels upstream are CUDA-only); kernels: `macai.patch.install()` before the model loads (the chunked forward and backward, chunk 16, the fused query / key normalization: `op.KERNEL = KERNEL_BWD = "chunked"`, `FUSED_NORM = True`) |
| runner | `bench/phase4_train.py`, each run its own process (`bench/phase4_run.sh`) under the bench's memory guard (stops past 96 GB footprint, 1 GB new swap or 3 h), `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.75`, `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.55`, one GPU job at a time |
| runner patches | the model on the Apple GPU (`device_map="mps"` for the trainer's `"cuda"`); gradient checkpointing off (the trainer turns it on: its own call made a no-op, the Trainer's argument forced to False, checked at the first step); the loss logged at every step (the trainer's `logging_steps=10` forced to 1); the pictures from the folder they were fetched into |

## The tests and their predictions (registered before any run)

| test | setup | prediction / pass rule |
|---|---|---|
| P4.1 loss parity | Qwen3.5-0.8B, 4 x 4 sequences a step, 4 epochs (100 steps), the loss at every step; the reference run twice, the kernels once | PASS when the kernels' loss is within 2x the reference's own run-to-run spread of the reference at every step, and the final loss within 1% |
| P4.2 time, 0.8B | the steady seconds a step (16 sequences) from the trainer's own clock, warm-up excluded; the P4.1 runs | kernels 10-12 s against the reference's ~14-15 s (>= 1.25x) |
| P4.2 time, 9B | Qwen3.5-9B, 2 x 8 sequences a step, 1 epoch (25 steps) | kernels 55-62 s against the reference's ~70-75 s (>= 1.2x) |
| P4.2 memory | the guard's peak physical footprint, both models | the kernels' peak within the reference's + 10% |

How the measures are read (fixed before the registered runs, after the smoke runs of entry 1):
- The loss at step s is the Trainer's logged training loss of that optimizer step (16 sequences). With the reference's two runs
  A and B: the reference ref(s) = (A(s) + B(s)) / 2, its spread(s) = |A(s) - B(s)|, the kernels' gap(s) = |K(s) - ref(s)|.
- Reading 1, the rule as registered, step by step: gap(s) <= 2 spread(s) at every step s. At steps 1 and 2 the spread is zero
  by construction (entry 1: the two runs compute the same loss until the first update), so this reading cannot pass there
  unless the kernels reproduce the reference's bits.
- Reading 2, the spread as one number for the run (its largest value): gap(s) <= 2 max_s spread(s) at every step.
- Beside them: the root mean square over the steps of gap(s) against that of spread(s). For a run drawn the same way as A and B,
  K - ref has 3/4 the variance of A - B, so the ratio expected is about 0.87.
- The final loss: |K(100) - ref(100)| / ref(100) <= 1% (the last step's loss; the run's mean loss beside it).
- An unregistered control, run after the registered runs: a third reference run C, judged against A and B by the same readings,
  to show whether the reference itself passes them.
- The steady seconds a step: (the trainer's clock at the last step - at step 10) / (steps after step 10): steps 11-100 for 0.8B,
  11-25 for 9B. The trainer's clock is its own progress log (whole seconds since training began, taken after the logging step
  has read the loss off the GPU); the runner's clock (the same moment, unrounded) is reported beside it. The 0.8B reference is
  the mean of its two runs.
- Memory: the guard's peak physical footprint (polled every 2 s; it includes the GPU buffers the process owns), with the Metal
  driver's allocation at each step beside it.

## The log

### 1. 2026-11-02 20:4x PST: the runner, smoke runs (the first 24 items: 42 examples, 3 steps)

| run | arm | gated delta rule calls | loss, steps 1 / 2 / 3 | gradient norm, steps 1 / 2 / 3 | LoRA A at the start (sum of magnitudes) | live tensors at a forward's end, GB | guard's peak footprint, GB |
|---|---|---|---|---|---|---|---|
| smoke_reference | reference | 198 | 2.8193 / 2.6874 / 2.4431 | 8.065 / 9.623 / 5.692 | 92,568.7 | 24.2 | 51.6 |
| smoke_kernels | kernels | 198 | 2.8196 / 2.6891 / 2.4467 | 7.940 / 9.409 / 5.565 | 92,571.1 | 21.6 | 55.6 |
| smoke_kernels_2 | kernels | (not counted) | 2.8196 / 2.6891 / 2.4427 | 7.990 / 9.563 / 5.609 | 92,600.7 | 21.7 | 55.6 |

What the smoke runs established, as causal chains:
- The patch reaches every linear-attention layer: 198 calls = 18 layers x 11 micro-batches, every one to
  `macai.gated_delta.op.chunk_gated_delta_rule`; with the patch absent the same count goes to transformers' function. No module
  checkpoints; the loss is logged at every step.
- Two runs of the same seed do not start alike: the trainer builds the LoRA before the Trainer seeds the run -> the A matrices
  differ in every process (the sums above) -> the loss is identical at steps 1 and 2 (B starts at zero, so the LoRA adds nothing
  until the first update, and the first step's learning rate is 0 under the warm-up) and differs from step 3 on (two kernel runs:
  2.4467 and 2.4427). So the reference's run-to-run spread is the spread of the LoRA's starting point, and it is zero at steps
  1-2.
- At steps 1-2 (the same weights, the same batches) the kernels' loss differs from the reference's by 1.4e-4 and 6.5e-4 of the
  loss: the Trainer runs the forward under bf16 autocast -> the reference's chunk products run in bfloat16 -> its output and
  gradients are 3.4e-3 to 7.2e-3 off a float64 reference under autocast and 1e-6 without, while the kernels stay at 3.5e-7 to
  6.2e-7 either way (`bench/phase4_autocast_check.py`, `bench/phase4_logs/autocast_check.json`). The kernels are the exact arm.
- Memory: the kernels hold fewer live tensors at a forward's end (21.6 against 24.2 GB) but the process's footprint peaks higher
  (55.6 against 51.6 GB): the footprint is mostly the Metal allocator's cache of freed buffers (it keeps them up to the low
  watermark, 0.55 x 115.4 = 63.5 GB), and the kernels' many per-layer buffers of new sizes enlarge that cache.
