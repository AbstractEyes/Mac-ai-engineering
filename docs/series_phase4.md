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
  driver's allocation at each step beside it, and the allocator's live tensors when each forward returns (a forward hook: the
  memory a step needs, which the footprint mixes with the allocator's cache). The 0.8B reference is the mean of its two runs.
- `bench/phase4_report.py` computes every figure below from the runs' records and writes `bench/phase4_logs/phase4_results.json`.

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

### 2. 20:48-20:53: the registered 0.8B configuration runs out of GPU memory (`p41_4x4_ref_a_oom`)
The reference at 4 x 4 sequences a step, checkpointing off, ran 7 steps (16.6 s a step with the warm-up) and stopped in step 8:
"MPS backend out of memory (MPS allocated: 78.17 GiB, other allocations: 1.69 GiB, max allowed: 80.64 GiB). Tried to allocate
4.12 GiB". The second reference run, started in the same configuration, was stopped by hand at step 4
(`p41_4x4_ref_b_stopped.log`).

The cause, as a chain:
- Without checkpointing every layer's activations stay alive until the backward -> at 4 sequences the live tensors when a
  forward returned were 32-39 GiB in steps 1-7.
- The largest single tensors are the vocabulary-wide ones: 4 sequences x ~1,100 tokens x 248,320 entries is 4.12 GiB in float32,
  and there are several (the model's loss upcasts the logits to float32; accelerate's mixed precision converts the returned bf16
  logits to float32 again: the traceback ends in its `convert_to_fp32`).
- The Metal allocator's heaps, fragmented by batches of changing shapes, held 63-76 GiB at a forward's end -> the float32 copy of
  step 8's longer batch did not fit under the high watermark (0.75 x 107.5 GiB = 80.6 GiB) -> out of memory.

The change (the bench's own micro-batch ladder; the same 16 sequences a step): the 0.8B runs at 2 x 8. Checkpointing stays off and
the watermarks stay as registered. Every step still takes the same 16 examples in the same order in every run, so the two arms
see identical steps; the split changes the padding and how many sequences run at once, so the step time and the memory are now
measured at 2 x 8, which the registered 0.8B time prediction (made for 4 x 4) did not price. The 9B pair runs first at its
registered 2 x 8 (its activations are larger: if its reference runs out of memory, the pair runs at 1 x 16).

### 3. 20:54-20:56: Qwen3.5-9B at its registered 2 x 8 runs out of GPU memory in step 1 (`p42_9b_ref`)
"MPS backend out of memory (MPS allocated: 61.23 GiB, other allocations: 17.63 GiB, max allowed: 80.64 GiB). Tried to allocate
2.31 GiB", 58 s after the start, in the first step. The model's bf16 weights (the 17.6 GiB of "other allocations") leave about
63 GiB under the high watermark for everything else, and two sequences without checkpointing needed more. The pair runs at 1 x 16
(the ladder's last rung; the same 16 sequences a step), the reference first: 54-60 s a step in its first steps, footprint 54-60 GB.
The registered 9B time prediction (made for 2 x 8) is read against 1 x 16 below.

### 4. 20:55-21:06: the 9B also runs out at 1 x 16; the cause is the allocator's fragmentation; the cache emptied each micro-batch
At 1 x 16 the 9B reference ran 4 steps (54-60 s a step) and ran out in the backward of step 5 ("MPS allocated: 62.67 GiB, other
allocations: 17.63 GiB, max allowed: 80.64 GiB. Tried to allocate 896.00 MiB"; `p42_9b_1x16_ref_oom`). Its kernels twin was stopped
by hand at step 3 (45 and 49 s for steps 1-2; `p42_9b_1x16_kernels_stopped.log`).

The cause, from the runner's memory records:

| run | live tensors when a forward returns, GiB | the allocator's pool at the step's end, GiB |
|---|---|---|
| 0.8B reference, 4 x 4, steps 1-7 | 32.1-38.6 (+1.7 of weights) | 63.1 -> 75.5 |
| 9B reference, 1 x 16, steps 1-4 | 22.5-26.8 (+17.6 of weights) | 51.9 -> 57.7 -> 60.6 -> 73.8 |

- The step needs far less than the allocator holds: about 45 GiB at 9B 1 x 16 against a pool of 74 GiB that grows with every step.
- Every micro-batch has a new length -> new tensor sizes -> the Metal allocator caches the freed buffers by size in heaps that
  stay partly occupied -> the pool outgrows the high watermark (80.6 GiB with the weights) although the live tensors fit.
- So the lever is the cache, not the batch: the Trainer's own option `torch_empty_cache_steps=1` releases the cached buffers
  between each micro-batch's forward and backward (`--empty-cache`; on PyTorch 2.15 it calls `torch.mps.empty_cache()` only).

Smoke runs with it (the first 24 items, 4 x 4; `smoke_*_emptycache`):

| arm | loss, steps 1 / 2 | seconds, steps 1 / 2 / 3: cache kept -> emptied | the pool at the steps' ends, GiB: kept -> emptied | live at a forward's end, GiB | guard's peak footprint, GB: kept -> emptied |
|---|---|---|---|---|---|
| reference | 2.81926 / 2.68739 either way | 9.82 / 8.86 / 4.87 -> 10.42 / 9.41 / 5.15 | 41-51 -> 26-33 | 20.4-24.2 either way | 51.6 -> 34.4 |
| kernels | 2.81964 / 2.68914 either way | 7.34 / 6.59 / 3.68 -> 7.75 / 6.87 / 3.92 | 45-54 -> 32-34 | 18.4-21.7 either way | 55.6 -> 41.8 |

The computation is unchanged (the same losses at steps 1-2), the step costs about 5-6% more in both arms (the kernels' advantage
per step stays 1.34-1.37x), and the pool shrinks by a third. Every registered run from here on uses it, at the registered micro-
batches: 0.8B 4 x 4 and 9B 2 x 8 (if the 9B reference still runs out at 2 x 8, the pair runs at 1 x 16, both arms alike). The
footprint stays the registered memory measure; with the cache emptied it shows the kernels' extra transient buffers more (41.8
against 34.4 GB here) while their live tensors are fewer (21.7 against 24.2 GiB).

### 5. 21:06-21:08: the 9B at 2 x 8 with the cache emptied: step 1 runs, step 2 runs out (`p42_9b_2x8_emptycache_ref_oom`)
Step 1 took 65 s; step 2 ran out ("MPS allocated: 61.86 GiB, other allocations: 17.63 GiB, max allowed: 80.64 GiB. Tried to
allocate 2.38 GiB"). With the cache emptied the pool is close to the live tensors, so at 2 sequences the live tensors themselves
outgrow the ~63 GiB the weights leave under the high watermark. The 9B pair runs at 1 x 16, both arms with the cache emptied:
the reference passed step 5 (where the run with the cache kept had stopped) at a steady 59-60 GB footprint.

### 6. 21:56-22:01: the 0.8B at 4 x 4 with the cache emptied runs out at step 18 (`p41_4x4_emptycache_ref_a_failed`)
17 steps at 16.0 s a step (steps 11-17), live tensors at a forward's end 32-39 GiB, the pool at the steps' ends 41 -> 76 GiB, then
"MPS allocated: 76.70 GiB, other allocations: 1.69 GiB, max allowed: 80.64 GiB. Tried to allocate 3.75 GiB". Emptying the cache
between forward and backward slows the pool's growth (step 8 passed) but does not stop it: the backward's freed buffers still
accumulate across micro-batches and steps. So every comparison below runs where the ladder fits, both arms alike, the cache
emptied each micro-batch: Qwen3.5-0.8B at 2 x 8 (P4.1 and the 0.8B half of P4.2) and Qwen3.5-9B at 1 x 16 (the 9B half).
Checkpointing stays off and the watermarks stay as registered.
