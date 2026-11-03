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
| runner patches | the model on the Apple GPU (`device_map="mps"` for the trainer's `"cuda"`); gradient checkpointing off (the trainer turns it on: its own call made a no-op, the Trainer's argument forced to False, checked at the first step); the loss logged at every step (the trainer's `logging_steps=10` forced to 1); the pictures from the folder they were fetched into; in every comparison (entry 4) the Trainer's own `torch_empty_cache_steps=1` (`--empty-cache`) |

## The tests and their predictions (registered before any run)

| test | setup | prediction / pass rule |
|---|---|---|
| P4.1 loss parity | Qwen3.5-0.8B, 4 x 4 sequences a step, 4 epochs (100 steps), the loss at every step; the reference run twice, the kernels once | PASS when the kernels' loss is within 2x the reference's own run-to-run spread of the reference at every step, and the final loss within 1% |
| P4.2 time, 0.8B | the steady seconds a step (16 sequences) from the trainer's own clock, warm-up excluded; the P4.1 runs | kernels 10-12 s against the reference's ~14-15 s (>= 1.25x) |
| P4.2 time, 9B | Qwen3.5-9B, 2 x 8 sequences a step, 1 epoch (25 steps) | kernels 55-62 s against the reference's ~70-75 s (>= 1.2x) |
| P4.2 memory | the guard's peak physical footprint, both models | the kernels' peak within the reference's + 10% |

## Where the registered tests stand (the runs of 2026-11-02, 20:48-23:38 PST)

| test | registered | as run | measured | verdict |
|---|---|---|---|---|
| the configurations | 0.8B 4 x 4, 9B 2 x 8, checkpointing off, watermarks 0.75 / 0.55 | | out of GPU memory: 0.8B at step 8 (step 18 with the cache emptied); 9B in step 1 (step 2 with the cache emptied; at 1 x 16 with it kept, step 5) | did not run (entries 2-6) |
| P4.1 loss, step by step | within 2x the reference's spread at every step | 0.8B at 2 x 8, the cache emptied each micro-batch | over the bar at 25 of 100 steps (2 of them with a zero bar by construction); a third reference run: 26 | NOT MET as written; the rule fails the reference itself (entry 10) |
| P4.1 loss, the run's spread as one bar | (fixed before the runs) | | the largest gap 0.0085 against 0.0159 | met |
| P4.1 final loss | within 1% | | 0.1972 against 0.1967: 0.27% (a third reference run: 2.19%) | MET |
| P4.2 time, 0.8B | kernels 10-12 s against ~14-15 s, >= 1.25x | 2 x 8, the cache emptied | 11.06 s against 15.21 s: 1.375x | MET |
| P4.2 time, 9B | kernels 55-62 s against ~70-75 s, >= 1.2x | 1 x 16, the cache emptied | 51.2 s against 62.5 s: 1.22x | the ratio MET; the seconds not comparable (made for 2 x 8) |
| P4.2 memory | the kernels' peak footprint within the reference's + 10% | | 0.8B 57.7 against 48.65 GB (+19%); 9B 79.5 against 67.8 GB (+17%); live tensors 10% fewer at both | NOT MET (the allocator's pool, entry 11) |

In plain words: the kernels drop into the readers' real LoRA training and train like the reference. Their loss curve is as close
to each reference run as the reference runs are to each other, and their final loss is within 0.3%. Training steps are 1.375x
faster at 0.8B and 1.22x faster at 9B. The registered configurations do not fit in the GPU memory the allocator allows with
checkpointing off, so the comparisons ran at 2 sequences a micro-batch (0.8B) and 1 (9B), with the allocator's cache emptied
each micro-batch; the 16 sequences a step, the data, the recipe and the watermarks are as registered. Memory missed its bar: the
kernels need 10% fewer live tensors, but the allocator holds 17-19% more for them; as far as measured (entry 11), their buffers
come in more sizes than its cache can reuse.

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

### 3. 20:54-20:56: Qwen3.5-9B at its registered 2 x 8 runs out of GPU memory in step 1 (`p42_9b_2x8_ref_oom`)
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

### 7. 21:08-21:56: P4.2 at 9B: Qwen3.5-9B, 1 x 16, 25 steps, the cache emptied (`p42_9b_ref_1x16`, `p42_9b_kernels_1x16`)

| | reference | kernels | kernels against the reference |
|---|---|---|---|
| steady seconds a step (the trainer's clock, steps 11-25) | 62.47 | 51.20 | 1.22x faster (median of the 15 paired steps: 1.22x) |
| guard's peak footprint, GB | 67.8 | 79.5 | 1.17x |
| live tensors when a forward returns, peak, GiB (weights not counted: +17.6) | 27.8 | 25.1 | 0.90x |
| the allocator's pool at a step's end, peak, GiB | 60.6 | 73.9 | 1.22x |
| loss at step 25; the run's mean loss | 0.5452; 0.8699 | 0.5435; 0.8684 | -0.3%; -0.2% |
| loss at steps 1-2 (the same weights, the same batches) | | | 9.4e-4, 2.5e-4 of the loss apart |

Verdicts:
- Time: the registered ratio (>= 1.2x) is MET at 1.22x. The registered seconds (kernels 55-62 s against ~70-75 s) were made for
  2 x 8, which does not fit (entries 3 and 5); at 1 x 16 both arms run faster than predicted (kernels 51.2 s, under the range;
  the reference 62.5 s, under its ~70-75 s).
- Memory: NOT MET by the registered measure: the footprint is 17% over the reference's against the 10% allowed. The step's live
  tensors are 10% fewer with the kernels; the excess is the allocator's pool, which grows over the run with the kernels (49.9 ->
  77.3 GiB when a forward returns) and levels off with the reference (50.6 -> 59-66 GiB). Entry 11 isolates the cause.

### 8. 22:01-23:12: P4.1, loss parity: Qwen3.5-0.8B, 2 x 8, 100 steps, the cache emptied (`p41_ref_a`, `p41_ref_b`, `p41_kernels`)

| step | reference A | reference B | kernels | spread A-B | kernels' gap | gap / spread |
|---|---|---|---|---|---|---|
| 1 | 2.5636 | 2.5636 | 2.5624 | 0.0000 | 0.0012 | (spread 0) |
| 2 | 2.4262 | 2.4262 | 2.4255 | 0.0000 | 0.0006 | (spread 0) |
| 3 | 2.0275 | 2.0287 | 2.0281 | 0.0012 | 0.0000 | 0.02 |
| 4 | 1.8676 | 1.8675 | 1.8678 | 0.0002 | 0.0003 | 1.70 |
| 5 | 1.3860 | 1.3926 | 1.3867 | 0.0066 | 0.0026 | 0.39 |
| 10 | 0.9889 | 0.9895 | 0.9898 | 0.0006 | 0.0006 | 1.09 |
| 18 (the largest spread) | 0.8288 | 0.8367 | 0.8319 | 0.0080 | 0.0009 | 0.11 |
| 25 | 0.7713 | 0.7772 | 0.7726 | 0.0059 | 0.0016 | 0.27 |
| 47 (the largest gap) | 0.4770 | 0.4837 | 0.4719 | 0.0066 | 0.0085 | 1.27 |
| 50 | 0.5672 | 0.5642 | 0.5646 | 0.0031 | 0.0011 | 0.37 |
| 75 | 0.3647 | 0.3619 | 0.3592 | 0.0028 | 0.0042 | 1.49 |
| 100 | 0.1955 | 0.1979 | 0.1972 | 0.0024 | 0.0005 | 0.22 |

Every step is in `bench/phase4_logs/phase4_results.json` (`p4_1.rows`).

| reading (fixed before the runs) | bar | measured | verdict |
|---|---|---|---|
| 1, step by step, as registered | gap(s) <= 2 spread(s) at every step | over the bar at 25 of 100 steps: steps 1-2 (spread 0 by construction) and 23 of the other 98 | NOT MET |
| 2, the run's spread as one number | gap(s) <= 2 x 0.0080 = 0.0159 at every step | the largest gap 0.0085 (step 47, 1.8% of the loss) | met |
| root mean square, gap against spread | (0.87 expected of a run drawn like the reference) | 0.0032 against 0.0034: 0.95 | |
| the final loss | within 1% of the reference | 0.1972 against 0.1967 (A 0.1955, B 0.1979): 0.27% | MET |
| the run's mean loss | | 0.5767 against 0.5776: 0.15% | |

Why reading 1 fails, as a chain:
- Steps 1-2 compute the same weights on the same batches (entry 1) -> the two reference runs agree to the bit -> the bar is zero
  -> the kernels, which compute the rule exactly where the reference rounds its products to bfloat16 under autocast, sit 4.7e-4
  and 2.5e-4 of the loss away -> over a zero bar.
- From step 3 the runs differ by their LoRA's starting point, so at each step the spread is one draw of a noisy difference -> for
  three runs drawn alike, the third run's gap from the mean of two exceeds twice their difference with probability about 0.26 at
  any one step (the gap / spread ratio of two independent normal differences, scales sqrt(3/2) and sqrt(2)) -> over 98 steps
  about 25 such steps are expected from a reference run itself; the kernels have 23.
- So reading 1 as registered cannot be met by a third run of the reference either; the control (entry 10) measures that.

### 9. P4.2 at 0.8B (the P4.1 runs: 2 x 8, the cache emptied)

| | reference A / B (mean) | kernels | kernels against the reference |
|---|---|---|---|
| steady seconds a step (the trainer's clock, steps 11-100) | 15.23 / 15.18 (15.21) | 11.06 | 1.375x faster (median of the 90 paired steps: 1.375x) |
| guard's peak footprint, GB | 47.3 / 50.0 (48.65) | 57.7 | 1.19x |
| live tensors when a forward returns, peak, GiB | 20.43 / 20.44 | 18.46 | 0.90x |
| the allocator's pool at a step's end, peak, GiB | 42.4 / 43.8 | 51.2 | 1.19x |

Verdicts:
- Time: MET: the kernels' 11.06 s is inside 10-12 s and 1.375x clears the 1.25x bar; the reference's 15.2 s is just above its
  expected ~14-15 s. Measured at 2 x 8 with the cache emptied (the registered 4 x 4 does not fit, entries 2 and 6); emptying the cache costs
  both arms about the same 5-6% (entry 4).
- Memory: NOT MET by the registered measure: the footprint is 19% over the reference's against the 10% allowed, while the live
  tensors are 10% fewer (as at 9B).

### 10. 23:12-23:37: the control, a third reference run judged like the kernels (`control_p41_ref_c`, unregistered)

| | kernels | third reference run (C) |
|---|---|---|
| reading 1: steps over 2x the spread, of the 98 with a spread | 23 (and steps 1-2, where the bar is zero) | 26 (its steps 1-2 equal A's and B's to the bit) |
| reading 2: the largest gap (bar 0.0159) | 0.0085 (step 47) | 0.0085 (step 57) |
| root mean square, gap against spread (0.87 expected) | 0.95 | 0.85 |
| the final loss against the reference's 0.1967 | 0.1972: 0.27% | 0.2010: 2.19%, over the 1% bar |
| the run's mean loss against the reference's 0.5776 | 0.15% | 0.04% |
| steady seconds a step | 11.06 | 14.92 (A 15.23, B 15.18) |
| guard's peak footprint, GB | 57.7 | 50.9 (A 47.3, B 50.0) |

Root mean square of the loss difference between two runs, over the 100 steps:

| A-B | A-C | B-C | kernels-A | kernels-B | kernels-C |
|---|---|---|---|---|---|
| 0.0034 | 0.0034 | 0.0034 | 0.0035 | 0.0038 | 0.0033 |

A third run of the reference fails the registered step rule at as many steps as the kernels do and misses the 1% final-loss bar
that the kernels meet. The kernels' loss is as far from each reference run as the reference runs are from one another.

### 11. 23:38: why the kernels' pool is larger (`bench/phase4_alloc_check.py`, `bench/phase4_logs/alloc_check.json`)
The training pattern of one linear-attention stack in isolation, at the 9B shape (1 x T x 32 heads x 128): 24 layer calls a
micro-batch with their saved tensors alive until one backward, 24 micro-batches of random lengths 400-1,299, the cache emptied
between forward and backward as in the runs.

| arm | live tensors when the forward returns, GiB | the pool after micro-batch 1 / 8 / 24, GiB |
|---|---|---|
| reference | 2.65-7.57 | 8.84 / 8.45 / 9.53 |
| kernels | 2.13-6.08 | 7.05 / 8.03 / 10.02 |
| kernels, every length padded up to a multiple of 64 | 2.30-6.55 | 7.09 / 8.03 / 9.03 |

- The kernels hold about 20% fewer live tensors, yet their pool grows over the micro-batches (+3.0 GiB against the reference's
  +0.7) and passes the reference's; padding every length to 64 tokens, as the reference does inside its call, cuts that growth by
  a third.
- The chain, partly inferred [I]: the kernels' large float32 buffers (the per-chunk states and their gradients, every 16 tokens)
  take a new size at almost every new length -> the allocator's cache fills with freed buffers the next lengths cannot reuse ->
  the pool grows. In the whole model the excess is larger (about 13 GiB at 9B) than this isolated stack shows, so the model's
  other buffers add to it.
- The lever is on the kernels' side (fewer distinct buffer sizes: lengths bucketed, or one workspace reused per shape class),
  measured in training like this phase. Not done here.
