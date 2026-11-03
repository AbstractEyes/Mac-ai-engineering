# Series: the kernels' memory (design D2m), a dated log

Phase 4 ([series_phase4.md](series_phase4.md)) ran the chunked gated delta rule in real LoRA training of Qwen3.5: the same loss as
the reference, steps 1.375x (0.8B) and 1.22x (9B) faster, but a process footprint 17-19% over the reference's while the live
tensors were 10% fewer. This entry of the series traces the cause to PyTorch's Apple-GPU allocator and tries the registered
designs against it, on one machine (an M3 Max, 40-core GPU, 128 GB, macOS 26, PyTorch 2.15, transformers 5.22.1, PEFT 0.21.2),
with Phase 4's trainer and runner. The designs, their tests, bars and predictions were registered before any code (series
journal, 2026-11-03 00:2x PST); this log gives each design and test: setup, prediction, measurement and verdict, the failed
design included.

## The cause

PyTorch 2.15's allocator for the Apple GPU (`aten/src/ATen/mps/MPSAllocator.{h,mm}`, read in full at the build's commit):

- buffers over 1 MiB are rounded up to 1/32 of their power of two;
- buffers of 10-512 MiB are cut out of shared 1 GiB heaps: the smallest free range that fits (best fit), a remainder of 1 MiB or
  more split off, adjacent free ranges merged when nothing else fits; buffers of 512 MiB or more get a heap of their own;
- a heap goes back to the system only when every buffer in it is free (`torch.mps.empty_cache()`, the low-watermark collection,
  the out-of-memory fallbacks).

So the chain, as Phase 4's records show it:

- the chunked kernels keep, for every linear-attention layer, the state entering each chunk of 16 tokens
  ([B, H, chunks, 128, 128] float32: 82-164 MiB a layer at the lengths of these runs), with Vn and Xu (about 21 MB each), from the
  forward to the backward;
- 18 (0.8B) or 24 (9B) such sets live at once at a forward's end, spread over the shared heaps between the model's own buffers ->
  the heaps they sit in cannot be released when the Trainer empties the cache between forward and backward -> the backward's
  buffers need new heaps beside them -> the pool runs 17-19% over the reference's, while the live tensors are 10% fewer.

## The designs (as built)

| design | setting (`macai.gated_delta.op`) | what it changes |
|---|---|---|
| M1 size classes | `SIZE_CLASSES = True` | every buffer the drop-in, the fused prologue and the chunked kernels allocate whose size follows the sequence length (saved, temporary, the float32 copies and the output, the gradients) is cut as the first bytes of a buffer of the smallest of 4 sizes a doubling that holds it (`_memory.py`: 2^(j/4) bytes rounded to a 16 KiB page; about 9% over on average, at most 19% from 10 MiB up). The tensor keeps its shape, contiguous at offset 0 (`resize_` to a smaller size keeps the larger storage; the tensor is not a view, so autograd treats it as before); the float32 copies of the inputs and the output's conversion go through a size-classed cast whose backward converts the gradient the same way. The computation is untouched. |
| M3 the states recomputed | `RECOMPUTE_STATES = True` | with the chunked forward and backward, the training forward keeps only its inputs (q, k, v, g, beta, S_0) and stores nothing a chunk; the backward first rebuilds G, A, M and KG (`gdc_prep`) and the chunk states, Vn and Xu (`gdc_fwd`) with the forward's own kernels on the same inputs, then runs the backward passes as before (flash-linear-attention's and MLX's choice). `gdc_fwd` takes a new flag (8: write neither the output nor the final state, do not read q) for that pass. |
| M1 + M3 | both | |
| M2 a stack arena | not built | registered to be built only if M1, M3 and M1 + M3 all missed |

Every design gives the stored path's bits exactly (entry 1): the same kernels on the same inputs. The edit to `gdc_fwd` was checked
against the original source: on the stored paths (flags 0, 1, 2, 4) the two give the same bits on 51 shape / kernel-pair cases.

## The tests, bars and predictions (registered before any code)

| test | setup | bar / prediction |
|---|---|---|
| parity | the 203-test harness, and `tests/test_memory.py`: for M1, M3 and M1 + M3 every output and gradient against the stored path on the harness's shapes and through the drop-in, on CPU copies, NaN-aware exact equality | a design that changes one bit is not adopted |
| function cost (informational) | the rule's forward + backward per design at the training shapes (`bench/bench_gated_delta.py --memory`) | M3: +20-30% on the rule (~10 -> 12-13 ms) |
| screen, each design | Qwen3.5-0.8B, 2 x 8 sequences, the cache emptied each micro-batch, 1 epoch (25 steps): the Metal driver's memory at each step's end, its maximum by step 20, read against Phase 4's first 20 steps (reference runs 35.6 / 36.6 / 39.4 GB, kernels 45.1); steps 1-2 losses against Phase 4's kernels' | PASS at <= 43.3 GB (39.4 + 10%); steps 1-2 equal to the bit |
| admission | the first of M1, M3, M1 + M3 that passes the screen goes to the confirmation; if it misses a confirmation bar, the next that passed (or M1 + M3) | |
| confirmation, 0.8B | 2 x 8, the cache emptied, 4 epochs (100 steps), read as Phase 4 read them | the guard's peak footprint <= 53.5 GB (the reference's mean 48.65 + 10%); steady seconds a step (the trainer's clock, steps 11-100) <= 11.39 (Phase 4's kernels + 3%); the loss as Phase 4 read it (root mean square of the step-loss difference against reference runs A, B, C, and theirs among themselves; the final and mean loss) |
| confirmation, 9B | Qwen3.5-9B, 1 x 16, the cache emptied, 4 epochs (100 steps) | steps 1-25 against Phase 4's pair: footprint <= 74.6 GB (67.8 + 10%), steady seconds a step over steps 11-25 <= 52.7 (51.2 + 3%); the long run: the driver's maximum over steps 51-100 <= its maximum over steps 1-50 + 2 GB |
| predictions, M1 | | bit-identical; the step within 1%; live tensors up by at most 5%; the heaps' excess over the reference at least halved |
| predictions, M3 | | bit-identical; the rule +20-30%; the training step +1-2% (0.8B 11.1-11.3 s, 9B 51.5-52.0 s); live tensors at a forward's end ~3 GiB fewer at 0.8B (18.5 -> ~15.5) and ~4 GiB fewer at 9B (25.1 -> ~21); the footprint at or below the reference's |

## Where the registered tests and predictions stand (the runs of 2026-11-03, 00:46-02:44 PST)

| test | registered | measured | verdict |
|---|---|---|---|
| parity, M1 / M3 / M1 + M3 | every output and gradient bit-identical to the stored path | 128 tests pass (each design on the harness's shapes and through the drop-in, plus checks of the comparison and of each design being in force); in all five training runs the steps 1-2 losses equal Phase 4's kernels' to the bit | MET, all three |
| the screen, M1 size classes | the driver's maximum by step 20 <= 43.3 GB | 46.37 (Phase 4's kernels 45.07) | MISS |
| the screen, M3 the states recomputed | | 33.09 | PASS |
| the screen, M1 + M3 | | 32.84 | PASS |
| admission | the first to pass, in the order M1, M3, M1 + M3 | M3 | M3 confirmed |
| 0.8B: the footprint | <= 53.5 GB | 46.32 (the reference's mean 48.65; Phase 4's kernels 57.70) | MET |
| 0.8B: the step | <= 11.39 s | 10.978 s (Phase 4's kernels 11.056) | MET |
| 0.8B: the loss, as Phase 4 read it | | root mean square against A / B / C .0029 / .0030 / .0035 (A-B, A-C, B-C .0034); final 0.1983 against 0.1967 (0.83%); mean 0.5773 against 0.5776 | like the reference |
| 9B: the footprint through step 25 | <= 74.6 GB | 64.35 (Phase 4's reference 67.78, kernels 79.49) | MET |
| 9B: the step, steps 11-25 | <= 52.7 s | 51.13 s (Phase 4's kernels 51.20) | MET |
| 9B: the long run | the driver's maximum over steps 51-100 <= its maximum over steps 1-50 + 2 GB | 61.86 against 65.84 (-3.98) | MET |

| prediction | registered | measured | verdict |
|---|---|---|---|
| M1 | bit-identical | (above) | met |
| M1 | the step within 1% | steps 11-25: 11.183 s against Phase 4's kernels' 11.254 on the same steps (-0.6%) | met |
| M1 | live tensors up by at most 5% | 19.02 against 18.46 GiB (+3.0%) | met |
| M1 | the heaps' excess over the reference at least halved | by step 20, the driver's maximum over the reference runs' mean 9.18 GB against Phase 4's kernels' 7.88 (+16%); the memory held above the live tensors 21.3 against 21.6 GB | missed |
| M3 | bit-identical | (above) | met |
| M3 | the rule +20-30% (~10 -> 12-13 ms) | +17.6% (9.94 -> 11.71 ms; 9B shape 10.01 -> 11.76) | missed (cheaper) |
| M3 | the training step +1-2%: 0.8B 11.1-11.3 s | 10.978 s (-0.7% against Phase 4's kernels) | missed (no cost measurable) |
| M3 | 9B 51.5-52.0 s | 51.13 s over steps 11-25 (-0.1%); 50.91 over steps 11-100 | missed (no cost measurable) |
| M3 | live at a forward's end 0.8B ~15.5 GiB (from 18.5) | 15.10 (-3.36) | met |
| M3 | 9B ~21 GiB (from 25.1) | 20.59 (-4.49) | met |
| M3 | the footprint at or below the reference's | 0.8B 46.32 against 48.65; 9B through step 25 64.35 against 67.78 | met |

In plain words: keeping the per-chunk states from the forward to the backward was what held the extra memory. Recomputing them
in the backward, as the prior art does, gives the same bits, costs the rule 18% in isolation and nothing measurable in a training
step, and brings the process's footprint under the reference implementation's at both model sizes (0.8B: 46.3 against 48.65
GB; 9B: 64.4 against 67.8 GB over the first 25 steps), with the kernels' speed kept (1.38x / 1.22x the reference). Size classes,
the first design tried, give the same bits but no memory: the allocator already rounds sizes and cuts larger free ranges down,
so the number of sizes was not the cause. M3 is now the default.

## The log

### 1. 2026-11-03 00:3x PST: the build and the parity (commit 11c1c99)
Built as in the table above: `src/macai/gated_delta/_memory.py` (the size classes and the size-classed cast), the two settings
and the recompute mode in `op.py`, the size-classed buffers in `_norm.py`, flag 8 in `gdc_fwd`.

- The edit to `gdc_fwd` against the original source (HEAD, 7f2cd07), both compiled in one process: the stored paths (the
  training forward with stored states, the forward without them, the 8-token states for the token-at-a-time backward, the
  backward's recomputation after a token-at-a-time forward) on 13 shapes, from 1 token to 2 x 1,300 x 16 heads and 2,048 tokens,
  key widths 64-256: every output and gradient the same bits in all 51 cases.
- `tests/test_memory.py`, 126 tests, all pass: each design and both together against the stored path, bit for bit, on the
  harness's shapes (lengths 1-2,048 on and off the chunk, batch 1-3, key / value widths 32-256, starting states or none, beta
  at 0 and near 1, no decay and very strong decay, nearly parallel keys), forward only and forward + backward; through the
  drop-in in float32, bfloat16 and float16, with the fused and the PyTorch prologue, on contiguous inputs and on the strided
  views the Qwen3.5 layer passes; the mixed kernel pairs; and checks that each design is in force (with size classes the output
  and every saved buffer sit in storage of their class's size; with the recomputation the forward keeps exactly its six
  inputs). The comparison is NaN-aware exact equality on CPU copies plus the bits themselves. Two tests added later (commit
  354abe5) check the comparison itself (a signed zero, one unit in the last place, NaN at another place are told apart) and a
  negative control (two settings that compute differently are told apart): 128 tests.
- The whole suite: 329 tests pass (the harness's 203 unchanged); 331 with the two later tests.

### 2. 00:4x: the function's cost (`bench/bench_gated_delta.py --memory`, `bench/memory_logs/function_cost.json`)
The rule through the drop-in, every input needing its gradient, bfloat16 inputs, wall clock, the median of 5 rounds that
interleave the designs (each round 5 calls after 2 warm-up calls):

| design | 0.8B shape (4 x 1,024 x 16 heads): forward / forward + backward, ms | 9B shape (2 x 1,024 x 32 heads) | against the stored path | kept from the forward to the backward, one call |
|---|---|---|---|---|
| stored | 3.55 / 9.94 | 3.54 / 10.01 | | 413.0 MiB |
| M1 size classes | 3.58 / 10.00 | 3.57 / 10.06 | +0.6% / +0.5% | 413.0 MiB (1,024 tokens: every size is already a class) |
| M3 the states recomputed | 3.56 / 11.71 | 3.55 / 11.76 | +17.8% / +17.5% | 116.8 MiB |
| M1 + M3 | 3.59 / 11.86 | 3.59 / 11.81 | +19.3% / +18.0% | 116.8 MiB |

Verdict on the registered prediction for M3 (+20-30%, ~10 -> 12-13 ms): missed, on the cheap side: +17.6%, 11.7 ms. The
recomputation adds 1.75 ms, about half of a forward through the drop-in (3.55 ms): it runs `gdc_prep` and the state pass
without the output's work (flag 8: no query staging, no Q S or M Vn products, no output or final state written) and none of the
drop-in's prologue.

### 3. 00:4x: the stack in isolation (`bench/phase4_alloc_check.py --designs`, `bench/memory_logs/alloc_check.json`)
Phase 4's diagnostic with the designs as arms: the 9B shape (1 x T x 32 heads x 128), 24 layer calls a micro-batch with their saved
tensors alive until one backward, 24 micro-batches of random lengths 400-1,299, the cache emptied between forward and backward.

| arm | live tensors when the forward returns, GiB | the pool after micro-batch 1 / 8 / 24, GiB |
|---|---|---|
| reference | 2.65-7.56 | 8.84 / 8.42 / 9.51 |
| kernels, stored | 2.13-6.08 | 7.05 / 7.02 / 9.02 |
| kernels, lengths padded to 64 | 2.30-6.56 | 7.09 / 9.05 / 12.03 |
| M1 size classes | 2.43-7.00 | 8.30 / 9.28 / 10.06 |
| M3 the states recomputed | 0.65-1.74 | 3.05 / 3.05 / 4.02 |
| M1 + M3 | 0.73-1.99 | 3.30 / 3.27 / 3.27 |

The pool moves in whole 1 GiB heaps, so one run's reading moves by a heap or two between runs (Phase 4's run of the same
script: kernels 7.05 / 8.03 / 10.02, padded 7.09 / 8.03 / 9.03); the recomputed states' 3-4 GiB against 9-10 is far beyond that.
Size classes hold 15% more live bytes at these lengths (the rounding) and do not shrink the pool.

### 4. 00:46-01:00: the screens (`screen_*_0.8b_2x8_25steps`, commits be84664, c11f32c)
Qwen3.5-0.8B, 2 x 8 sequences a step, the cache emptied each micro-batch, 1 epoch (25 steps), one run each.

| design | the driver's memory, maximum by step 20, GB (bar 43.3) | live tensors at a forward's end, peak by step 25, GiB | the guard's peak footprint, 25 steps, GB | seconds a step, steps 11-25 (runner's clock) | steps 1-2 losses against Phase 4's kernels | verdict |
|---|---|---|---|---|---|---|
| Phase 4's kernels (stored) | 45.07 | 18.46 | 49.81 (its first 25 steps) | 11.254 | | |
| Phase 4's reference runs A / B / C | 35.63 / 36.57 / 39.38 | 20.43-20.45 | 40.12 / 44.43 / 41.11 (first 25 steps) | | | |
| M1 size classes | 46.37 | 19.02 | 50.66 | 11.183 | equal to the bit | MISS |
| M3 the states recomputed | 33.09 | 15.08 | 37.53 | 11.130 | equal to the bit | PASS |
| M1 + M3 | 32.84 | 15.25 | 35.95 | 11.113 | equal to the bit | PASS |

What the screens show, as causal chains:
- Steps 1-2 compute the same weights on the same batches (B starts at zero; the learning rate is 0 at step 1) -> every design's
  losses there equal Phase 4's kernels' to the bit (2.5623860359191895, 2.4255459308624268) -> in the real model, too, the
  designs change no bit of the loss. (Gradients cannot be compared across runs: the trainer starts each run's LoRA unseeded;
  their bits are the tests' to check.)
- The memory the allocator holds above the step's live tensors (the driver's memory minus the live tensors at a forward's end,
  mean over steps 1-25): Phase 4's kernels 21.6 GB, reference run A 11.6, M1 21.3, M3 13.0, M1 + M3 12.1. Size classes leave it
  where it was; recomputing the states takes away nearly all of the kernels' excess over the reference.
- So the excess was not the number of distinct sizes: the allocator already rounds sizes to 1/32 of their power of two and cuts
  any larger free range down to a request. It was the per-chunk states held from the forward to the backward: 18 sets of
  82-164 MiB blocks spread over the shared 1 GiB heaps keep those heaps from being released when the cache is emptied -> the
  backward's buffers open new heaps beside them. Without the kept states (M3) the forward leaves 3.4 GiB fewer live tensors
  (15.1 against 18.5 GiB), the cache-emptying releases those heaps, and the backward's recomputed states live one layer at a
  time.
- All three runs are about 1% faster than Phase 4's kernels on the same steps (median paired step ratios 0.985-0.989), size
  classes included: the day's run-to-run variation is larger than the recomputation's cost in a whole training step.
- The registered admission takes the first design that passes, in the order M1, M3, M1 + M3: M3.

### 5. 01:00-01:19: the 0.8B confirmation of M3 (`confirm_recompute_0.8b_2x8_100steps`, commit 3075ec8)
Qwen3.5-0.8B, 2 x 8, the cache emptied each micro-batch, 4 epochs (100 steps); read as Phase 4 read its runs.

| | M3 | Phase 4's kernels (stored) | Phase 4's reference A / B / C | bar | verdict |
|---|---|---|---|---|---|
| the guard's peak footprint, GB | 46.32 | 57.70 | 47.3 / 50.0 / 50.9 (A-B mean 48.65) | <= 53.5 | MET (0.95x the reference's mean) |
| steady seconds a step (the trainer's clock, steps 11-100) | 10.978 | 11.056 | 15.23 / 15.18 / 14.92 | <= 11.39 | MET (median paired step ratio against Phase 4's kernels 0.994) |
| the driver's peak at a step's end, GB | 41.76 | 51.16 | 42.41 / 43.84 / 46.74 | | |
| live tensors at a forward's end, peak, GiB | 15.10 | 18.46 | 20.43 / 20.44 / 20.45 | | |
| the final loss (step 100); the run's mean loss | 0.1983; 0.5773 | 0.1972; 0.5767 | 0.1955 / 0.1979 / 0.2010; A-B mean 0.5776 | | 0.83% and 0.04% from the reference |

Root mean square of the loss difference between two runs, over the 100 steps:

| M3-A | M3-B | M3-C | M3-Phase 4's kernels | A-B | A-C | B-C |
|---|---|---|---|---|---|---|
| 0.0029 | 0.0030 | 0.0035 | 0.0030 | 0.0034 | 0.0034 | 0.0034 |

M3 sits as far from each reference run as the reference runs sit from each other; its steps 1-2 equal Phase 4's kernels' to
the bit, so from step 3 on the differences are the LoRA's unseeded start (series_phase4.md, entry 1), not the kernels. Both bars
met: the 9B confirmation ran next (the chain's gate read the two bars from the record and the guard's log).

### 6. 01:19-02:44: the 9B confirmation of M3 (`confirm_recompute_9b_1x16_100steps`, commit 3504223)
Qwen3.5-9B, 1 x 16 sequences a step, the cache emptied each micro-batch, 4 epochs (100 steps). Steps 1-25 are read against
Phase 4's 9B pair (25-step runs, the same batches); the long run against itself.

| | M3 | Phase 4's kernels (stored) | Phase 4's reference | bar | verdict |
|---|---|---|---|---|---|
| the guard's peak footprint through step 25, GB | 64.35 (the runner's 0.25 s poll: 64.35) | 79.49 | 67.78 | <= 74.6 | MET (0.95x the reference's) |
| steady seconds a step, steps 11-25 (the trainer's clock) | 51.13 | 51.20 | 62.47 | <= 52.7 | MET |
| the driver's maximum at a step's end, steps 1-50 / 51-100, GB | 65.84 / 61.86 | | | 51-100 <= 1-50 + 2 | MET (-3.98) |
| the driver's maximum, steps 1-25, GB | 60.22 | 73.93 | 60.65 | | |
| live tensors at a forward's end, peak, GiB (weights not counted: +17.6) | 20.59 | 25.08 | 27.84 | | |
| steady seconds a step, steps 11-100 | 50.91 | | | | |
| the guard's peak footprint over the 100 steps, GB | 68.13 (reached by step 50, then flat) | | | | |
| loss at steps 1-2 | equal to Phase 4's kernels' to the bit (2.2249152660369873, 2.0693628787994385) | | | | |

- The run's learning-rate schedule spans 100 steps where Phase 4's spanned 25, so from step 3 on its losses are not Phase 4's
  (final 0.0400, mean 0.3429 over four epochs); steps 1-2 come before any update and match to the bit.
- Over the first 25 steps the pool grows at about the stored path's pace, from lower: the driver's maximum over steps 1-5 /
  6-15 / 16-25 is 50.6 / 56.6 / 60.2 GB with the states recomputed, 63.4 / 70.1 / 73.9 for Phase 4's stored kernels (whose
  growth made Phase 4 warn that a whole 9B recipe might run out of memory) and 55.0 / 59.7 / 60.6 for the reference. Over the
  longer run it levels: 65.8 GB at most over steps 1-50 (reached by step 50), 61.9 over steps 51-100.
- The record's `repo_dirty` is true: at its start the report script (`bench/memory_report.py`) had uncommitted edits; `src/` was
  as built in entry 1 (unchanged from 11c1c99 to c11f32c).
- A fault in the procedure: at about 02:03 a pytest filter meant for the CPU-only tests of `tests/test_memory.py` also matched
  one GPU test, which ran beside this run for under a second in its own process (a 2 x 650 x 4-head forward and backward, a few
  MB). It fell in step 51 or 52 (59 s and 53 s, within the 43-65 s the batch lengths give); the steps 11-25 window and every
  memory reading (taken in the runner's own process) are untouched by it.

All three 9B bars met: M3 is admitted.

### 7. 02:4x: M3 becomes the default
`op.RECOMPUTE_STATES = True`; `op.SIZE_CLASSES` stays `False`; both stay selectable, and both off is Phase 3's stored path. The
whole suite, 331 tests (the harness's 203 and 128 here), passes before and after the change; after it the harness runs the
chunked pair through the recomputation against its float64 reference.

### Next (not done here)
1. The pool still fills over a run in every arm, the reference too: at 0.8B the memory held above the live tensors (mean of
   each quarter of the 100 steps) goes 12.4, 17.5, 20.7, 22.5 GB with the recomputed states and 11.6, 18.9, 18.3, 19.6 for
   reference run A (the stored kernels 21.6, 25.3, 27.4, 27.2). The trainer's batching (lengths grouped, padding to a multiple)
   is outside this entry: it would change the reference arm too.
2. Phase 4's deferred whole recipe (P4.3) is no longer held back by the kernels' memory at 9B: the footprint sits under the
   reference's.
3. The rest of the series' designs.
