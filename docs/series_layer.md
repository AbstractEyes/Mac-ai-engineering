# Series: the rest of the layer (D3-D5), a dated log

After the gated delta rule's kernels (Phase 3, [series_phase3.md](series_phase3.md)) and their memory design (D2m,
[series_memory.md](series_memory.md)), a Qwen3.5 LoRA training step on the M3 Max takes 10.98 s at 0.8B (2 x 8 sequences) and
50.91 s at 9B (1 x 16). Where the rest of that time goes was a count of FLOPs, not a measurement. This phase of the series
measures it first (step 0, an instrument), then builds the designs in the order the measurement gives: D4 the layer's
surroundings fused, D5 the matrix products, D3 the rule's chunking; and two entries outside them that step 0 found: D8 the LM
head and the loss at the labelled positions only, D9 the frozen vision tower's output cached per picture. One machine (an M3 Max, 40-core GPU, 128 GB, macOS 26,
PyTorch 2.15, transformers 5.22.1, PEFT 0.21.2, MLX as installed beside them), Phase 4's trainer and runner. The steps, their
predictions and bars were registered before any code (series journal, 2026-11-03 11:2x PST); this log gives each step's setup,
prediction, measurement and verdict.

## Step 0, the step profile (registered before any code)

| part | what | registered |
|---|---|---|
| (a) in situ | real micro-batches of the readers' recipe (the Phase 4 runner: LoRA rank 32 on the record's targets, bf16 autocast, the kernels with the recomputed states, checkpointing off, the cache emptied each micro-batch; 0.8B 2 x 8, 9B 1 x 16, the sample's batches): Metal events at module boundaries, forward and backward, the CPU's issue time beside each; the time by component | shares of a micro-batch, 0.8B / 9B: the layers' frozen products 35-45% / 55-65%; the LoRA adapters' extra work 8-15% / 5-10%; the surroundings 10-15% / 6-10%; the rule 6-9% / 3-5%; the LM head and the loss 12-20% / 8-14%; the vision tower 8-16% / 6-12%; data, optimizer and the Trainer 5-15% / 3-10% |
| validity | | the profiled micro-batch within 10% of an unprofiled one; the components summing to the measured whole within 10% |
| (b) isolated | each part of both decoder layers forward + backward at the bench shapes (0.8B 4 x 1,024, 9B 2 x 1,024) and the training micro-batch's; frozen and with LoRA rank 32; each part's bytes and FLOPs, its floor at 356 GB/s and at the measured product rate | (D4's predictions are graded against these floors) |
| (c) products | PyTorch bf16 at every projection, LoRA and LM-head shape, MLX's mx.matmul at the same (its own process), against the large-square rate | (D5's predictions read these) |
| riders | a segment is dispatch-bound when its CPU issue time is >= 80% of its event time; the parts outside D3-D5 at >= 5% of the step are reported with a proposed entry, not built | |

## The instrument (`bench/timeline.py`)

- A mark records a PyTorch MPS timing event (the point the GPU reached in the command stream) and the CPU's clock when the mark
  was issued. Module boundaries are marked by forward pre / post hooks, module-level functions by wrappers.
- The time between two consecutive marks belongs to the innermost component open there (exclusive time). The sum of all
  segments is the span from the first mark to the last, so the components add up to the whole by construction; the check that
  matters is the span against the wall clock of the same pass.
- The backward is marked by identity autograd nodes placed at the forward's marks, on the tensors that need a gradient. The
  autograd engine runs one device's nodes in the reverse of their creation order, so these nodes run in the reverse order of the
  forward's marks, interleaved with the backward of the operations between them: the backward segment that starts at the node
  of forward mark j belongs to the component that was innermost just before mark j.
- The fence (found while validating, entry 1): a timing event whose command buffer holds no GPU work, as when two marks follow
  each other, is reported late by up to ~0.1 ms while the GPU runs the next buffer. Short segments then come out negative and
  time moves between neighbours. One tiny kernel before each event (an add on a one-element tensor) gives every event work in
  its own command buffer, and the segments then match the operations' own times.
- The CPU's issue time of a segment is reported net of the marks' own cost. It cannot tell a CPU-bound segment from a GPU-bound
  one by itself: when the GPU is behind, Metal's queue of in-flight command buffers fills (each mark commits one), and the CPU
  waits for a free slot inside the next operation's issue. So beside the registered rule (CPU issue time >= 80% of the event
  time) the instrument reads the GPU's lag behind the CPU at each mark (the event's time minus the CPU's issue time, both from
  the pass's first mark): within 1 ms of its minimum the GPU had caught up and was waiting for the CPU. A component is read as
  GPU-waiting when that holds for >= 80% of its time.

### The instrument checked on known cases (`bench/timeline_check.py`, `bench/layer_logs/timeline_check.json`)

Each case with a mark before and after each operation, five rounds, without the fence and with it:

| case | the operation's own time | without the fence | with the fence (the instrument) |
|---|---|---|---|
| a chain of 2,048-cube bf16 products | 1.388 ms (back-to-back, `gpu_ms`) | segments 0.889 of it (median 1.233 ms) | 0.952 (median 1.322 ms, p10-p90 1.317-1.390) |
| 1,024-cube products | 0.186 ms | 0.559 | 0.967 (p10-p90 0.178-0.186) |
| 512-cube products | 0.039 ms | median negative; 145 of 200 segments negative | 0.803 (0.031 ms; the back-to-back figure carries the gaps between kernels) |
| adds of 1 M bf16 elements | 0.011 ms | median -0.060 ms; 1,672 of 2,000 negative | 0.0104 ms, none negative; CPU issue / event 0.86: dispatch-bound by the rule |
| adds of 1 K elements | 0.005 ms | 1,770 of 2,000 negative | 0.0043 ms; CPU / event 2.07: dispatch-bound |
| a stack of 8 linear layers (0.2-5.6 ms each), forward and backward under autocast, module marks and their backward nodes | each layer's product timed alone | each layer 0.47-1.05 of its own time (the small ones lose half) | 0.935-1.043 |
| the stack's components summed against its wall clock | | 0.998 | 0.992 |
| the marked stack's wall clock against the unmarked | | 1.005 | 1.006 |
| a mark's CPU cost | | 5.2 us | 20.8 us |
| the lag across 5 ms of CPU work, the GPU idle | | +0.02 ms | +0.50-0.75 ms (the fence kernel waits for the idle GPU to wake) |
| the lag after 27-28 ms of products issued in 0.4 ms | | +27.3-28.2 ms | +26.8-27.4 ms |

So, as a chain: an event in an empty command buffer is reported late -> without the fence the boundary between two segments
moves by up to ~0.1 ms -> segments shorter than that come out negative and the time of short components lands in their
neighbours -> with one tiny kernel before each event, every boundary sits where the work ends, to within a few percent down to
0.2 ms segments and about 1 us for single elementwise kernels. The one cost the fence brings: after the GPU has idled, the
fence kernel waits for it to wake (0.5-0.75 ms), so the GPU-waiting reading allows 1 ms above the lag's minimum.

## (a) in situ: how it was measured

`bench/profile_step.py` runs Phase 4's runner (`bench/phase4_train.py`, now with a hook that leaves it unchanged by default)
through the guard (`RUNNER=bench/profile_step.py sh bench/phase4_run.sh ...`): the trainer's own `run()`, the runner's
patches, the sample (400 examples), LoRA rank 32 on the record's targets, plain Adam, bf16 autocast, the Metal kernels with
the recomputed states (D2m's default), checkpointing off, the cache emptied each micro-batch, the D2m schedule (4 epochs).
On the profiled optimizer steps (0.8B: 3, 5, 7 of 8; 9B: 2, 4 of 5) every micro-batch runs three times on the same inputs:

| pass | what | used for |
|---|---|---|
| A | as in training, timed by the wall clock only; its loss and gradient are the ones the step keeps | the micro-batch as trained |
| B | again, unprofiled (the input's shapes now in PyTorch's caches) | the unprofiled reference for C; A - B is the cost of a new input shape |
| C | again, profiled: marks at every module and function boundary below, forward and backward | the time by component |

B and C alternate in order from one micro-batch to the next; after both, the gradients are set back to A's, so the run trains
as an unprofiled one does (only dropout's random draws move). The other steps run unprofiled and are compared with the D2m
confirmation runs' same steps (the same batches: the data order is seeded). The step's own parts outside the micro-batches
(collation, gradient clipping, the optimizer, the scheduler and zero_grad, logging, the Trainer's FLOP count and the rest of
its loop) are timed on the profiled steps with the same marks.

The marks (a component's time is its own, its marked children excluded):

| scope | components |
|---|---|
| each linear-attention layer | the two RMSNorms; each projection (in_proj_qkv, in_proj_z, out_proj, gate / up / down) split into its base product and its LoRA parts (PEFT's dropout, lora_A, lora_B, and the rest of PEFT's forward: the input cast, the scale and the add); in_proj_a / in_proj_b (frozen, not adapted); the convolution path (`causal_conv1d_fn`, wrapped); the rule (the patched `torch_chunk_gated_delta_rule`, wrapped); the gated norm; the glue (the linear-attention module's own time: the padding mask, transposes, split, reshapes, beta, g, the repeat at 9B); the SwiGLU (the MLP's own time); the residual adds (the decoder layer's own time) |
| each full-attention layer | the same norms, projections and MLP; q_norm / k_norm; the rotary embedding (`apply_rotary_pos_emb`, wrapped); the attention call (transformers' attention registry, wrapped); the attention module's own time (views, transposes, the output gate) |
| the model | the vision tower (its blocks, patch embedding, merger and the rest; its attention calls; forward only: it is frozen); the token embedding; the picture features' split, the placeholder mask, the 3-D positions; the masks; the rotary tables; the final norm; the LM head; the loss (the model's `loss_function`, wrapped); PEFT's and accelerate's wrappers (autocast, the logits converted to float32) |
| the micro-batch | the inputs moved to the GPU; the cache emptied between forward and backward (the Trainer's `clear_device_cache`, wrapped); the backward call's own time (the loss scaling's backward) |

## (a) in situ: the results

Records: `bench/layer_logs/profile_0.8b_2x8.json`, `profile_9b_1x16.json` (the runs, every profiled micro-batch's components
and two whole timelines each), `profile_results.json` (`bench/profile_report.py`: every figure below). The profiled micro-batches:
0.8B 24 (2 rows, padded to 811 tokens on average: 1,622 positions, 1,386 real tokens, 312 of them with a label, 1.96 pictures
of 3,801 patches); 9B 32 (1 row of 768 tokens on average, 163 with a label, 0.97 pictures of 2,150 patches).

### Validity

| check | bar | 0.8B | 9B | verdict |
|---|---|---|---|---|
| the profiled pass C against the unprofiled replay B of the same micro-batch (over all profiled micro-batches) | within 10% | 1.045 (median 1.042; single micro-batches 0.914-1.153, one of 24 over 1.10) | 1.012 (median 1.009; 0.962-1.079, none over 1.10) | MET |
| the components' sum (the events' span) against C's wall clock | within 10% | 0.9996 (lowest 0.9987) | 0.9998 (lowest 0.9988) | MET |
| this run's unprofiled steps against D2m's same steps (the same batches) | | steps 2 / 4 / 6 / 8: 1.009 / 1.005 / 0.987 / 0.999 | steps 3 / 5: 0.999 / 1.007 | |
| the profiled steps as trained (A and the step's own parts) against D2m's same steps | | steps 3 / 5 / 7: 1.000 / 0.979 / 0.974 | steps 2 / 4: 1.005 / 0.999 | |
| the step rebuilt from the components (less the instrument's C - B, plus A - B) against the profiled steps as trained | | 9.81 s against 9.82 s | 51.81 s against 51.83 s | |
| stack mismatches; negative segments | none | 0; 0 | 0; 0 | |

The instrument costs 4.2% of the profiled 0.8B micro-batch (4,067 marks in 1.23 s) and 1.2% at 9B (5,412 marks in 3.23 s).
A new input shape costs 26.5 ms a micro-batch at 0.8B (2.1%) and 23.4 ms at 9B (0.7%): the first pass of a micro-batch against
its replay (PyTorch builds and caches a graph per shape).

### The time by group, and the registered predictions graded

Shares of the profiled micro-batch with its part of the step's own parts (the registered reading); the new-shape cost stands
beside them on the same scale. A step is 8 micro-batches at 0.8B (1.253 s each: 10.02 s), 16 at 9B (3.253 s: 52.04 s).

| group | 0.8B ms a micro-batch | 0.8B share | predicted | verdict | 9B ms a micro-batch | 9B share | predicted | verdict |
|---|---|---|---|---|---|---|---|---|
| the layers' frozen products | 285.9 | 22.8% | 35-45% | MISSED (lower) | 1,785.1 | 54.9% | 55-65% | MISSED (lower, by 0.1 point) |
| the LoRA adapters' extra work | 160.4 | 12.8% | 8-15% | MET | 256.1 | 7.9% | 5-10% | MET |
| the surroundings | 186.8 | 14.9% | 10-15% | MET | 292.0 | 9.0% | 6-10% | MET |
| the rule | 90.3 | 7.2% | 6-9% | MET | 104.8 | 3.2% | 3-5% | MET |
| the LM head and the loss | 266.7 | 21.3% | 12-20% | MISSED (higher) | 347.6 | 10.7% | 8-14% | MET |
| the vision tower | 138.5 | 11.1% | 8-16% | MET | 317.5 | 9.8% | 6-12% | MET |
| data, optimizer and the Trainer | 57.6 | 4.6% | 5-15% | MISSED (lower) | 49.1 | 1.5% | 3-10% | MISSED (lower) |
| (not registered) the full-attention layers' attention | 55.0 | 4.4% | | | 66.6 | 2.0% | | |
| (not registered) the rest of the model | 11.9 | 0.9% | | | 33.9 | 1.0% | | |
| (beside the shares) a new input shape, A - B | 26.5 | 2.1% | | | 23.4 | 0.7% | | |

Groups as graded: the frozen products are every projection's base product, forward and the gradient to its input (in_proj_a /
in_proj_b included); the LoRA work is PEFT's dropout, lora_A, lora_B and the rest of its forward, forward and backward with the
adapters' weight gradients; the surroundings are both RMSNorms, the convolution path, the gated norm, the glue (beta / g
included), the SwiGLU, the residual adds and, in the full-attention layers, the q / k norms, the rotary embedding and the output
gate; the LM head and the loss include accelerate's conversion of the logits to float32; data, optimizer and the Trainer are
the collation, the inputs moved to the GPU, the cache emptied each micro-batch, clipping, the optimizer, the scheduler and
zero_grad, logging, and the Trainer's own work between micro-batches.

Why the misses, as causal chains:
- Frozen products, 0.8B: their time was counted right (the journal's 353 GFLOP a layer at 4,096 tokens at the product rate is
  ~10.7 ms a layer at the micro-batch's 1,622 positions, 257 ms for 24 layers; measured 286 ms, at 10.6-12.6 TFLOP/s in situ,
  in_proj_qkv's backward 9.4 with its transposed gradient) -> the miss is in the rest: the LM head and the loss (267 ms), the
  surroundings (187), the LoRA work (160) and the vision tower (139) are each larger than the prediction assumed beside a
  1,024-wide model -> the micro-batch is 1.25 s and the products 23% of it.
- Frozen products, 9B: 54.9% against a floor of 55%; the products run at 11.3-12.8 TFLOP/s in situ (the narrow k / v
  projections 9.4-11.9), the large-square rate of (c). Within the instrument's 1.2% and the spread between runs: at the edge,
  not a different picture.
- The LM head and the loss, 0.8B: the vocabulary (248,320) is 243 times the 0.8B's width -> the LM head's two products
  (forward and the gradient to its input, 151 ms) cost as much as 13 of the 24 layers' frozen products together, and the loss
  makes several float32 passes over a 1,622 x 248,320 tensor (109 ms) -> 21.3%. Of the 1,622 positions, 312 carry a label (19%).
- Data, optimizer and the Trainer: the optimizer and clipping take 23 ms a step at 0.8B (20 M adapter weights) and 40 ms at 9B
  (80 M); the collation 116 / 144 ms a step; the Trainer's FLOP count, which walks the model's parameters after every
  micro-batch, 46 / 126 ms a step; the cache emptied each micro-batch 28 / 22 ms a micro-batch -> 4.6% / 1.5%, under the
  registered floors.

### The components

0.8B, the components at 1% of the micro-batch or more (ms a micro-batch; the CPU's net issue time and the GPU's time waiting for
the CPU, forward / backward):

| component | forward | backward | share | CPU net | GPU waiting |
|---|---|---|---|---|---|
| the LM head | 74.1 | 76.5 | 12.0% | 0.3 / 2.4 | 0 / 0 |
| the loss | 25.5 | 83.4 | 8.7% | 0.1 / 3.3 | 0 / 0 |
| the rule | 33.6 | 56.7 | 7.2% | 6.5 / 3.3 | 0 / 0 |
| the vision blocks | 84.3 | | 6.7% | 110.9 | 37.6 |
| in_proj_qkv, base product | 30.4 | 39.0 | 5.5% | 0.6 / 3.2 | 0 / 0 |
| the full-attention layers' attention (sdpa) | 20.0 | 35.0 | 4.4% | 1.7 / 10.1 | 0 / 0 |
| the convolution path | 11.7 | 38.0 | 4.0% | 1.2 / 5.5 | 0 / 0 |
| the gated norm | 18.1 | 28.5 | 3.7% | 5.2 / 1.5 | 0 / 0 |
| down_proj / up_proj / gate_proj base (linear-attention layers) | 20.1 / 18.1 / 18.0 | 17.1 / 18.3 / 18.2 | 3.0 / 2.9 / 2.9% | | 0 |
| the vision tower's attention | 33.3 | | 2.7% | 1.2 | 3.7 |
| the cache emptied between forward and backward | 28.2 | | 2.2% | 143.1 (a synchronization) | 28.2 |
| in_proj_z / out_proj base | 11.6 / 11.1 | 10.8 / 10.1 | 1.8 / 1.7% | | 0 |
| in_proj_qkv, lora_B | 1.7 | 20.0 | 1.7% | 0.6 / 6.5 | 0 / 0.7 |
| the glue | 3.6 | 14.7 | 1.5% | 5.1 / 2.5 | 0.1 / 0 |
| in_proj_qkv, the rest of PEFT's forward | 8.2 | 6.9 | 1.2% | 1.5 / 0.9 | 0 / 0 |
| collation (the step's, spread over its micro-batches) | 14.5 | | 1.2% | 14.5 | 14.5 |
| post-attention / input RMSNorm (linear-attention layers) | 4.8 / 4.7 | 9.6 / 9.6 | 1.1 / 1.1% | | 0 |
| q_proj base | 6.8 | 7.0 | 1.1% | | 0 |
| the vision tower's own code (positions, interpolation) | 13.1 | | 1.1% | 12.7 | 12.2 |

9B, the same reading:

| component | forward | backward | share | CPU net | GPU waiting |
|---|---|---|---|---|---|
| up_proj / gate_proj / down_proj base (linear-attention layers) | 157.9 / 155.7 / 161.4 | 152.8 / 152.7 / 146.4 | 9.6 / 9.5 / 9.5% | | 0 |
| the LM head | 146.1 | 137.7 | 8.7% | 0.6 / 2.4 | 0 / 0 |
| the vision blocks | 224.6 | | 6.9% | 289.7 | 104.2 |
| in_proj_qkv base | 104.0 | 103.5 | 6.4% | 1.3 / 2.0 | 0 |
| in_proj_z / out_proj base | 53.7 / 53.6 | 52.4 / 51.2 | 3.3 / 3.2% | | 0 |
| the rule | 33.1 | 71.7 | 3.2% | 6.4 / 6.1 | 0 |
| down / up / gate base (full-attention layers) | 54.3 / 51.4 / 51.1 | 48.3 / 50.9 / 50.6 | 3.1% each | | 0 |
| the vision tower's attention | 74.5 | | 2.3% | 1.9 | 6.6 |
| q_proj base | 34.6 | 35.9 | 2.2% | | 0 |
| the full-attention layers' attention | 22.8 | 43.8 | 2.1% | 3.8 / 9.3 | 0 |
| the loss | 12.0 | 48.4 | 1.9% | 0.2 / 3.6 | 0 |
| the convolution path | 8.8 | 46.6 | 1.7% | 2.5 / 6.6 | 0 |
| the gated norm | 19.0 | 35.4 | 1.7% | 3.1 / 3.1 | 0 |
| post-attention / input RMSNorm (linear-attention layers) | 13.8 / 12.4 | 24.3 / 24.8 | 1.2 / 1.1% | | 0 |
| o_proj base | 18.2 | 17.9 | 1.1% | | 0 |

The surroundings by D4's parts, and the LoRA work by its parts (ms a micro-batch, forward + backward; share):

| part | 0.8B | 9B |
|---|---|---|
| the convolution path (depthwise conv + SiLU) | 11.7 + 38.0 = 49.7 (4.0%) | 8.8 + 46.6 = 55.4 (1.7%) |
| the zero-centered RMSNorms (input, post-attention, q / k norms) | 16.8 + 35.5 = 52.3 (4.2%) | 39.9 + 77.1 = 117.0 (3.6%) |
| the gated RMSNorm | 18.1 + 28.5 = 46.5 (3.7%) | 19.0 + 35.4 = 54.5 (1.7%) |
| the glue (padding mask, transposes, split, reshapes, beta, g, the repeat) | 3.6 + 14.7 = 18.2 (1.5%) | 10.5 + 16.7 = 27.2 (0.8%) |
| the SwiGLU | 3.1 + 7.0 = 10.1 (0.8%) | 7.2 + 15.5 = 22.7 (0.7%) |
| the full-attention layers' rotary embedding, views and output gate | 6.5 (0.5%) | 8.8 (0.3%) |
| the residual adds | 3.4 (0.3%) | 6.5 (0.2%) |
| LoRA: the rest of PEFT's forward (the input cast to float32, the scale, the add, the cast back) | 63.0 (5.0%) | 117.6 (3.6%) |
| LoRA: lora_B's products | 46.5 (3.7%) | 42.3 (1.3%) |
| LoRA: lora_A's products | 31.8 (2.5%) | 60.5 (1.9%) |
| LoRA: dropout | 19.1 (1.5%) | 35.8 (1.1%) |

In both models the convolution path's backward costs 3.2-5.3x its forward, and the norms' backwards 1.6-2.1x theirs. Of the
LoRA work, in_proj_qkv's lora_B backward alone is 20.0 ms at 0.8B: its gradient arrives in the transposed layout the
convolution path leaves (the isolated bench and the products below measure that layout).

### Where the GPU waits for the CPU (the dispatch-bound reading)

By the registered rule (the CPU's net issue time >= 80% of the event time, components of 0.5 ms or more) 35 components (each
direction counted apart) are dispatch-bound at 0.8B and 53 at 9B; most are small LoRA products in the backward, where the GPU
never waited: the CPU's issue time there holds its waits for a slot in the command queue while the GPU works through older
work. The GPU's own reading (it had caught up with the CPU for >= 80% of the component's time) keeps the ones that matter:

| component | 0.8B: event ms / GPU waiting ms | 9B | why |
|---|---|---|---|
| the vision blocks | 84.3 / 37.6 (45%; the rule: dispatch-bound) | 224.6 / 104.2 (46%; the rule: dispatch-bound) | `lengths.tolist()` in every block's attention (a GPU-to-CPU synchronization a block: 12 / 27 a micro-batch), then the per-picture attention calls |
| the cache emptied between forward and backward | 28.2 / 28.2 | 21.9 / 20.6 | a synchronization, then the release |
| collation (the step's, per micro-batch) | 14.5 / 14.5 | 9.0 / 9.0 | the collator on the main process; the GPU idle |
| the vision tower's own code | 13.1 / 12.2 | 10.5 / 9.2 | position interpolation and ids on the CPU |
| the 3-D position ids | 6.3 / 6.3 | 4.0 / 4.0 | `get_rope_index`'s Python loops |
| the Trainer's FLOP count | 5.7 / 5.3 | 7.9 / 7.4 | `num_parameters()` walks the model after every micro-batch |
| the rest: the training step's own code, the masks, the picture merge and split, the backward's start, the Trainer's loop | 10.5 / 10.0 | 17.1 / 15.0 | Python between the GPU's work |

In all, the GPU waits for the CPU 101 ms of a 0.8B micro-batch (8.1%) and 164 ms at 9B (5.0%), the vision tower 55 / 121 ms of
it; inside the decoder layers 2.4 / 2.9 ms (the CPU runs 4-10 ms ahead of the GPU there, as far as the command queue allows).

## (b) isolated: how it was measured

`bench/profile_layer.py`: one linear-attention and one full-attention decoder layer of each model, built from its config in
bf16 (weights drawn at the scale of trained ones), the Metal kernels installed, run under bf16 autocast as the Trainer runs
them. Each part runs on inputs laid out as the layer hands them (the convolution path on the projection's transpose; the rule
on the strided q / k / v the glue makes; in_proj_qkv a second time with its output transposed, so its backward gets the
gradient in the layer's layout). Forward, and forward + backward to the part's inputs (and to the LoRA weights in the LoRA
variant), Metal events, back-to-back calls filling at least ~10 ms, the median of 7 rounds. Two variants: frozen, and LoRA
rank 32 (alpha 64, dropout 0.05) injected by PEFT on the record's targets, the adapters in float32 as PEFT keeps them. Shapes:
the bench's (0.8B 4 x 1,024, 9B 2 x 1,024) and the training micro-batch's median padded one (from (a)); at a two-row
micro-batch shape the second row is padded from 60% of its length, so the masks take the padded paths the training takes.

Each part's bytes are the least traffic it needs as one pass: each input read once, each output written once; the backward
reads the output's gradient and what it needs of the inputs and writes the inputs' gradients (a projection with its adapter
counted as fused with the base product). Its floor is the larger of bytes / 356 GB/s (bench/hw_ceilings.py's copy rate) and
FLOPs / the large-square product rate measured by (c) in the same session.

## (b) isolated: the results (`bench/layer_logs/layer_parts.json`)

Forward / forward + backward in ms, and in brackets forward + backward against the part's floor (the floors at 356 GB/s and
12.9 TFLOP/s, the highest large-square rate PyTorch reached in the session's records). Frozen at the bench shapes (D4's
registered layer baseline); with LoRA at the micro-batch shapes (the training's configuration).

The linear-attention decoder layer:

| part | 0.8B frozen 4 x 1,024 | 0.8B LoRA 2 x 903 | 9B frozen 2 x 1,024 | 9B LoRA 1 x 859 |
|---|---|---|---|---|
| input RMSNorm | 0.73 / 2.27 (19.3x) | 0.29 / 0.91 (17.5x) | 1.44 / 4.44 (18.9x) | 0.56 / 1.88 (19.0x) |
| the padding mask | | 0.11 / 0.27 (6.6x) | | |
| in_proj_qkv | 4.29 / 8.39 (1.05x) | 2.56 / 5.26 (1.41x) | 11.16 / 22.03 (1.03x) | 5.64 / 11.06 (1.21x) |
| in_proj_qkv, its output transposed as the layer uses it | 4.12 / 9.24 (1.15x) | 2.72 / 6.76 (1.82x) | 11.30 / 22.05 (1.03x) | 5.63 / 11.12 (1.22x) |
| in_proj_z | 1.44 / 2.85 (1.07x) | 0.97 / 1.96 (1.55x) | 5.81 / 11.02 (1.03x) | 3.07 / 5.96 (1.30x) |
| in_proj_a / in_proj_b (each) | 0.04 / 0.08 (1.5-1.6x) | 0.03 / 0.07 (3.0x) | 0.11 / 0.19-0.20 (2.0x) | 0.07 / 0.10 (2.5x) |
| the convolution path | 1.58 / 5.99 (8.5x) | 0.73 / 3.21 (10.3x) | 1.07 / 4.21 (8.9x) | 0.40 / 2.47 (12.5x) |
| the glue (beta, g, the layout; the repeat at 9B) | 0.06 / 0.46 (1.6x) | 0.05 / 0.27 (2.1x) | 1.31 / 2.24 (5.9x) | 0.45 / 0.97 (6.1x) |
| the rule | 4.39 / 12.14 (5.4x) | 2.02 / 5.59 (5.6x) | 3.63 / 11.52 (5.1x) | 1.58 / 4.81 (5.1x) |
| the gated RMSNorm | 2.33 / 6.64 (17.6x) | 0.94 / 3.00 (18.0x) | 2.36 / 6.64 (17.6x) | 0.90 / 2.69 (17.1x) |
| out_proj | 1.45 / 2.93 (1.10x) | 1.10 / 2.20 (1.75x) | 5.72 / 11.03 (1.03x) | 2.97 / 5.99 (1.31x) |
| post-attention RMSNorm | 0.74 / 2.29 (19.4x) | 0.29 / 0.90 (17.4x) | 1.35 / 4.27 (18.1x) | 0.55 / 1.90 (19.3x) |
| gate_proj / up_proj / down_proj | 4.97 / 4.82 / 4.86 (1.03-1.06x) | 3.40 / 3.41 / 3.55 (1.56-1.63x) | 33.21 / 33.42 / 33.17 (1.04x) | 16.08 / 16.20 / 16.19 (1.18-1.19x) |
| the SwiGLU | 0.36 / 1.10 (1.7x) | 0.11 / 0.43 (1.5x) | 0.68 / 2.01 (1.8x) | 0.27 / 0.75 (1.6x) |
| the two residual adds | 0.06 + 0.06 (0.8x) | 0.04 + 0.04 (1.4x) | 0.13 + 0.13 (0.9x) | 0.05 + 0.06 (0.9-1.0x) |
| the parts summed / the whole layer | 60.0 / 61.6 | 34.6 / 35.4 | 179.9 / 180.3 | 87.3 / 85.6 |
| the floors summed | 32.1 | 14.9 | 144.3 | 61.5 |

The full-attention decoder layer:

| part | 0.8B frozen 4 x 1,024 | 0.8B LoRA 2 x 903 | 9B frozen 2 x 1,024 | 9B LoRA 1 x 859 |
|---|---|---|---|---|
| input / post-attention RMSNorm | 2.21 / 2.14 (18.8x / 18.2x) | 0.97 / 0.91 (18.6x / 17.4x) | 4.46 / 4.49 (18.9x / 19.1x) | 1.90 / 1.78 (19.2x / 18.1x) |
| q_proj | 2.77 / 5.49 (1.03x) | 1.90 / 3.79 (1.52x) | 10.88 / 21.41 (1.00x) | 5.52 / 11.04 (1.21x) |
| k_proj / v_proj (each) | 0.75 (1.13x) | 0.73 (2.19x) | 2.97-2.98 (1.11-1.12x) | 2.01-2.07 (1.7-1.75x) |
| the views before (the gate's copy) | 0.33 (1.16x) | 0.14 (1.12x) | 0.33 (1.17x) | 0.13 (1.11x) |
| q_norm / k_norm | 4.76 / 1.07 (20.2x / 18.3x) | 2.02 / 0.51 (19.5x / 19.6x) | 4.81 / 1.16 (20.4x / 19.7x) | 1.92 / 0.50 (19.5x / 20.1x) |
| the rotary embedding | 0.56 / 1.46 (6.2x) | 0.31 / 0.75 (7.2x) | 0.57 / 1.45 (6.2x) | 0.32 / 0.76 (7.7x) |
| the attention (sdpa; the floor: no T x T matrix, the causal half) | 8.28 / 23.87 (5.1x) | 3.93 / 10.58 (5.8x) | 8.39 / 23.90 (5.1x) | 3.28 / 9.41 (5.7x) |
| the output gate and views after | 0.65 (1.7x) | 0.21 (1.3x) | 0.62 (1.7x) | 0.20 (1.2x) |
| o_proj | 2.94 (1.10x) | 2.14 (1.70x) | 10.88 (1.02x) | 5.99 (1.31x) |
| gate_proj / up_proj / down_proj | 4.95 / 4.82 / 5.02 (1.03-1.07x) | 3.27 / 3.21 / 3.73 (1.47-1.71x) | 32.90 / 32.04 / 33.35 (1.00-1.04x) | 15.65 / 16.03 / 16.57 (1.15-1.21x) |
| the SwiGLU | 1.23 (1.9x) | 0.38 (1.3x) | 2.07 (1.8x) | 0.80 (1.7x) |
| the parts summed / the whole layer | 62.6 / 62.2 | 34.2 / 34.2 | 180.1 / 180.5 | 86.9 / 85.7 |
| the floors summed | 30.2 | 13.8 | 141.2 | 59.9 |

What the isolated parts show, as chains:
- The parts add up to the whole layer within 1-5% at every shape (0.956-1.020): the parts measured in isolation are the layer's.
- The frozen projections sit at 1.00-1.10x their compute floor (the products run at the rate). With LoRA at the micro-batch
  shapes they sit at 1.2-1.8x: the adapters' passes cost 20-60% of a projection, more at 0.8B (narrower products).
- Every norm runs at 17-20x its bandwidth floor: the zero-centered RMSNorm and the q / k norms upcast to float32 and run the
  mean, rsqrt, scale and cast as separate full passes, forward and backward; the gated norm adds the SiLU gate's passes. These
  are the largest gaps in the layer.
- The convolution path runs at 8.5-12.5x its floor, its backward 3-6x its forward (a depthwise conv1d backward on the transposed
  view, then the copies back to the projection's layout).
- The rotary embedding at 6-8x (slices, a cat and a rotate-half, each a pass); the glue at 9B at 6x (the repeat of q and k).
- The SwiGLU is already at 1.3-1.9x its floor and the residual adds at it (their small working sets stay in the chip's cache
  across back-to-back calls, so the adds run past the DRAM floor): fusing them can win little.
- in_proj_qkv with its output transposed as the layer uses it: at 0.8B the forward + backward costs 10% (frozen, 4 rows) to 29%
  (LoRA, 2 rows) more than with a plain gradient; at 9B nothing measurable (frozen at 2 rows, LoRA at 1 row). The products
  (c) locate it: with two or more rows the LoRA-B backward runs 5x slower on the transposed gradient; with one row the
  transpose is a plain transposed matrix and costs nothing.
- Run 1 of this bench (`layer_parts_run1.json`, the same times within a few percent) carried two byte formulas that counted
  copies the layer does not make (the glue's q / k / v and the attention's query are views) and fed the residual add the same
  memory twice; this run corrects them. Its compute floors used 11.5 TFLOP/s (F.linear's large squares) where this run uses
  12.9.

## (c) products: how it was measured

`bench/profile_products.py`: at each shape, every product the training step runs with frozen weights, as autograd runs it:
each projection's forward (F.linear) and its backward to the input; for the LoRA targets the rank-32 products (lora_A and
lora_B forward; lora_B's backward, dH = dY B and dB = dY^T H; lora_A's, dX = dH A and dA = dH^T X); the LM head (forward and
the backward to its input, 248,320 words); in_proj_qkv's backward products also with the gradient in the layer's layout.
PyTorch: Metal events and the wall clock over back-to-back calls; MLX: mx.matmul in bf16, the wall clock around mx.eval of
the same number of calls, in its own process after the PyTorch runs. The large-square products (2,048, 4,096 and 8,192 cubed,
bf16) at the start and the end of each session are the control; `--square-forms` measures them in each form PyTorch can run
(X W by torch.mm, X W^T by F.linear as nn.Linear runs it).

## (c) products: the results (`bench/layer_logs/products_torch.json`, `products_mlx.json`, `products_square_forms.json`)

The control first, because it moved:

| large squares, bf16, TFLOP/s | 2,048 | 4,096 | 8,192 |
|---|---|---|---|
| PyTorch F.linear (X W^T), the products run, start / end (12:20-12:23) | 11.46 / 11.48 | 11.39 / 11.34 | 11.51 / 11.43 |
| PyTorch torch.mm (X W, preallocated output), the instrument check (11:52) | 12.81-12.88 | | |
| MLX mx.matmul (X W^T), start / end (12:26-12:27) | 12.64 / 11.89 | 12.82 / 12.44 | 12.59 / 12.88 |
| PyTorch, each form, two interleaved rounds, after ~10 minutes of continuous GPU work (12:27) | X W 11.90-12.41; X W^T 11.21-11.33 | X W 10.45-12.88; X W^T 9.15-10.20 | X W 9.80-10.02; X W^T 8.87-9.20 |

- The form matters: where both ran side by side, torch.mm's X W is 6-10% faster than F.linear's X W^T (the form every nn.Linear
  runs). Stage 0's 12.8-12.9 TFLOP/s is the X W form.
- The chip's state matters more: after ten minutes of continuous products the large squares fell to 9-10 TFLOP/s. Rates from
  different minutes differ by up to ~20%; the readings below come from one session each (PyTorch, then MLX), and any design
  that claims a speed-up must time its baseline and itself in interleaved rounds.

The products carrying the most time in situ, at the micro-batch's median shape (PyTorch by Metal events, MLX by the wall
clock; the large-square references: PyTorch's F.linear 11.5 TFLOP/s in the same run, the best measured 12.9):

| model | product | in-situ ms a micro-batch (share) | PyTorch ms, TFLOP/s | MLX ms, TFLOP/s | PyTorch / MLX (wall clocks) |
|---|---|---|---|---|---|
| 0.8B 2 x 903 | the LM head, forward | 74.1 (5.9%) | 79.6, 11.54 | 72.4, 12.68 | 0.89 |
| | the LM head, backward (dX) | 76.5 (6.1%) | 83.7, 10.97 | 85.2, 10.78 | 1.07 |
| | in_proj_qkv backward, the layer's layout | 39.0 (3.1%) | 2.50, 9.07 | 2.04, 11.12 | 0.84 |
| | in_proj_qkv forward | 30.4 (2.4%) | 1.98, 11.49 | 1.84, 12.35 | 0.96 |
| | gate / up forward; backward | 18.1; 18.3 a projection (1.4-1.5%) | 1.15, 11.53; 1.15, 11.54 | 1.14, 11.59; 1.09, 12.16 | 0.99; 0.92 |
| | down forward; backward | 20.1; 17.1 (1.6%; 1.4%) | 1.20, 11.02; 1.17, 11.37 | 1.10, 12.04; 1.09, 12.18 | 0.90; 0.94 |
| 9B 1 x 859 | down forward; backward | 161.4; 146.4 (5.0%; 4.5%) | 7.45, 11.60; 6.78, 12.76 | 7.37, 11.73; 7.22, 11.98 | 0.98; 1.06 |
| | gate / up forward; backward | 157.9; 152.8 a projection (4.9%; 4.7%) | 7.04, 12.28; 7.16, 12.09 | 7.19, 12.03; 7.32, 11.82 | 1.01; 1.01 |
| | the LM head, forward; backward | 146.1; 137.7 (4.5%; 4.2%) | 157.0, 11.13; 156.2, 11.19 | 143.3, 12.19; 164.2, 10.64 | 0.91; 1.06 |
| | in_proj_qkv forward; backward (the layer's layout) | 104.0; 103.5 (3.2% each) | 4.91, 11.74; 4.96, 11.63 | 4.79, 12.03; 4.78, 12.06 | 0.98; 0.96 |

Across all projections and both token counts (`products_torch.json`, `products_mlx.json`):
- PyTorch's frozen products run at 8.9-12.9 TFLOP/s (in_proj_a / in_proj_b, 16-32 outputs wide, 1.3-4.9); at 0.83-1.07 of
  MLX's rate (the lowest: 9B's k / v forward 0.83, 0.8B's in_proj_qkv backward in the layer's layout 0.84-0.86, 0.8B's k / v
  forward 0.84). No shape that carries 5% of the step runs under 80% of MLX; the largest of the low ones, in_proj_qkv's backward
  in the layer's layout at 0.8B, carries 3.1%.
- The rank-32 products run at 1.1-8.9 TFLOP/s (0.10-0.77 of the 11.5 large-square rate, median 0.45). The outlier is
  in_proj_qkv's LoRA-B backward with the layer's transposed gradient at two rows or more: 1.2-2.5 ms against 0.24-0.48 ms with a
  plain gradient (5.0-5.3x), and MLX takes half PyTorch's time there (0.44-0.52 of its rate). At one row (the 9B micro-batch) it
  costs nothing.
- The in-situ rates agree with the isolated ones: the frozen products in the profiled 0.8B step ran at 10.6-12.6 TFLOP/s
  (in_proj_qkv's backward 9.4), at 9B at 11.3-12.8.

## Step 0's verdicts in short

| registered | 0.8B | 9B |
|---|---|---|
| validity: the profiled micro-batch within 10% of an unprofiled one | 1.045 | 1.012 |
| validity: the components summing to the whole within 10% | 0.9996 | 0.9998 |
| the layers' frozen products | 22.8%, MISSED (35-45%) | 54.9%, MISSED by 0.1 point (55-65%) |
| the LoRA adapters' extra work | 12.8%, met | 7.9%, met |
| the surroundings | 14.9%, met | 9.0%, met |
| the rule | 7.2%, met | 3.2%, met |
| the LM head and the loss | 21.3%, MISSED (12-20%) | 10.7%, met |
| the vision tower | 11.1%, met | 9.8%, met |
| data, optimizer and the Trainer | 4.6%, MISSED (5-15%) | 1.5%, MISSED (3-10%) |

In plain words: the instrument holds (it moves a profiled micro-batch by 4% at 0.8B and 1% at 9B, and its components add up to
the whole). At 9B the step is the matrix products (55%, at the hardware's product rate, PyTorch level with MLX); at 0.8B the
products are under a quarter of the step, and the vocabulary-wide work (the LM head and the loss, 21%) is as large as all the
layers' products together. The layer's surroundings, the LoRA adapters' extra passes and the vision tower each take about a
tenth; the rule a few percent.

## The parts outside D3-D5 at 5% of the step or more: proposed entries (not built)

| part | share 0.8B / 9B | the cause | proposed entry | expected |
|---|---|---|---|---|
| the LM head and the loss (the float32 logits included) | 21.3% / 10.7% (267 / 348 ms a micro-batch) | the logits are computed at every position (1,622 / 768 a micro-batch), while the collator labels only the answer: 312 / 163 positions carry a label (19% / 21%); the loss upcasts the whole logits tensor to float32 and accelerate converts the returned logits to float32 again | compute the LM head and the loss at the labeled positions only (the hidden states gathered where the shifted labels are not ignored, through the model's `logits_to_keep` index or a compute_loss of the runner's own), then, if still large, a chunked fused cross-entropy (Liger's fused linear cross-entropy, Cut Cross-Entropy) that never holds float32 logits | the vocabulary work cut ~5x: ~215 ms a 0.8B micro-batch (-17% of the step, ~1.7 s of 10.0) and ~275 ms at 9B (-8%, ~4.4 s of 52); the largest float32 tensors (1.6 GB each at 0.8B's micro-batch) cut as much |
| the vision tower | 11.1% / 9.8% (139 / 318 ms) | the tower is frozen and the pictures fixed, yet it runs again on every picture every epoch; and every block synchronizes the GPU with the CPU (`lengths.tolist()` before the per-picture attention), so the GPU waits for the CPU for ~45% of the blocks' time | compute each picture's merged features once (the model takes precomputed `mm_encoder_outputs`) and feed them from the collator; ~1 MB a picture at 0.8B, ~4.5 MB at 9B (372 pictures: 0.4 / 1.7 GB). A smaller step: the block lengths read once a forward | the tower's whole time from the second epoch on (-11% / -10% of the step), or from the first with a pass over the pictures before training (~0.5 / ~2 min once) |

Below 5%, for the record: the cache emptied each micro-batch (D2m's fix: 2.2% / 0.7%), the full-attention layers' attention
(sdpa, 4.4% / 2.0%, at 5-6x a flash-style floor), the collation (1.2% / 0.3%), the Trainer's FLOP count (it walks the model's
parameters after every micro-batch: 46 / 126 ms a step), a new input shape (2.1% / 0.7%).

## What step 0 hands to D4, D5 and D3

- D4, in the order of the measured shares (0.8B / 9B of the micro-batch; forward + backward against the floor in isolation):
  1. the zero-centered RMSNorms, input and post-attention, with the full-attention layers' q / k norms: 4.2% / 3.6%; 17-20x;
  2. the convolution path: 4.0% / 1.7%; 8.5-12.5x, its backward 3-6x its forward. Done in the projection's own layout it also
     takes away in_proj_qkv's transposed gradient, which costs its base backward 18% at 0.8B (9.1 against 11.1 TFLOP/s) and its
     LoRA-B backward 5x (20 ms a 0.8B micro-batch);
  3. the gated RMSNorm: 3.7% / 1.7%; 17-18x;
  4. the glue (beta, g, the layout; the repeat at 9B): 1.5% / 0.8%; 1.6-2.1x at 0.8B, 6x at 9B;
  5. the SwiGLU: 0.8% / 0.7%, already at 1.3-1.9x its floor: within D4's 3x bar as it is, and the registered 1.5x speed bar
     needs it at 0.9-1.3x its floor;
  (not registered: the rotary embedding, 0.5% / 0.3%, at 6-8x).
  The registered D4 predictions look reachable by the floors: the two RMSNorms, the gated norm and the convolution path at 3x
  their floors take ~13 ms from the 0.8B bench layer (61.6 -> ~48 ms; predicted 44-50) and, with the glue's repeat, ~17 ms from
  the 9B one (180.3 -> ~164 ms; predicted 158-170).
- D5: (a) the frozen products already run at the rate (10.6-12.8 TFLOP/s in situ) and PyTorch is within 0.83-1.07 of MLX at
  every shape: by the registered rule no custom product (no shape carrying 5% runs under 80% of MLX). (b) The LoRA work is 12.8%
  / 7.9% of the step (over the 5% threshold): the fused adapter path is in scope. By part: the rest of PEFT's forward (the input
  cast to float32, the scale, the add, the cast back) 5.0% / 3.6%, lora_B 3.7% / 1.3%, lora_A 2.5% / 1.9%, dropout 1.5% / 1.1%.
- D3: the rule is 7.2% / 3.2%, at 5.1-5.6x a rough compute floor (the chunk-16 algorithm's products): last, as registered.
- For every design: rates move by up to ~20% with the chip's state (the large squares fell from 11.5 to 9-10 TFLOP/s after ten
  minutes of continuous work), so each comparison runs the baseline and the design in interleaved rounds; and the step profile
  re-run after each admission reads the shares with the same instrument.

## Not done, and what is uncertain

- The profiled steps are early in the run (0.8B steps 3-7, 9B steps 2-4); D2m's early and steady steps agree (0.8B: steps 3-10
  11.03 s, steps 11-100 10.98 s), but a new shape's cost (A - B) is measured where the graph cache is coldest.
- The 9B frozen products' miss (54.9% against 55%) is inside the instrument's 1.2% and the run-to-run spread.
- The registered dispatch-bound rule reads the CPU's issue time, which holds waits for a free command-buffer slot when the GPU is
  behind: by it 35 / 53 components are dispatch-bound, most of them small backward products the GPU never waited on. The GPU's
  lag reading is added beside it; for the vision tower the two disagree (the rule: dispatch-bound; the GPU waits 45-46% of the
  tower's time).
- The floors: the rule's FLOP count is a rough count of the chunked algorithm (D3 will count it); the attention's floor assumes
  a flash-style kernel and the causal half; elementwise parts whose working set stays in the chip's cache can beat the DRAM
  floor (the residual adds at 0.8-1.0x).
- The isolated bench pads a two-row micro-batch's second row from 60% of its length, a stand-in for the real rows' lengths.
- The square-forms control ran after ten minutes of GPU work, throttled; its X W against X W^T comparison holds within its own
  interleaved rounds only.
- MLX's products are timed by the wall clock (PyTorch's by Metal events and the wall clock: the two agree within a few percent
  at these sizes).

## D4, the layer's surroundings fused (registered before any code)

| item | registered |
|---|---|
| the parts, in the order of step 0's measured shares | 1. the zero-centered RMSNorm (the input and post-attention norms of every layer, the full-attention q / k norms, the final norm): one kernel each way, the input saved and r recomputed, the weight's gradient only when it trains (float32 partials and a second pass, no float atomics); optional, measured apart: the residual add folded into the norm. 2. the convolution path: the depthwise causal convolution (kernel 4) and its SiLU reading the projection's [B, T, C] output in place, float32 accumulation, SiLU on the float32 pre-activation, q, k, v written so the rule's drop-in reads them without copies; the backward recomputes the pre-activation, dx with one writer, the weight's gradient only when it trains; measure whether the layout removes in_proj_qkv's slow LoRA-B backward. 3. the gated RMSNorm (rows of 128): one kernel each way (dx, dz), transformers' rounding points. 4. the glue: beta and g in one pass each way (A_log's and dt_bias's gradients only when they train); the transposes, split and reshape copies removed; the 9B repeat of q and k if it costs a copy. 5. SwiGLU (transformers' rounding): built only if it can meet the speed bar, else recorded as not adopted |
| installation | by macai.patch beside the rule; each part a setting, selectable off; a method replaced only while its source matches transformers 5.22.1 (else the original stays, with a warning); the projections stay PEFT's module calls (LoRA and its dropout unchanged); an admitted part becomes the default at the end |
| parity bar | the forward and every gradient against float64 on the CPU at the harness's shapes (odd lengths, B > 1, float32 / bfloat16 / float16, the real head counts and widths): the part's error at most twice the PyTorch path's own error against float64, both measured, at every shape and dtype; and one end-to-end check of whole decoder layers (both kinds, LoRA on) patched and unpatched, both against float64, at the bench shapes |
| speed bar | each part >= 1.5x its PyTorch path, forward + backward, at both bench shapes (0.8B 4 x 1,024, 9B 2 x 1,024), Metal events, the baseline and the design in interleaved rounds; else not adopted |
| predictions | each part within 3x its bandwidth floor (step 0's floors); the bench linear-attention layer 0.8B ~60 -> 44-50 ms, 9B ~180 -> 158-170 ms; the training step 0.8B 10.98 -> 9.6-10.4 s, 9B 50.9 -> 45-49 s |
| integration | after the admissions, through bench/phase4_train.py with an option that installs the parts: Qwen3.5-0.8B 2 x 8, the cache emptied, the recomputed states, 4 epochs (100 steps): the loss read as Phase 4 read it against references A, B, C, the guard's peak footprint <= 53.5 GB, the steady step against D2m's 10.978 s; Qwen3.5-9B 1 x 16, the cache emptied, 1 epoch (25 steps): footprint <= 74.6 GB, the steady step (11-25) against D2m's 51.13 s; step 0's profile again at 0.8B with the parts on |

## D4: the parts as built (`macai.layer`, [layer.md](layer.md))

Each part is one Metal kernel forward and one backward (`torch.mps.compile_shader`), instantiated for float32, bfloat16 and
float16 inputs (and each weight dtype), with 4-element vector loads and float32 arithmetic.

| part | setting | forward | backward | kept for the backward |
|---|---|---|---|---|
| zero-centered RMSNorm | `RMSNORM` (replaces `Qwen3_5RMSNorm.forward`) | y = x rsqrt(mean(x^2) + eps) (1 + w), float32, one rounding. A row is spread over S simdgroups (1 up to 512 wide, 2 at 1,024, 8 at 4,096), its values held in registers between the sum and the output; any row stride (the query norm reads every other 256-wide half of the projection's rows in place) | dx = r ((1 + w) dy - x r^2 <x, (1 + w) dy> / D), r recomputed; dw only when the weight trains: float32 partial rows, then a simdgroup a column | x (transformers' path keeps a float32 copy of x and the float32 normalized product) |
| convolution path | `CONV` (inside the replaced `Qwen3_5GatedDeltaNet.forward`) | a thread owns 4 adjacent channels and walks 32 time steps, the last 3 inputs in registers; u in float32, SiLU on it, one rounding; q and k written as [B, T, H_v, 128] in the input dtype, each key head once for each value head it serves (the 9B repeat, without its copy); v written in float32 (the rule computes in float32: its drop-in neither casts nor copies it) | the same walk with a 3-step look-ahead: u and du recomputed in registers, dx[t] = sum_j w_j du[t + 3 - j] written in the projection's own [B, T, C] layout, one writer an element; dq, dk summed over the value heads a key head serves; dw only when the weight trains (float32 partials a tile, a second pass) | the projection's output (transformers' path keeps it and the rounded pre-activation) |
| gated RMSNorm | `GATED_NORM` (replaces `Qwen3_5RMSNormGated.forward`) | one simdgroup a 128-wide row (4 elements a lane, one `simd_sum`), 8 rows a threadgroup; transformers' rounding points: x_hat rounded to the input dtype, times w rounded, times silu(z) in float32, rounded | dx and dz in one pass, float32, r recomputed; dw only when the weight trains | x and z |
| beta and g | `GATES` (inside the replaced forward) | beta = sigmoid(b), g = -exp(A_log) softplus(a + dt_bias), both written in float32 (softplus with PyTorch's threshold of 20, log1p accurate below it) | db, da in one pass; A_log's and dt_bias's gradients only when they train (PyTorch's reductions) | b, a |
| SwiGLU | `SWIGLU` (replaces `Qwen3_5MLP.forward`; the three projections stay module calls in the original order) | silu(gate) rounded to the input dtype, times up, rounded (transformers' rounding); 8 elements a thread | dgate, dup in one pass, silu recomputed | gate, up |

The replaced linear-attention forward takes the parts for a forward with no cache or with a cache that holds no state of the
layer yet (a prefill, and every training forward: see "The first chain" below), filling the cache as transformers fills it; with
a cache holding the layer's states (decoding) or packed rows, transformers' own forward runs. It calls the projections in the
original order, then the rule through the modeling module's own name for it (so macai's rule, and the step profiler's wrapper
of it, see the same call). Not built: the residual add folded
into the post-attention norm (optional in the registration): the residual adds are 0.3% / 0.2% of the step (step 0) and the
fold saves, by its bytes, about 0.07 ms a layer at the bench shapes; it would need the decoder layer's forward replaced too.

## D4: parity (`tests/test_layer.py`; the test run of record `bench/layer_logs/d4_tests.log`, every case's errors in `d4_parity.jsonl`)

Each part on its real widths (hidden 1,024 / 4,096; q / k heads of 256; 16 / 32 value heads of 128; convolution channels 6,144 /
8,192; MLP 3,584 / 12,288), (rows, tokens) (1, 1), (2, 7), (3, 33), (2, 129) (the convolution (1, 1), (2, 3), (2, 33), (3, 100):
inside, across and past a 32-step tile), float32 / bfloat16 / float16; inputs and upstream gradients drawn in the run's dtype,
so the float64 reference sees the same values. Error = the largest absolute difference over the reference's largest magnitude,
each output and each gradient separately; the bar: the part's error <= 2 x transformers' path's error on the same case.

| part (cases) | float32: worst error, part / PyTorch path; worst ratio | bfloat16 | float16 | verdict |
|---|---|---|---|---|
| zero-centered RMSNorm, hidden widths, q / k norms, trained weight (39) | 1.5e-7 / 2.3e-7; 1.05 | 3.4e-3 / 3.4e-3; 1.00 | 4.2e-4 / 4.2e-4; 1.00 | MET |
| convolution path, frozen and trained weight (30) | 1.9e-7 / 1.8e-7; 1.60 (the trained weight's gradient) | 3.3e-3 / 6.1e-3; 1.00 | 3.7e-4 / 7.9e-4; 1.00 | MET |
| gated RMSNorm, frozen and trained weight (30) | 2.2e-7 / 1.9e-7; 1.61 (the trained weight's gradient) | 5.8e-3 / 7.0e-3; 1.00 | 6.8e-4 / 7.4e-4; 1.00 | MET |
| beta and g, frozen and trained (30) | 2.2e-7 / 2.2e-7; 1.91 (dt_bias's gradient, PyTorch's sum in both paths) | 3.1e-3 / 6.2e-3; 1.00 | 3.5e-4 / 7.6e-4; 1.15 | MET |
| SwiGLU (24) | 2.3e-7 / 2.7e-7; 1.00 | 5.0e-3 / 5.3e-3; 1.00 | 5.6e-4 / 6.8e-4; 1.00 | MET |

The whole decoder layers end to end: LoRA rank 32 on the record's targets (dropout 0.05 with fixed masks, the same draws in
every run), bf16 weights and activations under bf16 autocast, the rule's kernels with the recomputed states; every part off
against every part on, both against a float64 CPU run of the same layer (transformers' code with its float32 casts kept at
float64, transformers' own rule):

| layer, bench shape | output: off / on | input gradient: off / on | LoRA weight gradients: on / off ratio, range (worst) | verdict |
|---|---|---|---|---|
| 0.8B linear attention, 4 x 1,024 | 6.4e-3 / 5.6e-3 | 6.9e-3 / 6.2e-3 | 0.75-1.17 over 12 (down_proj's A) | MET |
| 0.8B full attention | 5.1e-3 / 5.1e-3 | 7.2e-3 / 4.9e-3 | 0.90-1.08 over 14 | MET |
| 9B linear attention, 2 x 1,024 | 1.22e-2 / 1.03e-2 | 2.57e-2 / 2.11e-2 | 0.81-1.12 over 12 | MET |
| 9B full attention | 8.9e-3 / 8.5e-3 | 1.10e-2 / 9.0e-3 | 0.89-1.28 over 14 (v_proj's A) | MET |

Beside them: every part off gives transformers' bits exactly; every part on calls the same submodules in the same order (the
MLP's activation module aside, which the SwiGLU replaces) and draws the same LoRA dropout masks; a source that differs from 5.22.1
leaves the original method with a warning; a forward handed a fresh cache (as every training forward is, entry 9) takes the
parts and leaves the cache as transformers' own forward does. The suite: 490 passed (the 331 before D4 and 159 new).

As chains: the parts round once where transformers rounds two or three times (the convolution's u before SiLU; beta before the
rule; the gated norm's and SwiGLU's backward intermediates) and keep v and beta in float32 -> in bfloat16 their errors are at or
below the PyTorch path's (ratio <= 1.00 for every part); in float32 both paths sit at a few units of float32 rounding, and the
largest ratios (1.6-1.9) are the trained weights' gradients, sums over the rows.

## D4: speed (`bench/bench_layer.py`, `bench/layer_logs/d4_parts.json`, 13:10-13:13 PST)

The parts on the decoder layers step 0 timed (built from each model's config, bf16, frozen, bf16 autocast), inputs laid out as
the layer hands them, the gradient to the part's inputs; Metal events, back-to-back calls filling at least ~10 ms, the median of
7; the PyTorch path and the kernel alternating in each of 5 rounds (the order flipped every round), the median over the rounds.
The convolution path's PyTorch path is everything the kernel replaces: transformers' transposes, `causal_conv1d_fn`, split,
reshape and (9B) repeat, then what the rule's drop-in does before its kernels (the strided q and k copied contiguous at 0.8B, v
cast to float32); beta and g's includes the drop-in's cast of beta. Floors: step 0's (each part's least traffic at 356 GB/s).

Forward / forward + backward in ms:

| part | 0.8B, 4 x 1,024: PyTorch path | kernel | speed-up (f + b) | kernel / floor | 9B, 2 x 1,024: PyTorch path | kernel | speed-up | kernel / floor |
|---|---|---|---|---|---|---|---|---|
| zero-centered RMSNorm (input norm) | 0.733 / 2.297 | 0.050 / 0.143 | 16.06x | 1.21x | 1.482 / 4.529 | 0.079 / 0.240 | 18.87x | 1.02x |
| zero-centered RMSNorm (q_norm, strided rows) | 1.563 / 4.744 | 0.108 / 0.277 | 17.13x | 1.18x | 1.649 / 4.859 | 0.110 / 0.270 | 18.00x | 1.15x |
| zero-centered RMSNorm (k_norm) | 0.385 / 1.138 | 0.081 / 0.158 | 7.20x | 2.68x | 0.352 / 1.098 | 0.049 / 0.130 | 8.45x | 2.21x |
| convolution path | 2.783 / 9.126 | 0.365 / 0.978 | 9.33x | 1.38x | 2.841 / 8.235 | 0.296 / 0.829 | 9.93x | 1.76x |
| (the same against `causal_conv1d_fn` alone, step 0's part) | 1.674 / 5.979 | | 6.11x | | 1.141 / 4.279 | | 5.16x | |
| gated RMSNorm | 2.375 / 6.708 | 0.200 / 0.489 | 13.72x | 1.30x | 2.412 / 6.753 | 0.211 / 0.516 | 13.09x | 1.37x |
| beta and g | 0.057 / 0.131 | 0.026 / 0.082 | 1.60x | see below | 0.065 / 0.142 | 0.033 / 0.089 | 1.60x | see below |
| SwiGLU | 0.383 / 1.185 | 0.242 / 0.714 | 1.66x | 1.08x | 0.746 / 2.154 | 0.432 / 1.273 | 1.69x | 1.13x |

The whole decoder layers, every part off against every part on (the same interleaved rounds):

| layer, forward / forward + backward, ms | 0.8B off | 0.8B on | f + b | 9B off | 9B on | f + b |
|---|---|---|---|---|---|---|
| linear attention (registered) | 24.92 / 61.88 | 18.77 / 42.17 | 1.47x (-19.7 ms) | 87.57 / 183.17 | 79.90 / 159.56 | 1.15x (-23.6 ms) |
| full attention (not registered) | 25.87 / 63.38 | 22.75 / 52.97 | 1.20x (-10.4 ms) | 84.98 / 182.40 | 80.42 / 166.09 | 1.10x (-16.3 ms) |

The rounds' spread on the linear layer: 0.8B on 40.7-43.0 ms, off 61.0-63.0; 9B on 159.0-161.6, off 182.6-183.5.

The LoRA-B question (step 0: in_proj_qkv's LoRA-B backward 5x slower on transformers' transposed gradient at two rows or more):
in_proj_qkv with LoRA rank 32 at the 0.8B micro-batch shape 2 x 903, forward + backward, its output's gradient as transformers'
path hands it (the transpose of a [B, C, T] tensor) 6.746 ms against 5.327 ms as the fused convolution path hands it ([B, T, C]):
1.27x, the 1.4 ms step 0 located in the transposed layout gone (forward 2.70 / 2.66 ms).

Why, as chains:
- The norms: transformers' zero-centered RMSNorm runs the float32 copy of x, the square, the mean, the rsqrt, the product, the
  weight's 1 + w and the product with it, and the cast back, each a pass over float32 tensors, and their backwards -> 17-20x
  the least traffic -> one pass each way over the bf16 rows, the row kept in registers between its sum and its output -> 1.0-1.2x
  the floor at the hidden widths and on the query norm's strided rows, 16-19x faster. The key norm's rows are a fifth of the
  query's (two or four heads), so its two launches weigh more: 2.2-2.7x its floor, 7-8x faster.
- The convolution path: transformers transposes the projection's output, PyTorch's depthwise kernel copies that view into
  channel-first order, rounds u, SiLU runs as its own pass, the result is transposed back and split; at 0.8B the strided q and k
  are copied contiguous by the drop-in, at 9B repeated over the value heads; v is cast to float32; the backward runs MPSGraph's
  channel-first data gradient and copies back -> 9.1 / 8.2 ms a layer for 0.71 / 0.47 ms of least traffic -> one pass each way
  in the projection's own layout, the pre-activation recomputed in registers -> 1.4-1.8x the floor, 9.3-9.9x faster (5.2-6.1x
  against the convolution call alone).
- The gated norm: the same chain as the zero-centered norm plus the SiLU gate's passes -> one pass each way -> 13-14x, at
  1.3-1.4x the floor.
- Beta and g: eight small PyTorch kernels each way (sigmoid; A_log's cast and exp; a's cast, the add, softplus, the products;
  beta's cast in the drop-in) against one each way. The tensors are 65,536 / 65,536 elements (0.8B / 9B bench shapes): the
  kernels' own bytes need ~5 us, so both paths are launch-bound -> 1.60x at both shapes, on 0.13-0.14 ms. Against step 0's floor
  for its glue part (beta, g and the layout copies, now done by the convolution kernel) the kernel is under it (0.29x / 0.23x);
  the convolution and the gates together run at 1.07x / 1.08x of step 0's convolution and glue floors together.
- SwiGLU: transformers' path is two passes forward (silu, the product) and three backward, 28 bytes an element against the fused
  16 -> 1.66-1.69x, the kernel at 1.08-1.13x the floor: the speed bar met (step 0 had expected it out of reach).
- The layer: the parts' own savings add to 19.2 ms at 0.8B (the two norms 2 x 2.15, the convolution path 8.15, the gated norm
  6.22, SwiGLU 0.47, beta / g 0.05) against the layer's 19.7 ms; the rest is in_proj_qkv's backward on a gradient in its own
  layout (step 0 put that at 0.85 ms at this shape). At 9B they add to 23.2 ms against the layer's 23.6, of a layer whose frozen
  products alone need 144 ms.

## D4: admission

| part | speed bar (>= 1.5x, f + b, both shapes) | parity bar | verdict |
|---|---|---|---|
| zero-centered RMSNorm (`RMSNORM`) | 16.1x / 18.9x (input norm); 17.1x / 18.0x (q_norm); 7.2x / 8.4x (k_norm) | MET | ADMITTED |
| convolution path (`CONV`) | 9.33x / 9.93x | MET | ADMITTED |
| gated RMSNorm (`GATED_NORM`) | 13.72x / 13.09x | MET | ADMITTED |
| beta and g (`GATES`) | 1.60x / 1.60x | MET | ADMITTED |
| SwiGLU (`SWIGLU`) | 1.66x / 1.69x | MET | ADMITTED |
| the residual add folded into the norm (optional) | not built | | not adopted (not built) |

## D4: integration (`bench/layer_logs/d4_0.8b_2x8_100steps.json`, `d4_9b_1x16_25steps.json`, `profile_d4_0.8b_2x8.json`; `bench/layer_report.py` -> `d4_results.json`)

D2m's confirmation runs again with the admitted parts on (`bench/phase4_train.py --layer RMSNORM,CONV,GATED_NORM,GATES,SWIGLU`,
recorded with each run; every replaced method checked in force when training began), at commit 3564a41 (the fix of the first
chain's fault, below), the tree clean, one job at a time under the guard, read the way Phase 4 and D2m read them.

### Qwen3.5-0.8B, 2 x 8 sequences, the cache emptied each micro-batch, the recomputed states, 4 epochs (100 steps), 14:02-14:18 PST

| measure | bar / prediction | D4 | D2m (the same run without the parts) | verdict |
|---|---|---|---|---|
| the guard's peak footprint | <= 53.5 GB | 41.58 GB | 46.32 GB | MET |
| the steady step (trainer's clock, steps 11-100) | against 10.978 s; predicted 9.6-10.4 s | 9.178 s (-16.4%) | 10.978 s | prediction MISSED (faster) |
| the loss: root mean square of the step-loss difference against A / B / C | the reference runs among themselves: .0034 / .0034 / .0034 | .0034 / .0044 / .0041 (against D2m .0033) | .0029 / .0030 / .0035 | |
| Phase 4's reading 2: the largest gap from the reference pair's mean | <= 2 x the pair's largest spread = .0159 | .0086 (step 59) | .0077 | met |
| Phase 4's final loss within 1% of the pair's mean (0.1967) | <= 1% | 0.1948 (0.94%) | 0.1983 (0.83%) | met |
| Phase 4's root mean square, gap over spread (0.87 expected of a run drawn like the reference; the control run C 0.85, Phase 4's kernels 0.95) | | 1.03 | 0.72 | |
| mean loss (the pair's mean 0.5776) | | 0.5754 (-0.37%) | 0.5773 | |
| the allocator's live tensors at a forward's end (peak) / the driver's memory (peak) | | 12.94 GiB / 36.80 GB | 15.10 GiB / 41.76 GB | |

The step time by window, the same batches in both runs (the data order is seeded):

| steps | 1-10 | 11-25 | 26-50 | 51-75 | 76-100 | 11-100 |
|---|---|---|---|---|---|---|
| D4, s a step (trainer's clock) | 9.10 | 9.73 | 9.24 | 9.00 | 8.96 | 9.178 |
| D2m | 10.90 | 11.07 | 10.96 | 10.92 | 11.00 | 10.978 |
| D4 / D2m | 0.835 | 0.880 | 0.843 | 0.824 | 0.815 | 0.836 |

As chains:
- The step: the surroundings fall 153.5 ms a micro-batch, the rule's drop-in 14.7 ms (the copies of q, k and v it no longer
  makes) and in_proj_qkv's backward 18.3 ms (on the gradient in its own layout: LoRA-B 13.9, the base product 4.4): ~187 ms of a
  1,253 ms micro-batch (the profile below), ~1.5 s of a step of 8 micro-batches -> 10.98 -> 9.18 s, faster than the 9.6-10.4 s
  predicted (the prediction counted the surroundings at 3x their floors, and neither the drop-in's copies nor the LoRA-B layout).
- The loss: the parts round once where transformers rounds two or three times; the run's LoRA starts from its own draw (the
  trainer seeds nothing before it) -> its curve differs from A, B and C by about what they differ among themselves (Phase 4's
  reading 2 and the final loss met; the gap over the spread 1.03, at the top of the runs read so far: D2m 0.72, the control run
  C 0.85, Phase 4's kernels 0.95).
- Memory: the norms keep their bf16 input instead of float32 copies; the convolution keeps the projection's output and nothing of
  its own; the gated norm and SwiGLU keep their inputs -> the live tensors at a forward's end fall 2.2 GiB (15.10 -> 12.94) and
  the driver's pool 5.0 GB (41.76 -> 36.80) -> the footprint 46.32 -> 41.58 GB, 15% under the reference implementation's 48.65.
- Steps 19-26 gain least: 84.8 s against D2m's 93.0 (-8.8%, against -16.4% over the run); the first chain (the norms and SwiGLU
  alone) ran those same steps slower than D2m (97.1 s) while gaining 5% over the run. Both runs share the batches, so the cause
  sits in those batches (they are the run's slowest steps in every run); the profile covers steps 3-7 and does not reach them.

### Qwen3.5-9B, 1 x 16 sequences, the cache emptied, 1 epoch (25 steps)

| measure | bar / prediction | D4 | the norms and SwiGLU alone (the first chain) | D2m | verdict |
|---|---|---|---|---|---|
| the guard's peak footprint (25 steps) | <= 74.6 GB | 60.44 GB | 62.90 GB | 64.35 GB (through step 25) | MET |
| the steady step (trainer's clock, steps 11-25) | against 51.13 s; predicted 45-49 s | 46.20 s (-9.6%) | 48.27 s (-5.6%) | 51.13 s | prediction met |
| steps 1-10, s a step | | 45.6 | 47.8 | 50.4 | |
| the allocator's live tensors at a forward's end, peak through step 25 / the driver's memory | | 17.28 GiB / 54.46 GB | 17.78 GiB / 58.21 GB | 20.59 GiB / 60.22 GB | |
| the loss at steps 1-2 (the same weights: the adapters' B starts at zero) | | 2.2223 / 2.0706 | 2.2240 / 2.0703 | 2.2249 / 2.0694 | |

The 9B run is one epoch, so its learning rate falls over 25 steps where D2m's fell over 100: past step 2 the two losses are not
comparable step by step (registered: the footprint and the step only). As a chain: at 9B the frozen products are 55% of the step
(step 0) and the parts' share 9% -> the parts take 4.9 s of a 51.1 s step (-9.6%), inside the predicted 45-49 s; the memory moves
as at 0.8B (live tensors -3.3 GiB, the footprint -3.9 GB).

Phase 4's 9B runs are one epoch as well (the same 25-step schedule), so the loss does compare step by step there (a reading added
after the report, from the same records): the root mean square of the step-loss difference over steps 1-25 against Phase 4's 9B
reference run (`bench/phase4_logs/p42_9b_ref_1x16.json`) is .0013 (Phase 4's kernels against the same run .0021, the first chain
.0028); step 25 0.5455 against the reference's 0.5452.

### The step profile again, 0.8B with every part on (`profile_d4_0.8b_2x8.json`, steps 3, 5, 7)

The same profiler, options and steps as step 0's run of record (`bench/profile_step.py`, steps 3, 5, 7 of 8, every micro-batch
as A / B / C), with the fused convolution path marked under step 0's name for the convolution and beta / g apart. Valid as step
0 was: the profiled pass against its unprofiled replay 1.052 (median 1.053; 2 of 24 micro-batches over 1.10), the components'
sum against the wall clock 0.9995, no stack mismatches or negative segments; the profiled steps as trained 0.81-0.83 of D2m's same
steps, the unprofiled ones 0.81-0.87.

| group, ms a micro-batch (share) | step 0 | D4 | change |
|---|---|---|---|
| the layers' frozen products | 285.9 (22.8%) | 279.7 (26.6%) | -6.2 |
| the LM head and the loss | 266.7 (21.3%) | 269.6 (25.6%) | +2.9 |
| the surroundings | 186.8 (14.9%) | 33.3 (3.2%) | -153.5 |
| the LoRA adapters' extra work | 160.4 (12.8%) | 135.1 (12.8%) | -25.3 |
| the vision tower | 138.5 (11.1%) | 134.1 (12.7%) | -4.4 |
| the rule | 90.3 (7.2%) | 75.7 (7.2%) | -14.7 |
| data, optimizer and the Trainer | 57.6 (4.6%) | 61.0 (5.8%) | +3.4 |
| the full-attention layers' attention | 55.0 (4.4%) | 53.8 (5.1%) | -1.2 |
| (beside) a new input shape, A - B | 26.5 (2.1%) | 17.9 (1.7%) | -8.7 |
| the micro-batch | 1,253.1 | 1,052.8 | -200.3 (-16.0%) |

Where the time moved, by component (ms a micro-batch, forward / backward):

| component | step 0 | D4 | saved |
|---|---|---|---|
| the gated norm | 18.07 / 28.48 | 1.29 / 1.40 | 43.9 |
| the convolution path | 11.68 / 38.02 | 2.58 / 3.93 | 43.2 |
| the rule (the drop-in's copies of q and k and its cast of v gone) | 33.62 / 56.71 | 21.62 / 54.04 | 14.7 |
| in_proj_qkv's LoRA-B (its backward on the gradient in its own layout) | 1.74 / 20.04 | 1.74 / 6.15 | 13.9 |
| the linear-attention layers' post-attention / input norms | 4.78 / 9.61; 4.67 / 9.62 | 0.44 / 0.43; 0.41 / 0.58 | 13.5; 13.3 |
| the glue (the transposes, split, repeat and their backward copies gone; beta / g now apart) | 3.57 / 14.68 | 2.73 / 3.21 | 12.3 |
| the full-attention q_norm / k_norm | 3.49 / 8.11; 0.86 / 1.93 | 0.24 / 0.49; 0.07 / 0.16 | 10.9; 2.6 |
| the full-attention layers' input / post-attention norms | 1.52 / 3.09; 1.47 / 3.18 | 0.08 / 0.14; 0.08 / 0.14 | 4.4; 4.4 |
| in_proj_qkv's base product (its backward on the gradient in its own layout) | 30.36 / 38.96 | 33.18 / 31.76 | 4.4 |
| the SwiGLU (linear / full-attention layers) | 2.30 / 5.26; 0.79 / 1.76 | 1.58 / 3.07; 0.51 / 1.03 | 2.9; 1.0 |
| beta and g (new, apart) | | 0.12 / 0.13 | -0.25 |

What is left at 5% of the step or more outside D3-D5: the LM head and the loss (25.6%), the vision tower (12.7%), data, the
optimizer and the Trainer (5.8%): the two entries step 0 proposed, unchanged and now larger shares.

### The first chain, and why its convolution path never ran (records kept as `d4_norms_swiglu_*`)

The first integration chain (13:17-13:57 PST, commit a0d3d57) was launched with every part on and recorded so, but the step
profile it ended with showed the convolution path at 11.3 / 37.8 ms a micro-batch, step 0's 11.7 / 38.0, while the norms had
fallen tenfold. As a chain:

- qwen-symbolic-thinking's `run()` sets the model's `use_cache` off, then transformers' Trainer sets the model's `use_cache` from
  its own arguments as its constructor ends (trainer.py, "Finalize"), which in this recipe leaves it on;
- so every training forward builds a fresh cache and hands it to every layer (a probe: 72 of 72 linear-attention calls of a
  training step carried one, holding no state yet);
- the replaced linear-attention forward took any cache for decoding and ran transformers' own forward -> the convolution path and
  beta / g never ran in training, while the norms, the gated norm and SwiGLU (methods of their own modules) did.

The fix (commit 3564a41): a cache holding no state of the layer yet takes the parts and is filled as transformers fills it (the
convolution's window through the cache's own method, the rule's final state); a cache holding the layer's states (decoding, a
continued prefill) still takes transformers' forward. Tested against transformers' forward with a fresh cache: the parts run, the
cache's convolution state is the same bits, its recurrent state agrees to the rule's precision, and a decode step from either
cache gives the same output; a probe then counted 72 fused convolution and gate calls in a training step. The parts bench and the
end-to-end check had run without a cache and were not affected.

What the first chain measured is the norms of both kinds and SwiGLU alone:

| measure | D2m | the norms and SwiGLU (first chain) | every part (the run of record, below) |
|---|---|---|---|
| 0.8B steady step (steps 11-100) | 10.978 s | 10.433 s (-5.0%) | 9.178 s (-16.4%) |
| 0.8B guard's peak footprint | 46.32 GB | 48.52 GB | 41.58 GB |
| 0.8B final / mean loss | 0.1983 / 0.5773 | 0.1978 / 0.5765 | 0.1948 / 0.5754 |
| 9B steady step (steps 11-25) | 51.13 s | 48.27 s (-5.6%) | 46.20 s (-9.6%) |
| 9B guard's peak footprint (25 steps) | 64.35 GB (through step 25) | 62.90 GB | 60.44 GB |
| 0.8B profile: the surroundings, ms a micro-batch | 186.8 (step 0) | 88.3 | 33.3 |

## D4's verdicts in short

| registered | measured | verdict |
|---|---|---|
| parity: each part's error <= 2 x the PyTorch path's against float64, every shape and dtype | 157 cases; the worst ratio 1.91 (a trained parameter's gradient in float32); in bfloat16 every part at or below the PyTorch path (ratio <= 1.00) | MET |
| parity: whole decoder layers with LoRA, patched and unpatched against float64 | output and input gradient at or below the unpatched layer's error; LoRA gradients 0.75-1.28x | MET |
| speed: each part >= 1.5x its PyTorch path at both bench shapes | the norms 7.2-18.9x, the convolution path 9.3 / 9.9x, the gated norm 13.7 / 13.1x, beta and g 1.60 / 1.60x, SwiGLU 1.66 / 1.69x | MET: all five ADMITTED, now the default |
| prediction: each part within 3x its floor (step 0's floors) | 1.02-1.76x, the key norm 2.21-2.68x; beta / g with the convolution 1.07-1.08x of their step-0 floors together (beta / g alone launch-bound) | met |
| prediction: the bench linear-attention layer 0.8B ~60 -> 44-50 ms | 61.88 -> 42.17 ms | missed (faster) |
| prediction: 9B ~180 -> 158-170 ms | 183.17 -> 159.56 ms | met |
| prediction: the training step 0.8B 10.98 -> 9.6-10.4 s | 9.178 s (steps 11-100; D2m 10.978) | missed (faster) |
| prediction: 9B 50.9 -> 45-49 s | 46.20 s (steps 11-25; D2m 51.13) | met |
| integration 0.8B: footprint <= 53.5 GB; the loss as Phase 4 read it | 41.58 GB (D2m 46.32); the largest gap .0086 against the bar .0159, the final loss 0.94% from the reference pair's | MET; like the reference |
| integration 9B: footprint <= 74.6 GB | 60.44 GB (D2m 64.35 through step 25) | MET |

In plain words: every surrounding part now runs as one Metal pass each way, at or near the time its memory traffic needs, and
none loses precision against transformers' own path. A whole linear-attention layer at the bench shapes is 1.47x (0.8B) / 1.15x
(9B) faster for it, and a LoRA training step 16% (0.8B, 10.98 -> 9.18 s) / 10% (9B, 51.1 -> 46.2 s) faster, with a smaller
footprint (41.6 against 46.3 GB; 60.4 against 64.4) and the loss of the reference. The surroundings are now 3% of a 0.8B
micro-batch (from 15%); the step is the products, the vocabulary work, the adapters' passes and the vision tower.

## D4: not done, and what is uncertain

- The residual add folded into the post-attention norm (optional in the registration): not built (0.3% / 0.2% of the step;
  about 0.07 ms a layer at the bench shapes by its bytes; it needs the decoder layer's forward replaced as well).
- The q / k l2 normalization and the query scale are not folded into the convolution's epilogue (prior art s.2.4): the
  registered design writes q and k for the rule's drop-in, whose fused prologue still makes one pass each way over them.
- The gated norm reads the rule's output after the drop-in's cast to the input dtype (transformers' rounding point); reading the
  float32 output would save that cast each way (~0.3 ms a 0.8B bench layer) but changes the drop-in's return.
- Beta and g are launch-bound: their two kernels take ~0.08 ms for ~2 MB of traffic. Folding them into the rule's prologue and
  epilogue (flash-linear-attention's form, prior art s.5) would remove the launches; it changes the rule's interface.
- The rotary embedding (0.5% / 0.3% of the step, 6-8x its floor) and the full-attention output gate: not registered, not built.
- Steps 19-26 of the 0.8B run gain least (-8.8% against -16.4%), in both D4 runs (the first chain ran them slower than D2m):
  tied to those batches, not traced (the profile reaches steps 3-7 only).
- The first chain's footprint was 2.2 GB over D2m's with fewer live tensors (a larger driver pool); with every part on it is 4.7 GB
  under. The pool's response to the changed set of buffer sizes is not traced here.
- The 9B run is one epoch (registered): its loss past step 2 is not comparable step by step with D2m's four-epoch schedule.
- Qwen3.5-MoE and Qwen3-Next: the layer parts are installed into Qwen3.5 (dense) only; the rule into all three.
- The parts are timed at the bench shapes and in training; long sequences (D3's range) are not timed.

## D8 and D9, the LM head and the loss at the labelled positions; the frozen vision tower's output cached (registered before any code)

Step 0 proposed both as entries outside D3-D5 (above, "The parts outside D3-D5"); they were registered after step 0 (series
journal, 2026-11-03 13:1x PST), before any code, and come before D5 and D3 in the measured order.

| entry | design | predicted | bars |
|---|---|---|---|
| D8, the head and the loss at labelled positions | while training with labels, the hidden states gathered at the positions whose shifted label is not ignored before the head (transformers 5.22.1: a boolean `logits_to_keep` and `shift_labels`; prior art s.7), the loss reduced as the model's own loss function reduces it (`num_items_in_batch` kept); the gradient to every other position zero, as before. No kernel; installed by macai.patch, selectable off. Conditional D8b: a fused cross-entropy (MLX's form, prior art s.7.5) only if the head and the loss still take >= 3% of the step after D8 | the head and the loss a micro-batch 0.8B 266.7 -> 50-75 ms, 9B 347.6 -> 70-100 ms; the step 0.8B -14 to -18%, 9B -6 to -9% against the same configuration without it; the logits' memory down by the same fraction | parity: on the same weights and batch the loss within relative 1e-5 of the model's own path and every gradient (to the hidden states, the LoRA weights) within twice that path's own error against float64; steps 1-2 of a training run against the run without it; speed: the step >= 10% faster at 0.8B, >= 5% at 9B |
| D9, the frozen vision tower's output cached | the tower's output (whatever `get_image_features` returns) stored per picture on the first pass, keyed by the exact bytes of the picture's `pixel_values` rows and its `image_grid_thw` entry (never a near match), kept off the GPU's allocator, served in later passes; active only while every tower parameter is frozen and the model trains; transformers' own precomputed-encoder path read first; selectable off | the step in epochs after the first 0.8B -9 to -11%, 9B -8 to -10% (the tower's 138.5 / 317.5 ms a micro-batch down to a lookup of <= 15 ms); the first epoch within +2%; the cache ~0.6 MB a picture at 0.8B, ~2.3 MB at 9B | parity: the served output equal to a recompute of the same picture (bit for bit, or within the tower's own run-to-run spread, measured); a changed picture never served (one pixel changed; the same pixels at another grid); speed: the later epochs' step >= 7% faster at both sizes |

Integration, once both are admitted, with D4's parts on: Qwen3.5-0.8B 2 x 8, the cache emptied, 100 steps (4 epochs): the loss
read as Phase 4 read it against A, B, C, the guard's peak footprint <= 53.5 GB, the steady step and each epoch's window against
D4's run of record; Qwen3.5-9B 1 x 16, 50 steps (2 epochs, so the cache serves): footprint <= 74.6 GB, the steady step by epoch.
The baselines: D4's runs of record (0.8B 100 steps; 9B 25 steps); for a 9B second epoch without a cache, D2m's 9B run's
second-epoch over first-epoch ratio. An entry is admitted only when its bars are met; a failing entry's integration is re-run
without it. Both live in macai (the record's trainer untouched).

## D8 and D9: as built (`macai.training`, [training.md](training.md))

| setting | replaces | what runs |
|---|---|---|
| `HEAD_AT_LABELS` (D8) | `Qwen3_5ForConditionalGeneration.forward` and `Qwen3_5ForCausalLM.forward` | while the model trains with labels: the original forward called with `labels=None` and a boolean `logits_to_keep` marking the positions whose next token's label is not -100 (its `slice_logits_to_keep` gathers those rows of the final hidden states, flattened to [N, hidden], before the head), then the model's own loss function on those logits with `shift_labels` (the shifted labels at those positions) and every keyword argument the original passes it. The output carries the loss first and the logits of the labelled positions |
| `VISION_CACHE` (D9) | `Qwen3_5Model.forward` | while the model trains with every tower parameter frozen: the micro-batch's pictures looked up by key; all stored: their outputs (copied from CPU memory to the GPU) handed to the original forward as its `mm_encoder_outputs`, `pixel_values` then None, so the tower does not run; any picture missing: `get_image_features` called as the original forward calls it, its output handed over the same way, the new pictures stored |
| `VISION_CACHE_MAX_GB` | | the most a cache stores (16 GiB): a guard for the default, far above this sample's needs |

Neither copies a line of transformers' code: each replacement calls the original method, and each is installed only while the
sources of what it replaces and relies on (the forwards, `slice_logits_to_keep`, the causal LM loss and its cross-entropy,
`get_image_features`, the tower's forward) hash as in transformers 5.22.1.

What the design rests on, read first:
- transformers' precomputed-encoder path. `accepts_precomputed_kwargs` (on `get_image_features`) renames modality-prefixed
  keyword arguments for the encoder: precomputed encoder *inputs* (the attention boundaries, position ids, interpolation
  indices), not outputs. `mm_encoder_outputs` in the model's forward takes the encoder's *outputs* in place of `pixel_values`.
  The cache serves through `mm_encoder_outputs`, so the model's own code places the features, checks the placeholder count and
  builds the 3-D positions; no new path in the forward.
- The key. A picture's rows of `pixel_values` are its patches (t x h x w rows of 1,536 values); the SHA-256 of their bytes
  (in 1 MiB chunks on four threads, then the chunks' digests and the length), the rows' dtype and width, the grid entry, and the
  device, the tower's dtype, the autocast state and dtype and the attention implementation. The tower's weights are guarded by
  their storage addresses and version counters, read at every call (a change empties the cache).
- The Trainer and `num_items_in_batch`. `Qwen3_5ForConditionalGeneration` sets `accepts_loss_kwargs = False`, so transformers'
  Trainer computes no `num_items_in_batch` for it, passes none, and divides each micro-batch's mean loss by the accumulation
  steps. D8 passes the original's keyword arguments to the loss function, so either reduction is the model's own (both tested).
- Random numbers. The tower's attention dropout is fixed at 0 and it has no dropout module; measured in both towers: a tower
  pass in training mode leaves the CPU's and the GPU's generator states as they were. Serving a picture therefore changes no
  draw that follows (the LoRA dropout's masks are the same with and without the cache).
- Synchronization. D8's boolean gather needs the count of labelled positions on the CPU: one synchronization at the forward's
  end, where the loss follows anyway. D9's lookup copies the micro-batch's pixel rows to the CPU at the forward's start, where the
  tower's own position code synchronizes too; a store's copy to CPU memory waits for the next call.

## D8 and D9: parity (`tests/test_training.py`, the test run of record `bench/layer_logs/d8d9_tests.log`, D8's cases in `d8d9_parity.jsonl`; `bench/training_probe.py`, records `probe_d8d9_0.8b.json`, `probe_d8d9_9b.json`)

### D8, the head at the labelled positions

Error = the largest absolute difference over the reference's largest magnitude, each tensor apart; the bar: the head-at-labels
path's error at most twice the model's own path's, every gradient; the loss within relative 1e-5 of the model's own path.

| check | the loss: relative difference from the model's own path | gradients: head-at-labels error / model path's error against float64 (worst) | other |
|---|---|---|---|
| models at the real widths and vocabulary (two layers, LoRA rank 32, B drawn, fixed dropout masks; 2 x 97, the prompt and padding ignored), against float64 on the CPU: 0.8B | 7.5e-8 | 28 gradients (input embeddings, final hidden states, 26 LoRA weights) 0.86-1.31 (in_proj_qkv's LoRA B) | the logits at the labelled positions the same bits |
| the same at 9B's widths | 0.0 | 0.89-1.09 | the same bits |
| the real Qwen3.5-0.8B, a real micro-batch (2 x 446 positions, 126 labelled, two pictures), the adapters' B drawn, no dropout, against the whole model in float64 on the CPU | 3.5e-7 | 302 gradients 0.91-1.09 (layer 9's in_proj_qkv LoRA A); the model path's own errors 4.3e-2 to 0.33 | the logits at the labelled positions the same bits; the final hidden states the same bits |
| the real Qwen3.5-9B, a real micro-batch (536 positions, 170 labelled, one picture), against a float64 head and loss on the same final hidden states | 6.9e-7 | the final hidden states' gradient: 1.66e-2 on both paths (ratio 1.00); the LoRA gradients between the two paths within 3.8% (no float64 model at 9B: 75 GB) | the same bits |
| steps 1-2 of a training run against D4's run of record (the adapters' B is zero and the first update's learning rate zero: the same weights) | 0.8B: step 1 9.3e-8, step 2 0 (the same bits); 9B: 8.6e-7, 6.9e-7 | | |
| `num_items_in_batch` given (the sum over the labelled positions, then the model's division) | <= 1e-5 | | a third of the mean at three times the count |

Beside them: the loss function receives the model path's keyword arguments plus `shift_labels`; the model's own path runs, bit
for bit, in evaluation, without labels, with `shift_labels` or a tuple output asked (and, by the condition's test, with
`logits_to_keep` or positional arguments given); a micro-batch without a label gives NaN on both paths; the positions without a
label get no gradient on either path.

As chains:
- The head's product on the gathered rows gives each row the same bits as on every position (`bench/probe_head_rows.py`,
  `bench/layer_logs/d8_head_rows.log`: 0 of 73 M logits differ at 0.8B's shape, 0 of 36 M at 9B's) -> the loss differs only by
  the order of its float32 mean -> 1e-7 relative.
- The gradient to the final hidden states, dE = dZ W, sums over the 248,320 words in another order when the product has fewer
  rows (the GPU picks another split) -> 0.4% of its elements differ by one bf16 step in the probe's shapes, 1.1% in the real
  micro-batch (1,369 of 129,024 at 0.8B), at the same
  error against float64 -> every gradient below at the model path's own error (ratios 0.86-1.31 in the two-layer models,
  0.91-1.09 over the real 0.8B's 302).

### D9, the cache

| check | 0.8B | 9B |
|---|---|---|
| the tower draws random numbers in training mode (the CPU's and the GPU's generator states before and after a pass) | no (3 micro-batches) | no (3) |
| the same micro-batch twice | the same bits | the same bits |
| each picture alone against its rows in a two-picture micro-batch (0.8B), or paired with the previous micro-batch's picture (9B) | the same bits (6 of 6) | the same bits (2 of 2) |
| a served output (everything get_image_features returns) against a recompute | the same bits (3 of 3) | the same bits (3 of 3) |
| one pixel value moved to the next float32 | not served (computed) | not served |
| the same pixels at another grid (h and w swapped, or 40 x 40 read as 80 x 20) | not served | not served |
| the tower's output needs a gradient in training | no | no |
| tests on a small model: served = computed bit for bit and the model's outputs unchanged; one value changed, another grid, the pixels in another dtype not served; not in evaluation, not with a training tower parameter; a weight changed in place empties the cache; the same picture twice in a micro-batch stored once; the cap | passed | |

So the tower's own run-to-run spread is zero: its output for a picture is the same bits whatever the micro-batch it runs in, and
the bar's strict form (bit for bit) holds.

## D8 and D9: speed

All runs through `bench/phase4_train.py` with D4's parts on (`--layer RMSNORM,CONV,GATED_NORM,GATES,SWIGLU`), the recomputed
states, the cache emptied each micro-batch, at commit 69ca269 with the tracked tree clean, one job at a time under the guard
(15:29-16:52 PST). Step times on the trainer's clock (D4's windows' clock); the same steps run the same batches in every run
(the data order is seeded; the head's per-step position counts match step for step between the runs).

### D8 at 0.8B, 2 x 8 (`d8_0.8b_2x8_25steps`, `d8_0.8b_2x8_100steps`)

| steps | D4 (run of record) | D8, 1 epoch (first run) | D8, 4 epochs (repeat) | D8 / D4, first | D8 / D4, repeat |
|---|---|---|---|---|---|
| 1-10 | 9.10 | 7.50 | 7.50 | 0.824 | 0.824 |
| 11-25 | 9.73 | 8.07 | 7.67 | 0.829 | 0.788 |
| 16-25 | | 8.50 | 7.80 | | |
| 26-50 / 51-75 / 76-100 | 9.24 / 9.00 / 8.96 | | 7.44 / 7.52 / 7.48 | | 0.805 / 0.836 / 0.835 |
| 11-100 (the steady step) | 9.178 | | 7.511 | | 0.818 (-18.2%) |

The two D8 runs disagree on steps 16-25 (8.50 against 7.80 s a step) while steps 1-15 agree within 1-2%. The repeat matches the
integration run's first epoch step for step (7.46-8.54 s against 7.46-8.52), which has D8 on as well; the first run came
directly after the loss-head bench's minute and a half of full-rate products. Identical configurations elsewhere in the record
agree within 1% on these steps (D2m's 25- and 100-step runs: 11.40 / 11.40 s at 16-25), so the first run's late steps are read as
a machine state, not traced. Both runs meet the bar; the steady step is read from the repeat.

Steps 1-2 against D4's run of record (the adapters' B is zero and the first update's learning rate is zero, so both steps run
the same weights): 2.5624918937683105 against 2.5624916553497314 (9.3e-8), 2.426778793334961 against the same bits.

Memory: the live tensors at a forward's end 12.94 -> 9.12 GiB (peak over 100 steps), the driver's pool 36.80 -> 23.83 GB, the
guard's peak footprint 41.58 -> 27.30 GB. As a chain: the vocabulary-wide tensors of a micro-batch (the bf16 logits, the loss's
float32 copies and gradient, accelerate's float32 copy of the returned logits) shrink with the positions computed (17.9% of them
over the run; 1,622 -> 312 at the profile's average micro-batch) -> the live peak at a forward's end falls 3.8 GiB, and the
allocator's pool, which grows with the largest tensors, 13 GB.

D8's gain is largest on the steps D4 found slowest (19-26, the batches with the most positions: 0.73-0.79 of D4 there against
0.82-0.85 on steps 11-18). As a chain, read here: those batches carry the largest vocabulary-wide float32 tensors (up to 2 GB
each) -> D4's step there was slower than its token count alone explains -> without them the steps fall in line. Not traced
further (the profile covers steps 3-7).

### D8 at 9B, 1 x 16 (`d8_9b_1x16_25steps`, 1 epoch: Phase 4's 9B schedule)

| measure | D4 (run of record) | D8 | D8 / D4 |
|---|---|---|---|
| steps 1-10, s a step | 45.6 | 42.4 | 0.930 |
| steps 11-25 | 46.20 | 42.93 | 0.929 (-7.1%) |
| the loss at steps 1-2 (the same weights) | 2.2223074 / 2.0705986 | 2.2223094 / 2.0706000 | 8.6e-7 / 6.9e-7 |
| the loss over steps 1-25 against Phase 4's 9B reference run (the same schedule), root mean square of the difference | .0013 | .0012 | |
| the guard's peak footprint / the live tensors at a forward's end / the driver's pool | 60.44 GB / 17.28 GiB / 54.46 GB | 48.22 GB / 15.29 GiB / 46.01 GB | |

As a chain: at 9B the head and the loss were ~11% of the step (step 0) and the head computes 21.5% of the positions -> the bench's
whole chain 298 -> 84 ms a micro-batch, 3.4 s of 16 micro-batches -> 46.2 -> 42.9 s (-3.3 s measured).

### The head and the loss part by part (`bench/bench_loss_head.py`, `bench/layer_logs/d8_loss_head.json`)

At the profile's average micro-batch (0.8B: 1,622 positions, 312 labelled; 9B: 768 positions, 163 labelled), every position
against the labelled ones only, Metal events, the two alternating in 5 rounds, ms a micro-batch:

| part | 0.8B every position | 0.8B labelled | 9B every position | 9B labelled |
|---|---|---|---|---|
| the head's product (forward) | 72.7 | 18.4 (8.6 TFLOP/s) | 136.8 | 46.4 (7.1 TFLOP/s) |
| the head's gradient to its input (dE) | 71.3 | 13.4 (11.8 TFLOP/s) | 132.0 | 31.4 (10.6 TFLOP/s) |
| the logits cast to float32 | 6.9 | 1.35 | 3.3 | 0.69 |
| the cross-entropy forward / backward | 17.9 / 22.5 | 3.5 / 4.5 | 8.5 / 10.8 | 2.0 / 2.3 |
| the gradient cast back to bf16 | 6.8 | 1.3 | 3.2 | 0.67 |
| accelerate's float32 copy of the returned logits | 6.9 | 1.35 | 3.3 | 0.68 |
| the whole chain, forward and backward | 200.8 | 43.1 (0.21x) | 298.0 | 83.6 (0.28x) |

The products shrink with the rows, but not in proportion at 9B: 163 rows is a narrow product, and the head's forward runs at
7.1 TFLOP/s there (8.6 at 0.8B's 312), against 11-12 at the full positions.

### D9 at 0.8B and 9B: the integration runs (`d8d9_0.8b_2x8_100steps`, `d8d9_9b_1x16_50steps`)

| steps | D4 | D8 alone (repeat) | D8 + D9 | D9's own: against D8 alone | registered reading: the window's ratio to D4 over the first epoch's |
|---|---|---|---|---|---|
| 0.8B 1-10 | 9.10 | 7.50 | 7.60 | 1.013 | |
| 0.8B 11-25 (first epoch) | 9.73 | 7.67 | 7.67 | 1.000 | (0.788 to D4) |
| 0.8B 26-50 | 9.24 | 7.44 | 6.32 | 0.849 | 0.868 |
| 0.8B 51-75 | 9.00 | 7.52 | 6.28 | 0.835 | 0.886 |
| 0.8B 76-100 | 8.96 | 7.48 | 6.28 | 0.840 | 0.890 |
| 0.8B 26-100 (the later epochs) | 9.067 | 7.48 | 6.293 | 0.841 (-15.9%) | 0.881 (-11.9%) |
| 9B 1-10 | 45.6 | | 42.8 | | |
| 9B 11-25 (first epoch) | 46.20 | | 43.13 | | (0.934 to D4) |
| 9B 26-50 (second epoch) | 45.94 (D4's 11-25 times D2m's second-epoch ratio 0.9943) | | 37.48 | | 0.874 (-12.6%) |

The registered reading assumes D8's relative effect is the same in every epoch; at 0.8B, where D8 alone ran all four epochs, it is
not (D8 alone against D4: 0.788 in the first epoch, 0.805-0.836 in the later ones, because D4's first epoch holds its slow steps
19-26), so the direct reading against D8 alone is the cleaner one there; both clear the bar. At 9B only the registered reading
exists (D8 alone ran one epoch).

The cache in the runs (`training_counts` at every step):

| | 0.8B | 9B |
|---|---|---|
| pictures stored (first epoch) | 372 (1.68 GB) | 372 (3.56 GB) |
| a picture, in full (the tower's last hidden states and the merged features) / the merged features alone | 4.51 / 1.13 MB | 9.58 / 4.51 MB |
| pictures repeating inside an epoch (the sample's 18 items in two forms) | 18 an epoch; none served in the first epoch (each repeat shared its micro-batch with a picture not yet stored, so the tower ran) | 18; all 18 served in the first epoch (one picture a micro-batch) |
| served micro-batches, later epochs | 200 of 200 an epoch | 390 of 390 |
| the cache's own time a call: the key (the pixel rows copied to the CPU and hashed) + building a served output | 5.8 + 1.6 = 7.4 ms (two pictures) | 4.2 + 2.0 = 6.2 ms (one picture) |
| a store's copy to CPU memory | 1.0 ms a picture | 2.3 ms |
| the first epoch against D8 alone | +0.5% over steps 1-25 (1-10 +1.3%, 11-25 0.0%) | 1-10 +0.9%, 11-25 +0.5% (13 micro-batches served there) |

As chains:
- 0.8B: the tower took 134 ms of a micro-batch in D4's profile; with D8 the micro-batch is 839 ms (the profile below), so the
  tower is 16% of it -> served, it costs the cache's 7.4 ms -> ~126 ms saved a micro-batch, ~1.0 s of a 7.5 s step -> -13.5%
  expected, -15.9% measured against D8 alone (the rest not traced: the tower's CPU-side position work and its per-block
  synchronizations are partly hidden from the GPU timeline). The prediction (-9 to -11%) counted the tower's share of D4's
  longer step (11%), before D8 shortened it.
- 9B: the tower's 317.5 ms a micro-batch is ~12% of the D8 micro-batch (2.7 s) -> ~-11.5% expected, -12.6% measured by the
  registered reading; the same cause for the miss against -8 to -10%.
- The first epoch: every call pays the key (5.5 / 3.5 ms) and the stores (1.0 / 2.3 ms a picture) -> +0.8% expected on the
  0.8B step, +0.2% at 9B; measured +0.5% over steps 1-25 at 0.8B (+1.3% on steps 1-10, none on 11-25) and +0.5-0.9% at 9B against
  D8 alone, inside the +2% predicted.

## D8 and D9: admission

| entry | parity | speed bar | verdict |
|---|---|---|---|
| D8 (`HEAD_AT_LABELS`) | the loss within 7.5e-8 to 6.9e-7 of the model's own path (bar 1e-5); every gradient 0.86-1.31x the model path's own error against float64 (bar 2); steps 1-2 the same weights' losses (0.8B 9.3e-8 / 0; 9B 8.6e-7 / 6.9e-7) | 0.8B: 0.788-0.829 of D4 on the first epoch (two runs), 0.818 on steps 11-100 (bar <= 0.90); 9B: 0.929 (bar <= 0.95) | ADMITTED |
| D9 (`VISION_CACHE`) | a served output the same bits as a recompute, both towers; one pixel moved, another grid: not served; no random numbers drawn | later epochs 0.8B 0.841 of D8 alone (registered reading 0.881), 9B 0.874 (registered reading) (bar <= 0.93) | ADMITTED |

Both are now macai's defaults (`macai.training`: `HEAD_AT_LABELS = True`, `VISION_CACHE = True`), selectable off;
`bench/phase4_train.py` keeps both off unless its options say otherwise, so the runs before reproduce.

## D8 and D9: integration (`bench/layer_logs/d8d9_0.8b_2x8_100steps.json`, `d8d9_9b_1x16_50steps.json`; `bench/d8d9_report.py` -> `d8d9_results.json`)

Both entries on, with D4's parts (`--layer RMSNORM,CONV,GATED_NORM,GATES,SWIGLU --head-at-labels --vision-cache`), at commit
69ca269, the tracked tree clean, one job at a time under the guard; every training forward took the head at the labelled
positions (0.8B 800 of 800 micro-batches, 9B 800 of 800, no fallback) and went through the cache.

### Qwen3.5-0.8B, 2 x 8, the cache emptied each micro-batch, 4 epochs (100 steps), 15:34-15:45 PST

| measure | bar / prediction | D8 + D9 | D4 (run of record) | verdict |
|---|---|---|---|---|
| the guard's peak footprint | <= 53.5 GB | 29.20 GB | 41.58 GB | MET |
| the steady step (steps 11-100) | | 6.522 s (-28.9%) | 9.178 s | |
| steps 1-10 / 11-25 / 26-50 / 51-75 / 76-100, s a step | | 7.60 / 7.67 / 6.32 / 6.28 / 6.28 | 9.10 / 9.73 / 9.24 / 9.00 / 8.96 | |
| the loss: the largest gap from the reference pair's mean | <= 2 x the pair's largest spread = .0159 | .0100 (step 47) | .0086 | met |
| the final loss within 1% of the pair's mean (0.1967) | <= 1% | 0.1975 (0.44%) | 0.1948 (0.94%) | met |
| root mean square of the step-loss difference against A / B / C | the references among themselves .0034 each | .0034 / .0042 / .0037 | .0034 / .0044 / .0041 | |
| its root mean square gap over spread (0.87 expected of a run drawn like the reference) | | 1.01 | 1.03 | |
| mean loss (the pair's 0.5776) | | 0.5760 | 0.5754 | |
| the live tensors at a forward's end (peak) / the driver's pool (peak) | | 9.14 GiB / 25.15 GB | 12.94 GiB / 36.80 GB | |
| the cache | | 372 pictures, 1.68 GB of CPU memory | | |

D8 alone over the same 100 steps (`d8_0.8b_2x8_100steps`, the direct baseline of D9): steady 7.511 s, footprint 27.30 GB; its
loss by Phase 4's readings: the largest gap .0082 (met), the root mean square gap over spread 0.97, the final loss 0.1923, 2.2%
from the pair's mean (the 1% reading not met; the references' own final losses disagree by 1.2% (A, B) and 2.2% (C against the
pair), and this run's adapters start from their own unseeded draw).

### Qwen3.5-9B, 1 x 16, the cache emptied, 2 epochs (50 steps), 15:45-16:19 PST

| measure | bar / prediction | D8 + D9 | D4 (run of record, 25 steps) | verdict |
|---|---|---|---|---|
| the guard's peak footprint (50 steps) | <= 74.6 GB | 51.96 GB | 60.44 GB | MET |
| steps 1-10 / 11-25 (first epoch) | | 42.8 / 43.13 s | 45.6 / 46.20 s | |
| steps 26-50 (second epoch) | | 37.48 s (-18.4% against the 45.94 s baseline) | (45.94 s: D4's 11-25 times D2m's second-epoch ratio) | |
| steps 11-50 | | 39.60 s | | |
| the live tensors at a forward's end (peak) / the driver's pool (peak) | | 15.31 GiB / 46.22 GB | 17.28 GiB / 54.46 GB | |
| the loss at steps 1-2 (the same weights) | | 2.2223094 / 2.0706000 | 2.2223074 / 2.0705986 (8.6e-7, 6.9e-7) | |
| the cache | | 372 pictures, 3.56 GB of CPU memory | | |

The 50-step run's learning rate differs from Phase 4's 25-step 9B reference's after the first update (warmup 2 steps against 1:
1e-4 against 2e-4 at step 2), so its loss is compared with that reference at steps 1-2 only: 2.2223 / 2.0706 against the
reference implementation's 2.2228 / 2.0699 (2.3e-4 and 3.5e-4 apart, as D4's kernels were). D8 alone at 9B runs the reference's
schedule and was compared over all 25 steps (above: .0012, D4 .0013).

As chains:
- The step, 0.8B: D8 takes ~216 ms from a micro-batch (the head and the loss 269.6 -> 53.6 ms, the profile below) -> D4's
  9.18 s -> 7.51 s; D9 then serves the tower's 133 ms a micro-batch from a 7.4 ms lookup from the second epoch on -> 7.48 -> 6.29
  s in the later epochs -> 6.52 s over steps 11-100, 29% under D4's.
- The step, 9B: D8 takes the head and the loss from ~298 to ~84 ms a micro-batch (the bench) -> 46.2 -> 42.9 s (D8 alone); D9
  the tower's ~318 ms from the second epoch -> 37.5 s.
- Memory: D8 removes most of the vocabulary-wide float32 tensors (live -3.8 GiB at 0.8B, -2.0 GiB at 9B) -> the allocator's pool
  shrinks 11.7 / 8.2 GB -> the footprint 41.58 -> 29.20 GB and 60.44 -> 51.96 GB, the cache's CPU memory (1.68 / 3.56 GB)
  included.
- The loss: D8 changes the rounding of the head's gradient in one element of a hundred, D9 changes nothing (the served output
  is the computed one's bits) -> the curve stays within the references' spread by Phase 4's readings.

## The step profile with D4 + D8 (0.8B, `profile_d8_0.8b_2x8.json`, steps 3, 5, 7)

Step 0's profiler, options and steps as D4's re-profile, with D8 on (`--head-at-labels`); D9 off, because the profiler replays
each micro-batch (B and C) after training on it (A), and with the cache on the replays would be served from it while A computed.
Valid as before: the profiled pass against its unprofiled replay 0.999 (median 1.014; none over 1.10), the components' sum
against the wall clock 0.9993, no stack mismatches or negative segments.

| group, ms a micro-batch (share) | D4 | D4 + D8 | change |
|---|---|---|---|
| the layers' frozen products | 279.7 (26.6%) | 281.4 (33.5%) | +1.6 |
| the LM head and the loss | 269.6 (25.6%) | 53.6 (6.4%) | -216.0 |
| the LoRA adapters' extra work | 135.1 (12.8%) | 154.8 (18.4%) | +19.7 |
| the vision tower | 134.1 (12.7%) | 132.8 (15.8%) | -1.3 |
| the rule | 75.7 (7.2%) | 75.6 (9.0%) | -0.1 |
| data, optimizer and the Trainer | 61.0 (5.8%) | 39.9 (4.7%) | -21.1 |
| the full-attention layers' attention | 53.8 (5.1%) | 58.0 (6.9%) | +4.2 |
| the surroundings | 33.3 (3.2%) | 33.4 (4.0%) | +0.1 |
| (beside) a new input shape, A - B | 17.9 (1.7%) | 10.1 (1.2%) | -7.8 |
| the micro-batch | 1,052.8 | 839.4 | -213.4 (-20.3%) |

The head and the loss by component (ms a micro-batch, forward / backward): the head's product 72.2 / 72.5 -> 18.9 / 13.8; the
loss (the float32 cast, the cross-entropy and their backward) 24.7 / 93.2 -> 4.8 / 13.5; accelerate's float32 copy of the
returned logits 6.9 -> 1.4; the gather of the labelled rows 0.3 / 0.4 (new); the labels' shift and gather 0.5 (new). The data,
optimizer and Trainer group falls with them (the cache emptied between forward and backward releases 5x smaller tensors). The
LoRA adapters' and the attention's rises (+20, +4 ms) are not explained by D8, which does not touch them; the profiled steps run
early in the epoch and the two profiles' replay ratios differ (1.052 against 0.999).

## D8b, the fused cross-entropy (conditional): the trigger reading and a proposal (not built)

The trigger (registered): the head and the loss still at >= 3% of the step after D8. Read in situ at 0.8B: 53.6 ms, 6.4% of the
micro-batch -> the trigger fires. At 9B (not profiled again): the bench's 83.6 ms a micro-batch against the D8 micro-batch's
~2.7 s -> ~3.1%, at the threshold.

What the 6.4% is made of: the head's two products 32.7 ms (61%), the loss's own passes (the float32 cast, the cross-entropy's
forward and backward, the cast back) 18.3 ms (34%), accelerate's float32 copy 1.4 ms, the gather and labels 1.2 ms. A fused
cross-entropy touches only the loss's own passes and the copy.

The proposal (D8b, for its own entry):

| item | proposal |
|---|---|
| design | a Metal cross-entropy over the bf16 logits of the labelled rows, MLX's form (prior art s.7.5): forward, one pass a row (online max and sum in float32) writing the loss and the row's log-sum-exp; backward, one pass rebuilding softmax from the saved log-sum-exp and writing the bf16 gradient over the logits in place; no float32 copy of the logits; the logits not returned in training (accelerate's float32 copy removed with them), behind a setting |
| floor | three passes over the bf16 logits: 312 x 248,320 x 2 bytes x 3 = 465 MB a 0.8B micro-batch (1.3 ms at 356 GB/s); 9B 243 MB (0.7 ms) |
| predicted | the loss's own passes and the copy 19.7 -> ~2.6 ms a 0.8B micro-batch (2x its floor): ~-135 ms a step, -1.8% of the D8 step (7.5 s), -2.2% of the D8 + D9 later-epoch step (6.3 s); 9B (the bench's 6.3 ms -> ~1.4 ms; in situ perhaps twice the bench): ~-0.2 to -0.4% |
| bars (proposed) | parity as D8's (the loss within relative 1e-5 of the model's own loss; the gradient to the hidden states within twice its error against float64); speed >= 1.5% of the 0.8B step |
| the larger lever beside it | the head's products at these row counts run at 7.1-8.6 TFLOP/s forward (312 / 163 rows) against 11-12 at full positions: 61% of what is left; a matter for D5 (the products), not D8b |

## D8 and D9's verdicts in short

| registered | measured | verdict |
|---|---|---|
| D8 parity: the loss within relative 1e-5 of the model's own path | 7.5e-8 (0.8B widths), 0 (9B widths), 3.5e-7 (the real 0.8B), 6.9e-7 (the real 9B) | MET |
| D8 parity: every gradient within twice the model path's own error against float64 | 0.86-1.31x (two-layer models at both widths); 0.91-1.09x over 302 gradients of the real 0.8B against the whole model in float64 | MET |
| D8 parity: steps 1-2 against the run without it | 0.8B 9.3e-8 / the same bits; 9B 8.6e-7 / 6.9e-7 | MET |
| D8 speed: >= 10% faster at 0.8B | -17.1% / -21.2% on the first epoch (two runs), -18.2% on steps 11-100 | MET: ADMITTED |
| D8 speed: >= 5% faster at 9B | -7.1% (steps 11-25, D8 alone); -6.6% in the integration's first epoch | MET: ADMITTED |
| D8 prediction: the head and the loss a 0.8B micro-batch 266.7 -> 50-75 ms | 269.6 -> 53.6 ms | met |
| D8 prediction: 9B 347.6 -> 70-100 ms | not profiled in situ; the bench's whole chain 83.6 ms | met by the bench (not in situ) |
| D8 prediction: the step 0.8B -14 to -18% | -18.2% (steps 11-100), -17.1% / -21.2% (the first epoch, two runs) | missed by 0.2 point (faster) on the steady step |
| D8 prediction: the step 9B -6 to -9% | -7.1% | met |
| D8 prediction: the logits' memory down by the same fraction | the head computes 17.9% of the positions (0.8B) / 21.5% (9B), every vocabulary-wide tensor with them; the live tensors at a forward's end -3.8 / -2.0 GiB | met |
| D9 parity: the served output against a recompute | the same bits (both towers, every check) | MET |
| D9 parity: a changed picture never served | one pixel moved, another grid, another dtype: never served | MET |
| D9 speed: the later epochs >= 7% faster at both sizes | 0.8B -15.9% against D8 alone (-11.9% by the registered reading); 9B -12.6% (registered reading) | MET: ADMITTED |
| D9 prediction: the later epochs 0.8B -9 to -11%, 9B -8 to -10% | -15.9% (-11.9%); -12.6% | missed (faster): the prediction counted the tower's share of D4's longer step |
| D9 prediction: the first epoch within +2% | 0.8B +0.5%, 9B +0.5% (steps 11-25) against D8 alone | met |
| D9 prediction: the cache ~0.6 MB a picture at 0.8B, ~2.3 MB at 9B | in full 4.51 / 9.58 MB; the merged features alone 1.13 / 4.51 MB | missed (larger): the sample's 372 pictures average 2,201 patches (550 merged positions), and the cache keeps the tower's last hidden states too |
| D9 prediction: the lookup <= 15 ms | 7.4 ms (0.8B, two pictures), 6.2 ms (9B), hashing included | met |
| integration 0.8B: footprint <= 53.5 GB; the loss as Phase 4 read it | 29.20 GB; the largest gap .0100 against .0159; the final loss 0.44% | MET; like the reference |
| integration 9B: footprint <= 74.6 GB | 51.96 GB | MET |

In plain words: the head and the loss now run only where a label is, which takes four fifths of the vocabulary-wide work out of
a 0.8B micro-batch (three quarters at 9B) without changing the loss beyond float32 rounding; and a frozen tower's output is kept per picture and served
from the second epoch on, the same bits as a recompute. A 0.8B LoRA step of the readers' recipe falls from D4's 9.18 s to 7.51 s
with the first and to 6.29 s in the later epochs with both (6.52 s over steps 11-100, -29%); a 9B step from 46.2 s to 42.9 s,
then 37.5 s in the second epoch; the footprint falls 12 / 8 GB with the cache's CPU memory counted.

## D8 and D9: not done, and what is uncertain

- D8b (a fused cross-entropy) is not built: its trigger fires at 0.8B (6.4% of a micro-batch), but what it can remove is the
  loss's own passes and the output copy (~2% of the step); proposed above for its own entry. The head's products at small row
  counts (61% of what is left) belong to D5.
- The two D8-alone runs at 0.8B disagree on steps 16-25 by 5-12% a step; the repeat agrees with the integration run step for
  step, and the first came straight after the loss-head bench. The cause is not traced; the bars hold on either run.
- D8's larger gain on D4's slow steps (19-26, the largest batches) is read as the vocabulary-wide float32 tensors' cost there;
  not profiled (the profile covers steps 3-7).
- D9's later-epoch gain against D8 alone (-15.9%) is ~2 points more than the tower's profiled time predicts (-13.5%); the rest
  (the tower's CPU-side position work and its per-block synchronizations, partly hidden from the GPU timeline) is not traced.
- At 9B, D9's gain is the registered reading only (D8 alone ran one epoch there); the head and the loss after D8 were measured in
  the bench, not in situ.
- The 0.8B repeats inside an epoch (the 18 items in two forms) were never served in the first epoch: each shared its micro-batch
  with a picture not yet stored, and the cache serves only whole micro-batches. Computing only the missing pictures would serve
  them, at the cost of a tower pass whose micro-batch differs from the uncached one (the same bits, by the probes' composition
  checks).
- The cache keeps the tower's last hidden states (registered: what get_image_features returns, in full), which the model never
  reads in this path: 3.4 of the 4.5 MB a picture at 0.8B, 5.1 of 9.6 MB at 9B. Keeping the merged features alone would cut the
  cache to a quarter / a half.
- The key copies the micro-batch's pixel rows back from the GPU to hash them (5.8 / 4.2 ms of the 7.4 / 6.2 ms lookup);
  hashing them on the CPU before the Trainer moves them (in the collator) would remove the copy but needs the trainer's side.
- A tower weight changed through a tensor's `.data` is not seen by the cache's version counters; `clear_vision_cache()` after one.
- Qwen3.5-MoE and Qwen3-Next: not touched by D8 or D9.
- D8 alone over 100 steps ended 2.2% from the reference pair's final loss (the 1% reading not met there; the integration run with
  both met it at 0.44%): the references themselves disagree by 1.2% and 2.2% at the last step, and each run's adapters start from
  their own unseeded draw.

## D5, the products and the LoRA adapter path (registered before any code)

| part | registered | predicted | bars |
|---|---|---|---|
| (a) the frozen products | step 0's products reading decides (PyTorch's products tie MLX's: no custom product for the frozen projections); read again at the shapes D8 created: the LM head's product at the labelled row counts (~312 rows at 0.8B, ~163 at 9B, against 248,320 words), PyTorch beside MLX's mx.matmul | the big projections at >= 85% of the large-product rate at the bench's token counts, >= 70% at the micro-batch's; MLX within 10% of PyTorch there | |
| (b) the LoRA adapter path fused | the dropout, the rank-32 products, the scaled add, forward and backward, in place of PEFT 0.21.2's `lora.Linear.forward` for vanilla LoRA only (one active adapter, no variant or DoRA, not merged, adapters enabled, no adapter_names, an nn.Linear base, a plain nn.Dropout or Identity); anything else runs PEFT's own forward and is counted. Replaced only while its source hashes as in PEFT 0.21.2; selectable off. The masks drawn by torch's own dropout call, so the fused path and PEFT draw the same masks from the same generator state (no random stream of its own). Under bf16 autocast as the readers train: PEFT's semantics matched, or every difference documented with its parity reading | the rank-32 products <= 30% of the large-product rate in PyTorch; the fused adapter path >= 1.5x its PyTorch path | parity as D4 (the forward and every gradient against float64 on the CPU at the harness's shapes: odd lengths, B > 1, float32 / bfloat16 / float16 inputs; the fused path's error at most twice the PyTorch path's own, both measured); the same generator state gives the same masks; speed: >= 1.2x on its part (the adapter path, forward + backward) at both sizes, else not adopted |
| (c) a custom product | MLX's open kernel through compile_shader, its licence kept, only at a shape carrying >= 5% of the step where PyTorch reaches < 80% of MLX: the head's product is 3.9% of a D8 micro-batch, so built only if the measurement puts it at >= 5% of the step and under 80% of MLX; otherwise the reading is reported | | |

Integration, once admitted (as D4, D8 and D9): every run with the admitted entries (`--layer RMSNORM,CONV,GATED_NORM,GATES,SWIGLU
--head-at-labels --vision-cache`) and D5's new runner option. Qwen3.5-0.8B 2 x 8, the cache emptied, the recomputed states, 4 epochs
(100 steps): the loss read as Phase 4 read it against A, B, C, the guard's peak footprint <= 53.5 GB, the steady step and each
window against D8 and D9's run of record (`d8d9_0.8b_2x8_100steps`: 6.522 s over steps 11-100; windows 1-10 7.60, 11-25 7.67,
26-50 6.32, 51-75 6.28, 76-100 6.28 s). Qwen3.5-9B 1 x 16, the cache emptied, 25 steps (1 epoch): footprint <= 74.6 GB, the
steady step (11-25) against `d8d9_9b_1x16_50steps`' same steps (43.13 s; the first epoch, where the vision cache served 18
repeated pictures, as a 25-step run with it on does), the loss over steps 1-25 against Phase 4's 9B reference.

### D5 (b): the design, and its predictions written before any code

What PEFT runs today per adapted projection, under the readers' bf16 autocast (the adapters in float32, the base in bf16):

| PEFT's pass | bytes an input element (d_in wide) | bytes an output element (d_out wide) |
|---|---|---|
| forward: the float32 copy of x; dropout in float32 (a byte mask); autocast's cast of the dropped copy to bf16; the A product reads it | 6 + 9 + 6 + 2 = 23 | |
| forward: the B product's output; the scaled copy; the add to the base output | | 2 + 4 + 6 = 12 |
| backward: the scaled gradient; the two products that read it (dP, dB) | | 4 + 4 = 8 |
| backward: the A product's input gradient, its cast to float32, dropout's backward in float32, the cast back, the add to the base's input gradient; dA reads the dropped copy | 2 + 6 + 9 + 6 + 6 + 2 = 31 | |

The design (`macai.lora`), the base product left to PEFT's own module call:

- forward: `xd, mask = torch.native_dropout(x, p, True)` on x in its own dtype when that is the compute dtype (probe: the same
  mask as PEFT's draw from the same generator state at every dtype, and the dropped bf16 / fp16 values bit for bit PEFT's dropped
  float32 copy after autocast's cast, 0 of 5.4 M elements differing); P = xd A^T by PyTorch's product ([N, 32]); Y = R + s P B^T
  by one Metal kernel (simdgroup-matrix products of rank 32 in float32, the base output R read and Y written once, one rounding);
- backward: G = s dY B and dB = s dY^T P by PyTorch's products, the scale applied to the [N, 32] and [d_out, 32] results instead
  of to dY (exact for s = 2: the products give the same bits for a power-of-two scale, probed); dA = G^T xd by PyTorch's
  product; dX_lora = M (G A) / (1 - p) by one Metal kernel (the rank-32 product with the mask and the scale in its epilogue,
  dX written once); autograd adds it to the base's input gradient, as before.
- traffic: about 18 bytes an input element and 8 an output element against PEFT's 54 and 20.
- numerics: dA and dB the same products on the same bits as PEFT's (s = 2); Y and dX_lora rounded once where PEFT rounds three
  and two times.

Predicted before any code (in situ, a micro-batch; the products at step 0's measured rank-32 rates, the passes at the copy rate,
the kernels at 1.5x their traffic):

| | 0.8B 2 x 8 (1,622 positions) | 9B 1 x 16 (768 positions) |
|---|---|---|
| the adapter path now | 154.8 ms (the step profile with D4 + D8) | ~256 ms (step 0; not profiled since D4) |
| predicted with the fused path | 60-85 ms (point 70: the products ~38, dropout ~5, the two kernels ~14, the gradient add ~6, casts and launches ~7) | 105-150 ms (point 125) |
| the adapter path in isolation, forward + backward, all projections weighted by their calls | >= 1.8x | >= 1.8x |
| the step | -0.55 to -0.75 s (8 micro-batches): steady 6.52 -> 5.8-6.0 s (-8 to -11%); the later epochs 6.29 -> ~5.6 s | -1.7 to -2.4 s (16 micro-batches): steps 11-25 43.13 -> 40.7-41.4 s (-4 to -6%) |
| memory | unchanged or lower (the same tensors kept: the dropped input, the mask, P; the float32 copies gone) | the same |

The rank-32 products' registered prediction (<= 30% of the large-product rate) can be read from step 0's records already: at
the micro-batch shapes they ran at 15-67% of the 11.5 TFLOP/s large-square rate (median 36% at 0.8B, 45% at 9B; at the bench
shapes 29-77%, medians 49% / 48%); this entry's bench grades it again.

## D5 (a): the products, read again at the shapes D8 left (`products_head_rows_torch.json`, `products_head_rows_mlx.json`, 17:38-17:41 PST)

Step 0's products reading decided (a) before this entry: PyTorch's frozen products run at the rate and tie MLX's. Its registered
predictions, graded from step 0's records (`products_torch.json`, `products_mlx.json`; every frozen projection but the 16-32-wide
in_proj_a / in_proj_b; the rate: PyTorch's F.linear large squares in the same session, 11.5 TFLOP/s):

| shape | of the large-square rate (predicted >= 85% at the bench's, >= 70% at the micro-batch's) | PyTorch's rate over MLX's (predicted within 10%) |
|---|---|---|
| 0.8B 4 x 1,024 (bench) | 0.77-1.12, median 1.05: MISSED by o_proj's forward (0.77) | 0.88-1.01: missed by the narrow k / v forward (0.88) |
| 0.8B 2 x 903 (micro-batch) | 0.82-1.01, median 0.98: met | 0.84-0.99: missed by seven narrow products (out_proj, k / v, o_proj, down, 0.84-0.90) |
| 9B 2 x 1,024 (bench) | 0.99-1.12, median 1.08: met | 0.90-1.03: missed by k / v's backward (0.90) |
| 9B 1 x 859 (micro-batch) | 0.87-1.11, median 1.01: met | 0.83-1.06: missed by k / v's forward (0.83) |

Every product outside 10% of MLX is one where PyTorch is the slower, all on narrow projections, none carrying 5% of the step:
the rule (c) builds nothing for the frozen projections.

The LM head at the row counts D8 leaves it (the labelled positions flattened to [rows, hidden], against 248,320 words; PyTorch by
Metal events, MLX by the wall clock, each in its own process, the large squares as the control: PyTorch 11.1-12.0, MLX 11.4-12.9
TFLOP/s):

| rows | 0.8B forward Z = E W^T: PyTorch / MLX TFLOP/s | its gradient dE = dZ W | 9B rows | 9B forward | its gradient |
|---|---|---|---|---|---|
| 128 | 6.45 / 12.62 (0.51) | 12.37 / 8.57 (1.43) | 64 | 4.68 / 11.89 (0.39) | 12.21 / 7.66 (1.62) |
| 256 | 8.95 / 12.29 (0.72) | 12.46 / 10.33 (1.20) | 128 | 6.71 / 11.49 (0.58) | 12.45 / 8.60 (1.44) |
| **312** (the profile's average) | **8.70 / 11.87 (0.73)** | **12.07 / 9.90 (1.21)** | **163** | **7.25 / 9.77 (0.74)** | **10.61 / 9.20 (1.15)** |
| 512 | 10.42 / 11.75 (0.88) | 12.53 / 9.96 (1.24) | 256 | 9.10 / 11.45 (0.79) | 12.72 / 10.36 (1.19) |
| 1,024 | 11.09 / 11.69 (0.94) | 12.46 / 10.67 (1.15) | 512 | 10.77 / 11.53 (0.93) | 12.12 / 10.79 (1.13) |
| 1,622 (every position) | 11.45 / 11.33 (1.02) | 11.56 / 10.30 (1.15) | 768 | 11.44 / 11.67 (0.98) | 11.86 / 10.73 (1.10) |

As a chain: the head's forward is F.linear's X W^T form with a few hundred rows against a quarter-million outputs -> PyTorch's
product runs at 7-9 TFLOP/s there, MLX's at 10-12 -> at the labelled rows the forward is under 80% of MLX (0.73 / 0.74); its
gradient, the X W form on the same sizes, runs faster than MLX's (1.21 / 1.15) -> both products together are 0.94 / 0.91 of MLX.
(c)'s rule asks for a shape at >= 5% of the step under 80% of MLX: the forward carries 18.8 ms of a micro-batch, 2.2% with D4 +
D8 and 2.5% with D5 (the step profiles), both products 3.9% / 4.3% -> (c) is not triggered; no custom product is built. The
reading for a later entry: the slow part is the forward's form at narrow row counts, not the products' rate.

## D5 (b): as built (`macai.lora`, [lora.md](lora.md))

| piece | what runs |
|---|---|
| the replacement | PEFT 0.21.2's `lora.Linear.forward`, replaced while its source hashes as written against (`macai.lora._peft.SOURCES`); the setting `macai.lora.FUSED`, read at every call; for vanilla LoRA in the form the kernels take, the base layer's own module call (`self.base_layer(x)`, as PEFT calls it), then `adapter_path` (looked up at every call: the step profiler wraps it); anything else runs PEFT's own forward, counted by reason |
| the mask | `torch.native_dropout(x, p, True)`, the call nn.Dropout makes on the Apple GPU: with float32 adapters and x in the compute dtype (bf16 under the readers' autocast) on x itself, otherwise on x cast to the adapters' dtype, as PEFT runs it |
| forward | P = x_d A^T by `torch.mm` ([N, 32]); Y = R + s P B^T by the up kernel (a new tensor: R is not changed in place, so a hook that returns the base output as a view stays valid) |
| backward | G = dY B and dB = dY^T P by `torch.mm`, the scale s applied to the [N, 32] and [d_out, 32] results; dA = G^T x_d by `torch.mm`; the adapter's input gradient M (G A) / (1 - p) by the down kernel; dR = dY; autograd adds the input gradient to the base's |
| the kernels | one source, two epilogues, instantiated for every pair of output dtype and compute dtype (float32 / bfloat16 / float16) and every rank that is a multiple of 8 up to 64: 4 simdgroups a threadgroup own 64 columns and walk a range of rows 32 at a time, the right operand's 8 x 8 fragments for their columns in registers, the left operand's rows staged in threadgroup memory as float32, the accumulator tile through threadgroup memory to an epilogue of two 8-element vectors a thread; float32 outputs take four fma chains over the rank summed as a tree (one chain in rank order, which the simdgroup products give, measured 2.7x the platform product's float32 error at 14 rows: entry 18) |
| saved for the backward | the dropped input (bf16), the bool mask, P, the bf16 casts of A and B: the tensors PEFT's chain keeps; the float32 copies are gone |

What PEFT's forward reads beside its source (the active adapters, disabled, merged, the input cast) lives in PEFT's base tuner
layer; the replacement reads them through PEFT's own attributes and the tests hold the fused path to PEFT's own forward, without a
source hash of that file.

## D5 (b): parity (`tests/test_lora.py`; the test run of record `bench/layer_logs/d5_tests.log`, every case's errors in `d5_parity.jsonl`)

Error = the largest absolute difference over the reference's largest magnitude, each tensor apart; the bar: the fused path's error
at most twice PEFT's own on the same case. The float64 reference applies the mask the fused path drew (checked equal to PEFT's
draw from the same generator state).

| configuration (cases; tensors) | PEFT's own errors against float64 | the fused path's error over PEFT's: worst (median); tensors at or under 1 | verdict |
|---|---|---|---|
| bf16 x under bf16 autocast, the base weight drawn (16; 64) | 3.2e-3 to 7.1e-3 | 1.39, the output on one row (1.00); 61 of 64 | MET |
| the same, the base weight zero: the adapter term alone, every width of both models (24; 96) | 2.9e-3 to 6.5e-3 | 1.00 (1.00); 96 of 96 | MET |
| fp16 x under fp16 autocast, the base drawn / zero (16 / 24; 64 / 96) | 3.3e-4 to 7.6e-4 | 1.03 / 1.07 (1.00); 63 / 94 | MET |
| fp32 x without autocast, drawn / zero (16 / 24; 64 / 96) | 8.9e-8 to 4.6e-6 | 1.04 / 1.27 (1.00); 61 / 86 | MET |
| bf16 x without autocast (the adapters' float32 as the compute dtype), drawn / zero (16 / 24; 64 / 96) | 9.6e-8 to 6.0e-3 | 1.00 / 1.00; 64 / 96 | MET |
| no mask (an Identity dropout; nn.Dropout in evaluation), every configuration (8; 32) | 2.9e-7 to 4.2e-3 | 1.02 (1.00); 30 of 32 | MET |
| whole decoder layers with LoRA and dropout on, both kinds, both sizes at the bench shapes (4; 60) | 5.3e-3 to 2.3e-2 | 1.29, a LoRA weight's gradient (1.00); 32 of 60 | MET |

The widths: the drawn-base cases at four of the adapted widths (1,024 -> 6,144, 1,024 -> 2,048, 3,584 -> 1,024, 12,288 -> 4,096) and
every row shape (1, 1), (2, 7), (3, 33), (2, 129); the base-zero cases at every adapted width of both models, rows (2, 7) and (3,
33). 172 cases, 680 tensors, none over the bar; the test run of record: 606 passed (the 505 before and 101 new).

Beside them:
- the masks: the same generator state gives the fused path and PEFT the same mask, in one layer and in a stack of three where the
  stream moves between draws (3 of 3), and in whole decoder layers (6 / 7 of 6 / 7); the dropout on x in its own dtype gives PEFT's
  dropped values bit for bit (bf16 and fp16, p = 0.05, 0.1, 0.5, three shapes);
- the weight gradients of a layer under the readers' bf16 autocast: the same bits as PEFT's (dA and dB at every width and shape);
- every fallback condition (a LoRA variant, a LoRA bias, a rank of 12, adapters disabled, merged, another dropout module,
  adapter_names, two active adapters, the CPU) runs PEFT's forward, counted by its reason, with PEFT's bits; an Identity dropout
  and evaluation mode take the fused path; an input without a gradient and a frozen adapter weight give the gradients PEFT gives;
  a source that differs from 0.21.2 leaves PEFT's method with a warning.

As chains:
- The output: PEFT rounds the B product, the scaled copy and the sum (three bf16 roundings under autocast); the up kernel adds
  R + s P B^T in float32 and rounds once -> with the base weight zero the output is the same bits (the add of zero is exact) or
  closer; with a drawn base weight the base product's own rounding dominates both paths' errors (ratio 1.00 median, 1.39 at worst
  on a single row).
- The input gradient: PEFT rounds the product G A to bf16, converts to float32 for dropout's backward and rounds again; the down
  kernel applies the mask and the scale to the float32 accumulator and rounds once -> at or under PEFT's error.
- The weight gradients: G and dB are the products PEFT runs on the same bits (the scale 2 commutes with them exactly: probed),
  dA reads PEFT's dropped values bit for bit -> the same bits.

In real training runs (`bench/lora_probe.py`, records `probe_d5_0.8b_run2.json`, `probe_d5_9b.json`):

| check | 0.8B (2 x 2, 12 items, 1 epoch) | 9B (1 x 4, 8 items, 1 epoch) |
|---|---|---|
| adapted projections that took the fused path, calls each | 150 of 150, at every one of 12 micro-batches (1,800 calls) | 200 of 200, at every one of 16 (3,200) |
| PEFT's own forward | 2 calls, both the trainer's own check on the CPU before training | the same |
| dropout on, the same generator state (the adapters' B drawn for the check): masks the same | 150 of 150 | 200 of 200 |
| the loss, fused against PEFT | 5.713 / 5.690 (0.40%) | 2.6634 / 2.6645 (0.039%) |
| the LoRA weights' gradients against each other: median / worst relative difference | 32% / 104% | 4.9% / 95% |
| every dropout an identity, both paths against the whole model in float64 on the CPU (301 gradients) | the fused path's error over PEFT's 0.36-2.38, median 0.89, geometric mean 0.91; over 1.5: 12, over 2: 1; the errors' medians 0.214 (fused) / 0.238 (PEFT) | (no float64 model: 75 GB) |

The whole-model reading is noisier than any check above, and in both directions: through 24 bf16 layers a rounding that differs
in each adapted projection propagates, so each gradient's error against float64 moves with it (PEFT's own errors run from 0.13%
(the loss) to 100% of a gradient's largest value; the input embeddings' gradient is noise-dominated: PEFT 1.02, the fused path 0.36). The
fused path's errors are lower in the median (0.214 against 0.238) and its ratio spreads around 1 (geometric mean 0.91); one
gradient of 301 sits over 2 (layer 8's in_proj_z LoRA B: 0.137 / 0.326). The registered bar is read at the harness's shapes and in
whole decoder layers (above), where it is met.

## D5 (b): speed

### The adapter path in isolation (`bench/bench_lora.py`; `d5_adapter_run1.json` 18:15, `d5_adapter.json` 18:47 (the record))

Every adapted projection's shape at the readers' in-situ token counts (0.8B 2 x 811 = 1,622 positions, 9B 1 x 768), PEFT's LoRA
layer with the fused path off and on, bf16 autocast, dropout on, the gradients to x, A and B; Metal events, back-to-back calls,
the variants interleaved in 5 rounds; each projection weighted by its calls in a micro-batch. Two readings: the layer less its base
product (the base timed in the same rounds) and the part alone (PEFT's own adapter operations against the fused function on the
same leaves, no subtraction).

Forward + backward a call, ms (the projections and their calls a micro-batch in brackets):

| 0.8B, 2 x 811 | the layer less its base (run 2, the record) | the part alone (run 2) | the layer less its base (run 1) |
|---|---|---|---|
| 1,024 -> 6,144 (in_proj_qkv 18) | 1.191 -> 0.515 (2.31x) | 1.214 -> 0.521 (2.33x) | 1.189 -> 0.592 (2.01x) |
| 1,024 -> 2,048 (in_proj_z 18) | 0.541 -> 0.281 (1.93x) | 0.533 -> 0.322 (1.66x) | 0.564 -> 0.282 (2.00x) |
| 2,048 -> 1,024 (out_proj 18, o_proj 6) | 0.681 -> 0.313 (2.18x) | 0.621 -> 0.349 (1.78x) | 0.666 -> 0.313 (2.13x) |
| 1,024 -> 3,584 (gate / up 24 each) | 0.863 -> 0.414 (2.08x) | 0.858 -> 0.442 (1.94x) | 0.865 -> 0.401 (2.16x) |
| 3,584 -> 1,024 (down 24) | 1.072 -> 0.491 (2.18x) | 1.021 -> 0.483 (2.11x) | 1.076 -> 0.491 (2.19x) |
| 1,024 -> 4,096 (q_proj 6) | 0.994 -> 0.451 (2.20x) | 0.964 -> 0.471 (2.05x) | 0.957 -> 0.441 (2.17x) |
| 1,024 -> 512 (k_proj / v_proj 6 each) | 0.363 -> 0.210 (1.73x) | 1.224 -> 0.721 (1.70x; every variant read slow at that point) | 0.352 -> 0.203 (1.73x) |
| **a micro-batch, forward + backward** | **124.99 -> 58.72 (2.13x)** | **132.51 -> 67.84 (1.95x)** | 124.85 -> 59.36 (2.10x) |
| a micro-batch, forward alone | 60.31 -> 29.12 (2.07x) | 67.58 -> 27.46 (2.46x) | 61.58 -> 31.26 (1.97x) |

| 9B, 1 x 768 | the layer less its base (run 2) | the part alone (run 2) | the layer less its base (run 1) |
|---|---|---|---|
| 4,096 -> 8,192 (in_proj_qkv 24, q_proj 8) | 1.125 -> 0.543 (2.07x) | 1.111 -> 0.499 (2.23x) | 0.640 -> 0.827 (0.77x: the base product read 8.33-9.01 ms over the rounds) |
| 4,096 -> 4,096 (in_proj_z, out_proj 24 each; o_proj 8) | 0.807 -> 0.417 (1.94x) | 0.743 -> 0.388 (1.92x) | 0.932 -> 0.471 (1.98x) |
| 4,096 -> 12,288 (gate / up 32 each) | 1.331 -> 0.583 (2.28x) | 1.332 -> 0.585 (2.28x) | 1.457 -> 0.696 (2.09x) |
| 12,288 -> 4,096 (down 32) | 1.748 -> 0.700 (2.50x) | 1.712 -> 0.682 (2.51x) | 1.464 -> 0.506 (2.89x) |
| 4,096 -> 1,024 (k_proj / v_proj 8 each) | 0.553 -> 0.277 (2.00x) | 0.490 -> 0.317 (1.55x) | 0.564 -> 0.272 (2.07x) |
| **a micro-batch, forward + backward** | **231.16 -> 104.87 (2.20x)** | **225.03 -> 102.03 (2.21x)** | 221.79 -> 117.93 (1.88x) |
| a micro-batch, forward alone | 127.34 -> 57.29 (2.22x) | 120.20 -> 58.30 (2.06x) | 131.88 -> 87.34 (1.51x) |

Why two readings, as a chain: at 9B the base product of the widest projections takes ~9-13 ms and moves by ~0.5 ms from round to
round (the chip's state; the large squares fell from 11.7 to 10.3 TFLOP/s over run 1) -> subtracted from the layer, it can swallow
an adapter path of ~0.6 ms (run 1's 4,096 -> 8,192: 0.77x) -> the part alone, timed on the same leaves without a base, reads the
same shape at 2.23x, and the two readings of run 2 agree within 1% at 9B (2.20x / 2.21x) and within 9% at 0.8B (2.13x / 1.95x;
the gap is the 1,024 -> 512 shape, read slow in every variant there).

The kernels against their least traffic (the up kernel reads R and writes Y; the down kernel writes dx and reads the mask; at 356
GB/s), and torch's dropout call, ms:

| shape | up kernel (its floor) | down kernel (its floor) | torch's dropout call |
|---|---|---|---|
| 0.8B 1,024 -> 6,144 | 0.135 (0.112) | 0.049 (0.014) | 0.024 |
| 0.8B 1,024 -> 3,584 | 0.098 (0.065) | 0.057 (0.014) | 0.024 |
| 0.8B 3,584 -> 1,024 | 0.050 (0.019) | 0.090 (0.049) | 0.050 |
| 0.8B 2,048 -> 1,024 | 0.050 (0.019) | 0.066 (0.028) | 0.033 |
| 9B 4,096 -> 12,288 | 0.135 (0.106) | 0.073 (0.027) | 0.034 |
| 9B 4,096 -> 8,192 | 0.096 (0.071) | 0.069 (0.027) | 0.033 |
| 9B 12,288 -> 4,096 | 0.074 (0.035) | 0.137 (0.080) | 0.075 |
| 9B 4,096 -> 1,024 | 0.036 (0.009) | 0.065 (0.027) | 0.034 |

As a chain: at the widest outputs the up kernel runs at 1.2-1.4x its traffic (R read and Y written once); the narrow outputs and
the down kernels take 0.04-0.07 ms whatever their bytes (a launch and each threadgroup's setup: the right operand's fragments and
the first rows staged), so they sit at 2-7x a floor of a few microseconds; torch's dropout call on bf16 costs 0.02-0.08 ms where
PEFT's float32 copy and dropout cost 0.08-0.51 ms (probe 2, entry 17).

The rank-32 products the fused path leaves to PyTorch (P = x_d A^T, G = dY B, dB = dY^T P, dA = G^T x_d) at the in-situ shapes,
against the session's large-square rate (11.7 TFLOP/s), the median and the range: run 2 (the record) 0.8B 26% (3-65%), 9B 30%
(10-72%); run 1 26% / 35%; step 0's reading of PEFT's rank-32 products at its micro-batch shapes (1,806 / 859 positions) 36% /
45%. The registered prediction (<= 30%) is met by the record at both sizes, at its edge at 9B, and missed by run 1 at 9B and by
step 0's reading: the rank-32 products run at about a third of the rate, a little faster than predicted.

### In situ: the step profile (`profile_d5_0.8b_2x8.json` against `profile_d8_0.8b_2x8.json`; at 9B `profile_d5_9b_1x16.json` against `profile_d8_9b_1x16.json`, both run for this entry)

The same profiler and steps as before (0.8B steps 3, 5, 7 of 8; 9B steps 2 and 4 of 5, as step 0's 9B run), D4 + D8 on, D9 off
(the profiler replays each micro-batch), with and without the fused path; the fused call is marked per projection
(`<name>.lora_fused`), what is left in PEFT's layer's own time (the replaced forward's Python, the base call's marks, the
backward's add of the input gradients) stays `lora_rest`. Valid as before: the profiled pass against its unprofiled replay 1.006 /
0.998 (0.8B / 9B; one of 24 0.8B micro-batches over 1.10), the components against the wall clock 0.9994 / 0.9998.

| group, ms a micro-batch (share) | 0.8B D4 + D8 | 0.8B with D5 | 9B D4 + D8 | 9B with D5 |
|---|---|---|---|---|
| the LoRA adapters' extra work | 154.8 (18.4%) | 83.3 (10.9%): fused 72.7, the layer's own 10.6 | 256.5 (9.0%) | 128.7 (4.8%): fused 108.6, the layer's own 20.1 |
| the layers' frozen products | 281.4 (33.5%) | 281.3 (36.9%) | 1,884.8 (66.3%) | 1,868.6 (69.0%) |
| the vision tower | 132.8 (15.8%) | 131.9 (17.3%) | 311.4 (11.0%) | 312.3 (11.5%) |
| the rule | 75.6 (9.0%) | 75.8 (9.9%) | 99.9 (3.5%) | 99.2 (3.7%) |
| the full-attention layers' attention | 58.0 (6.9%) | 58.0 (7.6%) | 68.7 (2.4%) | 68.6 (2.5%) |
| the LM head and the loss | 53.6 (6.4%) | 52.2 (6.8%) | 111.5 (3.9%) | 117.0 (4.3%) |
| data, optimizer and the Trainer | 39.9 (4.7%) | 35.4 (4.7%) | 34.2 (1.2%) | 35.1 (1.3%) |
| the surroundings | 33.4 (4.0%) | 34.3 (4.5%) | 47.9 (1.7%) | 48.0 (1.8%) |
| (beside) a new input shape, A - B | 10.1 | 20.0 | -5.0 | -14.7 |
| the micro-batch | 839.4 | 761.9 (-9.2%) | 2,841.3 | 2,706.9 (-4.7%) |

The adapter path in situ, 1.86x at 0.8B and 1.99x at 9B (the bar 1.2x), by projection (ms a micro-batch, PEFT's layer's whole
LoRA time against the fused call and the layer's own time):

| projection | 0.8B before -> D5 | 9B before -> D5 |
|---|---|---|
| lin.in_proj_qkv | 24.4 -> 12.1 (2.01x) | 26.6 -> 13.9 (1.91x) |
| lin.down_proj | 22.8 -> 9.4 (2.43x) | 45.1 -> 19.6 (2.30x) |
| lin.gate_proj / lin.up_proj | 16.2 / 16.2 -> 7.5 / 7.5 (2.16x) | 34.9 / 33.6 -> 16.6 / 15.0 (2.17x) |
| full.down_proj | 14.5 -> 9.1 (1.60x) | 23.7 -> 14.0 (1.69x) |
| lin.out_proj | 12.6 -> 6.4 (1.96x) | 21.1 -> 14.3 (1.48x) |
| lin.in_proj_z | 12.1 -> 7.2 (1.68x) | 19.8 -> 10.3 (1.93x) |
| full.up_proj / full.gate_proj | 11.3 / 5.3 -> 6.7 / 2.5 (1.69x / 2.12x) | 12.6 / 11.9 -> 5.8 / 5.0 (2.17x / 2.38x) |
| full.q_proj | 7.9 -> 8.1 (0.97x) | 9.5 -> 5.3 (1.78x) |
| full.o_proj | 5.8 -> 2.7 (2.16x) | 6.8 -> 3.3 (2.07x) |
| full.k_proj / full.v_proj | 2.7 / 3.0 -> 1.6 / 2.6 | 5.2 / 5.5 -> 2.8 / 2.8 |

By kind before D5 (0.8B / 9B): the rest of PEFT's forward 57.2 / 104.1, lora_A 39.8 / 64.8, lora_B 38.0 / 51.4, dropout 19.8 /
36.3. The per-projection rows move with where the backward's adds of shared inputs land (the input of q, k and v, or of gate and
up, is one tensor whose gradients autograd adds in whichever segment comes last): 0.8B's full.q_proj reads 0.97x in situ against
2.20x isolated, while the group's total falls 46%.

## D5: admission

| part | parity bar | speed bar (>= 1.2x on the adapter path, forward + backward, both sizes) | verdict |
|---|---|---|---|
| (b) the LoRA adapter path fused (`macai.lora.FUSED`) | MET (172 cases, the worst ratio 1.39; whole decoder layers 1.29; the masks PEFT's) | isolated 2.13x / 2.20x (the part alone 1.95x / 2.21x; run 1 2.10x / 1.88x); in situ 1.86x / 1.99x | ADMITTED |
| (c) a custom product for the head | | not triggered (the head's forward 2.2-2.5% of a micro-batch, under 5%) | not built |

The fused path is now macai's default (`macai.lora.FUSED = True`), selectable off; `bench/phase4_train.py` keeps it off unless
`--lora-fused` is given, so the runs before reproduce.

## D5: integration (`bench/layer_logs/d5_0.8b_2x8_100steps.json`, `d5_9b_1x16_25steps.json`; `bench/d5_report.py` -> `d5_results.json`)

Every admitted entry on (`--layer RMSNORM,CONV,GATED_NORM,GATES,SWIGLU --head-at-labels --vision-cache --lora-fused`), at commit
7e0ef7a with the tracked tree clean, one job at a time under the guard; every training forward took the fused path at every
adapted projection (0.8B 120,000 calls = 800 micro-batches x 150; 9B 80,000 = 400 x 200), PEFT's own forward only for the
trainer's two CPU checks before training.

### Qwen3.5-0.8B, 2 x 8, the cache emptied each micro-batch, 4 epochs (100 steps), 18:16-18:26 PST

| measure | bar / prediction | D5 | D8 + D9 (run of record) | verdict |
|---|---|---|---|---|
| the guard's peak footprint | <= 53.5 GB | 31.27 GB | 29.20 GB | MET (the prediction "unchanged or lower" missed: +2.07 GB) |
| the steady step (steps 11-100) | predicted 5.8-6.0 s | 6.000 s (-8.0%) | 6.522 s | prediction met (at its edge) |
| steps 1-10 / 11-25 / 26-50 / 51-75 / 76-100, s a step | | 6.80 / 7.20 / 5.92 / 5.68 / 5.68 | 7.60 / 7.67 / 6.32 / 6.28 / 6.28 | |
| the later epochs (steps 26-100) | predicted ~5.6 s | 5.76 s (-8.5%) | 6.29 s | missed (slower) by 0.16 s |
| steps 1-2 (the adapters' B is zero: the same weights) | | 2.5624918937683105 / 2.426778793334961 | the same bits | |
| the loss: the largest gap from the reference pair's mean | <= .0159 | .0103 (step 22) | .0100 | met |
| the final loss within 1% of the pair's mean (0.1967) | <= 1% | 0.1978 (0.60%) | 0.1975 (0.44%) | met |
| root mean square of the step-loss difference against A / B / C (the references among themselves .0034) | | .0037 / .0038 / .0036 | .0034 / .0042 / .0037 | |
| its gap over spread (0.87 expected of a run drawn like the reference) | | 0.98 | 1.01 | |
| mean loss (the pair's 0.5776) | | 0.5763 | 0.5760 | |
| the live tensors at a forward's end (peak) / the driver's pool (peak) | | 9.09 GiB / 26.59 GB | 9.14 GiB / 25.15 GB | |

### Qwen3.5-9B, 1 x 16, the cache emptied, 1 epoch (25 steps), 18:26-18:43 PST

| measure | bar / prediction | D5 | D8 + D9 (the 50-step run's same steps) | verdict |
|---|---|---|---|---|
| the guard's peak footprint | <= 74.6 GB | 49.33 GB | 50.77 GB (through step 25) | MET |
| steps 11-25 | predicted 40.7-41.4 s | 40.93 s (-5.1%) | 43.13 s | prediction met |
| steps 1-10 | | 41.5 s (-3.0%) | 42.8 s | |
| the vision cache's served micro-batches through step 25 | | 18 | 18 | |
| steps 1-2 (the same weights) | | 2.2223093509674072 / 2.0706000328063965 | the same bits | |
| the loss over steps 1-25 against Phase 4's 9B reference (the same schedule), root mean square of the difference | | .0015 | (D8 alone .0012, D4 .0013) | |
| step 25 | | 0.5425 | (the reference 0.5452) | |
| the live tensors at a forward's end / the driver's pool (peaks through step 25) | | 15.33 GiB / 44.82 GB | 15.31 GiB / 46.13 GB | |

As chains:
- The step, 0.8B: the adapter path 154.8 -> 83.3 ms a micro-batch (the profile) -> ~-0.57 s a step of 8 micro-batches -> 6.52 ->
  6.00 s measured (-0.52 s); the later epochs 6.29 -> 5.76 s.
- The step, 9B: the adapter path 256.5 -> 128.7 ms a micro-batch (the step profile) -> ~-2.0 s a step of 16 micro-batches ->
  43.13 -> 40.93 s measured (-2.2 s).
- Memory: the same tensors are kept for the backward (live tensors at a forward's end 9.09 against 9.14 GiB at 0.8B, 15.33 against
  15.31 at 9B) -> the driver's pool grew 1.4 GB at 0.8B and shrank 1.3 GB at 9B -> the footprint +2.07 GB / -1.44 GB; not traced
  (the pool's response to the changed set of buffer sizes, as D4's first chain).
- The loss: the forward rounds the adapter term once where PEFT rounds it three times, the masks are PEFT's, the weight gradients
  of a layer the same products -> the curve stays within the references' spread by Phase 4's readings; steps 1-2 the same bits
  (B is zero, so the adapter path adds an exact zero).

## D5's verdicts in short

| registered (series journal) or the design's own (written before any code) | measured | verdict |
|---|---|---|
| (a) the big projections >= 85% of the large-product rate at the bench's token counts, >= 70% at the micro-batch's (step 0's records) | bench 0.77-1.12 (0.8B, o_proj's forward under) / 0.99-1.12; micro-batch 0.82-1.01 / 0.87-1.11 | met but for 0.8B o_proj's forward at the bench shape |
| (a) MLX within 10% of PyTorch there | PyTorch at 0.83-1.06 of MLX's rate; every product outside 10% a narrow one where PyTorch is slower | missed (narrow projections); no shape at 5% of the step under 80% |
| (c) a custom product at >= 5% of the step and under 80% of MLX | the head's forward at the labelled rows 0.73 / 0.74 of MLX, at 2.2-2.5% of a micro-batch (both products 3.4-4.3%, 0.94 / 0.91 of MLX) | not triggered: nothing built |
| (b) parity as D4, both paths against float64 | 172 cases: the worst ratio 1.39 (bf16, one row), 1.27 (fp32), 1.07 (fp16), 1.00 (bf16 without autocast); whole decoder layers 1.29 | MET |
| (b) the same generator state gives the same masks; outputs to rounding | the same masks in every check (one layer, a stack, whole decoder layers, 150 / 200 projections of real training steps); the dropped values PEFT's bits; dA and dB PEFT's bits | MET |
| (b) speed: >= 1.2x on the adapter path at both sizes | isolated 2.13x / 2.20x (part alone 1.95x / 2.21x); in situ 1.86x / 1.99x | MET: ADMITTED, the default |
| registered prediction: the fused adapter path >= 1.5x its PyTorch path | 1.86-2.21x (every reading at both sizes; run 1 at 9B 1.88x) | met |
| registered prediction: the rank-32 products <= 30% of the large-product rate in PyTorch | medians 26% / 30% (the record), 26% / 35% (run 1); step 0's reading 36% / 45% | met by the record (9B at the edge); missed by run 1 and step 0's reading at 9B |
| design prediction: the adapter path in situ 0.8B 60-85 ms, 9B 105-150 ms a micro-batch | 83.3 ms / 128.7 ms | met / met |
| design prediction: isolated >= 1.8x | 1.95-2.20x (run 2), 1.88-2.10x (run 1) | met |
| design prediction: the steady step at 0.8B 5.8-6.0 s; the later epochs ~5.6 s | 6.000 s; 5.76 s | met (at its edge); missed (slower) by 0.16 s |
| design prediction: 9B steps 11-25 40.7-41.4 s | 40.93 s | met |
| design prediction: memory unchanged or lower | live tensors 9.09 against 9.14 GiB / 15.33 against 15.31; footprint +2.07 GB at 0.8B, -1.44 GB at 9B | missed at 0.8B (the allocator's pool) |
| integration 0.8B: footprint <= 53.5 GB; the loss as Phase 4 read it | 31.27 GB; the largest gap .0103 against .0159, the final loss 0.60% | MET; like the reference |
| integration 9B: footprint <= 74.6 GB | 49.33 GB | MET |

In plain words: PEFT's LoRA layers now run their adapter path as two Metal kernels around PyTorch's small products, with the
masks PEFT itself would draw: the passes around the rank-32 products (the float32 copies, dropout in float32, the casts, the
scaled copy and the add) are gone, the products' results round once, and the weight gradients are PEFT's own bits. The adapter
path is about twice as fast (2.1x at 0.8B and 1.9x at 9B on its own; 154.8 -> 83.3 ms of a 0.8B micro-batch in training), a 0.8B
LoRA step of the readers' recipe falls from 6.52 s to 6.00 s (-8.0%) and a 9B step from 43.1 s to 40.9 s (-5.1%), with the loss
of the reference. The head's product at the labelled rows runs under MLX's rate in its forward, but at 2.5% of a micro-batch it
is under the share that would justify a custom product.

## D5: not done, and what is uncertain

- The reductions (x_d A^T, dY B, dY^T P, G^T x_d) stay PyTorch's products at 7-74% of the large-square rate; reading dY once for G
  and dB, or applying the mask as x is loaded for P and dA (the prior art's full design), would remove a pass over dY and the
  dropped input's write and read: not built.
- The bf16 casts of A and B are made at every call (PEFT's autocast makes them too); kept per optimizer step they would save two
  launches a projection on 7 of every 8 micro-batches: not built.
- The adapter's input gradient is still added to the base's by autograd (one pass over x a projection), as in PEFT; folding it into
  the down kernel needs the base product's backward inside the fused function, which moves the base's backward out of its own
  marks in the profile: not built.
- The whole-model float64 reading at 0.8B scatters around 1 (0.36-2.38, one gradient of 301 over 2); the bar is met at the
  harness's shapes and in whole decoder layers, where it is registered.
- At 9B the isolated reading by subtraction depends on the base product's spread for the widest projections (run 1's 4,096 ->
  8,192 read 0.77x; run 2 2.07x by subtraction and 2.23x as the part alone); the in-situ readings are one profile of three (0.8B)
  or two (9B) steps each way.
- The footprint at 0.8B rose 2.07 GB with the same live tensors; the allocator pool's response is not traced.
- A new input shape costs more with the fused path in the profile (A - B 10.1 -> 20.0 ms a 0.8B micro-batch): not traced.
- The first probe run stopped before the model's load and answered no SIGINT (entry 19); the same command in the foreground with
  stdin from /dev/null ran; the cause is not traced (no stack: the runner now leaves one after 5 minutes).
- Qwen3.5 is where it was measured; the replacement applies to every model PEFT adapts with vanilla LoRA (bias-free, rank a
  multiple of 8 up to 64), untimed elsewhere.

## The log

### 1. 2026-11-03 11:4x-11:5x PST: the instrument and its check (commit 8fb6609)
`bench/timeline.py` and `bench/timeline_check.py`. The first check (marks without the fence) gave negative segments for every
short operation and attributed 0.47-0.55 of their own time to the small layers of the stack. An exploration of the event's
behaviour (single marks, back-to-back marks, a kernel before each mark) located the cause: an event in a command buffer
without GPU work is reported late, by up to ~0.1 ms while the GPU runs the next buffer. The fence (a one-element add before
each event) fixes it; the check keeps both arms.

### 2. 11:58-12:04: the profiler, a smoke run and a first 0.8B run (commit ce28115)
`bench/profile_step.py` on Phase 4's runner (`phase4_train.main(argv, customize)`, `RUNNER=` in `phase4_run.sh`; both unchanged
by default). The smoke run (the first 24 items, step 2 profiled) gave C / B 0.99-1.11 and the events 0.9996 of the wall clock.
The first 0.8B run (steps 3, 5, 7) is kept as run 1 (`profile_0.8b_2x8_run1.json`): its event-pool warm-up fell inside step 3's
step-level time (187 ms), and it predates the GPU's lag reading; its shares agree with the record of reference within ~1 point.

### 3. 12:08-12:15: the 9B run (commit 220fa2d)
Qwen3.5-9B, 1 x 16, steps 2 and 4 profiled, stopped after step 5. Peak footprint 53.8 GB.

### 4. 12:17-12:27: the remaining runs in one chain, one job at a time
The 0.8B profile with the final code (`profile_0.8b_2x8.json`), the instrument check again (`timeline_check.json`), the
PyTorch products, the isolated parts (run 1, `layer_parts_run1.json`), the MLX products. All under the guard, all rc 0.

### 5. 12:27-12:31: the square forms; the isolated parts again
The large-square rate per form (`products_square_forms.json`): it showed the chip throttled after the chain. The isolated
parts run again with the byte formulas corrected (the glue's and the attention's views; the residual add fed the same memory
twice in run 1) and the floors at the best measured rate (12.9 TFLOP/s): `layer_parts.json`.

### 6. 2026-11-03 12:4x-13:00 PST: D4's code and a first check against float64 (commits ed4b988, 8fee8cf)
The prior art copied into the repo first (`docs/prior_art_layer.md`). Then `macai.layer`: five kernels each way through
`torch.mps.compile_shader` (a probe first: `vec<bfloat, 4>` loads, the storage offset applied by the dispatch, Metal's float ->
bfloat conversion equal to PyTorch's), their autograd functions, the four replaced methods with the source check. A smoke script
against float64 put every part at the expected error, with one fault: the weight gradients' second pass summed the partial rows
one after another, and a float32 RMSNorm weight gradient read 3x the PyTorch path's error (2.4e-7 against 7.9e-8 over 144 rows)
-> each second pass now gives every column a simdgroup (each lane every 32nd partial, then a tree): the same case at 1.05x.

### 7. 13:01-13:17: the tests, the bench, a trainer smoke run (commits 8fee8cf, a0d3d57)
Two faults in the first test run, both in the test: the float64 reference ran first on the very tensor the GPU runs then copied
(`Tensor.to` returns the tensor itself when nothing changes, so `requires_grad_` marked the shared CPU leaf and the GPU runs'
gradients flowed back into float64 on the GPU: "Undefined type Double"); and the dropout-mask comparison counted inputs that are
exactly zero (28 against 23 of 75,776, where the norms' outputs differ in the last bit) as dropped. Both fixed in the test; the
masks agree wherever the inputs are nonzero. The end-to-end check passed at both sizes on its first clean run. The parts bench:
a one-round dry run, then the record (5 interleaved rounds, 13:10-13:13). A trainer smoke run with every part on
(`smoke_d4_0.8b`: 16 items, 8 steps) trained, every replaced method in force.

### 8. 13:17-13:57: the first integration chain: the convolution path and beta / g fell back (records `d4_norms_swiglu_*`)
0.8B 13:17-13:35 (10.433 s a step), 9B 13:35-13:55 (48.27 s), the profile 13:55-13:57. The profile showed the convolution path
at step 0's time; the cause (a fresh cache in every training forward) is set out under "The first chain" above. The 0.8B run's
steps 19-26 ran slower than D2m's; first read here as the machine's load, then found again in the second chain's run on the same
steps (entry 10), so tied to the batches.

### 9. 13:57-14:02: the cause and the fix (commit 3564a41)
A probe run (the runner with a wrapper, 4 items, 2 steps): 72 of 72 linear-attention calls carried a fresh cache, none ran the
fused path. The fix, its test (the fresh cache against transformers' forward), a second probe (72 of 72 fused), the suite again
(490 passed), the first chain's records renamed (their content untouched: inside, their names are the originals).

### 10. 14:02-14:39: the second integration chain (commit 3564a41, the tree clean)
0.8B 14:02-14:18 (9.178 s a step, footprint 41.58 GB), 9B 14:18-14:37 (46.20 s, 60.44 GB), the profile 14:37-14:39. All under the
guard, all rc 0, one job at a time.

### 11. 14:4x: the report, the defaults
`bench/layer_report.py` -> `d4_results.json` (the first chain's reading beside). Every part admitted becomes the default
(`macai.layer`: all five on). `bench/bench_gated_delta.py` and `bench/profile_layer.py` now install the rule alone, so they keep
measuring what they measured (the rule against transformers' reference; transformers' parts in step 0's bench);
`bench/phase4_train.py` keeps every part off unless `--layer` says otherwise (the runs before D4 reproduce).

### 12. 2026-11-03 14:5x-15:29 PST: D8 and D9's code, the tests, the probes (commits 89b06ac, 69ca269)
A probe of the head's product first (the bf16 product on the gathered rows against every position, at both models' shapes: 0 of
73 M / 36 M logits differ; the gradient to the input: 0.4% of its elements differ by one bf16 step). Then `macai.training` (the
two replacements, the cache, the source checks), the runner's options and counts, the tests. Two faults in the first test runs,
both in the tests: a call with labels and `logits_to_keep=1`, which the model's own path rejects as well (its loss needs a label
row for every kept position), and a non-contiguous `shift_labels`, which the model's own loss cannot view; the condition is now
tested on the condition itself, and `shift_labels` passed contiguous. The probe in a real training run: two attempts failed on
faults of the probe script (a conversion to float64 on the GPU; a module moved to float64 before leaving the GPU), their records
kept (`probe_d8d9_0.8b_failed1`, `_failed2`); a first complete probe with the LoRA dropout on during the head check
(`probe_d8d9_0.8b_run1`: the two paths' LoRA gradients within 5.5% of each other, no float64 model); the probes of record with
the dropout an identity during the check and, at 0.8B, the whole model in float64 (21 s on the CPU). The probes ran the code of
89b06ac before it was committed (their records say dirty). The cache's cap (`VISION_CACHE_MAX_GB`, 16 GiB) was added before the
probes of record, as a guard for the default.

### 13. 15:29-16:21: the first run chain (commit 69ca269, the tracked tree clean)
The loss-head bench 15:30-15:31; D8 alone at 0.8B (25 steps) 15:31-15:34; the integration at 0.8B 15:34-15:45 (6.522 s a step);
at 9B 15:45-16:19 (43.13 / 37.48 s by epoch); the profile with D4 + D8 16:19-16:21. All under the guard, all rc 0. The first D8
run's steps 16-25 ran 5-12% slower than the integration's same steps, with D8 on in both and the cache serving nothing there:
not a cost of the cache's absence, as the repeat showed (entry 14).

### 14. 16:21-16:52: the second chain: D8 alone at 0.8B over 100 steps, at 9B over 25
The repeat at 0.8B matched the integration's first epoch step for step (steps 16-25: 7.80 / 7.80 s a step) and not the first D8
run (8.50 s); it is D9's direct baseline for the later epochs. The 9B run gives D8's own 9B reading on Phase 4's 9B schedule.
Both under the guard, rc 0.

### 15. 16:52-17:0x: the defaults, the test run of record, the report
D8 and D9 admitted and made the defaults (`macai.training`: `HEAD_AT_LABELS`, `VISION_CACHE` on); `bench/phase4_train.py`
keeps both off unless its options say otherwise. The test run of record (`d8d9_tests.log`: 505 passed, the 490 before and 15
new; D8's cases in `d8d9_parity.jsonl` beside D4's). `bench/d8d9_report.py` -> `d8d9_results.json`; the report sets the Hub's
offline mode itself (a reading of the step profile run by hand without it had let huggingface_hub look up a model's
configuration online).

### 16. 17:0x-17:1x: a review
The test suite run again on the same commit (505 passed, 137 s, peak footprint 27.0 GB). Every figure above recomputed from the
raw records by separate code: the head's parity rows and the probes' checks, steps 1-2, each window's step time, D9's gain by
both readings (against D8 alone over the later epochs: 0.840; the registered reading against D4: 0.881), the loss readings,
the footprints, the cache's counts, sizes and lookup, the profile's head and loss rows; all agree to rounding. One change:
`bench/profile_products.py`'s `text_config` reads the Hub cache only (`local_files_only=True`), so no reading of the step
profile can look a model up online, whatever the environment (without the cache it fails locally); `d8d9_results.json`
regenerated from the same records is unchanged.

### 17. 2026-11-03 17:2x-17:4x PST: the design and its predictions; two probes of the dropout; (a) (commits 05bb222, a9f0c98)
The design and its predicted times written into this log before any code (05bb222). Two probes of torch's dropout on the Apple
GPU first: `native_dropout` has a kernel there (nn.Dropout's call: `NativeDropoutBackward0`); from the same generator state the
mask is the same whatever the input's dtype, values or layout (float32, bf16, fp16, ones, an expanded view, even `bernoulli_` on
a bool tensor: 75,776 of 75,776 the same), so it depends on the shape alone; on bf16 / fp16 x the dropped values are PEFT's
dropped float32 copy after autocast's cast bit for bit (0 of 5.4 M differ, three shapes); a kept element is x times float32(1 /
(1 - p)); the products on the Apple GPU are deterministic, give the same bits on a transposed view, and commute with a scale of 2
exactly; PEFT's float32 copy and dropout cost 0.08-0.51 ms at the in-situ shapes against 0.02-0.08 ms for the call on bf16 x.
(a): the head's products at the labelled rows, PyTorch then MLX (17:38-17:41, a9f0c98).

### 18. 17:44-17:56: the code, a smoke run, the tests (commit 75e99a2)
`macai.lora` (the kernels, the autograd function, the replacement of PEFT's method with its source hash, the settings), the patch,
the tests, the bench. The smoke run: the kernels against float64 at every dtype pair and rank; one PEFT layer off and on from the
same seed: the masks the same, dA and dB the same bits, the output within rounding; a first timing (wall clock) 1.15x on the whole
0.8B in_proj_qkv layer, its base product included. The first full test run: one case over the bar, the float32 output at 14 rows
(2.74x PEFT's error, 3.5e-7 against 1.3e-7). Its cause, as a chain: P is the same bits on both paths (probed: F.linear and
torch.mm agree) -> the expanding product alone, from an exact P, read 2.1e-7 against MPS's 1.2e-7 at 14 rows -> a scalar fma chain
in rank order gave the identical error to the simdgroup products (they accumulate in rank order) -> float32 outputs now take four
fma chains over the rank summed as a tree (that case 1.27x; 16-bit outputs keep the simdgroup products, whose last float32 bits
their rounding removes). One fault in a test (adapter_names passed to the wrapper module, not the LoRA layer). 101 passed.

### 19. 17:59-18:14: the probes; a first probe run stopped (commits ea87479, 7e0ef7a)
The first 0.8B probe run (launched in the background, 17:59) stopped after the trainer's line for the sample, before the model's
load: its own resource use showed no CPU time over 5 s, 0.56 GB, no GPU memory; two SIGINTs to its process id (18:04, 18:08) were
not answered (blocked outside the interpreter). While it sat idle, three diagnostics with every thread's stack dumped on a timer
and a self-exit (the runner's early steps with the fused path installed and the processor's load: 2.2 s; Metal from a new
process: 0.07 s; the model's load as the trainer makes it: 0.45 s) all ran; the same probe in the foreground with stdin from
/dev/null ran (probe_d5_0.8b_run2), as did every run after (the chains take stdin from /dev/null). The cause is not traced; the
runner now leaves every thread's stack in its log when its Trainer is not built within 5 minutes. The stopped process ends at the
guard's wall stop (20:59 PST); its log up to 19:12 is kept (`probe_d5_0.8b_stopped.log`); it wrote no record. The 9B probe (18:13-18:14).

### 20. 18:15-18:45: the chain of record (commit 7e0ef7a, the tracked tree clean)
The bench (run 1) 18:15-18:16, 0.8B 100 steps 18:16-18:26, 9B 25 steps 18:26-18:43, the 0.8B profile 18:43-18:45. All under the
guard, all rc 0, one job at a time (the stopped probe process idle beside them). The records committed (b91c4fe); the
bench's record renamed run 1 when the part alone was added.

### 21. 18:4x-19:04: admission and the defaults; chain 2 (commit f2a7b55, the tracked tree clean)
D5 admitted and made macai's default (f2a7b55: `macai.lora.FUSED = True`; the runner keeps it off unless `--lora-fused`; the
earlier benches pinned to PEFT's own layers; the call-order test of D4 pinned to PEFT's forward). Chain 2: the bench with the
part alone 18:46-18:47 (the record), the test run of record 18:47-18:50 (606 passed, 177 s, peak footprint 26.8 GB), the 9B
profiles without and with D5 18:51-19:04. All under the guard, rc 0, one job at a time.

### 22. 19:0x: the report
`bench/d5_report.py` -> `d5_results.json` (offline by itself): every figure of this entry from the records.

### Next
D5 is done and on by default: the adapters' extra work is 10.9% / 4.8% of a micro-batch (0.8B / 9B) after it, the frozen products
36.9% / 69.0% at the platform's rate. Then D3 (the rule's chunking: 9.9% / 3.7%). D8b (a fused cross-entropy, ~2% of a 0.8B
step) and the vision cache's trim stay proposed for their own entries; so do the adapter path's next steps (dY read once for G and
dB; the mask applied as x is loaded; the weights' bf16 casts kept per optimizer step) and the head's forward in another form at
the labelled row counts.
