# Series: the rest of the layer (D3-D5), a dated log

After the gated delta rule's kernels (Phase 3, [series_phase3.md](series_phase3.md)) and their memory design (D2m,
[series_memory.md](series_memory.md)), a Qwen3.5 LoRA training step on the M3 Max takes 10.98 s at 0.8B (2 x 8 sequences) and
50.91 s at 9B (1 x 16). Where the rest of that time goes was a count of FLOPs, not a measurement. This phase of the series
measures it first (step 0, an instrument), then builds the designs in the order the measurement gives: D4 the layer's
surroundings fused, D5 the matrix products, D3 the rule's chunking. One machine (an M3 Max, 40-core GPU, 128 GB, macOS 26,
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

### Next
D4 at his word, in the order above; the two proposed entries outside D3-D5 wait for his word too.
