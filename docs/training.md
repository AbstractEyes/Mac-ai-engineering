# The training path around the layers (macai.training)

## Why
A LoRA training step of a Qwen3.5 picture reader does two kinds of work whose result is thrown away or already known:

- **The LM head and the loss at every position.** The model's forward computes the logits at every position of every row (a
  product over the whole 248,320-word vocabulary) and its loss upcasts all of them to float32, while only the positions whose next
  token carries a label count: the record's collator ignores the prompt, the picture and the padding (label -100). In the
  readers' micro-batches about a fifth of the positions carry a label. The step profile ([series_layer.md](series_layer.md))
  measured the head and the loss at 25.6% of a 0.8B micro-batch after D4.
- **The frozen vision tower in every epoch.** The tower is frozen under the LoRA, and a picture's pixels are the same in every
  epoch, so the tower computes the same output for the same picture again and again: 12.7% of a 0.8B micro-batch.

Neither needs a kernel: the first is transformers' own boolean `logits_to_keep` and the loss's `shift_labels`; the second is
the model's own precomputed-encoder input (`mm_encoder_outputs`) fed from a cache.

## What it does

| setting (`macai.training`) | replaces | what runs |
|---|---|---|
| `HEAD_AT_LABELS` | `Qwen3_5ForConditionalGeneration.forward`, `Qwen3_5ForCausalLM.forward` | while the model trains with labels: the original forward with `labels=None` and a boolean `logits_to_keep` marking the positions whose shifted label is not ignored (the head runs on those rows only, flattened to [N, hidden]), then the model's own loss function on those logits with `shift_labels` and the keyword arguments the original passes it (`num_items_in_batch` among them, when the caller passes it), so the loss is reduced as the model reduces it. The output carries the loss and the logits of the labelled positions ([N, vocabulary]). Every other position's gradient is zero, as before |
| `VISION_CACHE` | `Qwen3_5Model.forward` | while the model trains with every tower parameter frozen: each picture's output of `get_image_features` (the tower's last hidden states and the merged features, in full) kept in CPU memory on the picture's first pass; a micro-batch whose pictures are all stored is served from the cache, handed to the original forward as `mm_encoder_outputs` (the tower does not run); any other micro-batch runs the tower as the model does and its new pictures are stored |
| `VISION_CACHE_MAX_GB` | | the most one cache stores (16 GiB by default); past it, new pictures are computed every time |

The cache's key for a picture is exact: the SHA-256 digest of the bytes of its rows of `pixel_values` (in 1 MiB chunks hashed on
four threads, then the chunks' digests and the length), the rows' dtype and width, its grid entry (t, h, w), and what the
tower's output depends on besides: the device, the tower's dtype, the autocast state and dtype, the attention implementation.
One changed pixel, or the same pixels at another grid, is another key. A cache belongs to one model and one set of tower
weights: each call reads every tower tensor's storage address and version counter, and a change (a parameter loaded, changed in
place, replaced) empties the cache. A change made through a tensor's `.data` bypasses the version counter: after one, call
`macai.training.clear_vision_cache()`. Storing a picture's output copies it to CPU memory at the next call (where the call
synchronizes with the GPU anyway), so storing adds no synchronization.

## Install
    import macai.patch
    macai.patch.install()          # the rule, the layer parts and the training path (each method only while its source matches)
    from macai import training
    training.configure(VISION_CACHE=False)           # switch either off at any time, or training.configured(...) in a block
    training.stats()                                 # the head's calls, the cache's stores, hits and its own time

A replaced method reads its setting at every call and passes the call through unchanged when the setting is off, when the model
is not training, and (the head) without labels, with `logits_to_keep` or `shift_labels` given, with a tuple output asked, with
another loss function, or (the cache) without pictures, with videos or precomputed outputs given, with a tower parameter that
trains, with pixel values that need a gradient, or with an argument that changes the tower's output (attentions or hidden
states asked). The methods are replaced only while the source of what they replace and of what they rely on hashes as in
transformers 5.22.1 (`macai.training._qwen3_5.SOURCES`); otherwise the original stays and a warning says which. Only Qwen3.5
(dense) is touched.

## Parity
`tests/test_training.py`: the head at the labelled positions on models at the real widths and vocabulary of Qwen3.5-0.8B and 9B
(two decoder layers, LoRA rank 32) against the model's own path, both against a float64 CPU run: the loss, the input
embeddings', the final hidden states' and every LoRA weight's gradient; the logits at the labelled positions bit for bit; the
reduction over `num_items_in_batch`; the arguments the loss function receives; every condition under which the original runs.
The cache on a small model: the served output equal to the computed one bit for bit, one pixel changed or another grid not served,
not in evaluation or with a training tower, a weight change empties it, the tower draws no random numbers, the cap.
`bench/training_probe.py` repeats the checks inside a real training run of each model (the real tower, the sample's pictures,
and for 0.8B the whole model in float64 as the reference of every gradient).

## Measured
Both settings are on by default since series designs D8 and D9 admitted them ([series_layer.md](series_layer.md), D8 and D9). On
an M3 Max (PyTorch 2.15, transformers 5.22.1), LoRA training of the Qwen3.5 picture readers (rank 32, bf16 autocast, the record's
collator: prompt, picture and padding ignored), with the rest of macai on:

| | Qwen3.5-0.8B, 2 x 8 sequences | Qwen3.5-9B, 1 x 16 |
|---|---|---|
| positions whose next token carries a label | 17.9% | 21.5% |
| the head and the loss a micro-batch (in the step profile at 0.8B; the isolated bench at 9B) | 269.6 -> 53.6 ms | 298.0 -> 83.6 ms |
| the step, the head at the labelled positions (against the kernels before) | 9.18 -> 7.51 s (-18.2%) | 46.2 -> 42.9 s (-7.1%) |
| the step from the second epoch on, the cache too (against the head alone) | 7.48 -> 6.29 s (-15.9%) | 37.5 s (-12.6%, by the epochs' ratios to the runs before) |
| the cache: CPU memory for the sample's 372 pictures; a lookup, hashing included | 1.68 GB; 7.4 ms (two pictures) | 3.56 GB; 6.2 ms |
| the process's peak footprint (the cache included) | 41.6 -> 29.2 GB | 60.4 -> 52.0 GB |
| the loss | like the reference by Phase 4's readings (100 steps) | steps 1-2 within 1e-6; over 25 steps .0012 from the reference run |

The cache in numbers: the key of a picture is computed from its pixel rows copied back from the GPU (5.8 ms a 0.8B micro-batch of
two pictures, 4.2 ms at 9B); a served micro-batch costs 7.4 / 6.2 ms in all against the tower's ~134 / ~318 ms. In the first epoch
every call pays the key and the stores (+0.5% on the step). A picture takes 4.5 MB in full at 0.8B (1.1 MB of it the merged
features the model reads) and 9.6 MB at 9B (4.5 MB).
