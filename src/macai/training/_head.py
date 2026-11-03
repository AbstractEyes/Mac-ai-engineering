"""The LM head and the loss at the labelled positions only (series design D8).

A causal LM's loss at position t compares the logits there with the label of position t + 1; positions whose next label is
ignored (the prompt, the picture, the padding: -100) contribute nothing to the loss or to any gradient. Their logits are still
computed by the model's forward, a product over the whole vocabulary at every position, then upcast to float32 by the loss.
Here the hidden states are gathered at the positions whose shifted label is not ignored before the head (transformers' boolean
logits_to_keep, which flattens them to [N, hidden]) and the loss is the model's own loss function on those rows with the
matching shifted labels (its shift_labels argument), so its reduction (the mean over the labelled positions, or the sum over
num_items_in_batch when the caller passes it) is the one it computes on the whole tensor.
"""
import torch.nn.functional as F

# taken: forwards that ran the head at the labelled positions; fallback: training forwards with labels that ran the model's
# own path (the setting on, but a condition unmet); positions / labelled: the positions seen and the rows the head computed
STATS = {"taken": 0, "fallback": 0, "positions": 0, "labelled": 0}


def reset() -> None:
    for k in STATS:
        STATS[k] = 0


def labelled_positions(labels, ignore_index: int = -100):
    """(keep, shift): shift[b, t] is the label position t predicts (labels[b, t + 1], ignore_index past the end), shifted as
    the causal LM loss shifts it; keep marks the positions whose shifted label is not ignore_index. No synchronization."""
    shift = F.pad(labels, (0, 1), value=ignore_index)[..., 1:]
    return shift != ignore_index, shift


def targets_at(shift, keep):
    """The shifted labels at the kept positions, in the order a boolean index flattens the hidden states (row-major)."""
    return shift[keep]
