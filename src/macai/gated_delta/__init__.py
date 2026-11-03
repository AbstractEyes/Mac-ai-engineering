"""The gated delta rule (Qwen3.5 / Qwen3-Next linear attention) as Metal kernels for PyTorch on Apple GPUs."""
from .op import GatedDeltaRule, chunk_gated_delta_rule, l2norm, recurrent_reference

__all__ = ["GatedDeltaRule", "chunk_gated_delta_rule", "l2norm", "recurrent_reference"]
