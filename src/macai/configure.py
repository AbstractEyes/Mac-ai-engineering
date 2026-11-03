"""Settings measured on an M3 Max (128 GB) for long PyTorch training on the Apple GPU. Call configure() before importing torch.

- The Metal allocator keeps the buffers of every new tensor shape; a long training run with varying batch shapes grows past the
  physical memory. The watermarks (ratios of the GPU's recommended working set, 115.4 GB on this machine) make it release cached
  buffers past the low mark and raise an error past the high mark instead of swapping.
- CPU thread pools: OpenMP and Apple Accelerate read their thread counts at their first call.
"""
import os

DEFAULTS = {
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.75",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.55",
}


def configure(cpu_threads: int | None = None, **overrides) -> dict:
    """Set the environment for a training process (keeps any value already set); returns what was set."""
    if "torch" in __import__("sys").modules:
        raise RuntimeError("configure() must run before torch is imported (the settings are read once)")
    wanted = dict(DEFAULTS, **{k: str(v) for k, v in overrides.items()})
    if cpu_threads:
        wanted.update(OMP_NUM_THREADS=str(cpu_threads), VECLIB_MAXIMUM_THREADS=str(cpu_threads))
    done = {}
    for k, v in wanted.items():
        if k not in os.environ:
            os.environ[k] = v
            done[k] = v
    return done
