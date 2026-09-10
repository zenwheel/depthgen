"""Torch device selection (MPS -> CUDA -> CPU)."""
from __future__ import annotations

import functools


@functools.lru_cache(maxsize=None)
def pick_device(prefer: str = "auto") -> str:
    import torch

    if prefer != "auto":
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
