"""Deterministic seeding and device selection."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch so a run is byte-for-byte reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # torch not needed for twin-only runs
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(preference: str = "auto"):
    """Pick a torch device. ``auto`` prefers CUDA, then Apple MPS, then CPU.

    MPS is deliberately *not* auto-selected: Opacus' per-sample gradient hooks
    and several PyG scatter kernels silently fall back or error on MPS, so the
    safe default on Apple silicon is CPU. Request ``mps`` explicitly to opt in.
    """
    import torch

    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
