"""Reproducibility helpers.

Deterministic seeding is a core requirement of this project: the reported
metrics must be reproducible from a clean checkout. ``set_seed`` fixes every
RNG that the pipeline touches (Python, NumPy, PyTorch).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all relevant random number generators for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Favor determinism over the last few percent of throughput.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
