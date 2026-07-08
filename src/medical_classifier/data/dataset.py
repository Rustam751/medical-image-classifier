"""ImageFolder-style dataset that plugs into the preprocessing pipeline.

The dataset reads a directory laid out as ``root/<CLASS_KEY>/*.png`` and applies
the exact same ``preprocess_pil`` transform used at inference time. This is what
lets a user swap the synthetic data for their own real dataset without touching
any code: arrange images as ``<split>/<CLASS_KEY>/*.png`` and retrain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from medical_classifier.constants import CLASS_KEYS, IMAGE_SIZE
from medical_classifier.preprocessing.transforms import (
    NormalizationStats,
    _robust_percentile_scale,
    preprocess_pil,
)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_samples(root: Path) -> list[tuple[Path, int]]:
    """Collect (path, label) pairs from an ImageFolder-style directory."""
    samples: list[tuple[Path, int]] = []
    for label, class_key in enumerate(CLASS_KEYS):
        class_dir = root / class_key
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in _IMAGE_EXTENSIONS:
                samples.append((path, label))
    return samples


class ImageFolderDataset(Dataset):
    """Torch dataset over an ImageFolder-style directory of class subfolders."""

    def __init__(
        self,
        root: str | Path,
        stats: NormalizationStats,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        self.root = Path(root)
        self.stats = stats
        self.image_size = image_size
        self.samples = _list_samples(self.root)
        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.root!s}. Expected class subfolders "
                f"{list(CLASS_KEYS)}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as img:
            tensor = preprocess_pil(img, stats=self.stats, image_size=self.image_size)
        return tensor, label

    def class_counts(self) -> dict[str, int]:
        """Return the number of samples per class key (for imbalance reporting)."""
        counts = {key: 0 for key in CLASS_KEYS}
        for _, label in self.samples:
            counts[CLASS_KEYS[label]] += 1
        return counts


def compute_norm_stats(root: str | Path, image_size: int = IMAGE_SIZE) -> NormalizationStats:
    """Compute grayscale mean/std over a dataset split after percentile scaling.

    These statistics standardize model inputs and are saved into the checkpoint
    so that inference reproduces the exact training-time normalization.
    """
    root = Path(root)
    samples = _list_samples(root)
    if not samples:
        raise FileNotFoundError(f"No images found under {root!s} to compute stats.")

    # Running mean/variance via sums keeps memory flat regardless of dataset size.
    total = 0.0
    total_sq = 0.0
    count = 0
    for path, _ in samples:
        with Image.open(path) as img:
            gray = img.convert("L").resize((image_size, image_size), Image.BILINEAR)
        arr = _robust_percentile_scale(np.asarray(gray, dtype=np.float32))
        total += float(arr.sum())
        total_sq += float((arr**2).sum())
        count += arr.size

    mean = total / count
    var = max(total_sq / count - mean**2, 1e-8)
    return NormalizationStats(mean=mean, std=float(np.sqrt(var)))
