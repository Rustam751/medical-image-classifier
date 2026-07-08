"""Deterministic preprocessing pipeline: raw image bytes/PIL -> model tensor.

This is the single most important module for train/serve consistency. The exact
same function is used to prepare training images and to prepare an uploaded
image at inference time. The steps are:

1. Convert to single-channel grayscale.
2. Resize to ``IMAGE_SIZE`` x ``IMAGE_SIZE`` (bilinear).
3. Robust percentile normalization to [0, 1] (clips extreme intensities, a
   technique carried over from the original ADNI PET preprocessing).
4. Standardize with dataset mean/std stored in the model checkpoint.

Keeping normalization statistics inside the checkpoint (rather than hard-coding
them) guarantees that whatever the model was trained on is exactly what it sees
at serve time.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from medical_classifier.constants import IMAGE_SIZE


@dataclass(frozen=True)
class NormalizationStats:
    """Per-dataset grayscale mean/std used to standardize model inputs."""

    mean: float
    std: float

    def to_dict(self) -> dict[str, float]:
        return {"mean": float(self.mean), "std": float(self.std)}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> NormalizationStats:
        return cls(mean=float(d["mean"]), std=float(d["std"]))


# A neutral default used before dataset statistics are computed. Grayscale
# images normalized to [0, 1] have a natural midpoint around 0.5.
DEFAULT_STATS = NormalizationStats(mean=0.5, std=0.5)


def _robust_percentile_scale(
    array: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5, eps: float = 1e-7
) -> np.ndarray:
    """Scale intensities to [0, 1] using percentile clipping.

    This suppresses the influence of a few very bright/dark outlier pixels,
    which is common and problematic in medical intensity images.
    """
    lo = np.percentile(array, low_pct)
    hi = np.percentile(array, high_pct)
    scaled = (array - lo) / (hi - lo + eps)
    return np.clip(scaled, 0.0, 1.0)


def preprocess_pil(
    image: Image.Image,
    stats: NormalizationStats = DEFAULT_STATS,
    image_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """Convert a PIL image into a model-ready tensor of shape ``(1, H, W)``.

    The returned tensor is *not* batched. Callers that feed the model should add
    a batch dimension with ``tensor.unsqueeze(0)``.
    """
    gray = image.convert("L").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32)
    arr = _robust_percentile_scale(arr)
    arr = (arr - stats.mean) / (stats.std + 1e-7)
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
    return tensor.float()


def preprocess_image(
    data: bytes,
    stats: NormalizationStats = DEFAULT_STATS,
    image_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """Decode raw image bytes and preprocess them into a model-ready tensor.

    Raises:
        ValueError: if the bytes cannot be decoded as an image.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # force decode now so corrupt data fails here, not later
    except Exception as exc:  # noqa: BLE001 - re-raised as a clean domain error
        raise ValueError(f"Could not decode image data: {exc}") from exc
    return preprocess_pil(image, stats=stats, image_size=image_size)
