"""Synthetic PET-like phantom generator.

WHY THIS EXISTS
---------------
The original Alzheimer's project trained on the ADNI amyloid-PET dataset, which
is access-restricted and cannot be redistributed. To keep this repository fully
reproducible and runnable by anyone (recruiters included) *without* faking
results, we generate a synthetic dataset of brain-phantom-like grayscale images
with three classes that carry a genuine, learnable — but deliberately noisy —
signal.

IMPORTANT HONESTY NOTE
----------------------
These images are procedurally generated shapes. They are NOT real brain scans
and do NOT reproduce the true physiology of amyloid PET. The class-conditional
"uptake" patterns here are an arbitrary synthetic signal designed so the model
learns something real and the reported metrics are reproducible. Metrics on this
dataset are a demonstration of the *engineering pipeline*, not a clinical result.
To obtain a clinically meaningful model, point the training pipeline at a real,
validated dataset arranged in the same ImageFolder layout (see README).

The generator is fully deterministic given a seed, so ``generate`` produces the
exact same dataset every time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from medical_classifier.constants import CLASS_KEYS, IMAGE_SIZE

# Class-conditional modulation of "uptake" in the posterior/temporoparietal-like
# region. CN keeps near-full intensity there; AD shows a marked reduction. The
# discriminative feature is the ANTERIOR-vs-POSTERIOR contrast, which is
# invariant to overall image brightness — so a global intensity change cannot be
# confused with a class change. Per-image jitter (below) keeps adjacent classes
# overlapping enough that the task is non-trivial (the model does not reach
# 100%), but the gaps are large enough to be genuinely learnable.
_POSTERIOR_FACTOR: dict[str, float] = {
    "CN": 1.00,
    "MCI": 0.65,
    "AD": 0.35,
}

# Standard deviation of the per-image posterior factor. Roughly one third of the
# inter-class gap (~0.30), giving meaningful but not overwhelming class overlap.
_POSTERIOR_JITTER: float = 0.10


def _coordinate_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized (x, y) grids in [-1, 1]."""
    lin = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    return np.meshgrid(lin, lin)


def _brain_phantom(
    rng: np.random.Generator, class_key: str, size: int
) -> np.ndarray:
    """Generate one synthetic phantom image for a class as a float array [0,1]."""
    xx, yy = _coordinate_grid(size)

    # Random rigid jitter so no two images are identical: small translation,
    # anisotropic scale, and rotation of the underlying brain ellipse.
    dx, dy = rng.normal(0, 0.06, size=2)
    sx, sy = 1.0 + rng.normal(0, 0.05, size=2)
    theta = rng.normal(0, 0.15)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x = (xx - dx) * sx
    y = (yy - dy) * sy
    xr = x * cos_t - y * sin_t
    yr = x * sin_t + y * cos_t

    # Elliptical brain mask with a soft edge.
    r2 = (xr / 0.8) ** 2 + (yr / 0.95) ** 2
    brain = np.clip(1.0 - r2, 0.0, 1.0)
    brain = brain ** 0.5  # broaden the plateau

    # Gyral texture: overlapping sinusoids give a cortical, folded appearance.
    # This texture is class-independent, so it acts as a realistic distractor
    # that the model must learn to ignore.
    freq = 6.0 + rng.normal(0, 0.5)
    texture = 0.5 + 0.5 * np.sin(freq * xr + rng.normal()) * np.cos(
        freq * yr + rng.normal()
    )
    base = brain * (0.7 + 0.3 * texture)

    # Class-conditional posterior "uptake" reduction. The posterior region is a
    # soft Gaussian blob in the lower portion of the image; the anterior region
    # stays bright and acts as an implicit reference. The discriminative signal
    # is therefore the anterior-posterior *contrast*.
    posterior = np.exp(-((xr**2) / 0.6 + ((yr - 0.40) ** 2) / 0.25))
    factor = _POSTERIOR_FACTOR[class_key] + rng.normal(0, _POSTERIOR_JITTER)
    factor = float(np.clip(factor, 0.1, 1.15))
    image = base * (1.0 - (1.0 - factor) * posterior)

    # Small global intensity jitter (does NOT confound the contrast signal) plus
    # additive Gaussian noise for texture realism.
    image = image * (0.95 + rng.normal(0, 0.04))
    image = image + rng.normal(0, 0.035, size=image.shape)
    image = np.clip(image, 0.0, 1.0)

    return image.astype(np.float32)


def _save_png(array: np.ndarray, path: Path) -> None:
    """Save a float [0,1] array as an 8-bit grayscale PNG."""
    img = Image.fromarray((array * 255.0).round().astype(np.uint8), mode="L")
    img.save(path)


def generate_synthetic_dataset(
    output_dir: str | Path,
    per_split_counts: dict[str, int],
    image_size: int = IMAGE_SIZE,
    seed: int = 42,
) -> dict[str, int]:
    """Generate a synthetic ImageFolder dataset on disk.

    Layout produced::

        output_dir/
            train/{CN,MCI,AD}/*.png
            val/{CN,MCI,AD}/*.png
            test/{CN,MCI,AD}/*.png

    Args:
        output_dir: root directory to write into.
        per_split_counts: e.g. ``{"train": 180, "val": 45, "test": 45}`` — images
            *per class* in each split.
        image_size: output edge length in pixels.
        seed: master seed for full reproducibility.

    Returns:
        A mapping of split name -> total images written.
    """
    output_dir = Path(output_dir)
    rng = np.random.default_rng(seed)
    totals: dict[str, int] = {}

    for split, count in per_split_counts.items():
        split_total = 0
        for class_key in CLASS_KEYS:
            class_dir = output_dir / split / class_key
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                arr = _brain_phantom(rng, class_key, image_size)
                _save_png(arr, class_dir / f"{class_key.lower()}_{i:04d}.png")
                split_total += 1
        totals[split] = split_total

    return totals
