"""Project-wide constants: class taxonomy, disclaimer, and file conventions.

These values are the single source of truth for the label space. The training
pipeline, the inference layer, and the API all import from here so that the
class ordering can never drift between training and serving.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Class taxonomy
# ---------------------------------------------------------------------------
# The ordering of this tuple defines the integer label for each class:
# index 0 -> "CN", index 1 -> "MCI", index 2 -> "AD". The neural network's
# output logits follow the same order. Do not reorder without retraining.
CLASS_KEYS: tuple[str, ...] = ("CN", "MCI", "AD")

# Human-readable display names, keyed by the short class key above.
CLASS_DISPLAY_NAMES: dict[str, str] = {
    "CN": "Cognitively Normal",
    "MCI": "Mild Cognitive Impairment",
    "AD": "Alzheimer's Disease",
}

NUM_CLASSES: int = len(CLASS_KEYS)


def display_names() -> list[str]:
    """Return display names in canonical (label-index) order."""
    return [CLASS_DISPLAY_NAMES[key] for key in CLASS_KEYS]


def key_to_index(key: str) -> int:
    """Map a short class key (e.g. ``"AD"``) to its integer label."""
    return CLASS_KEYS.index(key)


# ---------------------------------------------------------------------------
# Image / preprocessing conventions
# ---------------------------------------------------------------------------
IMAGE_SIZE: int = 128  # square edge length fed to the model, in pixels
IMAGE_CHANNELS: int = 1  # grayscale; PET slices are single-channel intensity maps

# ---------------------------------------------------------------------------
# Disclaimer surfaced in every prediction response.
# ---------------------------------------------------------------------------
DISCLAIMER: str = (
    "This model is for educational and portfolio demonstration only and is "
    "not intended for clinical diagnosis. Predictions are not medical advice "
    "and must not be used for clinical decision-making."
)

# ---------------------------------------------------------------------------
# On-disk conventions
# ---------------------------------------------------------------------------
# Keys used inside the saved checkpoint dictionary.
CKPT_STATE_DICT: str = "state_dict"
CKPT_METADATA: str = "metadata"

# Accepted upload content types for the /predict endpoint.
ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff"}
)
