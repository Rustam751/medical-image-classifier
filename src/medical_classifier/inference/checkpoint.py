"""Checkpoint (de)serialization.

A checkpoint bundles everything needed to reproduce inference exactly:

* ``state_dict``  — model weights.
* ``metadata``    — architecture name, class keys/order, image size,
  normalization statistics, and the metrics recorded at training time.

Keeping metadata alongside the weights means the serving layer never has to
guess how the model was trained; it reads the class order and normalization
straight from the file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from medical_classifier.constants import (
    CKPT_METADATA,
    CKPT_STATE_DICT,
    CLASS_KEYS,
    IMAGE_SIZE,
)
from medical_classifier.preprocessing.transforms import NormalizationStats


@dataclass
class CheckpointMetadata:
    """Everything (besides weights) needed to serve or audit the model."""

    architecture: str
    class_keys: list[str]
    image_size: int
    norm_mean: float
    norm_std: float
    num_classes: int
    dropout: float = 0.4
    created_at: str = ""
    framework: str = "pytorch"
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def stats(self) -> NormalizationStats:
        return NormalizationStats(mean=self.norm_mean, std=self.norm_std)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CheckpointMetadata:
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    metadata: CheckpointMetadata,
) -> None:
    """Serialize model weights and metadata to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            CKPT_STATE_DICT: model.state_dict(),
            CKPT_METADATA: metadata.to_dict(),
        },
        path,
    )


def load_checkpoint(path: str | Path) -> tuple[dict[str, Any], CheckpointMetadata]:
    """Load a checkpoint, returning ``(state_dict, metadata)``.

    Raises:
        FileNotFoundError: if the checkpoint does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {path!s}. Train a model first with "
            f"`python -m medical_classifier.training.train` (or `make train`)."
        )
    # weights_only=False is required because we store a metadata dict alongside
    # the tensors. The file is produced by this project's own training run.
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = blob[CKPT_STATE_DICT]
    metadata = CheckpointMetadata.from_dict(blob[CKPT_METADATA])
    return state_dict, metadata


def default_metadata(
    architecture: str,
    stats: NormalizationStats,
    dropout: float,
) -> CheckpointMetadata:
    """Build metadata prefilled with the canonical class taxonomy."""
    return CheckpointMetadata(
        architecture=architecture,
        class_keys=list(CLASS_KEYS),
        image_size=IMAGE_SIZE,
        norm_mean=stats.mean,
        norm_std=stats.std,
        num_classes=len(CLASS_KEYS),
        dropout=dropout,
    )
