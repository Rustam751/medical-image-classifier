"""The ``Predictor``: the single entry point for turning an image into a result.

The API layer, the CLI, and the Streamlit demo all go through this class so that
prediction behavior is identical everywhere. A ``Predictor`` is constructed once
(loading the checkpoint), then reused for many predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from medical_classifier.constants import CLASS_DISPLAY_NAMES, DISCLAIMER
from medical_classifier.inference.checkpoint import CheckpointMetadata, load_checkpoint
from medical_classifier.models.cnn import build_model
from medical_classifier.preprocessing.transforms import preprocess_image, preprocess_pil


@dataclass
class PredictionResult:
    """Structured prediction output, independent of any web framework."""

    predicted_key: str
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]

    def to_response(self) -> dict:
        """Render as the public API response dictionary (with disclaimer)."""
        return {
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "class_probabilities": {
                k: round(v, 4) for k, v in self.class_probabilities.items()
            },
            "disclaimer": DISCLAIMER,
        }


class Predictor:
    """Loads a checkpoint and produces predictions from images.

    Args:
        model_path: path to a saved checkpoint.
        device: torch device string; defaults to CPU (deployment target).
    """

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        state_dict, metadata = load_checkpoint(self.model_path)
        self.metadata: CheckpointMetadata = metadata
        self.model = build_model(
            architecture=metadata.architecture,
            num_classes=metadata.num_classes,
            dropout=metadata.dropout,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    # -- Core prediction --------------------------------------------------------
    @torch.no_grad()
    def _forward(self, tensor: torch.Tensor) -> dict[str, float]:
        """Run the model on a single (1, H, W) tensor and return probabilities."""
        batch = tensor.unsqueeze(0).to(self.device)  # (1, 1, H, W)
        logits = self.model(batch)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        return {
            key: float(probs[i]) for i, key in enumerate(self.metadata.class_keys)
        }

    def _result_from_probs(self, probs: dict[str, float]) -> PredictionResult:
        top_key = max(probs, key=probs.get)
        return PredictionResult(
            predicted_key=top_key,
            predicted_class=CLASS_DISPLAY_NAMES.get(top_key, top_key),
            confidence=probs[top_key],
            class_probabilities={
                CLASS_DISPLAY_NAMES.get(k, k): v for k, v in probs.items()
            },
        )

    # -- Public API -------------------------------------------------------------
    def predict_bytes(self, data: bytes) -> PredictionResult:
        """Predict from raw image bytes (used by the API upload path)."""
        tensor = preprocess_image(
            data, stats=self.metadata.stats, image_size=self.metadata.image_size
        )
        return self._result_from_probs(self._forward(tensor))

    def predict_pil(self, image) -> PredictionResult:  # noqa: ANN001 - PIL.Image
        """Predict from a PIL image (used by the CLI / Streamlit demo)."""
        tensor = preprocess_pil(
            image, stats=self.metadata.stats, image_size=self.metadata.image_size
        )
        return self._result_from_probs(self._forward(tensor))

    @property
    def is_ready(self) -> bool:
        return self.model is not None
