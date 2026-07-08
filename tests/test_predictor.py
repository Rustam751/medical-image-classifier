"""Tests for the Predictor and Grad-CAM."""

from __future__ import annotations

import math

from PIL import Image

from medical_classifier.constants import display_names
from medical_classifier.inference.predictor import Predictor


def test_predictor_loads_and_predicts(checkpoint_path, sample_image_bytes) -> None:
    predictor = Predictor(checkpoint_path)
    result = predictor.predict_bytes(sample_image_bytes)

    assert result.predicted_class in display_names()
    assert 0.0 <= result.confidence <= 1.0
    # Probabilities should form a valid distribution over all classes.
    assert set(result.class_probabilities.keys()) == set(display_names())
    assert math.isclose(sum(result.class_probabilities.values()), 1.0, abs_tol=1e-4)


def test_prediction_response_contains_disclaimer(checkpoint_path, sample_image_bytes) -> None:
    predictor = Predictor(checkpoint_path)
    response = predictor.predict_bytes(sample_image_bytes).to_response()
    assert "disclaimer" in response
    assert "not intended for clinical" in response["disclaimer"].lower()


def test_gradcam_overlay_returns_png(checkpoint_path, sample_image: Image.Image) -> None:
    from medical_classifier.inference.gradcam import GradCAM

    predictor = Predictor(checkpoint_path)
    with GradCAM(predictor) as cam:
        png = cam.overlay_png(sample_image)
    # PNG magic number.
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
