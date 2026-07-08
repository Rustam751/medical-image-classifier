"""Tests for the model architecture and checkpoint round-trip."""

from __future__ import annotations

import torch

from medical_classifier.constants import IMAGE_SIZE, NUM_CLASSES
from medical_classifier.inference.checkpoint import (
    default_metadata,
    load_checkpoint,
    save_checkpoint,
)
from medical_classifier.models.cnn import build_model, count_parameters
from medical_classifier.preprocessing.transforms import DEFAULT_STATS


def test_model_forward_output_shape() -> None:
    model = build_model()
    model.eval()
    x = torch.randn(4, 1, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, NUM_CLASSES)


def test_model_accepts_variable_resolution() -> None:
    # Global average pooling should make the net resolution-agnostic.
    model = build_model()
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 96, 96))
    assert out.shape == (1, NUM_CLASSES)


def test_model_has_reasonable_parameter_count() -> None:
    model = build_model()
    n = count_parameters(model)
    # Lightweight by design: well under one million parameters.
    assert 0 < n < 1_000_000


def test_checkpoint_round_trip(tmp_path) -> None:
    model = build_model()
    meta = default_metadata("small_cnn", DEFAULT_STATS, dropout=0.4)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, meta)

    state_dict, loaded_meta = load_checkpoint(path)
    assert loaded_meta.architecture == "small_cnn"
    assert loaded_meta.num_classes == NUM_CLASSES
    assert set(state_dict.keys()) == set(model.state_dict().keys())
