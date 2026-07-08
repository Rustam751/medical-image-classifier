"""Tests for the preprocessing pipeline."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from medical_classifier.constants import IMAGE_SIZE
from medical_classifier.preprocessing.transforms import preprocess_image, preprocess_pil


def test_preprocess_pil_shape_and_dtype(sample_image: Image.Image) -> None:
    tensor = preprocess_pil(sample_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype == torch.float32


def test_preprocess_handles_rgb_input() -> None:
    # An RGB image must be reduced to a single channel.
    rgb = Image.new("RGB", (64, 200), color=(120, 30, 200))
    tensor = preprocess_pil(rgb)
    assert tensor.shape == (1, IMAGE_SIZE, IMAGE_SIZE)


def test_preprocess_image_from_bytes(sample_image_bytes: bytes) -> None:
    tensor = preprocess_image(sample_image_bytes)
    assert tensor.shape == (1, IMAGE_SIZE, IMAGE_SIZE)


def test_preprocess_image_rejects_corrupt_bytes() -> None:
    with pytest.raises(ValueError):
        preprocess_image(b"not an image at all")


def test_preprocess_is_deterministic(sample_image: Image.Image) -> None:
    a = preprocess_pil(sample_image)
    b = preprocess_pil(sample_image)
    assert torch.allclose(a, b)
