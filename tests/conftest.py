"""Shared pytest fixtures.

The fixtures build a tiny model and a synthetic image entirely in-memory / in a
temp directory, so the test suite is fast and has no dependency on the committed
checkpoint or dataset. A session-scoped trained checkpoint is created once and
reused across the API/predictor tests.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from medical_classifier.constants import IMAGE_SIZE
from medical_classifier.data.synthetic import _brain_phantom
from medical_classifier.inference.checkpoint import default_metadata, save_checkpoint
from medical_classifier.models.cnn import build_model
from medical_classifier.preprocessing.transforms import DEFAULT_STATS


@pytest.fixture(scope="session")
def sample_image_bytes() -> bytes:
    """A valid in-memory PNG resembling a synthetic phantom."""
    rng = np.random.default_rng(0)
    arr = _brain_phantom(rng, "AD", IMAGE_SIZE)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def sample_image(sample_image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(sample_image_bytes))


@pytest.fixture(scope="session")
def checkpoint_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a small untrained-but-valid checkpoint for serving tests.

    We don't need high accuracy here — only a loadable, structurally correct
    checkpoint so the Predictor and API can be exercised end to end quickly.
    """
    torch.manual_seed(0)
    model = build_model()
    meta = default_metadata("small_cnn", DEFAULT_STATS, dropout=0.4)
    meta.metrics = {"dataset": "synthetic-test-fixture", "accuracy": 0.0}
    path = tmp_path_factory.mktemp("models") / "model.pt"
    save_checkpoint(path, model, meta)
    return path
