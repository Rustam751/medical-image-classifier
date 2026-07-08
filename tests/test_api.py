"""End-to-end API tests using FastAPI's TestClient.

The app's model singleton is pointed at a freshly-created test checkpoint so the
tests never depend on the committed model artifact.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(scope="module")
def client(checkpoint_path):
    """A TestClient whose app state is loaded from the test checkpoint."""
    from medical_classifier.api import dependencies, main

    dependencies.state.config.api.model_path = str(checkpoint_path)
    dependencies.state._predictor = None  # reset any prior load
    dependencies.state.load_predictor()

    with TestClient(main.app) as c:
        yield c


def test_health_ok(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client) -> None:
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["architecture"] == "small_cnn"
    assert len(body["classes"]) == 3
    assert body["num_parameters"] > 0


def test_predict_valid_image(client, sample_image_bytes) -> None:
    resp = client.post(
        "/predict",
        files={"file": ("scan.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] in body["class_probabilities"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["input_metadata"]["size_bytes"] == len(sample_image_bytes)
    assert "disclaimer" in body


def test_predict_rejects_wrong_content_type(client) -> None:
    resp = client.post(
        "/predict",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 415


def test_predict_rejects_empty_file(client) -> None:
    resp = client.post(
        "/predict",
        files={"file": ("empty.png", b"", "image/png")},
    )
    # An empty body is caught as a 400 (empty) before decoding is attempted.
    assert resp.status_code == 400


def test_predict_rejects_corrupt_image(client) -> None:
    resp = client.post(
        "/predict",
        files={"file": ("bad.png", b"\x89PNG\r\n\x1a\n garbage", "image/png")},
    )
    assert resp.status_code == 422


def test_explain_returns_png(client, sample_image_bytes) -> None:
    resp = client.post(
        "/explain",
        files={"file": ("scan.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_predict_matches_generated_class_bias(client) -> None:
    """A blank image should still yield a valid, normalized distribution."""
    blank = Image.new("L", (128, 128), color=0)
    buf = io.BytesIO()
    blank.save(buf, format="PNG")
    resp = client.post("/predict", files={"file": ("blank.png", buf.getvalue(), "image/png")})
    assert resp.status_code == 200
    probs = resp.json()["class_probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-3
