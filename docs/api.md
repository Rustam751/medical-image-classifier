# API Reference

Base URL (local): `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs` (Swagger UI) and `/redoc`.

All prediction responses include a `disclaimer` field. This service is for
educational and portfolio demonstration only and is not for clinical use.

---

## `GET /health`

Liveness/readiness probe.

**Response `200`**
```json
{ "status": "ok", "model_loaded": true }
```

---

## `GET /model-info`

Metadata about the currently loaded model.

**Response `200`**
```json
{
  "architecture": "small_cnn",
  "classes": ["Cognitively Normal", "Mild Cognitive Impairment", "Alzheimer's Disease"],
  "image_size": 128,
  "num_parameters": 97779,
  "trained_on": "synthetic",
  "metrics": { "accuracy": 0.941, "macro_f1": 0.940, "dataset": "synthetic" },
  "created_at": "2025-01-01T00:00:00+00:00",
  "disclaimer": "..."
}
```

---

## `POST /predict`

Classify a single medical image.

**Request:** `multipart/form-data` with a single file field named `file`.
Accepted content types: PNG, JPEG, BMP, TIFF. Max size: 10 MB (configurable).

**Example**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@samples/sample_ad_1.png"
```

**Response `200`**
```json
{
  "predicted_class": "Alzheimer's Disease",
  "confidence": 0.87,
  "class_probabilities": {
    "Cognitively Normal": 0.05,
    "Mild Cognitive Impairment": 0.08,
    "Alzheimer's Disease": 0.87
  },
  "input_metadata": {
    "filename": "sample_ad_1.png",
    "content_type": "image/png",
    "size_bytes": 11391
  },
  "disclaimer": "This model is for educational and portfolio demonstration only and is not intended for clinical diagnosis. Predictions are not medical advice and must not be used for clinical decision-making."
}
```

**Error responses**

| Status | Meaning |
|--------|---------|
| `400`  | Empty file body. |
| `413`  | File exceeds the maximum upload size. |
| `415`  | Unsupported content type. |
| `422`  | File could not be decoded as an image. |
| `503`  | No model is loaded (train one first). |

---

## `POST /explain`

Return a Grad-CAM heatmap overlay (PNG) highlighting the regions that most
influenced the predicted class.

**Request:** same as `/predict`.

**Example**
```bash
curl -X POST http://localhost:8000/explain \
  -F "file=@samples/sample_ad_1.png" --output gradcam.png
```

**Response `200`:** `image/png` binary body.

---

## Configuration via environment variables

Any config value can be overridden with `MIC_<SECTION>_<FIELD>`:

| Variable | Effect |
|----------|--------|
| `MIC_API_MODEL_PATH` | Path to the checkpoint to serve. |
| `MIC_API_PORT` | Port the app binds to (when using the config default). |
| `MIC_API_MAX_UPLOAD_BYTES` | Maximum accepted upload size. |
| `MIC_API_LOG_LEVEL` | Log verbosity (`DEBUG`, `INFO`, ...). |
