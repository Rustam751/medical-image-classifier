"""FastAPI application exposing the classifier as an HTTP microservice.

Endpoints
---------
GET  /            -> service banner + links
GET  /health      -> liveness/readiness with model_loaded flag
GET  /model-info  -> architecture, classes, parameter count, training metrics
POST /predict     -> multipart image upload -> prediction + probabilities
POST /explain     -> multipart image upload -> Grad-CAM heatmap PNG

Interactive documentation is served at /docs (Swagger UI) and /redoc.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from medical_classifier.api.dependencies import state
from medical_classifier.api.schemas import (
    HealthResponse,
    InputMetadata,
    ModelInfoResponse,
    PredictionResponse,
)
from medical_classifier.constants import ALLOWED_CONTENT_TYPES, DISCLAIMER
from medical_classifier.models.cnn import count_parameters
from medical_classifier.utils.logging import get_logger

logger = get_logger("medical_classifier.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the app starts."""
    state.load_predictor()
    yield


app = FastAPI(
    title=state.config.api.title,
    version="0.1.0",
    description=(
        "A production-style medical image classification API inspired by an "
        "Alzheimer's PET classification workflow. **For educational and "
        "portfolio demonstration only — not for clinical use.**"
    ),
    lifespan=lifespan,
)


def _require_predictor():
    """Return the loaded predictor or raise a clean 503."""
    if state.predictor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. Train a model with `make train` (or "
                "`python -m medical_classifier.training.train`) so that a "
                "checkpoint exists, then restart the service. "
                f"Details: {state.load_error}"
            ),
        )
    return state.predictor


def _read_upload(file: UploadFile) -> bytes:
    """Validate an uploaded file's content type/size and return its bytes."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported content type '{file.content_type}'. Allowed: "
                f"{sorted(ALLOWED_CONTENT_TYPES)}."
            ),
        )
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    max_bytes = state.config.api.max_upload_bytes
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(data)} bytes). Max is {max_bytes} bytes.",
        )
    return data


@app.get("/", tags=["meta"])
def root() -> dict:
    """Service banner with links to docs and health."""
    return {
        "service": state.config.api.title,
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict (multipart form field 'file')",
        "disclaimer": DISCLAIMER,
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness/readiness probe."""
    return HealthResponse(status="ok", model_loaded=state.model_loaded)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["meta"])
def model_info() -> ModelInfoResponse:
    """Return metadata about the loaded model."""
    predictor = _require_predictor()
    meta = predictor.metadata
    return ModelInfoResponse(
        architecture=meta.architecture,
        classes=_display_names(predictor),
        image_size=meta.image_size,
        num_parameters=count_parameters(predictor.model),
        trained_on=meta.metrics.get("dataset", "unknown"),
        metrics=meta.metrics,
        created_at=meta.created_at,
    )


def _display_names(predictor) -> list[str]:  # noqa: ANN001
    from medical_classifier.constants import CLASS_DISPLAY_NAMES

    return [CLASS_DISPLAY_NAMES.get(k, k) for k in predictor.metadata.class_keys]


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    responses={
        400: {"description": "Empty or invalid file"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported media type"},
        422: {"description": "Image could not be decoded"},
        503: {"description": "Model not loaded"},
    },
)
def predict(file: UploadFile = File(..., description="Medical image file")) -> PredictionResponse:
    """Classify an uploaded medical image."""
    predictor = _require_predictor()
    data = _read_upload(file)
    try:
        result = predictor.predict_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictionResponse(
        predicted_class=result.predicted_class,
        confidence=round(result.confidence, 4),
        class_probabilities={k: round(v, 4) for k, v in result.class_probabilities.items()},
        input_metadata=InputMetadata(
            filename=file.filename,
            content_type=file.content_type,
            size_bytes=len(data),
        ),
    )


@app.post(
    "/explain",
    tags=["inference"],
    responses={
        200: {"content": {"image/png": {}}, "description": "Grad-CAM overlay PNG"},
        503: {"description": "Model not loaded"},
    },
)
def explain(file: UploadFile = File(..., description="Medical image file")) -> Response:
    """Return a Grad-CAM heatmap overlay for the predicted class as a PNG."""
    predictor = _require_predictor()
    data = _read_upload(file)

    # Imported here so the core prediction path has no hard dependency on it.
    import io

    from PIL import Image

    from medical_classifier.inference.gradcam import GradCAM

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not decode image: {exc}") from exc

    with GradCAM(predictor) as cam:
        png = cam.overlay_png(image)
    return Response(content=png, media_type="image/png")


@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc: HTTPException):
    """Ensure error responses are consistent JSON with a 'detail' field."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
