"""Pydantic response/request models for the API.

These schemas define the public contract of the service and drive the
auto-generated OpenAPI/Swagger documentation at ``/docs``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from medical_classifier.constants import DISCLAIMER


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool = Field(..., description="Whether a model checkpoint is loaded.")


class ModelInfoResponse(BaseModel):
    architecture: str
    classes: list[str]
    image_size: int
    num_parameters: int
    trained_on: str = Field(..., description="Dataset the checkpoint was trained on.")
    metrics: dict = Field(default_factory=dict)
    created_at: str = ""
    disclaimer: str = DISCLAIMER


class InputMetadata(BaseModel):
    filename: str | None = None
    content_type: str | None = None
    size_bytes: int


class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., examples=["Alzheimer's Disease"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.87])
    class_probabilities: dict[str, float] = Field(
        ...,
        examples=[
            {
                "Cognitively Normal": 0.05,
                "Mild Cognitive Impairment": 0.08,
                "Alzheimer's Disease": 0.87,
            }
        ],
    )
    input_metadata: InputMetadata
    disclaimer: str = DISCLAIMER


class ErrorResponse(BaseModel):
    detail: str
