# syntax=docker/dockerfile:1
#
# Production-style image for the classifier API.
# CPU-only PyTorch keeps the image lean and GPU-free.

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps: libgomp1 is required by the PyTorch CPU runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching). Use the PyTorch CPU
# wheel index so we don't pull the large CUDA build.
COPY requirements.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy the application source and configuration.
COPY src ./src
COPY config ./config
COPY models ./models
COPY samples ./samples

# Make the package importable.
ENV PYTHONPATH=/app/src \
    MIC_API_HOST=0.0.0.0 \
    MIC_API_PORT=8000

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Container-level healthcheck hits the API's /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "medical_classifier.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
