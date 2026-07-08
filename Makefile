# Convenience targets. On Windows without `make`, run the underlying commands
# shown in each recipe (also documented in the README).

PYTHON ?= python
export PYTHONPATH := src

.PHONY: help install install-dev data samples train evaluate test lint serve demo docker-build docker-run clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies (CPU PyTorch)
	$(PYTHON) -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

install-dev:  ## Install runtime + dev dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu

data:  ## Generate the synthetic dataset
	$(PYTHON) scripts/generate_synthetic_data.py

samples:  ## Create sample images in samples/
	$(PYTHON) scripts/make_samples.py

train:  ## Train the model (generates data if missing) and write reports
	$(PYTHON) -m medical_classifier.training.train

test:  ## Run the test suite
	$(PYTHON) -m pytest

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check src tests scripts

serve:  ## Run the API locally at http://localhost:8000
	$(PYTHON) -m uvicorn medical_classifier.api.main:app --host 0.0.0.0 --port 8000

demo:  ## Run the Streamlit demo UI
	$(PYTHON) -m streamlit run app/streamlit_app.py

docker-build:  ## Build the Docker image
	docker build -t medical-image-classifier-api:latest .

docker-run:  ## Run the Docker image
	docker run --rm -p 8000:8000 medical-image-classifier-api:latest

clean:  ## Remove generated data, caches, and artifacts (keeps the checkpoint)
	rm -rf data .pytest_cache .ruff_cache **/__pycache__
