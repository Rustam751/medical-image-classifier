"""Application state: a lazily-loaded, cached predictor singleton.

The model is loaded once at startup (or on first use) and reused across
requests. If the checkpoint is missing, the API still starts and reports
``model_loaded: false`` on ``/health`` — the prediction endpoints then return a
clean 503 instead of crashing the process. This makes the container resilient:
it can boot before a model volume is mounted.
"""

from __future__ import annotations

from medical_classifier.config import Config, get_config
from medical_classifier.inference.predictor import Predictor
from medical_classifier.utils.logging import get_logger

logger = get_logger("medical_classifier.api")


class AppState:
    """Holds process-wide singletons for the API."""

    def __init__(self, config: Config | None = None) -> None:
        self.config: Config = config or get_config()
        self._predictor: Predictor | None = None
        self._load_error: str | None = None

    def load_predictor(self) -> None:
        """Attempt to load the predictor; record failure instead of raising."""
        try:
            self._predictor = Predictor(self.config.model_file)
            logger.info("Model loaded from %s", self.config.model_file)
        except FileNotFoundError as exc:
            self._load_error = str(exc)
            logger.warning("Model not loaded: %s", exc)
        except Exception as exc:  # noqa: BLE001 - surface any load failure cleanly
            self._load_error = f"Failed to load model: {exc}"
            logger.error("Model load error: %s", exc)

    @property
    def predictor(self) -> Predictor | None:
        return self._predictor

    @property
    def model_loaded(self) -> bool:
        return self._predictor is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error


# Module-level state instance used by the FastAPI app.
state = AppState()
