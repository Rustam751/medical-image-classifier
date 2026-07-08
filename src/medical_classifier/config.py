"""Typed configuration loaded from ``config/config.yaml`` with env overrides.

Configuration is intentionally centralized so that training and serving read
the same values. Any field can be overridden at runtime with an environment
variable of the form ``MIC_<SECTION>_<FIELD>`` (uppercase), which is convenient
for containerized deployment where config is injected through the environment.

Example::

    MIC_API_MODEL_PATH=/models/model.pt
    MIC_TRAINING_EPOCHS=10
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# Repository root = three parents up from this file
# (src/medical_classifier/config.py -> repo root).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = REPO_ROOT / "config" / "config.yaml"

ENV_PREFIX = "MIC"


@dataclass
class PathsConfig:
    data_dir: str = "data"
    synthetic_dir: str = "data/synthetic"
    models_dir: str = "models"
    reports_dir: str = "reports"
    samples_dir: str = "samples"


@dataclass
class DataConfig:
    image_size: int = 128
    per_class_train: int = 180
    per_class_val: int = 45
    per_class_test: int = 45
    seed: int = 42


@dataclass
class ModelConfig:
    architecture: str = "small_cnn"  # currently the only supported architecture
    dropout: float = 0.4


@dataclass
class TrainingConfig:
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 4
    seed: int = 42
    num_workers: int = 0


@dataclass
class APIConfig:
    title: str = "Medical Image Disease Classifier API"
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "models/model.pt"
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB
    log_level: str = "INFO"


@dataclass
class Config:
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    api: APIConfig

    # -- Derived absolute paths -------------------------------------------------
    def abs_path(self, relative: str) -> Path:
        """Resolve a config-relative path against the repository root."""
        p = Path(relative)
        return p if p.is_absolute() else (REPO_ROOT / p)

    @property
    def model_file(self) -> Path:
        return self.abs_path(self.api.model_path)


def _coerce(value: Any, target_type: type) -> Any:
    """Coerce a string env value to the dataclass field's declared type."""
    if target_type is bool:
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def _apply_env_overrides(section_name: str, instance: Any) -> None:
    """Mutate a dataclass ``instance`` in place from ``MIC_<SECTION>_<FIELD>``."""
    if not is_dataclass(instance):
        return
    for f in fields(instance):
        env_key = f"{ENV_PREFIX}_{section_name.upper()}_{f.name.upper()}"
        if env_key in os.environ:
            raw = os.environ[env_key]
            setattr(instance, f.name, _coerce(raw, f.type if isinstance(f.type, type) else str))


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from YAML, then apply environment overrides.

    Missing YAML sections fall back to dataclass defaults, so the app runs even
    with a partial or absent config file.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    config = Config(
        paths=PathsConfig(**(raw.get("paths") or {})),
        data=DataConfig(**(raw.get("data") or {})),
        model=ModelConfig(**(raw.get("model") or {})),
        training=TrainingConfig(**(raw.get("training") or {})),
        api=APIConfig(**(raw.get("api") or {})),
    )

    for name, instance in (
        ("paths", config.paths),
        ("data", config.data),
        ("model", config.model),
        ("training", config.training),
        ("api", config.api),
    ):
        _apply_env_overrides(name, instance)

    return config


# A module-level default instance for convenience. Callers that need isolation
# (e.g. tests) should call ``load_config()`` explicitly instead.
_default: Config | None = None


def get_config() -> Config:
    """Return a lazily-initialized, cached default configuration."""
    global _default
    if _default is None:
        _default = load_config()
    return _default
