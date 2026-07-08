"""CLI: generate the synthetic dataset on disk.

Usage:
    python scripts/generate_synthetic_data.py [--config config/config.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_classifier.config import load_config  # noqa: E402
from medical_classifier.data.synthetic import generate_synthetic_dataset  # noqa: E402
from medical_classifier.utils.logging import get_logger  # noqa: E402

logger = get_logger("generate_synthetic_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic dataset.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = cfg.abs_path(cfg.paths.synthetic_dir)
    totals = generate_synthetic_dataset(
        output_dir=out,
        per_split_counts={
            "train": cfg.data.per_class_train,
            "val": cfg.data.per_class_val,
            "test": cfg.data.per_class_test,
        },
        image_size=cfg.data.image_size,
        seed=cfg.data.seed,
    )
    logger.info("Wrote synthetic dataset to %s", out)
    logger.info("Images per split: %s", totals)


if __name__ == "__main__":
    main()
