"""CLI: create a handful of sample images for the API demo and tests.

Copies a few representative synthetic images (one or two per class) into the
``samples/`` directory so the repository ships with ready-to-use inputs for
``/predict`` and for the test suite, without committing the full dataset.

Usage:
    python scripts/make_samples.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_classifier.config import load_config  # noqa: E402
from medical_classifier.constants import CLASS_KEYS  # noqa: E402
from medical_classifier.data.synthetic import generate_synthetic_dataset  # noqa: E402
from medical_classifier.utils.logging import get_logger  # noqa: E402

logger = get_logger("make_samples")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create sample images.")
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    samples_dir = cfg.abs_path(cfg.paths.samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Generate a small, separate batch (seed offset) so samples are distinct from
    # the training data yet fully reproducible.
    tmp = samples_dir / "_tmp"
    generate_synthetic_dataset(
        output_dir=tmp,
        per_split_counts={"test": args.per_class},
        image_size=cfg.data.image_size,
        seed=cfg.data.seed + 1000,
    )

    for class_key in CLASS_KEYS:
        src_dir = tmp / "test" / class_key
        for i, img in enumerate(sorted(src_dir.glob("*.png"))):
            dst = samples_dir / f"sample_{class_key.lower()}_{i + 1}.png"
            shutil.copy(img, dst)
            logger.info("Wrote %s", dst.name)

    shutil.rmtree(tmp, ignore_errors=True)
    logger.info("Samples written to %s", samples_dir)


if __name__ == "__main__":
    main()
