"""CLI: run a prediction on a single image without starting the API.

Usage:
    python scripts/predict_cli.py path/to/image.png
    python scripts/predict_cli.py path/to/image.png --gradcam out.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from medical_classifier.config import load_config  # noqa: E402
from medical_classifier.inference.predictor import Predictor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a single image.")
    parser.add_argument("image", type=str, help="Path to the input image.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--gradcam", type=str, default=None, help="Optional path to save a Grad-CAM overlay PNG."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    predictor = Predictor(cfg.model_file)

    image = Image.open(args.image)
    result = predictor.predict_pil(image)
    print(json.dumps(result.to_response(), indent=2))

    if args.gradcam:
        from medical_classifier.inference.gradcam import GradCAM

        with GradCAM(predictor) as cam:
            png = cam.overlay_png(image)
        Path(args.gradcam).write_bytes(png)
        print(f"Saved Grad-CAM overlay to {args.gradcam}")


if __name__ == "__main__":
    main()
