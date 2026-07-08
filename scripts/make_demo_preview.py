"""Render a demo preview panel from the real model, for the README.

Produces a single image that mirrors what the Streamlit demo shows for one
prediction: the input scan, the predicted class with confidence, a probability
bar chart, and the Grad-CAM explanation. Every value is a real output of the
committed model, so the preview stays honest and reproducible.

Usage:
    python scripts/make_demo_preview.py [--image samples/sample_ad_1.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from medical_classifier.config import load_config  # noqa: E402
from medical_classifier.constants import CLASS_DISPLAY_NAMES, CLASS_KEYS, DISCLAIMER  # noqa: E402
from medical_classifier.inference.gradcam import GradCAM  # noqa: E402
from medical_classifier.inference.predictor import Predictor  # noqa: E402

# Brand-ish palette (works on light and dark README backgrounds).
_ACCENT = "#009688"
_MUTED = "#9aa0a6"
_BARS = "#cfd8dc"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the demo preview image.")
    parser.add_argument("--image", type=str, default="samples/sample_ad_1.png")
    parser.add_argument("--out", type=str, default="docs/images/demo_preview.png")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    predictor = Predictor(cfg.model_file)

    image = Image.open(args.image)
    result = predictor.predict_pil(image)

    # Grad-CAM overlay for the predicted class.
    with GradCAM(predictor) as cam:
        heat = cam.heatmap(image)
    size = predictor.metadata.image_size
    gray = image.convert("L").resize((size, size), Image.BILINEAR)

    # Ordered probabilities (canonical class order).
    names = [CLASS_DISPLAY_NAMES[k] for k in CLASS_KEYS]
    probs = [result.class_probabilities[n] for n in names]
    colors = [_ACCENT if n == result.predicted_class else _BARS for n in names]

    fig = plt.figure(figsize=(11.5, 4.6))
    fig.suptitle(
        "Medical Image Disease Classifier — demo interface",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.885,
        f"Predicted:  {result.predicted_class}   ·   {result.confidence:.1%} confidence",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=_ACCENT,
    )
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.5, 1], wspace=0.3)

    # Panel 1: input scan.
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(gray, cmap="gray")
    ax0.set_title("Input scan", fontsize=11)
    ax0.set_xlabel(Path(args.image).name, fontsize=8, color=_MUTED)
    ax0.set_xticks([])
    ax0.set_yticks([])

    # Panel 2: probability bars, each labeled above its bar so long class names
    # never collide with the neighbouring panels.
    ax1 = fig.add_subplot(gs[1])
    y = list(range(len(names)))
    ax1.barh(y, probs, color=colors, height=0.5)
    ax1.set_xlim(0, 1.18)
    ax1.set_ylim(-0.6, len(names) - 0.3)
    ax1.invert_yaxis()
    ax1.set_yticks([])
    ax1.set_title("Class probabilities", fontsize=11)
    ax1.set_xlabel("probability", fontsize=9, color=_MUTED)
    for i, (n, p) in enumerate(zip(names, probs, strict=True)):
        ax1.text(0.0, i - 0.45, n, fontsize=9, va="bottom", ha="left")
        ax1.text(p + 0.02, i, f"{p:.1%}", va="center", fontsize=9, fontweight="bold")
    for spine in ("top", "right", "left"):
        ax1.spines[spine].set_visible(False)

    # Panel 3: Grad-CAM overlay.
    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(gray, cmap="gray")
    ax2.imshow(heat, cmap="jet", alpha=0.45)
    ax2.set_title("Grad-CAM explanation", fontsize=11)
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.text(
        0.5,
        0.015,
        DISCLAIMER,
        ha="center",
        fontsize=7,
        color=_MUTED,
    )
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.04, right=0.97)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
