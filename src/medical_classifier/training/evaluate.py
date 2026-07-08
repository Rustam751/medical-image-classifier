"""Evaluation: metrics computation and confusion-matrix rendering.

Produces the artifacts that back the README's "Evaluation" section:

* ``metrics.json``            — accuracy plus per-class precision/recall/F1.
* ``classification_report.txt`` — sklearn's human-readable report.
* ``confusion_matrix.png``    — a labeled confusion-matrix figure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from medical_classifier.constants import display_names


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader and return ``(y_true, y_pred)`` arrays."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(y.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute accuracy and per-class precision/recall/F1."""
    names = display_names()
    labels = list(range(len(names)))
    accuracy = float((y_true == y_pred).mean())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    per_class = {
        names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in labels
    }
    return {
        "accuracy": accuracy,
        "macro_avg": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
        },
        "per_class": per_class,
        "n_samples": int(len(y_true)),
    }


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path) -> None:
    """Render and save a labeled confusion-matrix figure."""
    import matplotlib

    matplotlib.use("Agg")  # headless backend for CI/containers
    import matplotlib.pyplot as plt

    names = display_names()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (synthetic test set)")

    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_reports(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reports_dir: str | Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write metrics.json, classification_report.txt, and confusion_matrix.png."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    names = display_names()

    metrics = compute_metrics(y_true, y_pred)
    if extra:
        metrics.update(extra)

    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    report_txt = classification_report(
        y_true, y_pred, labels=list(range(len(names))), target_names=names, zero_division=0
    )
    with open(reports_dir / "classification_report.txt", "w", encoding="utf-8") as fh:
        fh.write(report_txt)

    save_confusion_matrix(y_true, y_pred, reports_dir / "confusion_matrix.png")
    return metrics
