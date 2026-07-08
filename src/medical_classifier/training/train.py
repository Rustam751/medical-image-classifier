"""Training entry point.

Runnable as a module::

    python -m medical_classifier.training.train

Pipeline:
1. If the synthetic dataset is missing, generate it (deterministically).
2. Compute normalization statistics from the training split.
3. Train the SmallCNN with early stopping on validation loss.
4. Evaluate on the held-out test split and write reports.
5. Save the best checkpoint (weights + metadata + metrics).

Point ``paths.synthetic_dir`` at a directory of *real* data (same ImageFolder
layout) to train a real model instead — no code changes required.
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from medical_classifier.config import Config, load_config
from medical_classifier.constants import NUM_CLASSES
from medical_classifier.data.dataset import ImageFolderDataset, compute_norm_stats
from medical_classifier.data.synthetic import generate_synthetic_dataset
from medical_classifier.inference.checkpoint import default_metadata, save_checkpoint
from medical_classifier.models.cnn import build_model, count_parameters
from medical_classifier.training.evaluate import collect_predictions, write_reports
from medical_classifier.utils.logging import get_logger
from medical_classifier.utils.seed import set_seed

logger = get_logger("medical_classifier.train")


def _ensure_dataset(cfg: Config) -> Path:
    """Generate the synthetic dataset if the training split is absent."""
    root = cfg.abs_path(cfg.paths.synthetic_dir)
    train_dir = root / "train"
    if train_dir.exists() and any(train_dir.iterdir()):
        logger.info("Using existing dataset at %s", root)
        return root
    logger.info("Dataset not found; generating synthetic dataset at %s", root)
    totals = generate_synthetic_dataset(
        output_dir=root,
        per_split_counts={
            "train": cfg.data.per_class_train,
            "val": cfg.data.per_class_val,
            "test": cfg.data.per_class_test,
        },
        image_size=cfg.data.image_size,
        seed=cfg.data.seed,
    )
    logger.info("Generated images per split: %s", totals)
    return root


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Run one epoch. If ``optimizer`` is None, runs in eval mode."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def train(cfg: Config) -> dict:
    """Run the full training pipeline and return the recorded metrics."""
    set_seed(cfg.training.seed)
    device = torch.device("cpu")
    root = _ensure_dataset(cfg)

    logger.info("Computing normalization statistics from training split...")
    stats = compute_norm_stats(root / "train", image_size=cfg.data.image_size)
    logger.info("Normalization: mean=%.4f std=%.4f", stats.mean, stats.std)

    train_ds = ImageFolderDataset(root / "train", stats, cfg.data.image_size)
    val_ds = ImageFolderDataset(root / "val", stats, cfg.data.image_size)
    test_ds = ImageFolderDataset(root / "test", stats, cfg.data.image_size)
    logger.info(
        "Dataset sizes: train=%d val=%d test=%d | train class balance=%s",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        train_ds.class_counts(),
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size)

    model = build_model(
        architecture=cfg.model.architecture,
        num_classes=NUM_CLASSES,
        dropout=cfg.model.dropout,
    ).to(device)
    logger.info(
        "Model '%s' built with %d trainable parameters",
        cfg.model.architecture,
        count_parameters(model),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    best_val_loss = float("inf")
    best_state: dict | None = None
    patience = 0

    for epoch in range(1, cfg.training.epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, None, device)
        logger.info(
            "Epoch %02d/%02d | train_loss=%.4f train_acc=%.3f | val_loss=%.4f val_acc=%.3f",
            epoch, cfg.training.epochs, tr_loss, tr_acc, val_loss, val_acc,
        )
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.training.early_stopping_patience:
                logger.info("Early stopping at epoch %d (no val improvement).", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info("Evaluating best model on held-out test split...")
    y_true, y_pred = collect_predictions(model, test_loader, device)
    reports_dir = cfg.abs_path(cfg.paths.reports_dir)
    metrics = write_reports(
        y_true,
        y_pred,
        reports_dir,
        extra={
            "dataset": "synthetic",
            "note": (
                "Synthetic phantom dataset. Metrics demonstrate the pipeline, "
                "not clinical validity."
            ),
            "best_val_loss": round(best_val_loss, 4),
        },
    )
    logger.info(
        "Test accuracy=%.3f macro-F1=%.3f", metrics["accuracy"], metrics["macro_avg"]["f1"]
    )

    metadata = default_metadata(cfg.model.architecture, stats, cfg.model.dropout)
    metadata.created_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    metadata.metrics = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_avg"]["f1"],
        "dataset": "synthetic",
    }
    metadata.notes = (
        "Trained on synthetic PET-like phantoms for portfolio demonstration. "
        "Not clinically validated."
    )
    model_path = cfg.model_file
    save_checkpoint(model_path, model, metadata)
    logger.info("Saved checkpoint to %s", model_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the medical image classifier.")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    train(cfg)


if __name__ == "__main__":
    main()
