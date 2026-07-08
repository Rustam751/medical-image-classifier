"""Compact convolutional network for grayscale medical image classification.

Why a small custom CNN instead of VGG16/ResNet50 transfer learning?
--------------------------------------------------------------------
The original research project used ImageNet-pretrained backbones (VGG16,
InceptionV3, ResNet50). Those are appropriate for a research setting with a
GPU and the real ADNI dataset. For a *deployable demo service* the priorities
are different:

* Small artifact  — the checkpoint is a few MB and can live in the repo, so the
  API works immediately on clone with no external weight download.
* CPU-friendly    — inference runs in milliseconds on CPU, so the container has
  no GPU dependency.
* Self-contained  — no torchvision / ImageNet weight download at build time,
  which keeps the Docker image lean and the build reproducible offline.

The architecture is a standard 4-block conv stack (Conv -> BatchNorm -> ReLU ->
MaxPool) followed by global average pooling and a small classifier head with
dropout. Global average pooling makes the network agnostic to the exact input
resolution and keeps the parameter count low.
"""

from __future__ import annotations

import torch
from torch import nn

from medical_classifier.constants import IMAGE_CHANNELS, NUM_CLASSES


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU -> MaxPool(2)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """A lightweight CNN for single-channel image classification.

    Args:
        num_classes: number of output classes.
        in_channels: number of input channels (1 for grayscale).
        dropout: dropout probability applied before the final linear layer.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = IMAGE_CHANNELS,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_channels, 16),
            _conv_block(16, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> (N, 128, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(
    architecture: str = "small_cnn",
    num_classes: int = NUM_CLASSES,
    dropout: float = 0.4,
) -> nn.Module:
    """Factory that returns a model instance by architecture name.

    Currently only ``small_cnn`` is implemented. The factory indirection keeps a
    single seam for adding transfer-learning backbones later without touching
    the training or serving code.
    """
    if architecture == "small_cnn":
        return SmallCNN(num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unknown architecture: {architecture!r}")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
