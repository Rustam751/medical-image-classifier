"""Grad-CAM explainability for the SmallCNN.

Grad-CAM highlights the image regions that most influenced the predicted class
by weighting the final convolutional feature maps with the gradient of the class
score flowing into them. For medical imaging this is valuable: it lets a reviewer
sanity-check *where* the model is looking, rather than trusting a bare score.

This is an optional feature. The core prediction path does not depend on it.
"""

from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from medical_classifier.inference.predictor import Predictor
from medical_classifier.preprocessing.transforms import preprocess_pil


class GradCAM:
    """Compute Grad-CAM heatmaps for a loaded :class:`Predictor`.

    The hook targets the last convolutional block of ``SmallCNN.features``.
    """

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor
        self.model = predictor.model
        # Last conv block is the final entry of the feature extractor.
        self._target_layer = self.model.features[-1]
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._fwd_handle = self._target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self._target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, output) -> None:  # noqa: ANN001
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out) -> None:  # noqa: ANN001
        self._gradients = grad_out[0].detach()

    def close(self) -> None:
        """Remove hooks. Call when the GradCAM instance is no longer needed."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        self.close()

    def heatmap(self, image: Image.Image, class_index: int | None = None) -> np.ndarray:
        """Return a normalized [0,1] Grad-CAM heatmap at the input resolution.

        Args:
            image: input PIL image.
            class_index: class to explain; defaults to the predicted class.
        """
        self.model.zero_grad()
        tensor = preprocess_pil(
            image,
            stats=self.predictor.metadata.stats,
            image_size=self.predictor.metadata.image_size,
        )
        batch = tensor.unsqueeze(0).to(self.predictor.device).requires_grad_(True)

        logits = self.model(batch)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        score = logits[0, class_index]
        score.backward()

        # Global-average-pool the gradients to get per-channel weights, then take
        # a weighted combination of the activation maps.
        grads = self._gradients  # (1, C, h, w)
        acts = self._activations  # (1, C, h, w)
        assert grads is not None and acts is not None
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # (1, 1, h, w)

        cam = F.interpolate(
            cam,
            size=(self.predictor.metadata.image_size, self.predictor.metadata.image_size),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        denom = cam.max() if cam.max() > 0 else 1.0
        return cam / denom

    def overlay_png(self, image: Image.Image, class_index: int | None = None) -> bytes:
        """Return a PNG (bytes) of the heatmap blended over the grayscale image."""
        size = self.predictor.metadata.image_size
        cam = self.heatmap(image, class_index=class_index)
        base = np.asarray(
            image.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32
        ) / 255.0

        # Simple red-hot colormap: heat drives the red channel, base is the gray.
        rgb = np.stack([base, base, base], axis=-1)
        heat = np.stack([cam, np.zeros_like(cam), np.zeros_like(cam)], axis=-1)
        blended = np.clip(0.55 * rgb + 0.45 * heat, 0.0, 1.0)

        out = Image.fromarray((blended * 255).astype(np.uint8), mode="RGB")
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()
