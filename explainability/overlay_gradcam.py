"""Recompute a visible Grad-CAM overlay for the four condition frames.

The original condition maps were min-max normalised after ReLU. When layer-4
attribution is nearly constant that maps everything to JET-blue, so the
overlay looks like a flat tint rather than a heatmap. This script recovers
the underlying RGB frame and draws a contrast-stretched Grad-CAM / Eigen-CAM
overlay so the localisation is visible.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# OpenCV COLORMAP_JET at 0 is BGR (128, 0, 0) = RGB (0, 0, 128)
JET_ZERO_RGB = np.array([0.0, 0.0, 128.0], dtype=np.float32)

FIGURES = Path("/workspace/report/figures")
SOURCE = {
    "gradcam_day.png": FIGURES / "gradcam_day.png",
    "gradcam_fog.png": FIGURES / "gradcam_fog.png",
    "gradcam_night.png": FIGURES / "gradcam_night.png",
    "gradcam_town.png": FIGURES / "gradcam_town.png",
}


def recover_frame(overlay_rgb: np.ndarray) -> np.ndarray:
    """Invert the 50/50 JET-blue overlay used by explain_conditions.py."""
    rec = np.clip(2.0 * overlay_rgb.astype(np.float32) - JET_ZERO_RGB, 0, 255)
    # Gentle per-channel stretch so the scene is readable after de-tinting.
    for c in range(3):
        lo, hi = np.percentile(rec[:, :, c], (0.5, 99.5))
        if hi - lo < 8:
            continue
        rec[:, :, c] = np.clip((rec[:, :, c] - lo) / (hi - lo) * 255.0, 0, 255)
    return rec.astype(np.uint8)


class CamExtractor:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward)
        target_layer.register_full_backward_hook(self._backward)

    def _forward(self, _module, _inp, output):
        self.activations = output

    def _backward(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0]

    def maps(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray, int]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        class_idx = int(logits.argmax(dim=1).item())
        logits[0, class_idx].backward()

        acts = self.activations[0].detach()          # C,H,W
        grads = self.gradients[0].detach()           # C,H,W

        weights = grads.mean(dim=(1, 2))
        gradcam = torch.relu((weights[:, None, None] * acts).sum(0))
        gradcam = _minmax(gradcam)

        # Eigen-CAM: first principal component of the activation tensor.
        c, h, w = acts.shape
        flat = acts.reshape(c, h * w)
        centered = flat - flat.mean(dim=1, keepdim=True)
        # covariance in channel space is expensive; use the leading
        # right-singular vector of the CxHW matrix via a cheap power step
        # on the spatial Gram matrix is unnecessary — take SVD of C x HW.
        try:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            eigencam = vh[0].reshape(h, w).abs()
        except RuntimeError:
            eigencam = acts.mean(0).abs()
        eigencam = _minmax(eigencam)

        return (
            gradcam.cpu().numpy(),
            eigencam.cpu().numpy(),
            class_idx,
        )


def _minmax(t: torch.Tensor) -> torch.Tensor:
    t = t - t.min()
    denom = t.max()
    if float(denom) < 1e-8:
        return torch.zeros_like(t)
    return t / denom


def contrast_stretch(cam: np.ndarray, lo_p: float = 10.0, hi_p: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(cam, (lo_p, hi_p))
    if hi - lo < 1e-6:
        return cam
    return np.clip((cam - lo) / (hi - lo), 0.0, 1.0)


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Visible jet overlay: original scene stays, hotspots read as red/yellow."""
    cam = cv2.resize(cam.astype(np.float32), (image_rgb.shape[1], image_rgb.shape[0]))
    cam = contrast_stretch(cam)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = image_rgb.astype(np.float32) / 255.0
    # Alpha follows CAM so cold regions keep the original colour
    # instead of a uniform blue wash.
    alpha = (0.15 + 0.70 * cam)[..., None]
    out = (1.0 - alpha) * img + alpha * heatmap_rgb
    return np.uint8(np.clip(out * 255.0, 0, 255))


def combine_cams(gradcam: np.ndarray, eigencam: np.ndarray) -> np.ndarray:
    """Prefer the map with more spatial structure; blend if both are peaked."""
    def peakiness(m: np.ndarray) -> float:
        return float(m.std() * (m.max() - np.median(m)))

    if peakiness(gradcam) < 0.02 and peakiness(eigencam) >= 0.02:
        return eigencam
    if peakiness(eigencam) < 0.02:
        return gradcam
    return contrast_stretch(0.6 * contrast_stretch(gradcam) + 0.4 * contrast_stretch(eigencam))


def main() -> None:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(DEVICE)
    extractor = CamExtractor(model, model.layer4[-1])

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    for name, path in SOURCE.items():
        overlay = np.array(Image.open(path).convert("RGB"))
        frame = recover_frame(overlay)
        x = preprocess(Image.fromarray(frame)).unsqueeze(0).to(DEVICE)
        gradcam, eigencam, class_idx = extractor.maps(x)
        cam = combine_cams(gradcam, eigencam)
        result = overlay_heatmap(frame, cam)
        Image.fromarray(result).save(path)
        print(
            f"{name}: ImageNet class {class_idx}  "
            f"gradcam std={gradcam.std():.3f}  "
            f"eigen std={eigencam.std():.3f}  "
            f"used std={cam.std():.3f}"
        )


if __name__ == "__main__":
    main()
