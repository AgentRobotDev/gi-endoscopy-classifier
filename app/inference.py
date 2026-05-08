"""Model loading, inference, and Grad-CAM heatmap generation."""

import base64
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _jet(t: np.ndarray) -> np.ndarray:
    """Map a [0,1] float array to jet colormap RGB uint8."""
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


class _GradCAM:
    """Grad-CAM for a single target layer.

    Registers forward and backward hooks so one forward+backward pass
    captures both the activations and their gradients.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out) -> None:
        self._activations = out

    def _bwd(self, module, grad_in, grad_out) -> None:
        self._gradients = grad_out[0]

    def __call__(self, tensor: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
        """Returns (normalized cam array H×W in [0,1], softmax probabilities)."""
        self.model.zero_grad()
        logits = self.model(tensor)
        probs = F.softmax(logits.detach(), dim=1)[0]

        # Backpropagate the score for the predicted class
        logits[0, int(probs.argmax())].backward()

        # Global-average-pool the gradients over spatial dims → (C,)
        pooled = self._gradients.mean(dim=[0, 2, 3])

        # Weight each activation map by its pooled gradient
        cam = self._activations[0].clone()           # (C, H, W)
        cam = (cam * pooled[:, None, None]).mean(0)  # (H, W)
        cam = torch.clamp(cam, min=0)

        lo, hi = cam.min(), cam.max()
        if hi > lo:
            cam = (cam - lo) / (hi - lo)

        return cam.detach().numpy(), probs


class Predictor:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {model_path}. "
                "Train the model first and copy best_model.pt to app/model/."
            )

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.classes: list[str] = checkpoint["classes"]

        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, len(self.classes)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.model = model

        # Target the last conv block — richest spatial feature map before pooling
        self._gradcam = _GradCAM(model, model.features[-1])

    def predict(self, image: Image.Image) -> dict:
        tensor = PREPROCESS(image.convert("RGB")).unsqueeze(0)

        # Grad-CAM needs a real backward pass — no torch.no_grad()
        cam_array, probs = self._gradcam(tensor)

        predictions = sorted(
            [{"class": cls, "probability": float(p)}
             for cls, p in zip(self.classes, probs)],
            key=lambda x: x["probability"],
            reverse=True,
        )

        return {
            "predictions": predictions,
            "cam_overlay": self._overlay_b64(cam_array, image),
        }

    def _overlay_b64(self, cam: np.ndarray, original: Image.Image, alpha: float = 0.5) -> str:
        img = original.convert("RGB")
        w, h = img.size
        cam_up = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        ) / 255.0
        blended = (
            (1 - alpha) * np.array(img).astype(float)
            + alpha * _jet(cam_up).astype(float)
        ).clip(0, 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
