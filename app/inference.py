"""Model loading and inference for the Kvasir-v2 classifier."""

from pathlib import Path

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

    def predict(self, image: Image.Image) -> list[dict]:
        tensor = PREPROCESS(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(self.model(tensor), dim=1)[0]

        return sorted(
            [{"class": cls, "probability": float(p)} for cls, p in zip(self.classes, probs)],
            key=lambda x: x["probability"],
            reverse=True,
        )
