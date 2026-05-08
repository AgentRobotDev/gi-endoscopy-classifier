"""Shared fixtures for the test suite."""

import io
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import efficientnet_b0

CLASSES = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis",
]


@pytest.fixture(scope="session")
def dummy_model_path(tmp_path_factory):
    """Checkpoint with random weights — no trained model required."""
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, len(CLASSES)),
    )
    path = tmp_path_factory.mktemp("model") / "best_model.pt"
    torch.save(
        {"epoch": 1, "model_state_dict": model.state_dict(),
         "classes": CLASSES, "val_acc": 0.929},
        path,
    )
    return path


@pytest.fixture
def sample_image() -> Image.Image:
    """Random 300×400 RGB image."""
    arr = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def sample_image_bytes(sample_image) -> bytes:
    buf = io.BytesIO()
    sample_image.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def client(dummy_model_path, monkeypatch_session=None):
    """FastAPI TestClient with a random-weight model loaded via lifespan."""
    import os
    os.environ["MODEL_PATH"] = str(dummy_model_path)
    from fastapi.testclient import TestClient
    import main
    with TestClient(main.app) as c:
        yield c


# session-scoped monkeypatch not built-in; use env var approach above
