"""Unit tests for image preprocessing transforms."""

import numpy as np
import pytest
import torch
from PIL import Image

from dataset import TRAIN_TRANSFORM, VAL_TRANSFORM


def make_image(h=300, w=400) -> Image.Image:
    return Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))


# ── Output shape ───────────────────────────────────────────────────────────

def test_val_transform_produces_224x224():
    tensor = VAL_TRANSFORM(make_image())
    assert tensor.shape == (3, 224, 224)


def test_train_transform_produces_224x224():
    tensor = TRAIN_TRANSFORM(make_image())
    assert tensor.shape == (3, 224, 224)


def test_val_transform_works_on_small_image():
    tensor = VAL_TRANSFORM(make_image(h=50, w=50))
    assert tensor.shape == (3, 224, 224)


def test_val_transform_works_on_large_image():
    tensor = VAL_TRANSFORM(make_image(h=2000, w=3000))
    assert tensor.shape == (3, 224, 224)


# ── Determinism ────────────────────────────────────────────────────────────

def test_val_transform_is_deterministic():
    """Val transform must be deterministic — no random ops."""
    img = make_image()
    assert torch.equal(VAL_TRANSFORM(img), VAL_TRANSFORM(img))


def test_train_transform_is_stochastic():
    """Train transform must apply random augmentation."""
    img = make_image(h=300, w=300)
    results = [TRAIN_TRANSFORM(img) for _ in range(5)]
    # At least one pair should differ (extremely unlikely to all match)
    assert not all(torch.equal(results[0], r) for r in results[1:])


# ── Value range ────────────────────────────────────────────────────────────

def test_val_transform_output_is_normalized():
    """Normalized tensors should have values well outside [0, 1]."""
    tensor = VAL_TRANSFORM(make_image())
    # Raw pixels in [0,1] after ToTensor, then shifted by ImageNet stats
    # so some values should be negative (mean subtraction)
    assert tensor.min().item() < 0


def test_val_transform_output_dtype_is_float():
    tensor = VAL_TRANSFORM(make_image())
    assert tensor.dtype == torch.float32
