"""Unit tests for the inference pipeline (Predictor + GradCAM)."""

import numpy as np
import pytest
from PIL import Image

from inference import Predictor


@pytest.fixture(scope="module")
def predictor(dummy_model_path):
    return Predictor(dummy_model_path)


# ── Predictor output structure ─────────────────────────────────────────────

def test_predict_returns_all_eight_classes(predictor, sample_image):
    result = predictor.predict(sample_image)
    assert len(result["predictions"]) == 8


def test_probabilities_sum_to_one(predictor, sample_image):
    result = predictor.predict(sample_image)
    total = sum(p["probability"] for p in result["predictions"])
    assert abs(total - 1.0) < 1e-5


def test_predictions_sorted_descending(predictor, sample_image):
    result = predictor.predict(sample_image)
    probs = [p["probability"] for p in result["predictions"]]
    assert probs == sorted(probs, reverse=True)


def test_each_prediction_has_class_and_probability(predictor, sample_image):
    result = predictor.predict(sample_image)
    for p in result["predictions"]:
        assert "class" in p
        assert "probability" in p
        assert 0.0 <= p["probability"] <= 1.0


def test_predicted_classes_match_known_classes(predictor, sample_image):
    from tests.conftest import CLASSES
    result = predictor.predict(sample_image)
    returned = {p["class"] for p in result["predictions"]}
    assert returned == set(CLASSES)


# ── Grad-CAM overlay ───────────────────────────────────────────────────────

def test_cam_overlay_is_jpeg_data_url(predictor, sample_image):
    result = predictor.predict(sample_image)
    assert result["cam_overlay"].startswith("data:image/jpeg;base64,")


def test_cam_overlay_decodes_to_valid_image(predictor, sample_image):
    import base64, io
    result = predictor.predict(sample_image)
    b64 = result["cam_overlay"].split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert img.size == sample_image.size


# ── Edge cases ─────────────────────────────────────────────────────────────

def test_handles_rgba_image(predictor):
    rgba = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 4), dtype=np.uint8), "RGBA"
    )
    result = predictor.predict(rgba)
    assert len(result["predictions"]) == 8


def test_handles_small_image(predictor):
    tiny = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    result = predictor.predict(tiny)
    assert len(result["predictions"]) == 8


def test_missing_weights_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model weights not found"):
        Predictor(tmp_path / "nonexistent.pt")
