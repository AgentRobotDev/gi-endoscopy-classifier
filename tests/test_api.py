"""Integration tests for the FastAPI endpoints."""

import io

import pytest
from PIL import Image


# ── Basic endpoints ────────────────────────────────────────────────────────

def test_index_returns_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]


def test_health_returns_ok(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_samples_returns_dict(client):
    res = client.get("/samples")
    assert res.status_code == 200
    assert isinstance(res.json(), dict)


# ── Predict endpoint ───────────────────────────────────────────────────────

def test_predict_returns_200(client, sample_image_bytes):
    res = client.post(
        "/predict",
        files={"file": ("image.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert res.status_code == 200


def test_predict_response_has_predictions(client, sample_image_bytes):
    res = client.post(
        "/predict",
        files={"file": ("image.jpg", sample_image_bytes, "image/jpeg")},
    )
    data = res.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 8


def test_predict_response_has_cam_overlay(client, sample_image_bytes):
    res = client.post(
        "/predict",
        files={"file": ("image.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert "cam_overlay" in res.json()


def test_predict_probabilities_sum_to_one(client, sample_image_bytes):
    res = client.post(
        "/predict",
        files={"file": ("image.jpg", sample_image_bytes, "image/jpeg")},
    )
    probs = [p["probability"] for p in res.json()["predictions"]]
    assert abs(sum(probs) - 1.0) < 1e-5


def test_predict_invalid_file_returns_400(client):
    res = client.post(
        "/predict",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
    )
    assert res.status_code == 400


def test_predict_empty_file_returns_400(client):
    res = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert res.status_code == 400


# ── Upload size limit ──────────────────────────────────────────────────────

def test_predict_oversized_file_returns_413(client):
    oversized = b"x" * (10 * 1024 * 1024 + 1)
    res = client.post(
        "/predict",
        files={"file": ("big.jpg", oversized, "image/jpeg")},
    )
    assert res.status_code == 413


# ── Rate limiting ──────────────────────────────────────────────────────────

def test_predict_rate_limit_returns_429(client, sample_image_bytes):
    import main
    main._rate_store.clear()

    for _ in range(main._RATE_LIMIT):
        client.post("/predict", files={"file": ("img.jpg", sample_image_bytes, "image/jpeg")})

    res = client.post(
        "/predict",
        files={"file": ("img.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert res.status_code == 429
    assert "Retry-After" in res.headers
