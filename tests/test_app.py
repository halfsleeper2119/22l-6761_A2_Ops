"""Tests for the web app, model API, validation and preprocessing.
They use a fake predictor, so they run in CI without TensorFlow or the model file.

  python -m pytest -q
"""
import io

import pytest
from PIL import Image

from model_api import create_api
from visionguard.inference import Prediction
from visionguard.preprocess import load_for_model
from webapp import create_app


class FakePredictor:
    model_name, model_version, threshold, img_size, source = "fake", "0", 0.5, 64, "local"

    def predict(self, image, filename=None):
        return Prediction(label="ai", label_text="AI-generated", p_ai=0.9, confidence=0.9,
                          threshold=0.5, model_name="fake", model_version="0", latency_ms=1.0)


def png_bytes(size=(64, 48), mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, size, "white").save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    app = create_app(predictor=FakePredictor(), database_url=f"sqlite:///{tmp_path / 't.db'}")
    return app.test_client()


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    return create_api(predictor=FakePredictor()).test_client()


def upload(client, data, name, url="/predict", headers=None):
    return client.post(url, data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data", headers=headers or {})


# ---------- web app ----------
def test_pages_load(client):
    assert client.get("/").status_code == 200
    assert client.get("/history").status_code == 200


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["database"] == "ok"


def test_valid_upload_is_predicted_and_logged(client):
    r = upload(client, png_bytes(), "cat.png")
    assert r.status_code == 200 and b"AI-generated" in r.data
    assert b"cat.png" in client.get("/history").data


def test_wrong_file_type_rejected(client):
    assert upload(client, b"hello", "notes.txt").status_code == 415


def test_undecodable_image_rejected(client):
    assert upload(client, b"this is not really a png", "fake.png").status_code == 422


def test_too_large_rejected(client):
    assert upload(client, b"0" * (2 * 1024 * 1024), "big.png").status_code == 413


def test_filename_is_sanitized(client):
    upload(client, png_bytes(), "../../etc/passwd.png")
    page = client.get("/history").data
    assert b"../" not in page and b"etc_passwd.png" in page


# ---------- model API ----------
def test_api_requires_key(api):
    assert upload(api, png_bytes(), "a.png").status_code == 401


def test_api_predicts_with_key(api):
    r = upload(api, png_bytes(), "a.png", headers={"X-API-Key": "secret"})
    assert r.status_code == 200 and r.get_json()["prediction"]["label"] == "ai"


def test_api_health_is_public(api):
    assert api.get("/health").get_json()["status"] == "ok"


# ---------- preprocessing ----------
@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_preprocess_always_gives_rgb_square(mode):
    x = load_for_model(png_bytes((100, 40), mode), 64)
    assert x.shape == (64, 64, 3) and x.dtype == "float32" and 0 <= x.min() and x.max() <= 255
