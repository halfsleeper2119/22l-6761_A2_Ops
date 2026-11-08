"""Task 8 - client for the cloud-hosted model API. Same interface as visionguard.inference.Predictor,
so the Flask views don't care whether the model runs in-process or on another machine."""
from urllib.parse import urlparse

import requests

from visionguard.inference import Prediction


class RemoteError(RuntimeError):
    pass


class RemotePredictor:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 30):
        if not base_url:
            raise ValueError("MODEL_API_URL must be set when INFERENCE_MODE=remote")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.source = urlparse(self.base_url).netloc or self.base_url
        self.model_version = "remote"

    def _headers(self):
        return {"X-API-Key": self.api_key} if self.api_key else {}

    def predict(self, image: bytes, filename: str = "upload") -> Prediction:
        try:
            r = requests.post(f"{self.base_url}/predict", files={"image": (filename, image)},
                              headers=self._headers(), timeout=self.timeout)
        except requests.RequestException as exc:
            raise RemoteError(f"Model API unreachable: {type(exc).__name__}") from exc
        if r.status_code != 200:
            try:
                detail = r.json().get("error", r.text[:200])
            except ValueError:
                detail = r.text[:200]
            raise RemoteError(f"Model API returned {r.status_code}: {detail}")
        return Prediction(**r.json()["prediction"])

    def health(self) -> dict:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=10)
            return {"reachable": r.ok, **(r.json() if r.ok else {})}
        except requests.RequestException as exc:
            return {"reachable": False, "error": type(exc).__name__}
