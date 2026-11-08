"""Task 4.1 - inference module, shared by the web app (local mode) and the model API.

The model is loaded ONCE when a Predictor is created (at app start-up), never per request:
loading a Keras model reads tens of MB from disk and builds the TensorFlow graph, which
takes seconds. Doing that on every upload would make each request slow and waste memory.

Preprocessing is the SAME function used in training (visionguard.preprocess.load_for_model),
and pixel scaling lives inside the saved model, so serving cannot drift from training.
"""
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from visionguard.config import CLASSES, MODELS_DIR, REGISTERED_MODEL_NAME
from visionguard.preprocess import load_for_model

LABELS = {"real": "Real photo", "ai": "AI-generated"}


@dataclass
class Prediction:
    label: str            # "ai" or "real"
    label_text: str       # human-readable
    p_ai: float           # model's probability that the image is AI-generated
    confidence: float     # probability of the predicted label
    threshold: float
    model_name: str
    model_version: str
    latency_ms: float

    def to_dict(self):
        return asdict(self)


def _champion_info() -> dict:
    """Defaults written by `scripts/registry.py export` next to champion.keras."""
    info = {}
    f = MODELS_DIR / "champion.txt"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                info[k.strip()] = v.strip()
    return info


class Predictor:
    def __init__(self, model_path=None, threshold=None, model_name=None, model_version=None):
        import keras  # imported here so the web app can start in remote mode without TensorFlow

        info = _champion_info()
        self.model_path = Path(model_path or os.getenv("MODEL_PATH") or MODELS_DIR / "champion.keras")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}. "
                                    "Run `python scripts/registry.py export` or set MODEL_PATH.")
        self.threshold = float(threshold or os.getenv("THRESHOLD") or info.get("threshold", 0.5))
        self.model_name = model_name or os.getenv("MODEL_NAME") or info.get("model", REGISTERED_MODEL_NAME)
        self.model_version = str(model_version or os.getenv("MODEL_VERSION") or info.get("version", "unknown"))

        t = time.perf_counter()
        self.model = keras.models.load_model(self.model_path)
        self.img_size = int(self.model.input_shape[1])   # the model knows its own input size
        self.model.predict(np.zeros((1, self.img_size, self.img_size, 3), "float32"), verbose=0)  # warm-up
        print(f"[inference] loaded {self.model_path.name} (v{self.model_version}, "
              f"{self.img_size}px, threshold {self.threshold:.3f}) in {time.perf_counter() - t:.1f}s")

    def predict(self, image: bytes) -> Prediction:
        t = time.perf_counter()
        x = load_for_model(image, self.img_size)[np.newaxis]
        p_ai = float(self.model.predict(x, verbose=0)[0, 0])
        label = CLASSES[1] if p_ai >= self.threshold else CLASSES[0]
        return Prediction(label=label, label_text=LABELS[label], p_ai=p_ai,
                          confidence=p_ai if label == "ai" else 1.0 - p_ai,
                          threshold=self.threshold, model_name=self.model_name,
                          model_version=self.model_version,
                          latency_ms=(time.perf_counter() - t) * 1000)
