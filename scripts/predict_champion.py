"""Task 3.6 - load the registered model BY ALIAS and classify one image.

Run:  python scripts/predict_champion.py path\\to\\image.jpg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from visionguard.config import MLFLOW_EXPERIMENT, REGISTERED_MODEL_NAME  # noqa: E402
from visionguard.preprocess import load_for_model  # noqa: E402
from visionguard.tracking import setup_mlflow  # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/predict_champion.py <image>")
    image = Path(sys.argv[1])
    setup_mlflow(MLFLOW_EXPERIMENT, local="--local" in sys.argv)

    uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    print(f"Loading {uri} ...")
    model = mlflow.tensorflow.load_model(uri)
    mv = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    threshold = MlflowClient().get_run(mv.run_id).data.metrics.get("threshold", 0.5)

    img_size = model.input_shape[1]                     # the model knows its own input size
    x = load_for_model(image, img_size)[np.newaxis]     # same preprocessing as training
    p_ai = float(model.predict(x, verbose=0)[0, 0])
    label = "AI-generated" if p_ai >= threshold else "real"
    confidence = p_ai if label == "AI-generated" else 1 - p_ai
    print(f"Model version {mv.version} (alias champion), threshold {threshold:.3f}")
    print(f"{image.name}: {label}  (p_ai = {p_ai:.3f}, confidence {confidence:.1%})")


if __name__ == "__main__":
    main()
