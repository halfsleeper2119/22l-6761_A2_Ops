"""MLflow helpers: where to log, reproducibility tags, and a manual per-epoch logger."""
import os
import re
import subprocess
import sys

import keras
import mlflow
from dotenv import load_dotenv

from visionguard.config import ENV_DIR, ROOT


def setup_mlflow(experiment: str, local: bool = False) -> str:
    """Remote (DagsHub) by default. The address and token come from .env / environment
    variables - MLflow reads MLFLOW_TRACKING_URI/USERNAME/PASSWORD itself - never from code."""
    load_dotenv(ROOT / ".env")
    if local or not os.getenv("MLFLOW_TRACKING_URI"):
        if not local:
            print("MLFLOW_TRACKING_URI not set -> logging to local ./mlruns")
        mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment(experiment)
    uri = mlflow.get_tracking_uri()
    print(f"MLflow tracking URI: {uri}  |  experiment: {experiment}")
    return uri


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def dvc_data_version() -> str:
    """The md5 in data/raw.dvc identifies the exact dataset version used."""
    f = ROOT / "data" / "raw.dvc"
    if not f.exists():
        return "not-tracked"
    m = re.search(r"md5:\s*([0-9a-f]+(\.dir)?)", f.read_text())
    return m.group(1) if m else "unknown"


def save_pip_freeze() -> str:
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    out = ENV_DIR / "pip_freeze.txt"
    out.write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                  capture_output=True, text=True).stdout)
    return str(out)


class MLflowEpochLogger(keras.callbacks.Callback):
    """Manual logging (Task 3.3): send every epoch's metrics and learning rate to MLflow.
    Stage 2 of fine-tuning uses initial_epoch, so epoch numbers keep counting up."""

    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})
        logs["learning_rate"] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch + 1)
