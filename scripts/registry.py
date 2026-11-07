"""Task 3.6 - model registry.

  python scripts/registry.py list                      # versions of the registered model
  python scripts/registry.py promote --version 2       # give version 2 the alias "champion"
  python scripts/registry.py export                    # download champion -> models/champion.keras

`export` is how the model reaches the server (Task 6): we fetch the champion ONCE, at build /
deploy time, so the serving app never calls DagsHub per request.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from visionguard.config import MLFLOW_EXPERIMENT, MODELS_DIR, REGISTERED_MODEL_NAME  # noqa: E402
from visionguard.tracking import setup_mlflow  # noqa: E402

ALIAS = "champion"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["list", "promote", "export"])
    p.add_argument("--version", type=str)
    p.add_argument("--local", action="store_true")
    a = p.parse_args()

    setup_mlflow(MLFLOW_EXPERIMENT, local=a.local)
    client = MlflowClient()

    if a.action == "list":
        for v in sorted(client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"),
                        key=lambda v: int(v.version)):
            run = client.get_run(v.run_id)
            print(f"version {v.version}: run '{run.info.run_name}'  aliases={list(v.aliases)}  "
                  f"test_f1={run.data.metrics.get('test_f1', float('nan')):.3f}  "
                  f"test_roc_auc={run.data.metrics.get('test_roc_auc', float('nan')):.3f}")

    elif a.action == "promote":
        if not a.version:
            sys.exit("--version is required")
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS, a.version)
        print(f"{REGISTERED_MODEL_NAME} version {a.version} now has alias '{ALIAS}'")

    elif a.action == "export":
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS)
        local = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="keras_file")
        src = next(Path(local).glob("*.keras"))
        MODELS_DIR.mkdir(exist_ok=True)
        dst = MODELS_DIR / "champion.keras"
        shutil.copy(src, dst)
        run = client.get_run(mv.run_id)
        (MODELS_DIR / "champion.txt").write_text(
            f"model={REGISTERED_MODEL_NAME}\nversion={mv.version}\nrun={run.info.run_name}\n"
            f"threshold={run.data.metrics['threshold']:.4f}\n")
        print(f"Exported version {mv.version} ({run.info.run_name}) to {dst}")
        print(f"Its validation-chosen threshold is {run.data.metrics['threshold']:.4f} -> use it in the app config")


if __name__ == "__main__":
    main()
