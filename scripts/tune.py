"""Task 2.7 (apply) - Keras Tuner random search on VisionGuard, at least 8 trials.

Searches: learning rate, dropout, and how many top backbone layers are unfrozen (0 = frozen).
Each trial trains for a few epochs with early stopping; the objective is validation loss.
Results: reports/tables/tuner_results.csv, printed table, and one MLflow run with the table.

Run:  python scripts/tune.py               (about 20-40 min on CPU; fine to leave overnight)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras  # noqa: E402
import keras_tuner as kt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from visionguard.config import IMG_SIZE, MLFLOW_EXPERIMENT, ROOT, SEED, TABLES  # noqa: E402
from visionguard.data import make_dataset  # noqa: E402
from visionguard.models import build_transfer, compile_model, unfreeze_top  # noqa: E402
from visionguard.tracking import dvc_data_version, git_commit, setup_mlflow  # noqa: E402


def build(hp: kt.HyperParameters, weights="imagenet"):
    model, base = build_transfer(IMG_SIZE, "mobilenetv2",
                                 dropout=hp.Float("dropout", 0.1, 0.5, step=0.1), weights=weights)
    n = hp.Choice("unfrozen_layers", [0, 10, 20, 40])
    if n:
        unfreeze_top(base, "mobilenetv2", n_layers=n)
    compile_model(model, hp.Float("learning_rate", 1e-5, 1e-3, sampling="log"))
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--local", action="store_true")
    p.add_argument("--no-pretrained", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()

    keras.utils.set_random_seed(SEED)
    train_ds = make_dataset("train", IMG_SIZE, augment=True, shuffle=True)
    val_ds = make_dataset("val", IMG_SIZE)

    weights = None if a.no_pretrained else "imagenet"
    tuner = kt.RandomSearch(lambda hp: build(hp, weights), objective="val_loss",
                            max_trials=a.trials, seed=SEED, overwrite=True,
                            directory=str(ROOT / "tuner"), project_name="visionguard")
    tuner.search_space_summary()
    tuner.search(train_ds, validation_data=val_ds, epochs=a.epochs, verbose=2,
                 callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)])

    rows = []
    for t in tuner.oracle.get_best_trials(num_trials=a.trials):
        m = t.metrics
        rows.append({"trial": t.trial_id, **t.hyperparameters.values,
                     "val_loss": t.score,
                     "val_accuracy": m.get_best_value("val_accuracy") if m.exists("val_accuracy") else None,
                     "val_auc": m.get_best_value("val_auc") if m.exists("val_auc") else None})
    df = pd.DataFrame(rows).sort_values("val_loss").round(5)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "tuner_results.csv", index=False)
    print("\nAll trials, best first:\n" + df.to_string(index=False))
    print("\nBest hyperparameters:", tuner.get_best_hyperparameters(1)[0].values)

    # Which setting mattered most? Spread of val_loss across the values of each hyperparameter.
    print("\nMean val_loss per value (bigger spread = setting matters more):")
    df["lr_bucket"] = pd.cut(df["learning_rate"], [0, 3e-5, 1.5e-4, 1], labels=["<3e-5", "3e-5..1.5e-4", ">1.5e-4"])
    for col in ["lr_bucket", "dropout", "unfrozen_layers"]:
        print(df.groupby(col, observed=True)["val_loss"].agg(["mean", "count"]).round(4).to_string(), "\n")

    setup_mlflow(MLFLOW_EXPERIMENT + "_tuning", local=a.local)
    with mlflow.start_run(run_name="keras_tuner_random_search"):
        mlflow.set_tags({"git_commit": git_commit(), "dvc_data_version": dvc_data_version(), "seed": SEED})
        mlflow.log_params({"trials": a.trials, "epochs_per_trial": a.epochs,
                           **{f"best_{k}": v for k, v in tuner.get_best_hyperparameters(1)[0].values.items()}})
        mlflow.log_metric("best_val_loss", float(df["val_loss"].min()))
        mlflow.log_artifact(str(TABLES / "tuner_results.csv"))


if __name__ == "__main__":
    main()
