"""Tasks 2.2-2.6 and 3.3-3.5 - train one model, evaluate it once on test, log it all to MLflow.

Examples (each is one MLflow run; see GUIDE.md for the full run plan):
  python scripts/train.py --model cnn --run-name cnn_lr1e-3
  python scripts/train.py --model transfer --run-name mnv2_160 --register
  python scripts/train.py --model transfer --log-mode auto --run-name mnv2_160_autolog

What happens:
  1. seeds fixed, tf.data pipelines built from data/splits/*.csv
  2. CNN: one training stage.  Transfer: stage 1 = backbone frozen, head trained;
     stage 2 = last block unfrozen, ~100x smaller learning rate
  3. callbacks: EarlyStopping, ModelCheckpoint (best val_loss), ReduceLROnPlateau
  4. best checkpoint reloaded -> threshold chosen on VALIDATION -> test evaluated ONCE
  5. params, per-epoch metrics, test metrics, tags and artifacts logged to MLflow
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from visionguard import evaluate as ev  # noqa: E402
from visionguard.config import (BATCH_SIZE, IMG_SIZE, MLFLOW_EXPERIMENT, MODELS_DIR,  # noqa: E402
                                REGISTERED_MODEL_NAME, RUNS_DIR, SEED, SPLITS_DIR)
from visionguard.data import labels_of, make_dataset, paths_of  # noqa: E402
from visionguard.models import (build_cnn, build_transfer, compile_model,  # noqa: E402
                                trainable_report, unfreeze_top)
from visionguard.tracking import (MLflowEpochLogger, dvc_data_version, git_commit,  # noqa: E402
                                  save_pip_freeze, setup_mlflow)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "transfer"], required=True)
    p.add_argument("--backbone", choices=["mobilenetv2", "efficientnetb0"], default="mobilenetv2")
    p.add_argument("--run-name", required=True)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate (stage 1 for transfer)")
    p.add_argument("--ft-lr", type=float, default=1e-5, help="fine-tuning learning rate (stage 2)")
    p.add_argument("--epochs", type=int, default=30, help="max epochs (stage 1 for transfer)")
    p.add_argument("--ft-epochs", type=int, default=15, help="max fine-tuning epochs (stage 2)")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-finetune", action="store_true", help="transfer: skip stage 2")
    p.add_argument("--unfreeze", type=int, default=None,
                   help="transfer: unfreeze last N backbone layers instead of the last block")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-fpr", type=float, default=0.10,
                   help="max share of real photos we accept calling AI (threshold choice)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--log-mode", choices=["manual", "auto"], default="manual",
                   help="manual = our Keras callback, auto = mlflow.tensorflow.autolog()")
    p.add_argument("--local", action="store_true", help="log to ./mlruns instead of DagsHub")
    p.add_argument("--register", action="store_true", help="add to the MLflow model registry")
    p.add_argument("--no-pretrained", action="store_true", help=argparse.SUPPRESS)  # offline tests
    return p.parse_args()


def callbacks(ckpt: Path, patience: int, log_mode: str):
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]
    if log_mode == "manual":
        cbs.append(MLflowEpochLogger())
    return cbs


def merge(h1: dict, h2: dict) -> dict:
    return {k: list(h1.get(k, [])) + list(h2.get(k, [])) for k in h1}


def main():
    a = parse_args()
    keras.utils.set_random_seed(a.seed)  # Python, NumPy and TensorFlow seeds in one call
    out = RUNS_DIR / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    ckpt = MODELS_DIR / f"{a.run_name}.keras"

    train_ds = make_dataset("train", a.img_size, a.batch_size, augment=not a.no_augment,
                            shuffle=True, seed=a.seed)
    val_ds = make_dataset("val", a.img_size, a.batch_size)
    test_ds = make_dataset("test", a.img_size, a.batch_size)

    setup_mlflow(MLFLOW_EXPERIMENT, local=a.local)
    if a.log_mode == "auto":
        # Autolog patches model.fit: it records fit/optimizer params, per-epoch metrics,
        # the model summary and early-stopping info. We log the model file ourselves.
        mlflow.tensorflow.autolog(log_models=False, log_datasets=False)

    with mlflow.start_run(run_name=a.run_name) as run:
        mlflow.set_tags({
            "git_commit": git_commit(), "dvc_data_version": dvc_data_version(),
            "seed": a.seed, "model_type": a.model, "log_mode": a.log_mode,
            "backbone": a.backbone if a.model == "transfer" else "none",
        })
        if a.log_mode == "manual":
            mlflow.log_params({k: v for k, v in vars(a).items()
                               if k not in ("run_name", "local", "register", "log_mode")})

        stage_boundary = None
        if a.model == "cnn":
            model = build_cnn(a.img_size, a.dropout)
            compile_model(model, a.lr)
            (out / "trainable_layers.txt").write_text(trainable_report(model))
            hist = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, verbose=2, shuffle=False,
                             callbacks=callbacks(ckpt, a.patience, a.log_mode)).history
        else:
            weights = None if a.no_pretrained else "imagenet"
            model, base = build_transfer(a.img_size, a.backbone, a.dropout, weights=weights)
            compile_model(model, a.lr)
            (out / "trainable_layers_stage1.txt").write_text(trainable_report(model))
            print("\n=== Stage 1: backbone frozen, training the head ===")
            hist = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, verbose=2, shuffle=False,
                             callbacks=callbacks(ckpt, a.patience, a.log_mode)).history
            if not a.no_finetune:
                stage_boundary = len(hist["loss"])
                n = unfreeze_top(base, a.backbone, a.unfreeze)
                compile_model(model, a.ft_lr)  # recompile so the new trainable flags take effect
                (out / "trainable_layers_stage2.txt").write_text(trainable_report(model))
                mlflow.log_metric("stage2_unfrozen_layers", n)
                print(f"\n=== Stage 2: {n} backbone layers unfrozen, lr={a.ft_lr} ===")
                cb = callbacks(ckpt, a.patience, a.log_mode)
                cb[1].best = float(min(hist["val_loss"]))  # only overwrite ckpt if stage 2 beats stage 1
                h2 = model.fit(train_ds, validation_data=val_ds, verbose=2, shuffle=False,
                               epochs=stage_boundary + a.ft_epochs, initial_epoch=stage_boundary,
                               callbacks=cb).history
                hist = merge(hist, h2)

        # ---- best checkpoint, threshold on validation, single test evaluation ----
        model = keras.models.load_model(ckpt)
        best_epoch = int(np.argmin(hist["val_loss"]) + 1)
        p_val = model.predict(val_ds, verbose=0).ravel()
        p_test = model.predict(test_ds, verbose=0).ravel()
        y_val, y_test = labels_of("val"), labels_of("test")
        thr = ev.choose_threshold(y_val, p_val, a.max_fpr)
        val_m = ev.compute_metrics(y_val, p_val, thr)
        test_m = ev.compute_metrics(y_test, p_test, thr)

        title = f"{a.run_name}"
        ev.plot_curves(hist, best_epoch, stage_boundary, title, out / "curves.png")
        ev.plot_confusion(y_test, p_test, thr, f"Test confusion matrix - {title}", out / "confusion_matrix.png")
        ev.plot_roc(y_test, p_test, thr, f"Test ROC - {title}", out / "roc_curve.png")
        errors = ev.plot_errors(paths_of("test"), y_test, p_test, thr, out / "misclassified.png")
        with open(out / "model_summary.txt", "w", encoding="utf-8") as f:
            model.summary(print_fn=lambda s, **k: f.write(s + "\n"))
        summary = {"run_name": a.run_name, "best_epoch": best_epoch, "threshold": thr,
                   "test": test_m, "val": val_m, "misclassified": errors}
        (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))

        mlflow.log_metrics({"best_epoch": best_epoch, "threshold": thr,
                            **{f"val_{k}_at_thr": v for k, v in val_m.items()},
                            **{f"test_{k}": v for k, v in test_m.items()}})
        mlflow.log_artifacts(str(out), artifact_path="evaluation")
        mlflow.log_artifacts(str(SPLITS_DIR), artifact_path="splits")
        mlflow.log_artifact(save_pip_freeze(), artifact_path="environment")
        mlflow.log_artifact(str(ckpt), artifact_path="keras_file")  # plain .keras for serving
        mlflow.tensorflow.log_model(
            model, artifact_path="model",
            pip_requirements=[f"tensorflow=={tf.__version__}", f"keras=={keras.__version__}"],
            registered_model_name=REGISTERED_MODEL_NAME if a.register else None,
        )

        print("\n" + "=" * 60)
        print(f"Run {a.run_name}  (id {run.info.run_id})")
        print(f"Best epoch (lowest val_loss): {best_epoch}   threshold: {thr:.3f}")
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate"]:
            print(f"  test {k:<20} {test_m[k]:.3f}")
        print(f"  confusion: TN={test_m['tn']} FP={test_m['fp']} FN={test_m['fn']} TP={test_m['tp']}")
        print(f"Figures saved in {out}")


if __name__ == "__main__":
    main()
