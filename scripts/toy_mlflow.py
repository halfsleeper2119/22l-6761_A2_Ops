"""Task 3.1 / 3.2 - MLflow tool lab on MNIST.

Trains a tiny Keras model and logs: parameters, per-epoch metrics, one plot and the model.

  python scripts/toy_mlflow.py --lr 0.001 --local      # Task 3.1: local ./mlruns
  python scripts/toy_mlflow.py --lr 0.01  --local
  mlflow ui --backend-store-uri ./mlruns                # open http://127.0.0.1:5000

  python scripts/toy_mlflow.py --lr 0.001               # Task 3.2: same run on DagsHub (.env)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras import layers  # noqa: E402

from visionguard.config import SEED  # noqa: E402
from visionguard.tracking import setup_mlflow  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--units", type=int, default=64)
    p.add_argument("--local", action="store_true")
    a = p.parse_args()

    keras.utils.set_random_seed(SEED)
    (x, y), (xt, yt) = keras.datasets.mnist.load_data()
    x, y = x[:10000], y[:10000]   # small subset: the lab is about MLflow, not accuracy

    model = keras.Sequential([keras.Input((28, 28)), layers.Flatten(), layers.Rescaling(1 / 255.0),
                              layers.Dense(a.units, activation="relu"), layers.Dense(10, activation="softmax")])
    model.compile(keras.optimizers.Adam(a.lr), "sparse_categorical_crossentropy", metrics=["accuracy"])

    setup_mlflow("toy_mlflow_mnist", local=a.local)
    with mlflow.start_run(run_name=f"mnist_lr{a.lr}"):
        mlflow.set_tag("purpose", "tool lab")
        mlflow.log_params({"learning_rate": a.lr, "epochs": a.epochs, "units": a.units, "train_size": len(x)})

        class Log(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch + 1)

        h = model.fit(x, y, validation_split=0.2, epochs=a.epochs, verbose=2, callbacks=[Log()]).history
        test_loss, test_acc = model.evaluate(xt, yt, verbose=0)
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(h["loss"], color="black", label="train")
        ax.plot(h["val_loss"], color="0.55", label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(frameon=False)
        ax.set_title(f"MNIST loss, lr={a.lr}")
        fig.tight_layout()
        mlflow.log_figure(fig, "loss_curve.png")
        mlflow.tensorflow.log_model(model, artifact_path="model",
                                    pip_requirements=[f"tensorflow=={tf.__version__}"])
        print(f"lr={a.lr}: test accuracy {test_acc:.4f}")


if __name__ == "__main__":
    main()
