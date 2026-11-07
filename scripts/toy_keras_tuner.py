"""Task 2.7 (toy) - Keras Tuner on Fashion-MNIST.

A tiny dense network; the tuner searches the learning rate and the number of hidden units
with random search, then we read the best settings. Uses 10k training images so it finishes
in a couple of minutes.

Run:  python scripts/toy_keras_tuner.py   -> table printed + reports/tables/toy_tuner_results.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras  # noqa: E402
import keras_tuner as kt  # noqa: E402
import pandas as pd  # noqa: E402
from keras import layers  # noqa: E402

from visionguard.config import ROOT, SEED, TABLES  # noqa: E402


def build(hp):
    model = keras.Sequential([
        keras.Input((28, 28)),
        layers.Flatten(),
        layers.Rescaling(1 / 255.0),
        layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    keras.utils.set_random_seed(SEED)
    (x, y), _ = keras.datasets.fashion_mnist.load_data()
    x_train, y_train, x_val, y_val = x[:10000], y[:10000], x[50000:], y[50000:]

    tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=6, seed=SEED,
                            overwrite=True, directory=str(ROOT / "tuner"), project_name="toy_fashion")
    tuner.search(x_train, y_train, validation_data=(x_val, y_val), epochs=3, verbose=2)

    rows = [{"trial": t.trial_id, **t.hyperparameters.values, "val_accuracy": t.score}
            for t in tuner.oracle.get_best_trials(num_trials=6)]
    df = pd.DataFrame(rows).round(4)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "toy_tuner_results.csv", index=False)
    print("\nToy tuner results (best first):\n" + df.to_string(index=False))
    print("\nBest settings:", tuner.get_best_hyperparameters(1)[0].values)


if __name__ == "__main__":
    main()
