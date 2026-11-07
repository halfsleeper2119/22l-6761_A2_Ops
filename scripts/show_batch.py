"""Task 2.1 - figure of one augmented training batch, next to the same images un-augmented.

Run:  python scripts/show_batch.py   ->  reports/figures/augmented_batch.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import keras  # noqa: E402

from visionguard.config import CLASSES, FIGURES, IMG_SIZE, SEED  # noqa: E402
from visionguard.data import make_augmenter, make_dataset  # noqa: E402


def main(n: int = 8):
    keras.utils.set_random_seed(SEED)
    images, labels = next(iter(make_dataset("train", IMG_SIZE, batch_size=n, shuffle=True)))
    augmented = make_augmenter()(images, training=True)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.6, 3.6))
    for i in range(n):
        for row, batch in enumerate([images, augmented]):
            ax = axes[row, i]
            ax.imshow(keras.ops.convert_to_numpy(batch[i]).clip(0, 255).astype("uint8"))
            ax.axis("off")
        axes[0, i].set_title(CLASSES[int(labels[i])], fontsize=8)
    axes[0, 0].text(-0.15, 0.5, "original", transform=axes[0, 0].transAxes, rotation=90,
                    va="center", ha="right")
    axes[1, 0].text(-0.15, 0.5, "augmented", transform=axes[1, 0].transAxes, rotation=90,
                    va="center", ha="right")
    fig.suptitle(f"One training batch at {IMG_SIZE}x{IMG_SIZE}: before and after augmentation")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "augmented_batch.png", dpi=150)
    print("Saved", FIGURES / "augmented_batch.png")


if __name__ == "__main__":
    main()
