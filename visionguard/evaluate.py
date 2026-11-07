"""Task 2.4 / 2.5 / 2.6 - threshold choice, metrics and figures (black/grey styling)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score, roc_curve)

from visionguard.config import CLASSES, ROOT  # noqa: E402
from visionguard.preprocess import open_rgb  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def choose_threshold(y_val, p_val, max_fpr: float = 0.10) -> float:
    """Product decision: calling a REAL photo fake (false positive) is the worse error -
    it accuses a genuine photographer / news photo of being fabricated. So on VALIDATION
    data we pick the threshold that catches the most AI images (highest recall) while
    wrongly flagging at most `max_fpr` of real photos. Never chosen on the test set."""
    fpr, tpr, thr = roc_curve(y_val, p_val)
    ok = np.where(fpr <= max_fpr)[0]
    best = ok[np.argmax(tpr[ok])]
    return float(min(max(thr[best], 1e-6), 1.0))  # roc_curve's first threshold is +inf


def compute_metrics(y, p, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "false_positive_rate": fp / max(fp + tn, 1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def plot_confusion(y, p, threshold, title, out: Path):
    cm = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Greys")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], [f"pred {c}" for c in CLASSES])
    ax.set_yticks([0, 1], [f"true {c}" for c in CLASSES])
    ax.set_title(f"{title}\nthreshold = {threshold:.3f}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_roc(y, p, threshold, title, out: Path):
    fpr, tpr, thr = roc_curve(y, p)
    pred = p >= threshold
    op_fpr = (pred & (y == 0)).sum() / max((y == 0).sum(), 1)
    op_tpr = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.plot(fpr, tpr, color="black", lw=1.6, label=f"AUC = {roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1)
    ax.scatter([op_fpr], [op_tpr], color="black", zorder=3, label=f"threshold {threshold:.2f}")
    ax.set_xlabel("false positive rate (real called AI)")
    ax.set_ylabel("true positive rate (AI caught)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_curves(history: dict, best_epoch: int, stage_boundary, title, out: Path):
    """Loss and accuracy per epoch; dashed line = best epoch (lowest val_loss),
    dotted line = where fine-tuning started (transfer models only)."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, history[key], color="black", lw=1.5, label=f"train {key}")
        ax.plot(epochs, history[f"val_{key}"], color="0.55", lw=1.5, ls="-", marker="o",
                ms=3, label=f"val {key}")
        ax.axvline(best_epoch, color="black", ls="--", lw=1, label=f"best epoch {best_epoch}")
        if stage_boundary:
            ax.axvline(stage_boundary + 0.5, color="0.4", ls=":", lw=1, label="fine-tuning starts")
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_errors(paths, y, p, threshold, out: Path, n: int = 10):
    """Up to n misclassified test images, most confident mistakes first."""
    pred = (p >= threshold).astype(int)
    wrong = np.where(pred != y)[0]
    wrong = wrong[np.argsort(-np.abs(p[wrong] - threshold))][:n]
    if len(wrong) == 0:
        return []
    cols = 5
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6))
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    listed = []
    for k, (ax, i) in enumerate(zip(np.atleast_1d(axes).flat, wrong), start=1):
        img = open_rgb(ROOT / paths[i])
        w, h = img.size
        img.thumbnail((256, 256))
        ax.imshow(img)
        ax.set_title(f"#{k} true {CLASSES[y[i]]}\np(ai)={p[i]:.2f}  {w}x{h}", fontsize=7)
        listed.append({"n": k, "path": paths[i], "true": CLASSES[y[i]], "p_ai": round(float(p[i]), 3),
                       "width": w, "height": h})
    fig.suptitle("Misclassified test images (most confident first)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return listed
