"""Task 1.3 - Visual EDA. Uses the audit table, so run audit_dataset.py first.

Figures (saved to reports/figures/):
  samples_real.png, samples_ai.png   grid of random images per class (size and format in titles)
  distributions.png                  width, height, file size and aspect ratio, per class
  formats.png                        file formats per class (a possible "shortcut" for the model)

Plain black/grey styling so figures print well in a black-and-white report.
Run:  python scripts/eda.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionguard.config import FIGURES, ROOT, SEED, TABLES  # noqa: E402
from visionguard.preprocess import open_rgb  # noqa: E402

STYLE = {"real": dict(color="black", histtype="step", linewidth=1.6, label="real"),
         "ai": dict(color="0.55", histtype="stepfilled", alpha=0.6, label="ai")}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def sample_grid(df: pd.DataFrame, cls: str, rows: int = 3, cols: int = 6):
    subset = df[df["class"] == cls]
    picks = subset.sample(min(rows * cols, len(subset)), random_state=SEED)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.1))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, picks.iterrows()):
        img = open_rgb(ROOT / r["rel_path"])
        img.thumbnail((256, 256))
        ax.imshow(img)
        ax.set_title(f"{int(r['width'])}x{int(r['height'])} {r['format']}", fontsize=7)
    fig.suptitle(f"Random sample: {cls} ({len(subset)} images)", fontsize=11)
    fig.tight_layout()
    out = FIGURES / f"samples_{cls}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def distributions(df: pd.DataFrame):
    df = df.assign(aspect=df["width"] / df["height"])
    panels = [("width", "Width (px)", False), ("height", "Height (px)", False),
              ("size_kb", "File size (KB, log scale)", True), ("aspect", "Aspect ratio (w / h)", False)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, (col, xlabel, log) in zip(axes.flat, panels):
        values = df[col].clip(lower=1e-3)
        if log:
            bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 40)
            ax.set_xscale("log")
        else:
            upper = values.quantile(0.99)  # keep a few giant images from squashing the plot
            bins = np.linspace(values.min(), upper, 40)
        for cls in ["real", "ai"]:
            ax.hist(values[df["class"] == cls], bins=bins, **STYLE[cls])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("images")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Per-class distributions (readable, de-duplicated images)", fontsize=11)
    fig.tight_layout()
    out = FIGURES / "distributions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def formats(df: pd.DataFrame):
    table = pd.crosstab(df["format"], df["class"]).reindex(columns=["real", "ai"], fill_value=0)
    share = table / table.sum()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(share))
    ax.bar(x - 0.2, share["real"], 0.4, color="black", label="real")
    ax.bar(x + 0.2, share["ai"], 0.4, color="0.6", label="ai")
    ax.set_xticks(x, share.index)
    ax.set_ylabel("share of class")
    ax.legend(frameon=False)
    ax.set_title("File format per class")
    fig.tight_layout()
    out = FIGURES / "formats.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, table


def main():
    audit = pd.read_csv(TABLES / "audit.csv")
    df = audit[audit["readable"] & ~audit["is_duplicate_copy"]].copy()
    FIGURES.mkdir(parents=True, exist_ok=True)

    outputs = [sample_grid(df, "real"), sample_grid(df, "ai"), distributions(df)]
    fmt_fig, fmt_table = formats(df)
    outputs.append(fmt_fig)

    # Numbers to quote in the written interpretation
    df["aspect"] = df["width"] / df["height"]
    print("Medians per class (use these in your interpretation):")
    print(df.groupby("class")[["width", "height", "size_kb", "aspect"]].median().round(2).to_string())
    print("\nShare of square images (aspect 0.95-1.05):")
    print(df.groupby("class")["aspect"].apply(lambda a: a.between(0.95, 1.05).mean()).round(2).to_string())
    print("\nFormats per class:\n" + fmt_table.to_string())
    print("\nSaved:", *[str(p.relative_to(ROOT)) for p in outputs], sep="\n  ")


if __name__ == "__main__":
    main()
