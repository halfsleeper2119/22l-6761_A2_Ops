"""Task 1.2 - Dataset audit.

Walks data/raw/, and for EVERY file records: class, path, file size, width, height,
colour mode, format, frame count and a SHA-256 content hash. Then it reports:
  - counts per class (files, readable, corrupt, duplicates)
  - min / median / max of width, height and file size per class
  - format and colour-mode counts per class
  - exact duplicates (same hash), including duplicates that appear in BOTH classes

Outputs:
  reports/tables/audit.csv          one row per file (the full table)
  reports/tables/audit_summary.md   the summary tables, ready to paste into the report

Run:  python scripts/audit_dataset.py
"""
import hashlib
import sys
import warnings
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionguard.config import DATA_RAW, JUNK_FILES, ROOT, TABLES, folder_to_class  # noqa: E402
import visionguard.preprocess  # noqa: E402,F401  (sets Image.MAX_IMAGE_PIXELS)

EXT_TO_FORMAT = {
    ".jpg": "JPEG", ".jpeg": "JPEG", ".jfif": "JPEG", ".png": "PNG", ".webp": "WEBP",
    ".gif": "GIF", ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF",
}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Hash the raw bytes in 1 MB chunks so large files don't need to fit in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def inspect_image(path: Path) -> dict:
    """Two-pass check. verify() checks the file structure cheaply; load() then decodes
    every pixel, which catches truncated files that verify() lets through."""
    info = dict(readable=False, error="", format="", mode="", width=None, height=None, n_frames=1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # e.g. "Possibly corrupt EXIF data" - not fatal
            with Image.open(path) as im:
                info.update(format=im.format or "", mode=im.mode, width=im.width,
                            height=im.height, n_frames=getattr(im, "n_frames", 1))
                im.verify()
            with Image.open(path) as im:
                im.load()
        info["readable"] = True
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return info


def collect() -> pd.DataFrame:
    if not DATA_RAW.exists() or not any(DATA_RAW.iterdir()):
        sys.exit(f"No data found in {DATA_RAW}. Unzip the Kaggle dataset there first.")

    rows, skipped_junk = [], 0
    class_dirs = sorted(d for d in DATA_RAW.iterdir() if d.is_dir())
    print("Top-level folders found:", [d.name for d in class_dirs])
    for class_dir in class_dirs:
        cls = folder_to_class(class_dir.name)
        if cls is None:
            print(f"  WARNING: cannot tell the class of folder '{class_dir.name}', skipping it")
            continue
        files = sorted(p for p in class_dir.rglob("*") if p.is_file())
        print(f"  {class_dir.name:<20} -> class '{cls}', {len(files)} files")
        for p in files:
            if p.name.lower() in JUNK_FILES:
                skipped_junk += 1
                continue
            row = {
                "class": cls,
                "rel_path": p.relative_to(ROOT).as_posix(),
                "extension": p.suffix.lower(),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_of(p),
            }
            row.update(inspect_image(p))
            rows.append(row)
    if skipped_junk:
        print(f"  Skipped {skipped_junk} OS junk files (.DS_Store, Thumbs.db, ...)")
    return pd.DataFrame(rows)


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    # Extension says one thing, the file header says another (e.g. a PNG named .jpg)
    expected = df["extension"].map(EXT_TO_FORMAT)
    df["ext_mismatch"] = df["readable"] & expected.notna() & (expected != df["format"])

    # Exact duplicates: identical bytes => identical hash. The first copy (by path) is kept.
    df = df.sort_values("rel_path").reset_index(drop=True)
    df["dup_group_size"] = df.groupby("sha256")["sha256"].transform("size")
    df["is_duplicate_copy"] = df.duplicated("sha256", keep="first")

    # Same bytes filed under BOTH classes = contradictory label
    classes_per_hash = df.groupby("sha256")["class"].nunique()
    conflicting = classes_per_hash[classes_per_hash > 1].index
    df["cross_class_dup"] = df["sha256"].isin(conflicting)
    df["size_kb"] = (df["size_bytes"] / 1024).round(1)
    return df


def md_table(table: pd.DataFrame) -> str:
    """Minimal Markdown table writer (avoids an extra dependency)."""
    headers = [str(table.index.name or "")] + [str(c) for c in table.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for idx, row in table.iterrows():
        cells = [str(idx)] + [f"{v:,.1f}" if isinstance(v, float) else f"{v}" for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarise(df: pd.DataFrame) -> str:
    sections = []

    counts = df.groupby("class").agg(
        files=("rel_path", "size"),
        readable=("readable", "sum"),
        corrupt=("readable", lambda s: int((~s).sum())),
        duplicate_copies=("is_duplicate_copy", "sum"),
        cross_class_dups=("cross_class_dup", "sum"),
        ext_mismatch=("ext_mismatch", "sum"),
    )
    counts.loc["TOTAL"] = counts.sum()
    sections.append(("Counts per class", counts))

    ok = df[df["readable"]]
    stats = ok.groupby("class")[["width", "height", "size_kb"]].agg(["min", "median", "max"])
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    sections.append(("Width / height (px) and file size (KB), readable files", stats.T))

    sections.append(("Formats per class (readable files)",
                     pd.crosstab(ok["format"], ok["class"])))
    sections.append(("Colour modes per class (readable files)",
                     pd.crosstab(ok["mode"], ok["class"])))

    animated = ok[ok["n_frames"] > 1]
    text = []
    for title, table in sections:
        text.append(f"### {title}\n\n{md_table(table)}\n")
        print(f"\n== {title} ==\n{table.to_string()}")

    corrupt = df[~df["readable"]][["class", "rel_path", "error"]]
    print(f"\n== Corrupt / unreadable files: {len(corrupt)} ==")
    if len(corrupt):
        print(corrupt.to_string(index=False, max_colwidth=70))
    n_groups = df.loc[df["dup_group_size"] > 1, "sha256"].nunique()
    print(f"\n== Exact duplicates: {int(df['is_duplicate_copy'].sum())} extra copies in {n_groups} groups; "
          f"{df.loc[df['cross_class_dup'], 'sha256'].nunique()} hashes appear in BOTH classes ==")
    print(f"== Animated images (first frame will be used): {len(animated)} ==")

    text.append(f"Corrupt files: {len(corrupt)}. Duplicate copies: {int(df['is_duplicate_copy'].sum())} "
                f"(in {n_groups} groups). Cross-class duplicate hashes: "
                f"{df.loc[df['cross_class_dup'], 'sha256'].nunique()}. Animated images: {len(animated)}.\n")
    return "\n".join(text)


def main():
    df = add_flags(collect())
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "audit.csv", index=False)
    summary = summarise(df)
    (TABLES / "audit_summary.md").write_text("# Dataset audit summary\n\n" + summary)
    print(f"\nSaved {TABLES / 'audit.csv'} ({len(df)} rows) and {TABLES / 'audit_summary.md'}")


if __name__ == "__main__":
    main()
