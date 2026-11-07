"""Build the train / validation / test lists from the audit table.

Cleaning rules (each one is printed with a count, for the report):
  1. drop unreadable files
  2. drop every copy of an image whose bytes appear under BOTH classes (contradictory label)
  3. keep one copy of each exact duplicate, so the same image cannot land in train AND test
  4. drop images smaller than MIN_SIDE pixels on either side

Then a stratified 70 / 15 / 15 split with a fixed seed: every split keeps the same
real/ai ratio, and running the script again gives identical lists.

Outputs: data/splits/train.csv, val.csv, test.csv  (columns: rel_path, label, class, ...)
         reports/tables/split_summary.md
Run:  python scripts/make_splits.py
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionguard.config import LABEL_OF, MIN_SIDE, SEED, SPLITS_DIR, TABLES  # noqa: E402

COLUMNS = ["rel_path", "label", "class", "sha256", "format", "width", "height"]


def main():
    audit_path = TABLES / "audit.csv"
    if not audit_path.exists():
        sys.exit("Run scripts/audit_dataset.py first.")
    df = pd.read_csv(audit_path)
    log = [("files in audit", len(df))]

    df = df[df["readable"]]
    log.append(("after dropping unreadable", len(df)))

    df = df[~df["cross_class_dup"]]
    log.append(("after dropping cross-class duplicates", len(df)))

    df = df.sort_values("rel_path").drop_duplicates("sha256", keep="first")
    log.append(("after keeping one copy per duplicate group", len(df)))

    df = df[(df["width"] >= MIN_SIDE) & (df["height"] >= MIN_SIDE)].copy()
    log.append((f"after dropping images smaller than {MIN_SIDE}px", len(df)))

    df["label"] = df["class"].map(LABEL_OF)

    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part = part.sort_values("rel_path")[COLUMNS]
        part.to_csv(SPLITS_DIR / f"{name}.csv", index=False)
        counts = part["class"].value_counts()
        rows.append({"split": name, "total": len(part), "real": int(counts.get("real", 0)),
                     "ai": int(counts.get("ai", 0)), "ai_share": round(part["label"].mean(), 3)})

    # Safety check: no image (by hash) may appear in two splits
    assert not (set(train.sha256) & set(test.sha256) or set(train.sha256) & set(val.sha256)
                or set(val.sha256) & set(test.sha256)), "leak between splits!"

    split_table = pd.DataFrame(rows).set_index("split")
    print("Cleaning steps:")
    for step, n in log:
        print(f"  {step:<45} {n}")
    print("\nSplits:\n" + split_table.to_string())

    md = ["# Cleaning and splits", "", "| step | images |", "|---|---|"]
    md += [f"| {s} | {n} |" for s, n in log]
    md += ["", "| split | total | real | ai | ai share |", "|---|---|---|---|---|"]
    md += [f"| {i} | {r.total} | {r.real} | {r.ai} | {r.ai_share} |" for i, r in split_table.iterrows()]
    (TABLES / "split_summary.md").write_text("\n".join(md) + "\n")
    print(f"\nSaved split lists to {SPLITS_DIR} and summary to {TABLES / 'split_summary.md'}")


if __name__ == "__main__":
    main()
