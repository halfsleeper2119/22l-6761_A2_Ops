"""Task 2.5 / 3.5 - one comparison table of every training run (from reports/runs/*/metrics.json).

Run:  python scripts/compare_runs.py  ->  printed table + reports/tables/runs_comparison.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from visionguard.config import RUNS_DIR, TABLES  # noqa: E402

COLS = ["accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate"]


def main():
    rows = []
    for f in sorted(RUNS_DIR.glob("*/metrics.json")):
        m = json.loads(f.read_text())
        rows.append({"run": m["run_name"], "best_epoch": m["best_epoch"],
                     "threshold": m["threshold"], "val_auc": m["val"]["roc_auc"],
                     **{f"test_{c}": m["test"][c] for c in COLS}})
    if not rows:
        sys.exit("no runs found in reports/runs")
    df = pd.DataFrame(rows).sort_values("val_auc", ascending=False).round(3)
    print(df.to_string(index=False))
    TABLES.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False))
    (TABLES / "runs_comparison.md").write_text("# Run comparison (sorted by validation AUC)\n\n" + header + body + "\n")
    print(f"\nSaved {TABLES / 'runs_comparison.md'}")
    print("Choose the champion by VALIDATION metrics; the test columns are reported, not used to pick.")


if __name__ == "__main__":
    main()
