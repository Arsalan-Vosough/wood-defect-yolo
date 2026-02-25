from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


def load_best_metrics(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {"run": run_dir.name, "error": "results.csv not found"}

    df = pd.read_csv(csv_path)

    def find_col(contains_any):
        for c in df.columns:
            if any(s in c for s in contains_any):
                return c
        return None

    col_map50 = find_col(["metrics/mAP50(B)", "mAP50(B)", "mAP50 "])
    col_map5095 = find_col(["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95"])
    col_p = find_col(["metrics/precision(B)", "precision(B)", "precision"])
    col_r = find_col(["metrics/recall(B)", "recall(B)", "recall"])

    if col_map5095 is None:
        return {"run": run_dir.name, "error": "mAP50-95 column not found"}

    best_idx = df[col_map5095].idxmax()
    row = df.loc[best_idx]

    out = {
        "run": run_dir.name,
        "best_epoch": int(row["epoch"]) if "epoch" in df.columns else int(best_idx),
        "mAP50": float(row[col_map50]) if col_map50 else float("nan"),
        "mAP50-95": float(row[col_map5095]),
        "precision": float(row[col_p]) if col_p else float("nan"),
        "recall": float(row[col_r]) if col_r else float("nan"),
        "weights_best": str((run_dir / "weights" / "best.pt").resolve()),
        "error": "",
    }
    return out


def compare_runs(runs: Dict[str, Path], out_csv: Path) -> pd.DataFrame:
    rows = []
    for label, rdir in runs.items():
        m = load_best_metrics(Path(rdir))
        m["model"] = label
        rows.append(m)
    cmp = pd.DataFrame(rows)
    cmp = cmp[["model", "run", "best_epoch", "mAP50", "mAP50-95", "precision", "recall", "weights_best", "error"]]
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmp.to_csv(out_csv, index=False)
    return cmp
