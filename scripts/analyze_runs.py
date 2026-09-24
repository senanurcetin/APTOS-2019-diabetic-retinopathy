"""Compare runs and drill into one sweep's errors.

This used to read BigQuery, and it constructed the client at module import - so
without credentials for a project that no longer exists, the file could not even
print --help. It now reads the two places the record actually lives:

  * the run table, from MLflow (or reports/runs.csv when MLflow is absent);
  * per-image predictions, from each sweep's persisted fold arrays, joined back
    to image identity by the same order-checked loader the confound analysis
    uses.

Usage:
    python scripts/analyze_runs.py                               # leaderboard
    python scripts/analyze_runs.py --sweep models/cv/baseline-<id>   # one sweep
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aptos.config import GRADES, Config  # noqa: E402

COLUMNS = ["run_name", "variant", "split", "valid_qwk", "test_qwk",
           "test_qwk_mean", "test_qwk_std", "ensemble_qwk",
           "ensemble_referable_sensitivity", "ensemble_referable_specificity"]


def run_table(cfg: Config) -> pd.DataFrame:
    """Every tracked run. Falls back to the versioned CSV without MLflow."""
    try:
        from aptos import tracking

        tracking.export_runs(cfg)
    except ImportError:
        pass
    path = cfg.paths.reports / "runs.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run `python -m aptos.tracking export`")
    return pd.read_csv(path)


def leaderboard(cfg: Config) -> None:
    runs = run_table(cfg)
    summaries = runs[runs["split"] == "cv-summary"]
    singles = runs[runs["split"] == "single"]

    fmt = {"float_format": lambda x: f"{x:.4f}", "index": False}
    if not summaries.empty:
        cols = [c for c in COLUMNS if c in summaries and summaries[c].notna().any()]
        print("=== CROSS-VALIDATION SWEEPS (by ensemble QWK) ===")
        print(summaries.sort_values("ensemble_qwk", ascending=False)[cols].to_string(**fmt))
    if not singles.empty:
        cols = ["run_name", "variant", "valid_qwk", "test_qwk"]
        print("\n=== SINGLE-SPLIT RUNS (transcribed from RESULTS.md) ===")
        print(singles.sort_values("valid_qwk", ascending=False)[cols].to_string(**fmt))


def confusion(frame: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(frame["diagnosis"], frame["pred"]).reindex(
        index=range(5), columns=range(5), fill_value=0)
    table.index = [f"true {g} {GRADES[g]}" for g in range(5)]
    table.columns = [f"pred {g}" for g in range(5)]
    return table


def per_class(frame: pd.DataFrame) -> pd.DataFrame:
    """Recall per grade, and what each grade is most often mistaken for."""
    rows = []
    for grade in range(5):
        sub = frame[frame["diagnosis"] == grade]
        if sub.empty:
            continue
        wrong = sub[sub["pred"] != grade]["pred"].value_counts()
        rows.append({
            "class": f"{grade} {GRADES[grade]}",
            "n": len(sub),
            "recall": round(float((sub["pred"] == grade).mean()), 3),
            "confused_with": (f"{int(wrong.index[0])} ({int(wrong.iloc[0])})"
                              if not wrong.empty else "-"),
        })
    return pd.DataFrame(rows)


def detail(cfg: Config, sweep: pathlib.Path) -> None:
    from aptos.evaluation.confound import load_sweep_predictions

    frame = load_sweep_predictions(cfg, sweep)
    print(f"\n{'=' * 64}\nSWEEP {sweep.name} - fold ensemble on the held-out test split\n{'=' * 64}")
    print("\n--- confusion matrix ---")
    print(confusion(frame).to_string())
    print("\n--- per class ---")
    print(per_class(frame).to_string(index=False))

    missed = frame[(frame["diagnosis"] >= 3) & (frame["pred"] <= 1)]
    print(f"\n--- missed severe cases (true >= 3, predicted <= 1): "
          f"{len(missed)} of {int((frame['diagnosis'] >= 3).sum())} ---")
    if not missed.empty:
        print(missed[["id_code", "diagnosis", "pred", "raw"]].to_string(index=False))

    spread = np.abs(frame["raw"] - frame["diagnosis"])
    print("\nlargest raw-score errors:")
    print(frame.assign(error=spread).nlargest(5, "error")[
        ["id_code", "diagnosis", "pred", "raw"]].to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sweep", help="a models/cv/<sweep> directory to drill into")
    args = parser.parse_args()

    cfg = Config.load("configs/cv.yaml")
    leaderboard(cfg)
    if args.sweep:
        sweep = pathlib.Path(args.sweep)
        cfg = Config.load("configs/cv.yaml", variant=sweep.name.split("-")[0])
        detail(cfg, sweep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
