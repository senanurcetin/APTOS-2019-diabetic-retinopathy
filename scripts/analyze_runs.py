"""Compare training runs and analyse errors, reading from BigQuery.

train.py and train_cv.py write every run to aptos_predictions / aptos_metrics,
so runs from Colab and from a local GPU are compared from the same place.

Usage:
    python scripts/analyze_runs.py                 # list every run
    python scripts/analyze_runs.py --run 3f9a2b1c  # drill into one run
"""
import argparse
import os

import pandas as pd
from google.cloud import bigquery

# Cloud identifiers are read from the environment so the project runs against
# any BigQuery project and bucket. The defaults are the ones this work used.
PROJECT_ID = os.environ.get("APTOS_GCP_PROJECT", "datascientis")
BQ_DATASET = os.environ.get("APTOS_BQ_DATASET", "APTOS_2019")
GRADES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

client = bigquery.Client(project=PROJECT_ID)


def q(sql):
    return client.query(sql.format(p=PROJECT_ID, d=BQ_DATASET)).to_dataframe()


def leaderboard():
    """Every run, ordered by validation QWK."""
    df = q("""
        SELECT run_id, author, model, mode, IFNULL(variant, 'baseline') AS variant,
               img_size, epochs, seed,
               MAX(IF(split='valid', qwk, NULL))      AS valid_qwk,
               MAX(IF(split='test',  qwk, NULL))      AS test_qwk,
               MAX(IF(split='valid', accuracy, NULL)) AS valid_acc,
               MAX(IF(split='valid', macro_f1, NULL)) AS valid_f1,
               MAX(created_at) AS created_at
        FROM `{p}.{d}.aptos_metrics`
        GROUP BY run_id, author, model, mode, variant, img_size, epochs, seed
        ORDER BY valid_qwk DESC
    """)
    if df.empty:
        print("no runs recorded yet")
        return df
    print("=== RUNS (by validation QWK) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df


def confusion(run_id, split):
    df = q(f"""
        SELECT y_true, y_pred, COUNT(*) n
        FROM `{{p}}.{{d}}.aptos_predictions`
        WHERE run_id = '{run_id}' AND split = '{split}'
        GROUP BY y_true, y_pred
    """)
    if df.empty:
        return None
    m = (df.pivot(index="y_true", columns="y_pred", values="n")
           .reindex(index=range(5), columns=range(5), fill_value=0)
           .fillna(0).astype(int))
    m.index = [f"true {g}" for g in GRADES]
    m.columns = [f"pred {i}" for i in range(5)]
    return m


def per_class(run_id, split):
    """Per-class recall, and which class each is most often confused with."""
    df = q(f"""
        SELECT y_true, y_pred, COUNT(*) n
        FROM `{{p}}.{{d}}.aptos_predictions`
        WHERE run_id = '{run_id}' AND split = '{split}'
        GROUP BY y_true, y_pred
    """)
    rows = []
    for grade in range(5):
        sub = df[df.y_true == grade]
        total = sub.n.sum()
        if total == 0:
            continue
        correct = sub[sub.y_pred == grade].n.sum()
        wrong = sub[sub.y_pred != grade]
        worst = wrong.loc[wrong.n.idxmax()] if not wrong.empty else None
        rows.append({
            "class": f"{grade} {GRADES[grade]}",
            "n": int(total),
            "recall": round(correct / total, 3),
            "confused_with": (f"{int(worst.y_pred)} ({int(worst.n)})"
                              if worst is not None else "-"),
        })
    return pd.DataFrame(rows)


def missed_severe(run_id):
    """The clinically costly error: calling advanced DR healthy."""
    return q(f"""
        SELECT split, COUNT(*) n
        FROM `{{p}}.{{d}}.aptos_predictions`
        WHERE run_id = '{run_id}' AND y_true >= 3 AND y_pred <= 1
        GROUP BY split
    """)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run_id; defaults to the best validation QWK")
    args = ap.parse_args()

    board = leaderboard()
    if board.empty:
        return

    run_id = args.run or board.iloc[0]["run_id"]
    print(f"\n{'=' * 60}\nDETAIL: run_id={run_id}\n{'=' * 60}")

    for split in ("valid", "test"):
        m = confusion(run_id, split)
        if m is None:
            continue
        print(f"\n--- {split.upper()} confusion matrix ---")
        print(m.to_string())
        print(f"\n--- {split.upper()} per class ---")
        print(per_class(run_id, split).to_string(index=False))

    missed = missed_severe(run_id)
    print("\n--- MISSED SEVERE CASES (true >= 3, predicted <= 1) ---")
    print(missed.to_string(index=False) if not missed.empty
          else "  none - no severe case was classified as healthy")


if __name__ == "__main__":
    main()
