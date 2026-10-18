"""Read-only BigQuery SQL templates for the versioned APTOS labels export.

Offline generation requires no Google credentials. --dry-run uses a BigQuery
client solely to validate query syntax/cost; it never creates any tables,
datasets, views or model predictions.
"""
from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from pathlib import Path

PROJECT_RE = re.compile(r"^[a-z][a-z0-9-]{4,61}[a-z0-9]$")
DATASET_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,1023}$")
PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,100}$")


def render_queries(project: str, dataset: str = "APTOS_2019",
                   prefix: str = "aptos_") -> dict[str, str]:
    """Build predictable SELECT-only queries for the existing exported schema."""
    if not PROJECT_RE.fullmatch(project):
        raise ValueError("invalid GCP project ID")
    if not DATASET_RE.fullmatch(dataset):
        raise ValueError("invalid BigQuery dataset ID")
    if not PREFIX_RE.fullmatch(prefix):
        raise ValueError("invalid table prefix")
    table = f"`{project}.{dataset}.{prefix}labels`"
    return OrderedDict([
        ("label_integrity", f"""
SELECT
  COUNT(*) AS label_rows,
  COUNT(DISTINCT id_code) AS distinct_ids,
  COUNTIF(id_code IS NULL OR TRIM(id_code) = '') AS missing_ids,
  COUNTIF(diagnosis NOT BETWEEN 0 AND 4 OR diagnosis IS NULL) AS invalid_grades,
  COUNTIF(split NOT IN ('train', 'valid', 'test') OR split IS NULL) AS invalid_splits,
  COUNTIF(is_referable != (diagnosis >= 2) OR is_referable IS NULL) AS invalid_referral_flags
FROM {table}""".strip()),
        ("grade_distribution", f"""
SELECT
  split, diagnosis, diagnosis_label,
  COUNT(*) AS n,
  SAFE_DIVIDE(COUNT(*), SUM(COUNT(*)) OVER (PARTITION BY split)) AS split_share
FROM {table}
GROUP BY split, diagnosis, diagnosis_label
ORDER BY split, diagnosis""".strip()),
        ("referral_prevalence", f"""
SELECT
  split,
  COUNT(*) AS eyes,
  COUNTIF(is_referable) AS referable,
  SAFE_DIVIDE(COUNTIF(is_referable), COUNT(*)) AS referable_share
FROM {table}
GROUP BY split
ORDER BY split""".strip()),
        ("repeated_id_across_splits", f"""
SELECT id_code, COUNT(DISTINCT split) AS split_count, COUNT(*) AS occurrences
FROM {table}
GROUP BY id_code
HAVING COUNT(DISTINCT split) > 1
ORDER BY split_count DESC, id_code""".strip()),
    ])


def dry_run_queries(queries: dict[str, str], project: str) -> dict[str, int]:
    """Validate via BigQuery without executing a data-query job."""
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise RuntimeError('install optional dependencies: pip install -e ".[bigquery]"') from exc
    client = bigquery.Client(project=project)
    costs = {}
    for name, sql in queries.items():
        config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        job = client.query(sql, job_config=config)
        costs[name] = int(job.total_bytes_processed or 0)
    return costs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", required=True, help="GCP project ID (no credentials required for SQL output)")
    parser.add_argument("--dataset", default="APTOS_2019")
    parser.add_argument("--prefix", default="aptos_")
    parser.add_argument("--out-dir", type=Path, help="write one .sql file per query")
    parser.add_argument("--dry-run", action="store_true", help="BigQuery cost/schema validation only")
    args = parser.parse_args(argv)
    queries = render_queries(args.project, args.dataset, args.prefix)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, sql in queries.items():
            (args.out_dir / f"{name}.sql").write_text(sql + "\n", encoding="utf-8")
        print(f"{len(queries)} SELECT-only SQL files written to {args.out_dir}")
    else:
        for name, sql in queries.items():
            print(f"-- {name}\n{sql};\n")
    if args.dry_run:
        for name, cost in dry_run_queries(queries, args.project).items():
            print(f"{name}: BigQuery dry-run estimate {cost} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
