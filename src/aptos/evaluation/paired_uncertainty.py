"""Paired, uncertainty-aware research comparisons on the SAME retinal images.

A percentile cluster bootstrap resamples patients (all eyes together) when a
real patient linkage key exists. An eye-level bootstrap is explicit opt-in.
Only binary referable metrics are compared. No site recalibration, model
training, clinical claim, or synthetic per-eye reconstruction is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

METRICS = ("sensitivity", "specificity", "referral_rate")


def _grades(values, label: str) -> np.ndarray:
    try:
        a = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric grades") from exc
    if (a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a))
            or np.any((a < 0) | (a > 4)) or np.any(a != np.floor(a))):
        raise ValueError(f"{label} must be a nonempty 1-D array of integer grades 0–4")
    return a.astype(np.int8)


def _rates(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_ref = truth >= 2
    called_ref = predicted >= 2
    return {
        "sensitivity": float(called_ref[actual_ref].mean()),
        "specificity": float((~called_ref[~actual_ref]).mean()),
        "referral_rate": float(called_ref.mean()),
    }


def paired_bootstrap(
    diagnosis,
    baseline_grade,
    candidate_grade,
    *,
    unit: str,
    patient_ids=None,
    n_resamples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """Percentile intervals for candidate-minus-baseline *paired* metrics.

    Both models must have predictions for the exact same eyes in the same
    order. Patient resampling requires patient IDs; it must never be inferred
    from eye IDs. No clustering is possible from aggregate reports.
    """
    truth = _grades(diagnosis, "diagnosis")
    baseline = _grades(baseline_grade, "baseline_grade")
    candidate = _grades(candidate_grade, "candidate_grade")
    n = len(truth)
    if not (len(baseline) == len(candidate) == n):
        raise ValueError("paired predictions must have equal lengths and aligned image IDs")
    positives = int((truth >= 2).sum())
    if positives == 0 or positives == n:
        raise ValueError("both referable and non-referable eyes are required")
    if unit not in ("eye", "patient"):
        raise ValueError("unit must be explicitly 'eye' or 'patient'")
    if not isinstance(n_resamples, int) or n_resamples < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    if not isinstance(seed, int) or not isinstance(confidence, (int, float)):
        raise ValueError("invalid bootstrap seed or confidence level")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")

    if unit == "patient":
        if patient_ids is None or len(patient_ids) != n:
            raise ValueError("patient unit requires one real patient ID per eye")
        names = np.asarray(patient_ids, dtype=str)
        if any(not name.strip() or name.lower() in ("nan", "none") for name in names):
            raise ValueError("patient IDs cannot be empty")
        _, inverse = np.unique(names, return_inverse=True)
        clusters = [np.flatnonzero(inverse == k) for k in range(int(inverse.max()) + 1)]
    else:
        if patient_ids is not None:
            raise ValueError("patient IDs supplied: explicitly choose unit='patient'")
        clusters = [np.array([i]) for i in range(n)]
    if len(clusters) < 2:
        raise ValueError("at least two independent resampling units are required")

    first = _rates(truth, baseline)
    second = _rates(truth, candidate)
    rng = np.random.default_rng(seed)
    sampled = {name: [] for name in METRICS}
    rejected = 0
    for _ in range(n_resamples):
        choices = rng.integers(0, len(clusters), size=len(clusters))
        idx = np.concatenate([clusters[int(i)] for i in choices])
        selected = truth[idx]
        if np.all(selected >= 2) or np.all(selected < 2):
            rejected += 1
            continue
        a, b = _rates(selected, baseline[idx]), _rates(selected, candidate[idx])
        for metric in METRICS:
            sampled[metric].append(b[metric] - a[metric])
    used = n_resamples - rejected
    if used < max(50, int(n_resamples * 0.8)):
        raise ValueError("too many invalid bootstrap samples; cannot support a stable interval")

    alpha = (1 - confidence) / 2
    metrics = {}
    for name in METRICS:
        lower, upper = np.quantile(sampled[name], [alpha, 1 - alpha])
        metrics[name] = {
            "baseline": first[name],
            "candidate": second[name],
            "difference": second[name] - first[name],
            "ci_lower": float(lower),
            "ci_upper": float(upper),
        }
    return {
        "schema_version": 1,
        "analysis_unit": unit,
        "n_eyes": n,
        "n_units": len(clusters),
        "n_referable": positives,
        "threshold": "diagnosis >= 2",
        "effect": "candidate minus baseline",
        "confidence": float(confidence),
        "seed": seed,
        "draws_requested": n_resamples,
        "draws_used": used,
        "draws_rejected_single_class": rejected,
        "metrics": metrics,
        "limitations": [
            "Percentile bootstrap describes the sampled cohort; it is not a clinical safety bound.",
            "Patient-level resampling only accounts for correlation within known patient IDs.",
            "Eye-level sampling assumes independent eyes and may understate uncertainty.",
            "Fixed model predictions only: hyperparameter selection on this test set is invalid.",
        ],
    }


def read_paired_csv(path: Path, *, unit: str) -> tuple[list[int], list[int], list[int], list[str] | None]:
    """Read aligned grade predictions without retaining identifying rows in output."""
    required = {"id_code", "diagnosis", "baseline_grade", "candidate_grade"}
    if unit not in ("eye", "patient"):
        raise ValueError("choose resampling unit eye or patient")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("CSV needs id_code, diagnosis, baseline_grade, candidate_grade")
        if unit == "patient" and "patient_id" not in reader.fieldnames:
            raise ValueError("patient-level analysis needs a real patient_id column")
        rows = list(reader)
    if not rows:
        raise ValueError("empty prediction CSV")
    ids = [row["id_code"].strip() for row in rows]
    if any(not id_code for id_code in ids) or len(set(ids)) != len(ids):
        raise ValueError("image IDs must be nonempty and unique")
    columns = []
    for key in ("diagnosis", "baseline_grade", "candidate_grade"):
        values = [row[key].strip() for row in rows]
        if any(value not in ("0", "1", "2", "3", "4") for value in values):
            raise ValueError(f"{key} must contain integer grades 0–4")
        columns.append([int(value) for value in values])
    patients = [row["patient_id"].strip() for row in rows] if unit == "patient" else None
    return *columns, patients


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", required=True, type=Path,
                        help="CSV with aligned image IDs, diagnoses and two prediction columns")
    parser.add_argument("--unit", choices=("eye", "patient"), required=True,
                        help="choose eye-level iid or real patient-cluster bootstrap")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="optional aggregate-only JSON output")
    args = parser.parse_args(argv)
    truth, baseline, candidate, patients = read_paired_csv(args.predictions, unit=args.unit)
    result = paired_bootstrap(truth, baseline, candidate, unit=args.unit,
                              patient_ids=patients, n_resamples=args.draws, seed=args.seed)
    result["source_sha256"] = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"aggregate report written to {args.output}")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
