"""Reproducible evidence manifest for APTOS research results.

No torch, medical images, credentials or network access are required. This
module validates recorded results, not newly executed model performance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from aptos.reporting.clinical_audit import cohort_summary

COHORT_FILES = ("external_validation.json", "external_validation_messidor2.json")
REQUIRED_RUN_FIELDS = frozenset({
    "run_name", "variant", "split", "fold", "sweep",
    "test_qwk", "test_qwk_mean", "test_qwk_std", "ensemble_qwk",
})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_float(value: str, context: str) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{context} missing or not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite")
    return number


def audit_sweeps(path: Path) -> list[dict]:
    """Check persisted per-fold metrics against the reported CV summaries."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED_RUN_FIELDS.issubset(reader.fieldnames):
            raise ValueError("reports/runs.csv missing required columns")
        rows = list(reader)
    names = [row["run_name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate run_name in reports/runs.csv")

    folds: dict[tuple[str, str], list[dict]] = defaultdict(list)
    summaries: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["variant"], row["sweep"])
        if row["split"] == "cv-fold":
            folds[key].append(row)
        elif row["split"] == "cv-summary":
            if key in summaries:
                raise ValueError(f"duplicate summary for {key}")
            summaries[key] = row

    if not folds or set(folds) != set(summaries):
        raise ValueError("CV folds and summaries do not match")

    results = []
    for key in sorted(folds):
        fold_rows = folds[key]
        numbers = sorted(int(row["fold"]) for row in fold_rows)
        if numbers != list(range(1, len(numbers) + 1)) or len(numbers) < 2:
            raise ValueError(f"{key}: fold IDs must be unique and contiguous")
        qwk = [_as_float(row["test_qwk"], f"{key}: test_qwk") for row in fold_rows]
        if any(not -1 <= score <= 1 for score in qwk):
            raise ValueError(f"{key}: QWK outside [-1,1]")
        mean, std = statistics.mean(qwk), statistics.stdev(qwk)
        summary = summaries[key]
        reported_mean = _as_float(summary["test_qwk_mean"], f"{key}: summary mean")
        reported_std = _as_float(summary["test_qwk_std"], f"{key}: summary std")
        # CSV values are rounded independently to six decimals.
        if abs(mean - reported_mean) > 0.00005:
            raise ValueError(f"{key}: fold mean differs from reported summary")
        if abs(std - reported_std) > 0.00005:
            raise ValueError(f"{key}: fold SD differs from reported summary")
        ensemble = _as_float(summary["ensemble_qwk"], f"{key}: ensemble_qwk")
        if not -1 <= ensemble <= 1:
            raise ValueError(f"{key}: ensemble QWK outside [-1,1]")
        results.append({
            "variant": key[0], "sweep": key[1], "folds": len(numbers),
            "fold_test_qwk_mean": round(mean, 6),
            "fold_test_qwk_sd": round(std, 6),
            "ensemble_test_qwk": ensemble,
            "source": "reports/runs.csv",
        })
    return results


def audit_external(directory: Path) -> list[dict]:
    """Verify independent cohorts' confusion, distributions and fixed thresholds."""
    reports = [json.loads((directory / name).read_text(encoding="utf-8"))
               for name in COHORT_FILES]
    baseline_sweep = reports[0]["sweep"]
    baseline_thresholds = reports[0]["thresholds"]
    summaries = []
    for report in reports:
        summary = cohort_summary(report)
        dataset = summary["dataset"]
        if summary["thresholds_refitted"]:
            raise ValueError(f"{dataset}: external thresholds were refitted")
        if report["sweep"] != baseline_sweep or report["thresholds"] != baseline_thresholds:
            raise ValueError(f"{dataset}: external cohorts use different model or thresholds")
        true = {str(k): int(v) for k, v in report["true_distribution"].items()}
        pred = {str(k): int(v) for k, v in report["predicted_distribution"].items()}
        expected_keys = {str(i) for i in range(5)}
        if set(true) != expected_keys or set(pred) != expected_keys:
            raise ValueError(f"{dataset}: expected grade distributions 0-4")
        if any(v < 0 for v in (*true.values(), *pred.values())):
            raise ValueError(f"{dataset}: negative distribution counts")
        if sum(true.values()) != summary["n"] or sum(pred.values()) != summary["n"]:
            raise ValueError(f"{dataset}: grade distribution does not sum to n")
        actual_ref = sum(true[str(i)] for i in (2, 3, 4))
        if actual_ref != summary["referable"]:
            raise ValueError(f"{dataset}: referable count differs from grade distribution")
        tp, fp = int(report["ref_tp"]), int(report["ref_fp"])
        predicted_ref = sum(pred[str(i)] for i in (2, 3, 4))
        if predicted_ref != tp + fp:
            raise ValueError(f"{dataset}: predicted referable count disagrees")
        if true["0"] != int(report["healthy_n"]):
            raise ValueError(f"{dataset}: healthy count disagrees")
        if abs(float(report["ref_referral_rate"]) - predicted_ref / summary["n"]) > 1e-6:
            raise ValueError(f"{dataset}: referral rate disagrees")
        summaries.append(summary)
    return summaries


def build_evidence(directory: Path) -> dict:
    """A deterministic, content-addressed description of committed evidence."""
    filenames = ("runs.csv", *COHORT_FILES)
    result = {
        "schema_version": 1,
        "research_only": True,
        "inputs_sha256": {
            filename: file_sha256(directory / filename) for filename in filenames
        },
        "cross_validation": audit_sweeps(directory / "runs.csv"),
        "external": audit_external(directory),
        "limitations": [
            "No new model evaluation was executed to produce this manifest.",
            "Wilson intervals are descriptive and assume independent eyes.",
            "Kaggle mirrored image copies and distinct grading standards limit transportability.",
            "No clinical clearance or prospective validation has been established.",
        ],
    }
    a, b = result["external"]
    result["external_sensitivity_gap_percentage_points"] = round(
        (a["sensitivity"] - b["sensitivity"]) * 100, 2
    )
    return result


def render_markdown(evidence: dict) -> str:
    lines = [
        "# APTOS research evidence audit",
        "",
        "All values below are re-derived from **committed reports**, not a new model run.",
        "This is an open-source research artifact, not a clinical product.",
        "",
        "## Cross-validation provenance",
        "",
        "| Variant | Sweep | Folds | Mean test QWK | Fold SD | Ensemble QWK |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in evidence["cross_validation"]:
        lines.append(
            f'| {row["variant"]} | `{row["sweep"]}` | {row["folds"]} | '
            f'{row["fold_test_qwk_mean"]:.4f} | {row["fold_test_qwk_sd"]:.4f} | '
            f'{row["ensemble_test_qwk"]:.4f} |'
        )
    lines.extend([
        "", "## External referral performance (original thresholds)", "",
        "| Cohort | Eyes | Referable | Missed | Sensitivity | Wilson 95% interval | Specificity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in evidence["external"]:
        lower, upper = row["sensitivity_wilson95"]
        lines.append(
            f'| {row["dataset"]} | {row["n"]} | {row["referable"]} | '
            f'{row["missed_referable"]} | {row["sensitivity"]:.1%} | '
            f'{lower:.1%}–{upper:.1%} | {row["specificity"]:.1%} |'
        )
    lines += [
        "", f'**Observed sensitivity gap:** '
        f'{evidence["external_sensitivity_gap_percentage_points"]:.2f} percentage points.',
        "",
        "**Interpretation:** This gap compares non-exchangeable cohorts and cannot be "
        "attributed to a single cause. Patient-cluster-adjusted uncertainty is "
        "not available from aggregate files.",
        "",
        "## Input provenance (SHA-256)", "",
    ]
    for filename, digest in evidence["inputs_sha256"].items():
        lines.append(f"- `reports/{filename}`: `{digest}`")
    lines += ["", "## Limitations", ""]
    lines.extend(f"- {limitation}" for limitation in evidence["limitations"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path, help="optional output path, no files modified by default")
    args = parser.parse_args(argv)
    result = build_evidence(args.reports_dir)
    content = (json.dumps(result, indent=2, ensure_ascii=False) + "\n"
               if args.format == "json" else render_markdown(result))
    if args.output is None:
        print(content, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
