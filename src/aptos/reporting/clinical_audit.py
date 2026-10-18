"""Descriptive, reproducible referral audit for published external cohorts.

Reads committed cohort-level counts rather than rerunning imaging models.
Wilson intervals assume independent observations; bilateral or repeated eyes
can invalidate that assumption. These are descriptive uncertainty summaries,
NOT patient-cluster-adjusted clinical confidence intervals.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (nominal 95% when z=1.96)."""
    if not isinstance(successes, int) or not isinstance(total, int):
        raise ValueError("successes and total must be integers")
    if total <= 0 or not (0 <= successes <= total) or z <= 0:
        raise ValueError("invalid binomial count or z-score")
    p = successes / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def cohort_summary(report: dict) -> dict:
    """Validate confusion counts before deriving estimates and intervals."""
    dataset = str(report["dataset"])
    counts = {key: int(report[key]) for key in ("ref_tp", "ref_fn", "ref_tn", "ref_fp")}
    if any(v < 0 for v in counts.values()):
        raise ValueError(f"{dataset}: negative confusion count")
    tp, fn, tn, fp = (counts[k] for k in ("ref_tp", "ref_fn", "ref_tn", "ref_fp"))
    n = tp + fn + tn + fp
    if n != int(report["n"]):
        raise ValueError(f"{dataset}: confusion matrix sums to {n}, report says {report['n']}")
    if not tp + fn or not tn + fp:
        raise ValueError(f"{dataset}: both referable and non-referable eyes are required")
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    if abs(sensitivity - float(report["ref_sensitivity"])) > 1e-6:
        raise ValueError(f"{dataset}: sensitivity and counts disagree")
    if abs(specificity - float(report["ref_specificity"])) > 1e-6:
        raise ValueError(f"{dataset}: specificity and counts disagree")
    return {
        "dataset": dataset, "n": n,
        "sweep": str(report["sweep"]), "thresholds_refitted": bool(report["refitted"]),
        "referable": tp + fn, "missed_referable": fn,
        "sensitivity": sensitivity, "sensitivity_wilson95": wilson_interval(tp, tp + fn),
        "specificity": specificity, "specificity_wilson95": wilson_interval(tn, tn + fp),
        "qwk": float(report["qwk"]),
    }


def format_markdown(cohorts: list[dict]) -> str:
    rows = [
        "# External referral audit — descriptive uncertainty",
        "",
        "Source: committed `reports/external_validation*.json`; original APTOS thresholds were not refitted.",
        "Wilson 95% intervals are **illustrative binomial intervals**, NOT patient-cluster-adjusted",
        "clinical uncertainty estimates. Cohorts are not directly exchangeable.",
        "",
        "| Cohort | n | Referable eyes | Missed | Sensitivity [Wilson 95%] | Specificity [Wilson 95%] | QWK |",
        "|:--|--:|--:|--:|:--|:--|--:|",
    ]
    for c in cohorts:
        s0, s1 = c["sensitivity_wilson95"]
        p0, p1 = c["specificity_wilson95"]
        rows.append(
            f'| {c["dataset"]} | {c["n"]} | {c["referable"]} | {c["missed_referable"]} | '
            f'{c["sensitivity"]:.1%} [{s0:.1%}, {s1:.1%}] | '
            f'{c["specificity"]:.1%} [{p0:.1%}, {p1:.1%}] | {c["qwk"]:.3f} |'
        )
    rows.extend([
        "", "**Research only.** External performance deteriorates sharply on Messidor-2.",
        "No observed metric should be interpreted as clinical clearance or safety.",
    ])
    return "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = ap.parse_args()
    paths = [args.reports_dir / "external_validation.json",
             args.reports_dir / "external_validation_messidor2.json"]
    cohorts = [cohort_summary(json.loads(p.read_text(encoding="utf-8"))) for p in paths]
    if args.format == "json":
        print(json.dumps(cohorts, indent=2))
    else:
        print(format_markdown(cohorts), end="")


if __name__ == "__main__":
    main()
