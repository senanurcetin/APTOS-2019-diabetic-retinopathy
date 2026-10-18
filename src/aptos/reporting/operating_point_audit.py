"""Audit frozen referral operating-point results from committed aggregate evidence.

This module reconstructs counts only when published high-precision ratios
uniquely round to integers. It does not recreate image-level predictions,
evaluate new models, or imply patient-independent clinical validity.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from aptos.reporting.clinical_audit import cohort_summary, wilson_interval
from aptos.reporting.research_evidence import file_sha256

EXTERNAL_FILES = (
    "external_validation.json",
    "external_validation_messidor2.json",
)
SELECTION_SET = "APTOS out-of-fold (selection)"


def _probability(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected numeric probability") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{label}: probability must be finite and in [0, 1]")
    return number


def _integer_count(rate: object, denominator: int, label: str) -> int:
    """Recover a count only when aggregate precision supports it."""
    fraction = _probability(rate, label)
    if denominator < 1:
        raise ValueError(f"{label}: denominator must be positive")
    product = fraction * denominator
    count = round(product)
    if abs(product - count) > 1e-6:
        raise ValueError(f"{label}: fraction does not correspond to an integer count")
    return count


def reconstruct_confusion(record: dict) -> dict:
    """Recover a 2x2 confusion matrix and cross-check all published metrics."""
    name = str(record["set"])
    n = record["n"]
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError(f"{name}: n must be an integer of at least two")
    positives = _integer_count(record["prevalence"], n, f"{name} prevalence")
    negatives = n - positives
    if positives < 1 or negatives < 1:
        raise ValueError(f"{name}: both binary classes are required")
    tp = _integer_count(record["sensitivity"], positives, f"{name} sensitivity")
    tn = _integer_count(record["specificity"], negatives, f"{name} specificity")
    fn, fp = positives - tp, negatives - tn
    predicted_positive = tp + fp
    _integer_count(record["referral rate"], n, f"{name} referral rate")
    if abs(_probability(record["referral rate"], f"{name} referral rate")
           - predicted_positive / n) > 1e-8:
        raise ValueError(f"{name}: referral rate contradicts confusion matrix")
    if predicted_positive:
        ppv = tp / predicted_positive
        if abs(_probability(record["PPV"], f"{name} PPV") - ppv) > 1e-8:
            raise ValueError(f"{name}: PPV contradicts confusion matrix")
    else:
        # The saved rows all have predicted positives; prevent silently
        # accepting an undefined PPV for a newly added zero-referral row.
        raise ValueError(f"{name}: PPV undefined with zero predicted referrals")
    return {
        "cohort": name,
        "n": n,
        "referable": positives,
        "non_referable": negatives,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": tp / positives,
        "specificity": tn / negatives,
        "referral_rate": predicted_positive / n,
        "sensitivity_wilson95": wilson_interval(tp, positives),
        "specificity_wilson95": wilson_interval(tn, negatives),
        "source": "reports/calibration.json",
    }


def _transfer_summary(rows: list[dict], applied: dict[str, dict], cut: float) -> list[dict]:
    seen = set()
    results = []
    for record in rows:
        fit_name, target_name = str(record["fitted on"]), str(record["applied to"])
        key = (fit_name, target_name)
        if fit_name == target_name or key in seen or fit_name not in applied:
            raise ValueError("transfer: invalid/duplicate fitting cohort")
        if target_name not in applied or target_name == SELECTION_SET:
            raise ValueError("transfer: unknown external evaluation cohort")
        seen.add(key)
        if not math.isclose(float(record["cut_aptos"]), cut, rel_tol=0, abs_tol=1e-8):
            raise ValueError("transfer: frozen APTOS cut changed")
        local_cut = float(record["cut_refitted"])
        if not math.isfinite(local_cut):
            raise ValueError("transfer: local cut must be finite")
        original = applied[target_name]
        for name, field in (
            ("sensitivity_aptos_cut", "sensitivity"),
            ("specificity_aptos_cut", "specificity"),
        ):
            p = _probability(record[name], f"transfer {name}")
            if abs(p - original[field]) > 1e-8:
                raise ValueError(f"transfer: original {field} contradicts target cohort")
        before = _probability(record["ece_aptos_calibrator"], "original ECE")
        after = _probability(record["ece_refitted"], "refitted ECE")
        sensitivity = _probability(record["sensitivity_refitted_cut"], "transferred sensitivity")
        specificity = _probability(record["specificity_refitted_cut"], "transferred specificity")
        results.append({
            "fitted_on": fit_name,
            "applied_to": target_name,
            "original_cut": cut,
            "refitted_cut": local_cut,
            "ece_delta": after - before,
            "sensitivity_delta": sensitivity - original["sensitivity"],
            "specificity_delta": specificity - original["specificity"],
            "interpretation": "exploratory cross-site transfer, not unbiased site-specific validation",
        })
    return results


def audit_operating_points(reports_dir: Path) -> dict:
    """Distinguish the ordinal grade referral rule from OOF-selected raw cut."""
    calibration = json.loads((reports_dir / "calibration.json").read_text(encoding="utf-8"))
    point = calibration["operating_point"]
    minimum = _probability(point["min_sensitivity"], "minimum sensitivity")
    if minimum <= 0:
        raise ValueError("minimum sensitivity must be positive")
    cut = float(point["cut"])
    if not math.isfinite(cut):
        raise ValueError("operating-point cut must be finite")
    original = calibration["applied"]
    if not original or original[0]["set"] != SELECTION_SET:
        raise ValueError("operating point must have out-of-fold selection provenance")
    applied = [reconstruct_confusion(record) for record in original]
    index = {row["cohort"]: row for row in applied}
    if len(index) != len(applied):
        raise ValueError("duplicate calibration cohort")
    if index[SELECTION_SET]["sensitivity"] < minimum:
        raise ValueError("out-of-fold selection failed its sensitivity constraint")
    if abs(index[SELECTION_SET]["sensitivity"] - _probability(point["sensitivity"], "selected sensitivity")) > 1e-8:
        raise ValueError("selection summary sensitivity mismatch")
    if abs(index[SELECTION_SET]["specificity"] - _probability(point["specificity"], "selected specificity")) > 1e-8:
        raise ValueError("selection summary specificity mismatch")

    grade_comparison = []
    for filename in EXTERNAL_FILES:
        grade = cohort_summary(json.loads((reports_dir / filename).read_text(encoding="utf-8")))
        cohort = grade["dataset"]
        if cohort not in index:
            raise ValueError(f"{cohort}: no matching raw-cut evaluation cohort")
        raw = index[cohort]
        if grade["n"] != raw["n"] or grade["referable"] != raw["referable"]:
            raise ValueError(f"{cohort}: external cohort counts disagree between reports")
        grade_comparison.append({
            "cohort": cohort,
            "n": raw["n"],
            "n_referable": raw["referable"],
            "grade_rule_missed": grade["missed_referable"],
            "raw_cut_missed": raw["fn"],
            "difference_missed_raw_minus_grade": raw["fn"] - grade["missed_referable"],
            "grade_rule_sensitivity": grade["sensitivity"],
            "raw_cut_sensitivity": raw["sensitivity"],
            "rules": {
                "grade_rule": "predicted ordinal grade >= 2",
                "raw_cut": "ensemble raw score >= OOF-selected cut",
            },
        })

    transfers = _transfer_summary(calibration["transfer"], index, cut)
    return {
        "schema_version": 1,
        "research_only": True,
        "operating_point": {
            "selection_set": SELECTION_SET,
            "sensitivity_constraint": minimum,
            "raw_score_cut": cut,
        },
        "applied_cohorts": applied,
        "grade_rule_vs_raw_cut": grade_comparison,
        "transfer": transfers,
        "sha256": {
            name: file_sha256(reports_dir / name)
            for name in ("calibration.json", *EXTERNAL_FILES)
        },
        "limitations": [
            "Recovered confusion counts derive from recorded full-precision ratios.",
            "No per-eye data or paired uncertainty is reconstructed from aggregate records.",
            "Wilson intervals assume independent eyes, not verified patient clusters.",
            "The out-of-fold cut was selected on the APTOS training pool, not on external sites.",
            "Cross-site differences do not establish model transportability or clinical safety.",
        ],
    }


def format_markdown(audit: dict) -> str:
    lines = [
        "# Research-only referral operating-point audit",
        "",
        "Reconstructed from committed reports; **no new model inference**.",
        f"APTOS out-of-fold selected raw-score cut: "
        f"{audit['operating_point']['raw_score_cut']:.6f}.",
        "",
        "| Cohort | n | TP | FN | TN | FP | Sensitivity | Specificity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit["applied_cohorts"]:
        lines.append(
            f"| {row['cohort']} | {row['n']} | {row['tp']} | {row['fn']} | "
            f"{row['tn']} | {row['fp']} | {row['sensitivity']:.1%} | "
            f"{row['specificity']:.1%} |"
        )
    lines.extend([
        "", "## Two different referral rules", "",
        "| External cohort | Missed by ordinal-grade rule | Missed by OOF raw cut | Extra missed by raw cut |",
        "|---|---:|---:|---:|",
    ])
    for row in audit["grade_rule_vs_raw_cut"]:
        lines.append(
            f"| {row['cohort']} | {row['grade_rule_missed']} | "
            f"{row['raw_cut_missed']} | {row['difference_missed_raw_minus_grade']:+d} |"
        )
    lines.extend([
        "", "These counts come from **different operating rules** applied to the "
        "same named external cohorts. Their case-level overlap is unknown.",
        "", "## Exploratory cross-site recalibration (not a generalizable improvement)", "",
        "| Refit cohort | Evaluation cohort | Sensitivity change (pp) | Specificity change (pp) | ECE change |",
        "|---|---|---:|---:|---:|",
    ])
    for row in audit["transfer"]:
        lines.append(
            f"| {row['fitted_on']} | {row['applied_to']} | "
            f"{100 * row['sensitivity_delta']:+.2f} | "
            f"{100 * row['specificity_delta']:+.2f} | {row['ece_delta']:+.4f} |"
        )
    lines.extend([
        "", "## Evidence limitations", "",
        *(f"- {item}" for item in audit["limitations"]),
        "", "No claims of clinical screening utility or prospective validation are made.",
    ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    audit = audit_operating_points(args.reports_dir)
    result = (json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
              if args.format == "json" else format_markdown(audit))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
