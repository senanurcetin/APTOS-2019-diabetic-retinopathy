"""Render only the actual, independently recomputed examination follow-up."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aptos.training.examination_followup import ARMS, digest


def render(results_path: Path, audit_path: Path) -> str:
    results = json.loads(results_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (audit.get("status") != "PASS_PER_EYE_RECOMPUTATION"
            or audit.get("inputs_sha256", {}).get("results.json") != digest(results_path)):
        raise ValueError("a matching per-eye recomputation audit is required before publication")
    primary = results["paired_auc"]["contrasts"]["adjudicated_minus_control"]
    if primary["ci_lower"] > 0:
        interpretation = "The adjudicated arm has higher referable AUC than the control in this fitted comparison."
    elif primary["ci_upper"] < 0:
        interpretation = "The control arm has higher referable AUC than the adjudicated arm in this fitted comparison."
    else:
        interpretation = "The interval includes zero; this comparison does not establish a clear referable-AUC advantage."
    lines = [
        "# Messidor-2: examination-disjoint follow-up results", "",
        "Noncommercial exploratory research. New paired fine-tuning run; distinct from the historical AUC 0.925 experiment.", "",
        f"Baseline: `{results['baseline_sweep']}`. Test: **{results['n_test_eyes']} eyes / "
        f"{results['n_test_examinations']} supplied examinations**. "
        f"[Protocol](../../docs/MESSIDOR2_V2_PROTOCOL.md) · [frozen allocation](../../docs/MESSIDOR2_FOLLOWUP.md).", "",
        "## Primary contrast", "",
        f"Adjudicated minus control referable AUC: **{primary['difference']:+.4f}**, "
        f"95% examination-bootstrap interval **[{primary['ci_lower']:+.4f}, {primary['ci_upper']:+.4f}]**.", "",
        interpretation + " The cohort was inspected previously; this is not fresh external confirmation.", "",
        "## Same original APTOS cuts for all arms", "",
        "AUC uses continuous scores. Classification metrics use the same original APTOS thresholds, without external refitting.", "",
        "| Arm | Referable AUC | QWK | Macro F1 | Sensitivity | Specificity | False negatives |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = results["fixed_original_cuts"][arm]
        referral = row["referable"]
        lines.append(f"| {arm} | {row['auc']:.4f} | {row['qwk']:.4f} | {row['macro_f1']:.4f} | "
                     f"{referral['sensitivity']:.1%} | {referral['specificity']:.1%} | {referral['fn']} |")
    lines += ["", "Original cuts: `" + ", ".join(f"{x:.6g}" for x in
              results["fixed_original_cuts"]["baseline"]["thresholds"]) + "`.", "",
              "## Separately fitted Messidor validation cuts", "",
              "Both arms use adjudicated validation targets; only training targets differ. "
              "The control's training labels are the unchanged model's own grades.", "",
              "| Arm | QWK | Sensitivity | Specificity | False negatives | Validation cuts |",
              "|---|---:|---:|---:|---:|---|"]
    for arm in ARMS:
        row = results["validation_fitted_cuts"][arm]
        ref = row["referable"]
        cuts = ", ".join(f"{value:.4f}" for value in row["thresholds"])
        lines.append(f"| {arm} | {row['qwk']:.4f} | {ref['sensitivity']:.1%} | "
                     f"{ref['specificity']:.1%} | {ref['fn']} | {cuts} |")
    lines += ["", "## Severity support and recall at original cuts", "",
              "| True grade | Eyes | Baseline recall | Adjudicated recall | Control recall |",
              "|---|---:|---:|---:|---:|"]
    for grade in range(5):
        recalls = [results["fixed_original_cuts"][arm][f"recall_grade_{grade}"] for arm in ARMS]
        lines.append(f"| {grade} | {audit['test_grade_counts'][str(grade)]} | "
                     + " | ".join(f"{value:.1%}" for value in recalls) + " |")
    lines += ["", "## APTOS retention: reused 366-eye holdout", "",
              "These diagnostics were not used for epoch selection or a new model promotion. "
              "All rows below use the original APTOS cuts.", "",
              "| Arm | Referable AUC | QWK | Accuracy | Macro F1 |",
              "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        row = results["aptos_retention"]["fixed_original_cuts"][arm]
        lines.append(f"| {arm} | {row['auc']:.4f} | {row['qwk']:.4f} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |")
    lines += ["", f"Baseline re-inference differs from its saved scores by at most "
              f"{results['aptos_retention']['baseline_rescore_max_abs_difference']:.6f} raw-score units, "
              f"with {results['aptos_retention']['baseline_rescore_grade_mismatches']} grade mismatch(es). "
              "This comparison consistently uses the re-inferred baseline; numerical parity is not bit-exact."]
    lines += ["", "## Provenance and limitations", "",
              f"GPU source commit: `{audit['source_commit']}`. Ten new fold histories "
              "were checked; results and paired intervals were independently recomputed from saved scores. "
              "Weight tensors were not loaded by the independent audit.", "",
              "Bootstrap: 2,000 paired examination draws, seed 42, percentile 95% intervals. "
              "The primary contrast was fixed before the run; secondary contrasts are descriptive. "
              "Validation has only two grade-4 eyes; test has 24. Examination pairing is a third-party "
              "ADCIS transcription, not verified longitudinal patient identity.", ""]
    lines += [f"- {item}" for item in results["limitations"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    content = render(args.results, args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
