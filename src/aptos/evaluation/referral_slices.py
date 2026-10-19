"""Aggregate-only false-referral and missed-referral audit by true DR grade."""
from __future__ import annotations

import numpy as np

from aptos.evaluation.paired_uncertainty import _grades


def grade_referral_breakdown(diagnosis, predicted_grade) -> dict[str, dict]:
    """Referral error counts by true grade with explicit absent-class nulls."""
    truth = _grades(diagnosis, "diagnosis")
    prediction = _grades(predicted_grade, "predicted_grade")
    if len(truth) != len(prediction):
        raise ValueError("grade referral breakdown requires aligned prediction lengths")
    result = {}
    for grade in range(5):
        subset = truth == grade
        n = int(subset.sum())
        referred = int((prediction[subset] >= 2).sum())
        is_referable = grade >= 2
        result[str(grade)] = {
            "n": n,
            "referred": referred,
            "not_referred": n - referred,
            "missed_referable": n - referred if is_referable else None,
            "true_grade_referral_sensitivity": referred / n if is_referable and n else None,
            "false_referral_fraction": referred / n if not is_referable and n else None,
        }
    return result


def paired_referral_discordance(diagnosis, baseline_grade, candidate_grade) -> dict[str, dict]:
    """Show paired referral misses, particularly grade 2, without eye IDs."""
    truth = _grades(diagnosis, "diagnosis")
    baseline = _grades(baseline_grade, "baseline_grade")
    candidate = _grades(candidate_grade, "candidate_grade")
    if len(truth) != len(baseline) or len(truth) != len(candidate):
        raise ValueError("paired referral discordance requires aligned prediction lengths")
    rows = {}
    for name, selected in (
        ("grade_2", truth == 2),
        ("grade_3", truth == 3),
        ("grade_4", truth == 4),
        ("all_referable", truth >= 2),
    ):
        n = int(selected.sum())
        missed_a = baseline[selected] < 2
        missed_b = candidate[selected] < 2
        rows[name] = {
            "n": n,
            "both_missed": int(np.sum(missed_a & missed_b)),
            "baseline_only_missed": int(np.sum(missed_a & ~missed_b)),
            "candidate_only_missed": int(np.sum(~missed_a & missed_b)),
            "neither_missed": int(np.sum(~missed_a & ~missed_b)),
        }
    return rows
