"""Ordinal metrics and the threshold search, with no torch import.

Splitting these out of `train.py` is not cosmetic. They previously lived in a
module that imports torch and timm, so CI - which installs neither - could not
test them at all. `qwk` and `optimize_thresholds` decide every number this
project reports, and they were the least tested code in it.

Everything here is numpy and scikit-learn, so it runs in the fast CI job on
three Python versions.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

from aptos.config import N_GRADES, REFERABLE_FROM, ThresholdConfig


def qwk(y_true, y_pred) -> float:
    """Quadratic weighted kappa.

    The right metric for this task because the grades are ordered: predicting
    Severe when the truth is Proliferative is a smaller error than predicting
    No DR, and QWK is the standard metric that says so.

    It is also why macro F1 is always reported alongside. QWK forgives
    neighbouring-grade errors, which is exactly where this model is weakest -
    QWK 0.90 against macro F1 0.57 is the gap that tells you so.
    """
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def apply_thresholds(raw, thresholds) -> np.ndarray:
    """Turn continuous regression output into ordinal grades."""
    return np.digitize(np.asarray(raw, dtype=float), np.asarray(thresholds, dtype=float))


def optimize_thresholds(
    y_true, raw, cfg: ThresholdConfig | None = None
) -> tuple[np.ndarray, float]:
    """Coordinate search for the cut points that map regression output onto 0-4.

    Deliberately unchanged in behaviour from the original implementation so the
    cross-validation results stay comparable with the six recorded single-split
    runs; the magic numbers simply moved into config.

    Fit this on validation only. Because the thresholds are themselves fitted,
    validation QWK is partly a quantity we optimised - which is the documented
    reason CLAHE won on validation in all three seeds and did not carry to test.

    Known limitation, measured and left in place. The search is purely local and
    its largest step is 0.12, so if no raw value lies within one step of a cut
    point, every candidate scores the same as the incumbent and the search
    returns its starting point. On a synthetic set whose outputs sit exactly
    0.3 above the default cut points it returns QWK 0.80 while 1.00 was
    available. Real regression output is continuous and dense enough that this
    does not arise, and changing the search would break comparability with the
    recorded runs - so it is documented rather than fixed. See
    `tests/test_thresholds.py::test_optimize_is_local_and_stalls_without_nearby_points`.
    """
    cfg = cfg or ThresholdConfig()
    y_true = np.asarray(y_true)
    raw = np.asarray(raw, dtype=float)

    if y_true.shape[0] != raw.shape[0]:
        raise ValueError(f"length mismatch: {y_true.shape[0]} labels, {raw.shape[0]} predictions")
    if y_true.size == 0:
        raise ValueError("cannot optimise thresholds on an empty set")

    best = np.array(cfg.start, dtype=float)
    best_score = qwk(y_true, apply_thresholds(raw, best))

    for _ in range(cfg.rounds):
        improved = False
        for i in range(len(best)):
            for delta in cfg.deltas:
                cand = best.copy()
                cand[i] += delta
                if not np.all(np.diff(cand) > cfg.min_gap):
                    continue
                score = qwk(y_true, apply_thresholds(raw, cand))
                if score > best_score:
                    best, best_score, improved = cand, score, True
        if not improved:
            break

    return best, best_score


# ----------------------------------------------------------------- report views

def per_class_recall(y_true, y_pred) -> dict[int, float]:
    """Recall for every grade, including grades the model never predicts.

    Reported because the test split holds 366 images and only 17 of them are
    Severe: a single case moves that class's recall by 0.06, so these figures
    are indicative rather than precise, and saying so requires showing them.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {}
    for grade in range(N_GRADES):
        mask = y_true == grade
        out[grade] = float((y_pred[mask] == grade).mean()) if mask.any() else float("nan")
    return out


def grade_metrics(y_true, y_pred) -> dict[str, float]:
    """The standard block reported for every run."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    metrics = {
        "qwk": qwk(y_true, y_pred),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(N_GRADES)), zero_division=0)),
    }
    for grade, recall in per_class_recall(y_true, y_pred).items():
        metrics[f"recall_grade_{grade}"] = recall
    return metrics


def clinically_costly_errors(y_true, y_pred, *, true_from: int = 3, pred_upto: int = 1) -> dict[str, int]:
    """Cases graded Severe or worse that the model called Mild or No DR.

    The error that actually matters in screening: a missed referral, not a
    confusion between two referable grades.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    severe = y_true >= true_from
    missed = int((severe & (y_pred <= pred_upto)).sum())
    return {"severe_total": int(severe.sum()), "severe_missed": missed}


def referable_confusion(y_true, y_pred, *, threshold: int = REFERABLE_FROM) -> dict[str, float]:
    """Binary referable-DR view: sensitivity, specificity, and the workload it implies.

    Screening tolerates false positives far better than missed referrals, so
    sensitivity is the constraint and specificity is what it costs.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    true_ref, pred_ref = y_true >= threshold, y_pred >= threshold

    tp = int((true_ref & pred_ref).sum())
    fn = int((true_ref & ~pred_ref).sum())
    tn = int((~true_ref & ~pred_ref).sum())
    fp = int((~true_ref & pred_ref).sum())

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        # Share of the population a clinician would have to look at.
        "referral_rate": float(pred_ref.mean()),
    }
