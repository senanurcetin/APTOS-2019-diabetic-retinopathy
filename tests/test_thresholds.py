"""Tests for the ordinal metrics and the threshold search.

These decide every number the project reports and had no test coverage at all,
because they used to live in a module that imports torch.
"""
import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from aptos.config import ThresholdConfig
from aptos.modeling.thresholds import (
    apply_thresholds,
    clinically_costly_errors,
    grade_metrics,
    optimize_thresholds,
    per_class_recall,
    qwk,
    referable_confusion,
)

pytestmark = pytest.mark.pure


# ------------------------------------------------------------------------- qwk

def test_qwk_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, 200)
    y_pred = rng.integers(0, 5, 200)
    assert qwk(y_true, y_pred) == pytest.approx(
        cohen_kappa_score(y_true, y_pred, weights="quadratic")
    )


def test_qwk_is_one_for_perfect_prediction():
    y = np.array([0, 1, 2, 3, 4, 4, 2, 0])
    assert qwk(y, y) == pytest.approx(1.0)


def test_qwk_punishes_distant_errors_more_than_near_ones():
    """The whole reason QWK is used instead of accuracy: the grades are ordered."""
    y_true = np.array([0, 1, 2, 3, 4] * 8)
    near = np.clip(y_true + 1, 0, 4)
    far = np.clip(y_true + 3, 0, 4)
    assert qwk(y_true, near) > qwk(y_true, far)


# ------------------------------------------------------------ apply_thresholds

def test_apply_thresholds_maps_to_grade_range():
    raw = np.linspace(-1, 6, 50)
    graded = apply_thresholds(raw, [0.5, 1.5, 2.5, 3.5])
    assert graded.min() >= 0
    assert graded.max() <= 4


def test_apply_thresholds_is_monotone_in_raw():
    """A higher regression output can never produce a lower grade."""
    raw = np.sort(np.random.default_rng(1).normal(2, 1.5, 100))
    graded = apply_thresholds(raw, [0.4, 1.7, 2.3, 3.8])
    assert np.all(np.diff(graded) >= 0)


# ---------------------------------------------------------- optimize_thresholds

def test_optimize_recovers_a_known_offset():
    """A systematic bias the default cut points get wrong should be undone.

    The raw output carries a little noise, which is what real regression output
    looks like and what the search needs in order to move at all - see
    `test_optimize_is_local_and_stalls_without_nearby_points` below.
    """
    rng = np.random.default_rng(11)
    y_true = np.repeat(np.arange(5), 40)
    raw = y_true + 0.8 + rng.normal(0, 0.15, y_true.size)
    default_score = qwk(y_true, apply_thresholds(raw, ThresholdConfig().start))
    thresholds, score = optimize_thresholds(y_true, raw)
    assert score > default_score
    assert score > 0.97
    # The recovered cut points should sit near the true boundaries at x.3
    assert np.all(np.abs(thresholds - np.array([1.3, 2.3, 3.3, 4.3])) < 0.35)


def test_optimize_is_local_and_stalls_without_nearby_points():
    """Documents a real limitation of the search rather than hiding it.

    The largest step is 0.12. If no raw value lies within one step of a cut
    point, no single move changes any prediction, every candidate scores the
    same as the incumbent, and the search terminates at its starting point -
    even when a much better solution exists.

    Here the raw values sit exactly 0.3 above the default cut points, so QWK
    1.0 is reachable at [1.3, 2.3, 3.3, 4.3] and the search returns 0.8.

    This is not triggered by real model output, which is continuous and dense
    enough that small moves always reclassify something. The behaviour is left
    as it is on purpose: the six recorded single-split runs used this exact
    search, and changing it would make the cross-validation results
    incomparable with the history they exist to strengthen.
    """
    y_true = np.repeat(np.arange(5), 40)
    raw = y_true + 0.8  # discrete: only five distinct values, none near a cut point

    thresholds, score = optimize_thresholds(y_true, raw)

    assert np.array_equal(thresholds, np.array(ThresholdConfig().start))
    assert score == pytest.approx(0.8)
    # ... while a better answer was available all along.
    assert qwk(y_true, apply_thresholds(raw, [1.3, 2.3, 3.3, 4.3])) == pytest.approx(1.0)


def test_optimize_returns_strictly_increasing_thresholds():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 5, 300)
    raw = y_true + rng.normal(0, 0.6, 300)
    thresholds, _ = optimize_thresholds(y_true, raw)
    assert np.all(np.diff(thresholds) > ThresholdConfig().min_gap)


def test_optimize_never_returns_worse_than_the_starting_point():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 5, 200)
    raw = rng.normal(2, 1, 200)
    start_score = qwk(y_true, apply_thresholds(raw, ThresholdConfig().start))
    _, score = optimize_thresholds(y_true, raw)
    assert score >= start_score


def test_optimize_reports_the_score_of_the_thresholds_it_returns():
    """A returned score that does not belong to the returned cut points would
    silently misreport every validation figure."""
    rng = np.random.default_rng(4)
    y_true = rng.integers(0, 5, 250)
    raw = y_true + rng.normal(0, 0.8, 250)
    thresholds, score = optimize_thresholds(y_true, raw)
    assert qwk(y_true, apply_thresholds(raw, thresholds)) == pytest.approx(score)


def test_optimize_is_deterministic():
    rng = np.random.default_rng(5)
    y_true = rng.integers(0, 5, 150)
    raw = y_true + rng.normal(0, 0.7, 150)
    a, sa = optimize_thresholds(y_true, raw)
    b, sb = optimize_thresholds(y_true, raw)
    assert np.array_equal(a, b) and sa == sb


def test_optimize_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        optimize_thresholds(np.array([]), np.array([]))


def test_optimize_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        optimize_thresholds(np.array([0, 1, 2]), np.array([0.1, 0.2]))


# --------------------------------------------------------------- report views

def test_per_class_recall_handles_an_absent_grade():
    """The test split holds 17 Severe cases; a split with none must not crash."""
    y_true = np.array([0, 0, 1, 2, 2])  # no grade 3 or 4
    y_pred = np.array([0, 0, 1, 2, 1])
    recalls = per_class_recall(y_true, y_pred)
    assert recalls[0] == pytest.approx(1.0)
    assert np.isnan(recalls[3]) and np.isnan(recalls[4])


def test_grade_metrics_reports_qwk_and_macro_f1_together():
    """QWK alone hides minority-class weakness - the pairing is the point."""
    y_true = np.repeat(np.arange(5), 20)
    y_pred = np.clip(y_true + np.tile([0, 0, 1, 0], 25), 0, 4)
    m = grade_metrics(y_true, y_pred)
    assert {"qwk", "accuracy", "macro_f1"} <= set(m)
    assert all(f"recall_grade_{g}" in m for g in range(5))


def test_clinically_costly_errors_counts_missed_referrals():
    y_true = np.array([4, 3, 3, 0, 1])
    y_pred = np.array([0, 1, 3, 0, 1])  # two severe cases called non-referable
    out = clinically_costly_errors(y_true, y_pred)
    assert out == {"severe_total": 3, "severe_missed": 2}


def test_referable_confusion_matches_a_hand_counted_case():
    y_true = np.array([0, 1, 2, 3, 4, 0, 2])
    y_pred = np.array([0, 2, 2, 1, 4, 0, 1])
    #  referable truth: F F T T T F T   (4 positives)
    #  referable pred : F T T F T F F   (3 positives)
    out = referable_confusion(y_true, y_pred)
    assert (out["tp"], out["fp"], out["tn"], out["fn"]) == (2, 1, 2, 2)
    assert out["sensitivity"] == pytest.approx(0.5)
    assert out["specificity"] == pytest.approx(2 / 3)
    assert out["referral_rate"] == pytest.approx(3 / 7)


def test_referable_is_insensitive_to_confusion_within_referable_grades():
    """The clinical framing's whole advantage: confusing 3 with 4 costs nothing.

    Non-referable cases are included so specificity is defined; comparing two
    dicts that both carry NaN would pass or fail for the wrong reason.
    """
    y_true = np.array([0, 1, 2, 3, 4, 4, 3])
    perfect = y_true.copy()
    # Grades 0/1 unchanged; the referable grades are shuffled among themselves.
    shuffled = np.array([0, 1, 4, 4, 2, 3, 2])

    assert not np.array_equal(perfect, shuffled)  # the predictions really differ
    assert referable_confusion(y_true, perfect) == referable_confusion(y_true, shuffled)
    # and the five-way metric does notice the difference
    assert qwk(y_true, perfect) > qwk(y_true, shuffled)
