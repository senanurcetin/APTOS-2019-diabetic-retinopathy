"""No-dataset grade-specific missed-referral and paired-discordance tests."""
import json

import pytest

from aptos.evaluation.referral_slices import (
    grade_referral_breakdown,
    paired_referral_discordance,
)

pytestmark = pytest.mark.pure


def test_grade_2_recall_and_grade_1_false_referrals():
    truth = [0, 1, 2, 2, 2, 3, 4]
    pred = [0, 2, 1, 2, 3, 1, 4]
    grades = grade_referral_breakdown(truth, pred)
    assert grades["2"]["n"] == 3
    assert grades["2"]["missed_referable"] == 1
    assert grades["2"]["true_grade_referral_sensitivity"] == pytest.approx(2 / 3)
    assert grades["1"]["false_referral_fraction"] == 1.0
    assert grades["0"]["missed_referable"] is None
    json.dumps(grades, allow_nan=False)


def test_absent_true_grade_is_explicit_null_not_zero_rate():
    grades = grade_referral_breakdown([0, 2], [0, 2])
    assert grades["1"]["n"] == 0
    assert grades["1"]["false_referral_fraction"] is None
    assert grades["3"]["true_grade_referral_sensitivity"] is None
    json.dumps(grades, allow_nan=False)


def test_paired_grade2_discordance_decomposes_every_case():
    truth = [0, 2, 2, 2, 3, 3, 4]
    base = [0, 1, 1, 2, 2, 0, 3]
    cand = [0, 1, 2, 1, 2, 1, 1]
    out = paired_referral_discordance(truth, base, cand)
    assert out["grade_2"] == {
        "n": 3, "both_missed": 1,
        "baseline_only_missed": 1,
        "candidate_only_missed": 1,
        "neither_missed": 0,
    }
    assert out["all_referable"]["n"] == 6
    for grade in out.values():
        assert grade["n"] == sum(val for key, val in grade.items() if key != "n")
    json.dumps(out, allow_nan=False)


def test_identical_models_have_no_paired_discordance():
    out = paired_referral_discordance([0, 2, 3], [0, 1, 3], [0, 1, 3])
    assert all(item["baseline_only_missed"] == item["candidate_only_missed"] == 0
               for item in out.values())


@pytest.mark.parametrize("analysis,values", [
    (grade_referral_breakdown, ([0, 1], [0])),
    (paired_referral_discordance, ([0, 2], [0, 2], [0])),
])
def test_reject_misaligned_predictions(analysis, values):
    with pytest.raises(ValueError, match="aligned prediction lengths"):
        analysis(*values)
