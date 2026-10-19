"""Synthetic, no-dataset uncertainty checks for ordinal model comparisons."""
import json

import numpy as np
import pytest

from aptos.evaluation.grade_uncertainty import paired_grade_bootstrap


@pytest.fixture
def paired_grades():
    truth = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 2])
    baseline = np.maximum(truth - 1, 0)
    candidate = truth.copy()
    patients = [f"patient-{i // 2}" for i in range(len(truth))]
    return truth, baseline, candidate, patients


@pytest.mark.pure
def test_identical_predictions_give_zero_width_intervals(paired_grades):
    truth, _, same, _ = paired_grades
    result = paired_grade_bootstrap(truth, same, same, unit="eye", n_resamples=150)
    for metric in result["metrics"].values():
        assert metric["ci_lower"] == metric["ci_upper"] == metric["difference"] == 0
        assert metric["valid_draws"] == 150
    json.dumps(result, allow_nan=False)


@pytest.mark.pure
def test_candidate_benefit_and_repeatable_bootstrap(paired_grades):
    truth, baseline, candidate, _ = paired_grades
    a = paired_grade_bootstrap(truth, baseline, candidate, unit="eye",
                               seed=12, n_resamples=200)
    b = paired_grade_bootstrap(truth, baseline, candidate, unit="eye",
                               seed=12, n_resamples=200)
    assert a == b
    assert a["metrics"]["qwk"]["difference"] > 0
    assert a["metrics"]["macro_f1"]["difference"] > 0
    assert a["metrics"]["qwk"]["ci_lower"] <= a["metrics"]["qwk"]["ci_upper"]


@pytest.mark.pure
def test_patient_cluster_uses_real_linkage(paired_grades):
    truth, baseline, candidate, ids = paired_grades
    report = paired_grade_bootstrap(truth, baseline, candidate, unit="patient",
                                    patient_ids=ids, n_resamples=150)
    assert report["analysis_unit"] == "patient"
    assert report["n_units"] == 6 and report["n_eyes"] == 12
    assert "patient-0" not in json.dumps(report)


@pytest.mark.pure
@pytest.mark.parametrize("kwargs,match", [
    ({"unit": "patient"}, "one patient ID"),
    ({"unit": "eye", "patient_ids": ["x"] * 12}, "select patient sampling"),
    ({"unit": "eye", "n_resamples": 10}, "at least 100"),
    ({"unit": "eye", "confidence": 1}, "confidence must"),
    ({"unit": "eye", "seed": True}, "seed must"),
])
def test_rejects_bad_resampling_contract(paired_grades, kwargs, match):
    truth, baseline, candidate, _ = paired_grades
    with pytest.raises(ValueError, match=match):
        paired_grade_bootstrap(truth, baseline, candidate, **kwargs)


@pytest.mark.pure
def test_rejects_undefined_qwk_single_grade(paired_grades):
    truth, baseline, candidate, _ = paired_grades
    with pytest.raises(ValueError, match="two observed grades"):
        paired_grade_bootstrap(np.zeros_like(truth), baseline, candidate,
                               unit="eye", n_resamples=150)
