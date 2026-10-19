"""Pure external-scoring report boundaries with no retinal dataset or GPU."""
import json

import numpy as np
import pytest

from aptos.evaluation.external_report import metric_string, summarize_external_grades

pytestmark = pytest.mark.pure


def test_missing_healthy_cohort_does_not_invent_healthy_rates():
    summary = summarize_external_grades([2, 3, 4], [2, 3, 4])
    assert summary["healthy_n"] == 0
    assert summary["healthy_called_healthy"] is None
    assert summary["healthy_called_referable"] is None
    assert summary["ref_specificity"] is None
    assert summary["ref_sensitivity"] == 1
    assert summary["per_class_recall"]["0"] is None
    json.dumps(summary, allow_nan=False)


def test_without_referable_eyes_sensitivity_is_not_reported_as_zero():
    summary = summarize_external_grades([0, 1, 0], [0, 0, 0])
    assert summary["ref_sensitivity"] is None
    assert summary["ref_specificity"] == 1
    assert summary["ref_precision"] is None
    assert summary["ref_referral_rate"] == 0
    json.dumps(summary, allow_nan=False)


def test_existing_defined_metrics_and_counts_are_preserved():
    summary = summarize_external_grades([0, 1, 2, 4], [0, 2, 2, 1])
    assert summary["healthy_n"] == 1
    assert summary["healthy_called_healthy"] == 1
    assert summary["healthy_called_referable"] == 0
    assert summary["severe_total"] == 1
    assert summary["severe_missed"] == 1
    assert summary["true_distribution"]["4"] == 1
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("predicted", [[0], [0, 1, 2], [float("nan"), 0]])
def test_misaligned_or_invalid_grades_are_rejected(predicted):
    with pytest.raises(ValueError):
        summarize_external_grades([0, 1], predicted)


def test_null_and_nonfinite_values_format_as_na():
    assert metric_string(None) == "N/A"
    assert metric_string(np.nan) == "N/A"
    assert metric_string(float("inf")) == "N/A"
    assert metric_string(0.25, percent=True) == "25.0%"
    assert metric_string(0.25) == "0.2500"


def test_markdown_report_handles_absent_healthy_stratum_without_nan():
    from aptos.evaluation.external import format_report

    values = summarize_external_grades([2, 3, 4], [2, 3, 4])
    report = {
        "dataset": "Synthetic", "n": 3, "variant": "baseline",
        "sweep": "test-only", "thresholds": [0.5, 1.5, 2.5, 3.5],
        **values,
    }
    markdown = format_report(report, {"qwk": float("nan")})
    assert "| referable specificity | N/A | N/A |" in markdown
    assert "no healthy eyes in this cohort" in markdown
    assert "nan" not in markdown.lower()


def test_one_grade_report_preserves_undefined_kappa_as_null():
    summary = summarize_external_grades([0, 0], [0, 0])
    assert summary["qwk"] is None
    assert summary["per_class_recall"]["4"] is None
    json.dumps(summary, allow_nan=False)
