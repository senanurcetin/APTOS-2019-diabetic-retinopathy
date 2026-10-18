"""Offline reproducibility tests for competing referral operating rules."""
import json
from pathlib import Path

import pytest

from aptos.reporting.operating_point_audit import (
    audit_operating_points,
    format_markdown,
    reconstruct_confusion,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


@pytest.mark.pure
def test_committed_operating_point_counts_are_consistent():
    audit = audit_operating_points(REPORTS)
    assert audit["research_only"] is True
    assert len(audit["applied_cohorts"]) == 4
    by_cohort = {row["cohort"]: row for row in audit["applied_cohorts"]}
    assert (by_cohort["APTOS test"]["tp"], by_cohort["APTOS test"]["fn"]) == (126, 11)
    assert (by_cohort["IDRiD"]["tp"], by_cohort["IDRiD"]["fn"]) == (248, 56)
    assert (by_cohort["Messidor-2"]["tp"], by_cohort["Messidor-2"]["fn"]) == (119, 338)
    assert all(len(row["sensitivity_wilson95"]) == 2 for row in by_cohort.values())
    comparisons = {row["cohort"]: row for row in audit["grade_rule_vs_raw_cut"]}
    assert comparisons["IDRiD"]["grade_rule_missed"] == 35
    assert comparisons["IDRiD"]["difference_missed_raw_minus_grade"] == 21
    assert comparisons["Messidor-2"]["grade_rule_missed"] == 307
    assert comparisons["Messidor-2"]["difference_missed_raw_minus_grade"] == 31
    assert "extra" not in str(audit["limitations"]).lower()
    assert len(audit["transfer"]) == 2
    assert audit["transfer"][1]["specificity_delta"] < -0.8
    report = format_markdown(audit)
    assert "No claims of clinical screening utility" in report
    assert "338" in report and "307" in report


@pytest.mark.pure
def test_report_is_deterministic():
    left = json.dumps(audit_operating_points(REPORTS), sort_keys=True)
    right = json.dumps(audit_operating_points(REPORTS), sort_keys=True)
    assert left == right


def _write_test_reports(tmp_path):
    for name in (
        "calibration.json",
        "external_validation.json",
        "external_validation_messidor2.json",
    ):
        content = (REPORTS / name).read_text(encoding="utf-8")
        (tmp_path / name).write_text(content, encoding="utf-8")
    return tmp_path


@pytest.mark.pure
def test_rejects_tampered_referral_rate(tmp_path):
    root = _write_test_reports(tmp_path)
    path = root / "calibration.json"
    report = json.loads(path.read_text())
    report["applied"][1]["referral rate"] = 0.01
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="referral rate contradicts"):
        audit_operating_points(root)


@pytest.mark.pure
def test_rejects_non_integer_counts(tmp_path):
    root = _write_test_reports(tmp_path)
    path = root / "calibration.json"
    report = json.loads(path.read_text())
    report["applied"][1]["prevalence"] = 0.375
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="integer count"):
        audit_operating_points(root)


@pytest.mark.pure
def test_rejects_changed_external_reference_cohort(tmp_path):
    root = _write_test_reports(tmp_path)
    path = root / "calibration.json"
    report = json.loads(path.read_text())
    report["applied"][2]["n"] = 454
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        audit_operating_points(root)


@pytest.mark.pure
def test_rejects_tampered_site_transfer_baseline(tmp_path):
    root = _write_test_reports(tmp_path)
    path = root / "calibration.json"
    report = json.loads(path.read_text())
    report["transfer"][0]["sensitivity_aptos_cut"] = 0.8
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="original sensitivity contradicts"):
        audit_operating_points(root)


@pytest.mark.pure
@pytest.mark.parametrize("field,value", [
    ("sensitivity", 1.1),
    ("specificity", -0.1),
    ("prevalence", float("nan")),
])
def test_rejects_invalid_rates(field, value):
    original = json.loads((REPORTS / "calibration.json").read_text())
    record = dict(original["applied"][0])
    record[field] = value
    with pytest.raises(ValueError, match="probability"):
        reconstruct_confusion(record)


@pytest.mark.pure
def test_rejects_duplicate_calibration_cohorts(tmp_path):
    root = _write_test_reports(tmp_path)
    path = root / "calibration.json"
    report = json.loads(path.read_text())
    report["applied"].append(dict(report["applied"][1]))
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="duplicate calibration cohort"):
        audit_operating_points(root)
