"""Evidence integrity and uncertainty checks for external cohorts (no GPU required)."""
import copy
import json
from pathlib import Path

import pytest

from aptos.reporting.clinical_audit import cohort_summary, format_markdown, wilson_interval

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.pure
def test_wilson_sanity_and_limits():
    low, high = wilson_interval(0, 100)
    assert low == 0 and 0 < high < 0.05
    low, high = wilson_interval(100, 100)
    assert high == pytest.approx(1.0) and 0.95 < low < 1
    low, high = wilson_interval(50, 100)
    assert 0.4 < low < 0.5 < high < 0.6


@pytest.mark.pure
@pytest.mark.parametrize("success,total", [(1, 0), (2, 1), (-1, 4)])
def test_wilson_rejects_invalid_counts(success, total):
    with pytest.raises(ValueError):
        wilson_interval(success, total)


@pytest.mark.pure
def test_committed_external_cohorts_match_counts():
    cohorts = []
    for filename in ("external_validation.json", "external_validation_messidor2.json"):
        record = json.loads((ROOT / "reports" / filename).read_text(encoding="utf-8"))
        cohorts.append(cohort_summary(record))
    assert [c["n"] for c in cohorts] == [455, 1744]
    assert [c["missed_referable"] for c in cohorts] == [35, 307]
    assert cohorts[0]["sensitivity"] > cohorts[1]["sensitivity"]
    rendered = format_markdown(cohorts)
    assert "88.5%" in rendered and "32.8%" in rendered
    assert "NOT patient-cluster-adjusted" in rendered


@pytest.mark.pure
def test_audit_rejects_tampered_metrics():
    record = json.loads((ROOT / "reports" / "external_validation.json").read_text())
    wrong_counts = copy.deepcopy(record)
    wrong_counts["ref_fn"] += 1
    with pytest.raises(ValueError, match="sums"):
        cohort_summary(wrong_counts)
    wrong_sensitivity = copy.deepcopy(record)
    wrong_sensitivity["ref_sensitivity"] = 0.99
    with pytest.raises(ValueError, match="sensitivity"):
        cohort_summary(wrong_sensitivity)
