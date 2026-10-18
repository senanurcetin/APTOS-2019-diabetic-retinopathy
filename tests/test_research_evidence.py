"""Independent, dataset-free regression tests for committed APTOS evidence."""
import csv
import json
from pathlib import Path

import pytest

from aptos.reporting.research_evidence import (
    audit_external,
    audit_sweeps,
    build_evidence,
    file_sha256,
    render_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


@pytest.mark.pure
def test_audit_committed_results():
    result = build_evidence(REPORTS)
    assert len(result["cross_validation"]) == 4
    assert all(row["folds"] == 5 for row in result["cross_validation"])
    assert [row["n"] for row in result["external"]] == [455, 1744]
    assert [row["missed_referable"] for row in result["external"]] == [35, 307]
    assert result["external_sensitivity_gap_percentage_points"] > 50
    assert len(result["inputs_sha256"]["runs.csv"]) == 64
    assert file_sha256(REPORTS / "runs.csv") == result["inputs_sha256"]["runs.csv"]
    assert "not a clinical product" in render_markdown(result)


@pytest.mark.pure
def test_audit_is_deterministic():
    a = json.dumps(build_evidence(REPORTS), sort_keys=True)
    b = json.dumps(build_evidence(REPORTS), sort_keys=True)
    assert a == b


@pytest.mark.pure
def test_detects_tampered_fold_qwk(tmp_path):
    with (REPORTS / "runs.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)
    next(row for row in rows if row["split"] == "cv-fold")["test_qwk"] = "0.123456"
    target = tmp_path / "runs.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="fold mean"):
        audit_sweeps(target)


@pytest.mark.pure
def test_detects_tampered_external_grade_counts(tmp_path):
    for name in ("external_validation.json", "external_validation_messidor2.json"):
        data = json.loads((REPORTS / name).read_text(encoding="utf-8"))
        if name == "external_validation_messidor2.json":
            data["predicted_distribution"]["4"] += 1
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="distribution does not sum"):
        audit_external(tmp_path)


@pytest.mark.pure
def test_detects_external_model_mismatch(tmp_path):
    for name in ("external_validation.json", "external_validation_messidor2.json"):
        data = json.loads((REPORTS / name).read_text(encoding="utf-8"))
        if name == "external_validation_messidor2.json":
            data["thresholds"][0] += 0.1
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="different model or thresholds"):
        audit_external(tmp_path)
