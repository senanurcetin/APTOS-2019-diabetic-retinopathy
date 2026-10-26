"""Result audits must reject changed cohorts, numbers and incomplete runs."""
import json

import pandas as pd
import pytest

from aptos.evaluation.followup_audit import aligned, assert_same, audit

pytestmark = pytest.mark.pure


def test_numeric_comparison_rejects_optimistic_or_nonfinite_edits():
    expected = {"auc": 0.8, "thresholds": [0.5, 1.5]}
    assert_same({"auc": 0.80000000001, "thresholds": [0.5, 1.5]}, expected)
    for actual in ({"auc": 0.9, "thresholds": [0.5, 1.5]},
                   {"auc": float("nan"), "thresholds": [0.5, 1.5]},
                   {"auc": 0.8}, {"auc": True, "thresholds": [0.5, 1.5]}):
        with pytest.raises(ValueError):
            assert_same(actual, expected)


def test_alignment_preserves_reference_and_rejects_missing_or_changed_eyes():
    reference = pd.DataFrame({"id_code": ["a", "b"], "diagnosis": [0, 2]})
    frames = reference.iloc[::-1].copy()
    assert aligned(frames, reference, ["id_code", "diagnosis"]).id_code.tolist() == ["a", "b"]
    with pytest.raises(ValueError):
        aligned(frames.iloc[:1], reference, ["id_code", "diagnosis"])
    frames.loc[frames.id_code.eq("a"), "diagnosis"] = 4
    with pytest.raises(ValueError):
        aligned(frames, reference, ["id_code", "diagnosis"])


def test_incomplete_or_unpinned_execution_is_not_a_result(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "RUNNING", "source_commit": "abc"}))
    (tmp_path / "results.json").write_text("{}")
    with pytest.raises(ValueError, match="incomplete"):
        audit(tmp_path, tmp_path / "split.csv", tmp_path, expected_source="abc")
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "COMPLETE", "source_commit": "wrong"}))
    with pytest.raises(ValueError, match="source differs"):
        audit(tmp_path, tmp_path / "split.csv", tmp_path, expected_source="abc")
