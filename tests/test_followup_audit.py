"""Result audits must reject changed cohorts, numbers and incomplete runs."""
import json

import numpy as np
import pandas as pd
import pytest

from aptos.evaluation.followup_audit import aligned, assert_same, audit
from aptos.training.examination_followup import ARMS, CONTRASTS, digest, metric_block

pytestmark = pytest.mark.pure


def test_numeric_comparison_rejects_optimistic_or_nonfinite_edits():
    expected = {"auc": 0.8, "thresholds": [0.5, 1.5]}
    assert_same({"auc": 0.80000000001, "thresholds": [0.5, 1.5]}, expected)
    for actual in ({"auc": 0.9, "thresholds": [0.5, 1.5]},
                   {"auc": float("nan"), "thresholds": [0.5, 1.5]},
                   {"auc": 0.8}, {"auc": True, "thresholds": [0.5, 1.5]}):
        with pytest.raises(ValueError):
            assert_same(actual, expected)


def test_publication_refuses_missing_or_mismatched_prediction_audit(tmp_path):
    from aptos.reporting.followup_report import render

    results, record = tmp_path / "result.json", tmp_path / "audit.json"
    results.write_text("{}")
    for audit_record in ({"status": "RUNNING"}, {"status": "PASS_PER_EYE_RECOMPUTATION",
                                                "inputs_sha256": {"results.json": "wrong"}}):
        record.write_text(json.dumps(audit_record))
        with pytest.raises(ValueError, match="matching per-eye"):
            render(results, record)


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


@pytest.fixture
def completed_run(tmp_path, monkeypatch):
    # Statistical bootstrap behavior is tested separately; here exercise the
    # downloaded-file audit, including deliberately forged aggregate numbers.
    monkeypatch.setattr("aptos.evaluation.followup_audit.paired_auc_intervals", lambda frame: {"fixture": 1})
    monkeypatch.setattr("aptos.evaluation.followup_audit.paired_bootstrap", lambda *a, **k: {"fixture": 2})
    out, base = tmp_path / "run", tmp_path / "base"
    out.mkdir()
    base.mkdir()
    cuts = [0.5, 1.5, 2.5, 3.5]
    frame = pd.DataFrame({"id_code": [f"eye{i}" for i in range(872)],
                          "diagnosis": np.arange(872) % 5,
                          "exam_id": [f"exam{i // 2 if i < 870 else i - 435}" for i in range(872)],
                          "part": "test"})
    split_path = tmp_path / "split.csv"
    frame.to_csv(split_path, index=False)
    scores = frame.diagnosis.to_numpy(dtype=np.float32) + np.float32(0.1)
    report = {"fixed_original_cuts": {}, "validation_fitted_cuts": {},
              "aptos_retention": {"fixed_original_cuts": {}, "messidor_validation_cuts": {}},
              "paired_auc": {"fixture": 1},
              "paired_fixed_cut_referral": {key: {"fixture": 2} for key in CONTRASTS}}
    raw = frame.drop(columns="part").copy()
    for arm in ARMS:
        raw[f"{arm}_raw"] = scores
        raw[f"{arm}_grade_frozen"] = raw.diagnosis
        raw[f"{arm}_grade_validation"] = raw.diagnosis
        report["fixed_original_cuts"][arm] = metric_block(raw.diagnosis, scores, cuts)
        report["validation_fitted_cuts"][arm] = metric_block(raw.diagnosis, scores, cuts)
    raw.to_csv(out / "test_predictions.csv", index=False)
    frame.assign(baseline_raw=scores, control_grade=frame.diagnosis).to_csv(out / "baseline_all_predictions.csv", index=False)
    for name in CONTRASTS:
        raw[["id_code", "exam_id", "diagnosis"]].assign(
            baseline_grade=raw.diagnosis, candidate_grade=raw.diagnosis,
        ).to_csv(out / f"paired_{name}.csv", index=False)
    for arm in ("adjudicated", "control"):
        (out / arm).mkdir()
        for fold in range(1, 6):
            record = {"fold": fold, "history": [{"epoch": epoch, "valid_qwk": epoch / 10} for epoch in range(1, 9)],
                      "best_epoch": 8, "valid_qwk": 0.8, "thresholds": cuts}
            (out / arm / f"fold{fold}.json").write_text(json.dumps(record))
    retention = raw.iloc[:366][["id_code", "diagnosis"]].copy()
    saved_raw = retention.diagnosis.to_numpy(dtype=np.float32) + np.float32(0.1)
    for fold in range(1, 6):
        np.savez(base / f"fold{fold}.npz", test_ids=retention.id_code.to_numpy(dtype=str),
                 test_true=retention.diagnosis.to_numpy(), test_raw=saved_raw)
    current = np.mean([saved_raw] * 5, axis=0)
    for arm in ARMS:
        retention[f"{arm}_raw"] = current
        report["aptos_retention"]["fixed_original_cuts"][arm] = metric_block(retention.diagnosis, current, cuts)
        report["aptos_retention"]["messidor_validation_cuts"][arm] = metric_block(retention.diagnosis, current, cuts)
    report["aptos_retention"].update(baseline_rescore_max_abs_difference=0, baseline_rescore_grade_mismatches=0)
    retention.to_csv(out / "aptos_retention_predictions.csv", index=False)
    (base / "summary.json").write_text(json.dumps({"ensemble": {"thresholds": cuts}}))
    report["prediction_sha256"] = digest(out / "test_predictions.csv")
    (out / "results.json").write_text(json.dumps(report))
    manifest = {"status": "COMPLETE", "source_commit": "abc", "result_sha256": digest(out / "results.json"),
                "split_sha256": digest(split_path), "validation_targets": "adjudicated in both arms; only training targets vary",
                "fine_tuning": {"lr": 1e-4, "weight_decay": 1e-4, "epochs": 8, "seed": 42},
                "fine_tuned_checkpoint_sha256": {f"{arm}/fold{i}.pt": "0" * 64 for arm in ("adjudicated", "control") for i in range(1, 6)}}
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out, split_path, base


def test_complete_downloaded_predictions_recompute(completed_run):
    result = audit(*completed_run, expected_source="abc")
    assert result["status"] == "PASS_PER_EYE_RECOMPUTATION"
    assert result["new_fold_histories_checked"] == 10


def test_optimistic_result_edit_fails_even_with_updated_checksums(completed_run):
    out, split, base = completed_run
    report = json.loads((out / "results.json").read_text())
    report["fixed_original_cuts"]["adjudicated"]["qwk"] = 0.99
    (out / "results.json").write_text(json.dumps(report))
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["result_sha256"] = digest(out / "results.json")
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="numeric disagreement"):
        audit(out, split, base, expected_source="abc")
