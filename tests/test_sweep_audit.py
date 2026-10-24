"""A result audit must reject broken folds and optimistic summary edits."""
import json

import numpy as np
import pytest

from aptos.modeling.thresholds import clinically_costly_errors, grade_metrics, referable_confusion
from aptos.reporting.sweep_audit import audit_sweep

pytestmark = pytest.mark.pure


@pytest.fixture
def sweep(tmp_path):
    truth = np.tile(np.arange(5), 2)
    ids = np.array([f"eye_{i}" for i in range(len(truth))])
    cuts = [0.5, 1.5, 2.5, 3.5]
    metrics = grade_metrics(truth, truth)
    binary = referable_confusion(truth, truth)
    severe = clinically_costly_errors(truth, truth)
    for fold in range(1, 6):
        record = {"fold": fold, "best_epoch": 1, "epochs_run": 1,
                  "history": [{"epoch": 1, "valid_qwk": 1}], "valid_qwk": 1,
                  "thresholds": cuts, **severe,
                  **{f"test_{k}": v for k, v in metrics.items()},
                  **{f"test_{k}": v for k, v in binary.items()}}
        (tmp_path / f"fold{fold}.json").write_text(json.dumps(record))
        np.savez(tmp_path / f"fold{fold}.npz", test_raw=truth.astype(float),
                 test_true=truth, test_ids=ids)
    summary = {"folds": 5, "ensemble": {"thresholds": cuts, **metrics,
               **{f"referable_{k}": v for k, v in binary.items()}, **severe,
               "gain_over_fold_mean": 0}}
    for prefix in ("valid_qwk", "test_qwk", "test_acc", "test_macro_f1"):
        summary[f"{prefix}_mean"], summary[f"{prefix}_std"] = 1, 0
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "config.json").write_text(json.dumps({"cv.folds": 5, "train.epochs": 1}))
    return tmp_path


def test_complete_audit_exports_aggregates_without_eye_ids(sweep):
    result = audit_sweep(sweep, expected_n=10)
    assert result["recomputed_ensemble"]["qwk"] == 1
    assert result["recomputed_ensemble"]["referable_fn"] == 0
    assert len(result["inputs_sha256"]) == 12
    assert "eye_0" not in json.dumps(result)
    assert "test_ids" not in result


def test_rejects_missing_fold(sweep):
    (sweep / "fold5.npz").unlink()
    with pytest.raises(FileNotFoundError):
        audit_sweep(sweep)


def test_rejects_unaligned_identifiers(sweep):
    path = sweep / "fold3.npz"
    with np.load(path) as arrays:
        payload = {key: arrays[key] for key in arrays.files}
    payload["test_ids"] = payload["test_ids"][::-1]
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="not aligned"):
        audit_sweep(sweep)


def test_rejects_nonfinite_scores(sweep):
    path = sweep / "fold1.npz"
    with np.load(path) as arrays:
        payload = {key: arrays[key] for key in arrays.files}
    payload["test_raw"][0] = np.nan
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="invalid arrays"):
        audit_sweep(sweep)


def test_rejects_edited_ensemble_score(sweep):
    path = sweep / "summary.json"
    summary = json.loads(path.read_text())
    summary["ensemble"]["qwk"] = 0.98
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="inconsistent metric qwk"):
        audit_sweep(sweep)


def test_rejects_wrong_cohort_size(sweep):
    with pytest.raises(ValueError, match="row count"):
        audit_sweep(sweep, expected_n=11)
