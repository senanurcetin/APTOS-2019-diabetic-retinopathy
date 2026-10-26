"""Test the follow-up contracts without asserting clinical validity from toy data."""
import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aptos.training.examination_followup import (
    align_source_filenames,
    metric_block,
    paired_auc_intervals,
    training_frames,
    verify_baseline,
)

pytestmark = pytest.mark.pure


@pytest.fixture
def predictions():
    truth = np.tile([0, 1, 2, 3], 10)
    return pd.DataFrame({
        "id_code": [f"eye{i}" for i in range(40)],
        "exam_id": [f"exam{i // 2}" for i in range(40)], "diagnosis": truth,
        "baseline_raw": truth.astype(float), "adjudicated_raw": truth.astype(float),
        "control_raw": truth.astype(float),
    })


def test_identical_scores_have_zero_auc_contrasts(predictions):
    result = paired_auc_intervals(predictions, draws=100)
    assert result["n_eyes"] == 40
    assert result["n_examinations"] == 20
    assert result["unit"] == "examination"
    for row in result["contrasts"].values():
        assert row == {"difference": 0, "ci_lower": 0, "ci_upper": 0}
    assert "eye0" not in json.dumps(result)


def test_model_score_change_has_correct_paired_sign(predictions):
    predictions["control_raw"] = -predictions.baseline_raw
    first = paired_auc_intervals(predictions, draws=100, seed=5)
    second = paired_auc_intervals(predictions, draws=100, seed=5)
    assert first == second
    assert first["contrasts"]["adjudicated_minus_control"]["difference"] == 1
    assert first["contrasts"]["control_minus_baseline"]["difference"] == -1


@pytest.mark.parametrize("defect", ["duplicate", "missing_exam", "nan_score", "one_class", "fractional"])
def test_unaligned_or_invalid_bootstrap_inputs_refused(predictions, defect):
    if defect == "duplicate":
        predictions.loc[0, "id_code"] = predictions.loc[1, "id_code"]
    elif defect == "missing_exam":
        predictions.loc[0, "exam_id"] = None
    elif defect == "nan_score":
        predictions.loc[0, "control_raw"] = np.nan
    elif defect == "fractional":
        predictions["diagnosis"] = 1.5
    else:
        predictions["diagnosis"] = 0
    with pytest.raises(ValueError):
        paired_auc_intervals(predictions, draws=100)


def test_only_training_targets_differ_between_arms():
    split = pd.DataFrame({"id_code": ["a", "b", "c", "d", "e"],
                          "diagnosis": [0, 2, 1, 3, 4],
                          "part": ["tune-train", "tune-train", "tune-valid", "tune-valid", "test"]})
    adjudicated, valid_a = training_frames(split, [4, 1, 4, 0, 0], "adjudicated")
    control, valid_b = training_frames(split, [4, 1, 4, 0, 0], "control")
    assert adjudicated.diagnosis.tolist() == [0, 2]
    assert control.diagnosis.tolist() == [4, 1]
    pd.testing.assert_frame_equal(valid_a, valid_b)
    assert valid_b.diagnosis.tolist() == [1, 3]
    assert "e" not in set(adjudicated.id_code) | set(valid_a.id_code)


def test_missing_baseline_metadata_rejected_before_tensor_loading(tmp_path):
    with pytest.raises(ValueError, match="identity"):
        verify_baseline(tmp_path, {"sweep": "another"})
    with pytest.raises(ValueError, match="missing or changed"):
        verify_baseline(tmp_path, {"sweep": tmp_path.name, "inputs_sha256": {"config.json": "wrong"}})


def test_frozen_cut_confusion_is_recomputed():
    result = metric_block(np.array([0, 1, 2, 3, 4]), np.array([0.1, 0.9, 1.1, 3.1, 3.9]),
                          [0.5, 1.5, 2.5, 3.5])
    assert result["referable"]["fn"] == 1
    assert result["referable"]["tp"] == 2
    assert result["auc"] == 1


def test_source_casing_is_resolved_without_changing_reference_grades(tmp_path):
    (tmp_path / "a.JPG").write_bytes(b"a")
    (tmp_path / "b.png").write_bytes(b"b")
    reference = pd.DataFrame({"image_id": ["a.jpg", "b.PNG"], "adjudicated_dr_grade": [2, 0]})
    aligned = align_source_filenames(reference, tmp_path)
    assert aligned.image_id.tolist() == ["a.JPG", "b.png"]
    assert aligned.adjudicated_dr_grade.tolist() == [2, 0]
    (tmp_path / "b.jpg").write_bytes(b"duplicate")
    with pytest.raises(ValueError, match="ambiguous"):
        align_source_filenames(reference, tmp_path)


def test_private_bootstrap_is_valid_python_and_has_fixed_sources():
    path = Path(__file__).resolve().parents[1] / "scripts/package_examination_followup.py"
    spec = importlib.util.spec_from_file_location("package_followup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ast.parse(module.BOOTSTRAP.replace("__COMMIT__", "a" * 40))
    assert "image_tree_sha256" in module.BOOTSTRAP
    assert "split_csv_sha256" in module.BOOTSTRAP
    assert "scikit-learn==1.8.0" in module.BOOTSTRAP
