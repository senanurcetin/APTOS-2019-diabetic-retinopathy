"""Pure statistical regression tests for paired, patient-aware comparisons."""
import csv
import json

import numpy as np
import pytest

from aptos.evaluation.paired_uncertainty import (
    paired_bootstrap,
    read_paired_csv,
)


def _cohort():
    # 10 patients, 2 eyes each; repeat labels per patient for demonstrable
    # cluster-vs-eye sampling differences, but the method also allows mixed eyes.
    truth = np.repeat([0, 0, 1, 2, 3, 4, 0, 2, 3, 1], 2)
    baseline = truth.copy()
    baseline[[6, 10, 14, 16]] = [0, 1, 1, 1]
    candidate = truth.copy()
    candidate[[6, 10, 12, 18]] = [2, 2, 2, 2]
    patients = [f"patient-{i // 2}" for i in range(len(truth))]
    return truth, baseline, candidate, patients


@pytest.mark.pure
def test_same_eye_comparison_is_deterministic_and_bounded():
    y, a, b, patients = _cohort()
    first = paired_bootstrap(y, a, b, unit="patient",
                             patient_ids=patients, n_resamples=400, seed=8)
    again = paired_bootstrap(y, a, b, unit="patient",
                             patient_ids=patients, n_resamples=400, seed=8)
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)
    assert first["n_eyes"] == 20
    assert first["n_units"] == 10
    assert first["n_referable"] == 10
    assert first["draws_used"] >= 320
    for metric in first["metrics"].values():
        assert -1 <= metric["ci_lower"] <= metric["ci_upper"] <= 1
        assert metric["difference"] == pytest.approx(metric["candidate"] - metric["baseline"])


@pytest.mark.pure
def test_identical_models_have_zero_width_interval():
    y, a, _, _ = _cohort()
    result = paired_bootstrap(y, a, a, unit="eye", n_resamples=200)
    assert result["n_units"] == len(y)
    for row in result["metrics"].values():
        assert (row["difference"], row["ci_lower"], row["ci_upper"]) == (0, 0, 0)


@pytest.mark.pure
def test_examination_cluster_uses_pairs_without_claiming_patient_identity():
    y, a, b, ids = _cohort()
    result = paired_bootstrap(y, a, b, unit="examination", examination_ids=ids,
                              n_resamples=200, seed=8)
    same_clusters = paired_bootstrap(y, a, b, unit="patient", patient_ids=ids,
                                     n_resamples=200, seed=8)
    assert result["analysis_unit"] == "examination"
    assert result["n_units"] == 10
    assert result["metrics"] == same_clusters["metrics"]
    with pytest.raises(ValueError, match="substituted"):
        paired_bootstrap(y, a, b, unit="examination", patient_ids=ids)
    with pytest.raises(ValueError, match="examination unit requires"):
        paired_bootstrap(y, a, b, unit="examination")
    with pytest.raises(ValueError, match="explicitly choose"):
        paired_bootstrap(y, a, b, unit="eye", examination_ids=ids)


@pytest.mark.pure
def test_examination_cli_reads_explicit_exam_column(tmp_path):
    from aptos.evaluation.paired_uncertainty import main

    y, a, b, ids = _cohort()
    path, output = tmp_path / "scores.csv", tmp_path / "aggregate.json"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id_code", "diagnosis", "baseline_grade", "candidate_grade", "exam_id"])
        writer.writerows((f"eye-{i}", y[i], a[i], b[i], ids[i]) for i in range(len(y)))
    assert read_paired_csv(path, unit="examination")[-1] == ids
    assert main(["--predictions", str(path), "--unit", "examination", "--draws", "200",
                 "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["analysis_unit"] == "examination"
    assert "eye-0" not in output.read_text()
    with pytest.raises(ValueError, match="patient_id"):
        read_paired_csv(path, unit="patient")


@pytest.mark.pure
def test_swapping_models_negates_point_estimates():
    y, a, b, p = _cohort()
    forward = paired_bootstrap(y, a, b, unit="patient", patient_ids=p, n_resamples=300)
    reverse = paired_bootstrap(y, b, a, unit="patient", patient_ids=p, n_resamples=300)
    for metric in ("sensitivity", "specificity", "referral_rate"):
        assert forward["metrics"][metric]["difference"] == pytest.approx(
            -reverse["metrics"][metric]["difference"]
        )
        assert forward["metrics"][metric]["ci_lower"] == pytest.approx(
            -reverse["metrics"][metric]["ci_upper"]
        )


@pytest.mark.pure
@pytest.mark.parametrize("kwargs,reason", [
    ({"unit": "patient"}, "patient unit requires"),
    ({"unit": "eye", "patient_ids": ["x"] * 20}, "explicitly choose"),
    ({"unit": "something"}, "unit must"),
    ({"unit": "eye", "n_resamples": 5}, "at least 100"),
    ({"unit": "eye", "confidence": 1}, "confidence must"),
])
def test_explicit_sampling_contract(kwargs, reason):
    y, a, b, _ = _cohort()
    with pytest.raises(ValueError, match=reason):
        paired_bootstrap(y, a, b, **kwargs)


@pytest.mark.pure
def test_one_class_and_invalid_grades_refused():
    y, a, b, _ = _cohort()
    with pytest.raises(ValueError, match="both referable"):
        paired_bootstrap(np.zeros_like(y), a, b, unit="eye")
    with pytest.raises(ValueError, match="grades"):
        paired_bootstrap(np.array([0, 2.5] + list(y[2:])), a, b, unit="eye")
    with pytest.raises(ValueError, match="equal lengths"):
        paired_bootstrap(y[:-1], a, b, unit="eye")


@pytest.mark.pure
def test_patient_cluster_not_fake_eye_level():
    y, a, b, patient = _cohort()
    grouped = paired_bootstrap(y, a, b, unit="patient", patient_ids=patient,
                               n_resamples=200, seed=2)
    eyes = paired_bootstrap(y, a, b, unit="eye", n_resamples=200, seed=2)
    assert grouped["n_units"] == 10
    assert eyes["n_units"] == 20
    assert grouped["metrics"]["sensitivity"]["difference"] == pytest.approx(
        eyes["metrics"]["sensitivity"]["difference"]
    )


def _csv(tmp_path, columns=("id_code", "diagnosis", "baseline_grade", "candidate_grade",
                             "patient_id")):
    path = tmp_path / "test_predictions.csv"
    rows = [
        ("eye-a", "0", "0", "2", "person-1"),
        ("eye-b", "2", "0", "2", "person-1"),
        ("eye-c", "0", "0", "0", "person-2"),
        ("eye-d", "3", "3", "3", "person-3"),
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(zip(("id_code", "diagnosis", "baseline_grade",
                                      "candidate_grade", "patient_id"), row, strict=True)))
    return path


@pytest.mark.pure
def test_csv_contract_and_real_patient_linkage(tmp_path):
    path = _csv(tmp_path)
    y, a, b, patients = read_paired_csv(path, unit="patient")
    assert y == [0, 2, 0, 3]
    assert a == [0, 0, 0, 3]
    assert b == [2, 2, 0, 3]
    assert patients == ["person-1", "person-1", "person-2", "person-3"]
    assert read_paired_csv(path, unit="eye")[-1] is None


@pytest.mark.pure
def test_rejects_duplicate_image_identifiers(tmp_path):
    path = _csv(tmp_path)
    text = path.read_text().replace("eye-d", "eye-a")
    path.write_text(text)
    with pytest.raises(ValueError, match="unique"):
        read_paired_csv(path, unit="eye")


@pytest.mark.pure
def test_rejects_non_numeric_csv_grade(tmp_path):
    path = _csv(tmp_path)
    path.write_text(path.read_text().replace("eye-d,3,3,3", "eye-d,3,invalid,3"))
    with pytest.raises(ValueError, match="baseline_grade"):
        read_paired_csv(path, unit="eye")


@pytest.mark.pure
def test_patient_group_requires_more_than_one_unit():
    with pytest.raises(ValueError, match="at least two"):
        paired_bootstrap([0, 2], [0, 2], [2, 2], unit="patient",
                         patient_ids=["same", "same"], n_resamples=200)
