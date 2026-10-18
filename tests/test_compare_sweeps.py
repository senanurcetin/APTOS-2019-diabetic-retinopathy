"""Pure no-dataset comparisons for ordinal versus MSE predictions."""
import copy

import numpy as np
import pandas as pd
import pytest

from aptos.evaluation.compare_sweeps import (
    compare_frames,
    compare_sweep_configs,
    validated_thresholds,
)


@pytest.fixture
def sample():
    grades = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 2]
    ids = [f"eye-{i:02}" for i in range(len(grades))]
    a = pd.DataFrame({"id_code": ids, "diagnosis": grades,
                      "raw": [float(x) - 0.7 for x in grades]})
    b = pd.DataFrame({"id_code": ids, "diagnosis": grades,
                      "raw": [float(x) + 0.1 for x in grades]})
    return a, b


@pytest.mark.pure
def test_paired_comparison_is_invariant_to_candidate_row_order(sample):
    a, b = sample
    cuts = [0.5, 1.5, 2.5, 3.5]
    x = compare_frames(a, b, cuts, cuts, n_resamples=300, seed=14)
    y = compare_frames(a, b.iloc[::-1], cuts, cuts, n_resamples=300, seed=14)
    assert x == y
    assert x["n_eyes"] == 12 and x["n_referable"] == 7
    assert x["ordinal_minus_baseline"]["qwk"] > 0
    assert x["paired_referral_uncertainty"]["analysis_unit"] == "eye"


@pytest.mark.pure
def test_duplicate_image_ids_rejected(sample):
    a, b = sample
    b = b.copy()
    b.loc[1, "id_code"] = b.loc[0, "id_code"]
    with pytest.raises(ValueError, match="unique"):
        compare_frames(a, b, [0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5, 3.5], n_resamples=100)


@pytest.mark.pure
def test_truth_mismatch_rejected(sample):
    a, b = sample
    b = b.copy()
    b.loc[1, "diagnosis"] = 4
    with pytest.raises(ValueError, match="inconsistent"):
        compare_frames(a, b, [0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5, 3.5], n_resamples=100)


@pytest.mark.pure
def test_unmatched_eye_rejected(sample):
    a, b = sample
    b = b.copy()
    b.loc[0, "id_code"] = "unmatched"
    with pytest.raises(ValueError, match="different sets of eyes"):
        compare_frames(a, b, [0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5, 3.5], n_resamples=100)


@pytest.mark.pure
@pytest.mark.parametrize("broken", [float("nan"), float("inf")])
def test_non_finite_raw_scores_rejected(sample, broken):
    a, b = sample
    a = a.copy()
    a.loc[0, "raw"] = broken
    with pytest.raises(ValueError, match="finite"):
        compare_frames(a, b, [0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5, 3.5], n_resamples=100)


@pytest.mark.pure
def test_patient_cluster_linkage_is_explicit(sample):
    a, b = sample
    patient_map = {i: f"p{k // 2}" for k, i in enumerate(a["id_code"])}
    result = compare_frames(a, b, [0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5, 3.5],
                            n_resamples=300, patient_map=patient_map)
    assert result["paired_referral_uncertainty"]["analysis_unit"] == "patient"
    assert result["paired_referral_uncertainty"]["n_units"] == 6


@pytest.mark.pure
def test_no_partial_patient_map_allowed(sample):
    a, b = sample
    with pytest.raises(ValueError, match="patient linkage"):
        compare_frames(a, b, [0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5, 3.5],
                       n_resamples=100, patient_map={"eye-00": "p1"})


@pytest.mark.pure
def test_validated_thresholds_fail_closed():
    with pytest.raises(ValueError, match="four ordered"):
        validated_thresholds({"ensemble": {"thresholds": [0.5, 0.4, 2, 3]}})
    with pytest.raises(ValueError, match="four"):
        validated_thresholds({"ensemble": {"thresholds": [0.5, 1.5]}})
    assert np.allclose(validated_thresholds({"ensemble": {"thresholds": [0, 1, 2, 3]}}),
                       [0, 1, 2, 3])


@pytest.mark.pure
def test_experiment_metadata_matching():
    from aptos.config import Config

    cfg = Config.load("configs/cv.yaml").to_dict()
    a, b = copy.deepcopy(cfg), copy.deepcopy(cfg)
    b["train.mode"] = "ordinal"
    assert compare_sweep_configs(a, b)["train.model"] == a["train.model"]
    b["train.seed"] += 1
    with pytest.raises(ValueError, match="train.seed"):
        compare_sweep_configs(a, b)
    b["train.seed"] = a["train.seed"]
    b["augment.rotation_degrees"] += 5
    with pytest.raises(ValueError, match="augment.rotation_degrees"):
        compare_sweep_configs(a, b)


@pytest.mark.pure
def test_model_families_are_not_interchangeable():
    from aptos.config import Config

    cfg = Config.load().to_dict()
    with pytest.raises(ValueError, match="reg baseline and ordinal"):
        compare_sweep_configs(cfg, cfg)
