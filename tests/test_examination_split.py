"""A follow-up split must keep every examination intact without rewriting history."""
import json

import numpy as np
import pandas as pd
import pytest

from aptos.evaluation.examination_split import (
    compare_historical_split,
    main,
    make_examination_split,
    split_plan,
    validate_examination_split,
)

pytestmark = pytest.mark.pure


@pytest.fixture
def cohort():
    pairs = pd.DataFrame({"left": [f"eye{i}_L.png" for i in range(100)],
                          "right": [f"eye{i}_R.png" for i in range(100)]})
    ids = [f"eye{i}_{side}" for i in range(100) for side in ("L", "R")]
    labels = pd.DataFrame({"id_code": ids, "diagnosis": np.tile([0, 2, 1, 3, 4], 40)})
    return labels, pairs


def test_all_eyes_kept_and_exams_disjoint_independent_of_input_order(cohort):
    labels, pairs = cohort
    split = make_examination_split(labels, pairs)
    reordered = make_examination_split(
        labels.sample(frac=1, random_state=8),
        pairs.rename(columns={"left": "right", "right": "left"}).sample(frac=1, random_state=9),
    )
    pd.testing.assert_frame_equal(split, reordered)
    assert split.groupby("exam_id").part.nunique().max() == 1
    assert set(split.id_code) == set(labels.id_code)
    assert split.diagnosis.tolist() == labels.set_index("id_code").loc[split.id_code].diagnosis.tolist()
    plan = split_plan(split, pairs, seed=42)
    assert plan["audit"]["examination_disjoint"]
    assert plan["audit"]["cross_partition_examinations"] == 0
    assert not any(ident in json.dumps(plan) for ident in labels.id_code)


@pytest.mark.parametrize("corruption", ["missing_eye", "grade", "key", "overlap", "part"])
def test_changed_or_leaky_split_rejected(cohort, corruption):
    labels, pairs = cohort
    split = make_examination_split(labels, pairs)
    if corruption == "missing_eye":
        split = split.iloc[:-1]
    elif corruption == "grade":
        split.loc[0, "diagnosis"] = (split.loc[0, "diagnosis"] + 1) % 5
    elif corruption == "key":
        split.loc[0, "exam_id"] = "made-up-patient"
    elif corruption == "overlap":
        split.loc[0, "part"] = "test" if split.loc[0, "part"] != "test" else "tune-train"
    else:
        split.loc[0, "part"] = "validation"
    with pytest.raises(ValueError):
        validate_examination_split(split, labels, pairs)


def test_unmapped_eye_or_insufficient_groups_rejected(cohort):
    labels, pairs = cohort
    with pytest.raises(ValueError, match="coverage"):
        make_examination_split(labels, pairs.iloc[:-1])
    with pytest.raises(ValueError, match="20"):
        make_examination_split(labels.iloc[:10], pairs)
    labels.diagnosis = 0
    with pytest.raises(ValueError, match="both referable"):
        make_examination_split(labels, pairs)


def test_old_checkpoint_exposure_is_recorded_not_treated_as_clean(cohort):
    labels, pairs = cohort
    followup = make_examination_split(labels, pairs)
    historical = followup.copy()
    test_ids = historical.index[historical.part.eq("test")]
    historical.loc[test_ids[:3], "part"] = "tune-train"
    historical.loc[test_ids[3:5], "part"] = "tune-valid"
    comparison = compare_historical_split(historical, followup)
    assert comparison["new_test_eyes_previously_in_training"] == 3
    assert comparison["new_test_eyes_previously_in_validation"] == 2
    assert not comparison["old_finetuned_checkpoint_reuse_allowed"]
    with pytest.raises(ValueError, match="match exactly"):
        compare_historical_split(historical.iloc[:-1], followup)


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5])
def test_invalid_seed_rejected(cohort, seed):
    with pytest.raises(ValueError, match="seed"):
        make_examination_split(*cohort, seed=seed)


def test_cli_writes_only_new_artifacts_and_refuses_overwrite(cohort, tmp_path):
    labels, pairs = cohort
    labels_path, pairs_path = tmp_path / "labels.csv", tmp_path / "pairs.csv"
    labels.to_csv(labels_path, index=False)
    pairs.to_csv(pairs_path, sep=";", index=False)
    csv, plan = tmp_path / "split.csv", tmp_path / "plan.json"
    args = ["--pairs", str(pairs_path), "--labels", str(labels_path),
            "--output", str(csv), "--plan", str(plan)]
    assert main(args) == 0
    original_csv, original_plan = csv.read_bytes(), plan.read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    assert csv.read_bytes() == original_csv
    assert plan.read_bytes() == original_plan
    assert json.loads(original_plan)["audit"]["cross_partition_examinations"] == 0


@pytest.mark.parametrize("command", ["split", "run"])
def test_historical_cli_refuses_implicit_proxy_replay(monkeypatch, command):
    from aptos.training import finetune

    def unavailable_config(*args, **kwargs):
        pytest.fail("archival guard must run before reading data or importing training code")

    monkeypatch.setattr(finetune.Config, "load", unavailable_config)
    with pytest.raises(SystemExit) as exc:
        finetune.main([command])
    assert exc.value.code == 2
