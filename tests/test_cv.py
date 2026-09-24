"""Tests for the cross-validation machinery.

The sweep that these protect died three times and lost everything each time.
Two causes were environmental; the third was that per-fold predictions were
never persisted and the run id was random, so a restart could not recognise its
own earlier work. These tests cover that third cause.
"""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from aptos.config import Config  # noqa: E402
from aptos.training.cv import (  # noqa: E402
    build_folds,
    fold_is_complete,
    fold_paths,
    load_fold,
    run_id_for,
    save_fold,
    summarise,
)

pytestmark = pytest.mark.torch


MANIFEST = {"size": 512, "clahe": False, "clip_limit": None, "square_mode": "pad"}


def _pool(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id_code": [f"img{i:04d}" for i in range(n)],
        "diagnosis": rng.integers(0, 5, n),
        "split": rng.choice(["train", "valid"], n),
        "orig_split": rng.choice(["train", "valid"], n),
        "resolution_bucket": rng.choice(["1050x1050", "other"], n),
    })


# ------------------------------------------------------------------- run ids

def test_run_id_is_stable_across_calls():
    """The whole point: re-invoking the same sweep must land in the same place."""
    cfg = Config.load("configs/cv.yaml")
    assert run_id_for(cfg, MANIFEST) == run_id_for(cfg, MANIFEST)


def test_run_id_changes_with_the_variant():
    a = run_id_for(Config.load("configs/cv.yaml", variant="baseline"), MANIFEST)
    b = run_id_for(Config.load("configs/cv.yaml", variant="clahe"), MANIFEST)
    assert a != b


def test_run_id_changes_with_the_seed_and_fold_count():
    base = Config.load("configs/cv.yaml")
    assert run_id_for(base, MANIFEST) != run_id_for(
        Config.load("configs/cv.yaml", train={"seed": 43}), MANIFEST)
    assert run_id_for(base, MANIFEST) != run_id_for(
        Config.load("configs/cv.yaml", cv={"folds": 10}), MANIFEST)


def test_run_id_changes_with_the_stratification():
    """A confound-aware sweep is a different experiment and must not collide
    with the plain one."""
    a = run_id_for(Config.load("configs/cv.yaml"), MANIFEST)
    b = run_id_for(Config.load("configs/cv_resolution.yaml"), MANIFEST)
    assert a != b


def test_run_id_changes_with_the_cache_settings():
    """Same config, different pixels - that is a different sweep."""
    cfg = Config.load("configs/cv.yaml")
    other = dict(MANIFEST, square_mode="squash")
    assert run_id_for(cfg, MANIFEST) != run_id_for(cfg, other)


# ------------------------------------------------------------------ fold state

def test_a_fold_needs_both_metrics_and_predictions_to_count_as_done(tmp_path):
    """Weights alone are not enough. The original saved a checkpoint without
    `test_raw`, so a resumed sweep could not rebuild the ensemble - which is how
    a partial failure became a total one."""
    paths = fold_paths(tmp_path, 1)
    assert fold_is_complete(tmp_path, 1) is False

    paths["metrics"].write_text("{}")
    assert fold_is_complete(tmp_path, 1) is False, "metrics alone must not count"

    np.savez(paths["preds"], test_raw=np.zeros(3), test_true=np.zeros(3))
    assert fold_is_complete(tmp_path, 1) is True


def test_saved_fold_round_trips(tmp_path):
    result = {"fold": 2, "valid_qwk": 0.91, "test_qwk": 0.89,
              "thresholds": [0.5, 1.5, 2.5, 3.5]}
    raw = np.array([0.2, 1.8, 3.4])
    true = np.array([0, 2, 3])

    save_fold(tmp_path, 2, result, {"w": torch.zeros(1)}, raw, true)
    back, raw2, true2 = load_fold(tmp_path, 2)

    assert back["valid_qwk"] == pytest.approx(0.91)
    assert np.allclose(raw2, raw)
    assert np.array_equal(true2, true)
    assert fold_paths(tmp_path, 2)["checkpoint"].exists()


def test_smoke_runs_do_not_leave_weights_behind(tmp_path):
    """`--limit` results are meaningless; they must not be mistaken for a real
    checkpoint later."""
    save_fold(tmp_path, 1, {"fold": 1, "thresholds": []}, {"w": torch.zeros(1)},
              np.zeros(2), np.zeros(2), save_weights=False)
    assert not fold_paths(tmp_path, 1)["checkpoint"].exists()
    assert fold_is_complete(tmp_path, 1) is True


# ---------------------------------------------------------------- fold splits

def test_every_image_appears_in_exactly_one_validation_fold():
    cfg = Config.load("configs/cv.yaml")
    pool = _pool(300)
    splits = build_folds(cfg, pool)

    seen = np.concatenate([va for _, va in splits])
    assert len(seen) == len(pool)
    assert len(set(seen)) == len(pool), "an image landed in two validation folds"


def test_train_and_validation_never_overlap_within_a_fold():
    cfg = Config.load("configs/cv.yaml")
    pool = _pool(300)
    for tr, va in build_folds(cfg, pool):
        assert not set(tr) & set(va)
        assert len(tr) + len(va) == len(pool)


def test_folds_are_deterministic_for_a_given_seed():
    cfg = Config.load("configs/cv.yaml")
    pool = _pool(300)
    a = [va.tolist() for _, va in build_folds(cfg, pool)]
    b = [va.tolist() for _, va in build_folds(cfg, pool)]
    assert a == b


def test_resolution_stratification_balances_the_confounded_bucket():
    """The shortcut reaches QWK 0.652 on metadata alone. Stratifying jointly is
    what stops a fold exploiting a resolution-to-label mapping the others lack."""
    cfg = Config.load("configs/cv_resolution.yaml")
    pool = _pool(500)
    shares = [
        (pool.iloc[va]["resolution_bucket"] == "1050x1050").mean()
        for _, va in build_folds(cfg, pool)
    ]
    overall = (pool["resolution_bucket"] == "1050x1050").mean()
    assert max(abs(s - overall) for s in shares) < 0.06


def test_resolution_stratification_needs_the_column():
    cfg = Config.load("configs/cv_resolution.yaml")
    pool = _pool(100).drop(columns=["resolution_bucket"])
    with pytest.raises(KeyError, match="resolution_bucket"):
        build_folds(cfg, pool)


# -------------------------------------------------------------------- summary

def _fold_result(fold, qwk_value, thresholds=(0.5, 1.5, 2.5, 3.5)):
    return {
        "fold": fold, "valid_qwk": qwk_value, "test_qwk": qwk_value,
        "test_accuracy": 0.8, "test_macro_f1": 0.55,
        "thresholds": list(thresholds),
    }


def test_summary_rebuilds_the_ensemble_from_persisted_arrays():
    """This is what a resumed sweep relies on - the ensemble is reconstructed
    from what was written to disk, not from anything held in memory."""
    cfg = Config.load("configs/cv.yaml")
    true = np.array([0, 1, 2, 3, 4] * 4)
    raws = [true + 0.1, true - 0.1, true + 0.05]
    results = [_fold_result(i + 1, 0.9) for i in range(3)]

    summary = summarise(cfg, "abc123", results, raws, [true] * 3)

    assert summary["folds"] == 3
    assert summary["test_qwk_mean"] == pytest.approx(0.9)
    assert "ensemble" in summary
    assert summary["ensemble"]["qwk"] == pytest.approx(1.0)


def test_summary_refuses_folds_that_disagree_on_the_test_set():
    """If the held-out set changed between folds the ensemble is meaningless,
    and averaging would hide that rather than reveal it."""
    cfg = Config.load("configs/cv.yaml")
    results = [_fold_result(1, 0.9), _fold_result(2, 0.9)]
    raws = [np.zeros(5), np.zeros(5)]
    trues = [np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 3])]

    with pytest.raises(RuntimeError, match="disagree on the test labels"):
        summarise(cfg, "abc123", results, raws, trues)


def test_summary_std_is_zero_for_a_single_fold():
    cfg = Config.load("configs/cv.yaml")
    true = np.array([0, 1, 2, 3, 4])
    summary = summarise(cfg, "x", [_fold_result(1, 0.9)], [true + 0.1], [true])
    assert summary["test_qwk_std"] == 0.0


def test_summary_is_json_serialisable(tmp_path):
    """It gets written to summary.json; numpy floats would break that."""
    cfg = Config.load("configs/cv.yaml")
    true = np.array([0, 1, 2, 3, 4] * 3)
    summary = summarise(cfg, "x", [_fold_result(1, 0.9)], [true + 0.1], [true])
    (tmp_path / "s.json").write_text(json.dumps(summary, default=float))
    assert json.loads((tmp_path / "s.json").read_text())["run_id"] == "x"
