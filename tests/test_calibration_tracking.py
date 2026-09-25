"""The clinical operating point, calibration, and the tracking store.

The operating point and ECE are the numbers the README quotes for screening
use; the tracking layer is where every other number is recorded. Both are
checked on inputs whose answers are known by construction, and the tracking
store is a throwaway SQLite database under tmp_path, never the repository's.
"""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from aptos.config import Config, Paths

# ------------------------------------------------------------------ calibration


@pytest.fixture
def cal():
    pytest.importorskip("torch")   # calibration imports the fold machinery
    from aptos.evaluation import calibration
    return calibration


@pytest.mark.torch
def test_operating_point_meets_the_sensitivity_floor_at_best_specificity(cal):
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(100)].astype(int)
    score = np.r_[rng.normal(0, 1, 200), rng.normal(2, 1, 100)]
    point = cal.pick_operating_point(y, score, min_sensitivity=0.9)
    assert point["sensitivity"] >= 0.9

    # Brute force over every possible cut: none that reaches the floor is more
    # specific. (roc_curve keeps only the curve's corners, so the chosen point
    # can beat the floor by a case at no cost in specificity - it dominates.)
    best = max(cal.apply_operating_point(y, score, c)["specificity"]
               for c in np.unique(score)
               if cal.apply_operating_point(y, score, c)["sensitivity"] >= 0.9)
    assert point["specificity"] == pytest.approx(best)
    assert 0.9 < point["roc_auc"] <= 1.0


@pytest.mark.torch
def test_applied_operating_point_counts_every_case_once(cal):
    y = np.array([1, 1, 0, 0, 1])
    score = np.array([0.9, 0.2, 0.8, 0.1, 0.7])
    out = cal.apply_operating_point(y, score, cut=0.5)
    assert (out["tp"], out["fn"], out["fp"], out["tn"]) == (2, 1, 1, 1)
    assert out["referral_rate"] == 0.6
    assert out["ppv"] == pytest.approx(2 / 3)


@pytest.mark.torch
def test_applied_operating_point_survives_a_single_class(cal):
    out = cal.apply_operating_point(np.zeros(3), np.array([0.1, 0.2, 0.9]), cut=0.5)
    assert np.isnan(out["sensitivity"]) and np.isnan(out["roc_auc"])


@pytest.mark.torch
def test_calibrator_maps_higher_scores_to_higher_probability(cal):
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    score = np.r_[np.linspace(0, 1.5, 50), np.linspace(1, 4, 50)]
    prob = cal.probabilities(cal.fit_calibrator(y, score), np.array([0.0, 2.0, 4.0]))
    assert np.all(np.diff(prob) > 0) and 0 < prob[0] < prob[-1] < 1


@pytest.mark.torch
def test_ece_is_zero_when_probabilities_are_the_observed_rates(cal):
    prob = np.r_[np.full(10, 0.2), np.full(10, 0.8)]
    y = np.r_[[1, 1] + [0] * 8, [1] * 8 + [0, 0]]
    error, table = cal.expected_calibration_error(y, prob)
    assert error == pytest.approx(0.0)
    assert table["n"].sum() == 20


@pytest.mark.torch
def test_ece_measures_and_signs_overconfidence(cal):
    """Direction matters: the table keeps the sign that one ECE number hides."""
    error, table = cal.expected_calibration_error(np.r_[np.ones(5), np.zeros(5)], np.full(10, 0.9))
    assert error == pytest.approx(0.4)
    assert table["gap"].iloc[0] == pytest.approx(-0.4)


@pytest.mark.torch
def test_perfect_model_beats_both_trivial_policies(cal):
    y = np.r_[np.zeros(70), np.ones(30)].astype(int)
    table = cal.decision_curve(y, y.astype(float))
    assert table["model_beats_both"].all()
    # Treat-all at pt is prevalence - (1 - prevalence) * odds(pt).
    row = table.iloc[0]
    assert row["treat_all"] == pytest.approx(0.3 - 0.7 * row["threshold"] / (1 - row["threshold"]))


@pytest.mark.torch
def test_markdown_table_formats_floats_only(cal):
    lines = cal._md_table(pd.DataFrame({"name": ["a"], "value": [0.12345], "n": [3]}))
    assert lines == ["| name | value | n |", "|---|---|---|", "| a | 0.123 | 3 |"]


# --------------------------------------------------------------------- tracking

@pytest.fixture
def store(tmp_path):
    """A config whose tracking URI points at a fresh SQLite file in tmp_path."""
    pytest.importorskip("mlflow")
    from aptos import tracking

    cfg = dataclasses.replace(Config.load(), paths=Paths(root=tmp_path))
    yield tracking, cfg
    tracking._mlflow().end_run()


def test_backfill_writes_the_six_historical_runs_once(store):
    tracking, cfg = store
    assert tracking.backfill_history(cfg) == 6
    assert tracking.backfill_history(cfg) == 0   # idempotent


def test_export_is_sorted_text_with_the_recorded_metrics(store, tmp_path):
    tracking, cfg = store
    tracking.backfill_history(cfg)
    out = tracking.export_runs(cfg)
    frame = pd.read_csv(out)
    assert len(frame) == 6
    assert set(frame["source"]) == {"results_md"}
    seed42 = frame[frame["run_name"] == "historical-baseline-seed42"].iloc[0]
    assert seed42["test_qwk"] == pytest.approx(0.8960)
    assert b"\r\n" not in out.read_bytes()   # stable line endings in a versioned file


def test_export_maps_rewritten_commits(store):
    tracking, cfg = store
    with tracking.run(cfg, "mapped", tags={"split": "single", "git_commit": "old"}):
        tracking.log_metrics({"test_qwk": 0.9})
    cfg.paths.reports.mkdir(parents=True, exist_ok=True)
    (cfg.paths.reports / "commit_map.csv").write_text("pre_rewrite,post_rewrite\nold,new\n")
    frame = pd.read_csv(tracking.export_runs(cfg))
    assert frame.loc[frame["run_name"] == "mapped", "git_commit"].item() == "new"


def test_export_refuses_an_empty_store(store):
    tracking, cfg = store
    with pytest.raises(RuntimeError, match="no runs"):
        tracking.export_runs(cfg)


def test_a_run_records_its_config_and_drops_nan_metrics(store):
    tracking, cfg = store
    with tracking.run(cfg, "probe", tags={"split": "single"}) as active:
        tracking.log_metrics({"good": 1.0, "missing": float("nan"), "label": "text"})
        tracking.log_dict({"thresholds": [0.5]}, "thresholds.json")
        run_id = active.info.run_id

    client = tracking._mlflow().tracking.MlflowClient()
    run = client.get_run(run_id)
    assert run.data.metrics == {"good": 1.0}
    assert run.data.params["train.seed"] == str(cfg.train.seed)
    assert run.data.tags["variant"] == cfg.variant
    assert [a.path for a in client.list_artifacts(run_id)] == ["thresholds.json"]


def test_git_commit_is_the_short_hash_or_says_unknown(tmp_path):
    from aptos import tracking
    from aptos.config import ROOT

    assert tracking.git_commit(tmp_path) == "unknown"   # not a repository
    head = tracking.git_commit(ROOT)
    assert head == "unknown" or (len(head) >= 7 and int(head, 16) >= 0)
