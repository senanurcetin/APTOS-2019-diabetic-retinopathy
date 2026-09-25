"""The single-split trainer's configuration and persistence.

Training itself is exercised by a --limit smoke run. These cover the two parts
that run does not reach or cannot check by eye: how the old command line turns
into a config - the cache decides the variant, the variant decides what the
manifest is checked against - and what a real run writes to disk.
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aptos.config import Config  # noqa: E402
from aptos.training.single import (  # noqa: E402
    build_config,
    build_parser,
    run_name,
    save_run,
    variant_for,
)

pytestmark = pytest.mark.torch


def _cfg(argv):
    return build_config(build_parser().parse_args(argv))


def test_the_cache_directory_decides_the_variant(tmp_path):
    for directory, variant in [("processed", "baseline"),
                               ("processed_clahe", "clahe"),
                               ("processed_squash", "squash")]:
        assert variant_for(tmp_path / directory) == variant
    assert variant_for(tmp_path / "somewhere_else") is None


def test_the_variant_brings_its_own_preprocessing_settings():
    """So the manifest check compares a CLAHE cache against CLAHE settings - the
    whole point of checking it."""
    cfg, data_dir = _cfg(["--data-dir", "data/processed_clahe"])
    assert cfg.variant == "clahe"
    assert cfg.preprocess.clahe is True and cfg.preprocess.clip_limit == 2.0
    assert cfg.preprocess.square_mode == "pad"
    assert data_dir.name == "processed_clahe" and data_dir.is_absolute()


def test_the_old_flags_all_land_in_the_config():
    cfg, _ = _cfg(["--mode", "cls", "--size", "320", "--batch", "8", "--epochs", "3",
                   "--lr", "0.001", "--workers", "0", "--patience", "2",
                   "--min-delta", "0.01", "--seed", "7", "--exclude-leaked"])
    t = cfg.train
    assert (t.mode, t.size, t.batch, t.epochs, t.lr, t.workers, t.patience,
            t.min_delta, t.seed, t.exclude_leaked) == ("cls", 320, 8, 3, 0.001, 0, 2,
                                                       0.01, 7, True)


def test_compatibility_flags_are_accepted():
    """run_seeds.sh passes --no-bq and --author; they must not be errors."""
    args = build_parser().parse_args(["--no-bq", "--author", "someone", "--variant", "x"])
    assert args.no_bq and args.author == "someone"


def test_run_name_is_stable_and_seed_sensitive():
    a, _ = _cfg(["--seed", "42"])
    b, _ = _cfg(["--seed", "42"])
    c, _ = _cfg(["--seed", "43"])
    assert run_name(a, "baseline") == run_name(b, "baseline")
    assert run_name(a, "baseline") != run_name(c, "baseline")


def test_leak_handling_changes_the_run_name():
    """Runs with and without leak exclusion are different experiments; the
    historical ones were made without it."""
    with_, _ = _cfg(["--exclude-leaked"])
    without, _ = _cfg([])
    assert run_name(with_, "baseline") != run_name(without, "baseline")


def test_save_run_writes_everything_a_later_analysis_needs(tmp_path):
    cfg = Config.load()
    state = {"w": torch.ones(2)}
    result = {"thresholds": [0.5, 1.5, 2.5, 3.5], "test_qwk": 0.9, "exclude_leaked": True}
    predictions = {"valid": (np.array([0.1, 2.2]), np.array([0, 2])),
                   "test": (np.array([3.3]), np.array([3]))}
    history = [{"epoch": 1, "valid_qwk": 0.8}]

    out = save_run(tmp_path / "run", cfg, state, result, predictions, history)

    blob = torch.load(out / "model.pt", weights_only=False)
    assert torch.equal(blob["state_dict"]["w"], torch.ones(2))
    assert blob["thresholds"] == [0.5, 1.5, 2.5, 3.5]
    arrays = np.load(out / "predictions.npz")
    assert sorted(arrays.files) == ["test_raw", "test_true", "valid_raw", "valid_true"]
    assert np.allclose(arrays["valid_raw"], [0.1, 2.2])
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["test_qwk"] == 0.9 and metrics["history"][0]["epoch"] == 1


def test_save_run_handles_classification_mode_without_thresholds(tmp_path):
    result = {"thresholds": None, "test_qwk": 0.5}
    out = save_run(tmp_path / "cls", Config.load(), {"w": torch.zeros(1)}, result,
                   {"test": (np.array([1.0]), np.array([1]))}, [])
    assert torch.load(out / "model.pt", weights_only=False)["thresholds"] is None
