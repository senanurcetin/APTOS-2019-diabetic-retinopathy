"""The training path end to end, on a synthetic cache small enough for CI.

Every other test stops at the edge of training, because the real run needs the
dataset and a GPU. That left the code that produced every number in RESULTS.md -
the single-split trainer and the resumable cross-validation sweep - checked only
by running it for real. Here it runs on about fifty 64px synthetic fundus
images with a tiny backbone on CPU: not a test of how well it learns, but of
everything around the learning - the manifest gate, label handling, fold
construction, persistence, resume, and the ensemble built from saved arrays.

The resume test is the verification the rebuild plan asked for: delete one
finished fold, restart, and confirm only that fold is retrained and the
ensemble comes back the same.
"""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from aptos.config import Config, Paths  # noqa: E402
from aptos.data import manifest as manifest_mod  # noqa: E402
from aptos.training import cv as cv_mod  # noqa: E402
from aptos.training import loop, single  # noqa: E402

pytestmark = pytest.mark.torch

PER_GRADE = {"train": 5, "valid": 3, "test": 2}   # 50 images across five grades


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A repository root holding labels, a leak list and a processed cache."""
    import cv2

    cfg = Config.load("configs/baseline.yaml",
                      train={"model": "resnet10t", "size": 64, "batch": 8, "epochs": 2,
                             "workers": 0, "patience": 0},
                      cv={"folds": 2})
    cfg = dataclasses.replace(cfg, paths=Paths(root=tmp_path))

    rng = np.random.default_rng(0)
    rows = []
    for split, n in PER_GRADE.items():
        for grade in range(5):
            for i in range(n):
                id_code = f"{split}{grade}{i}"
                rows.append({"id_code": id_code, "diagnosis": grade, "split": split})
                # Brightness rises with grade, so there is something to learn.
                img = np.zeros((64, 64, 3), np.uint8)
                cv2.circle(img, (32, 32), 28, (40 + 40 * grade,) * 3, -1)
                img = np.clip(img + rng.integers(0, 20, img.shape), 0, 255).astype(np.uint8)
                target = cfg.data_dir / split / f"{id_code}.jpg"
                target.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target), img)
    labels = pd.DataFrame(rows)
    cfg.paths.labels_csv.parent.mkdir(parents=True)
    labels.to_csv(cfg.paths.labels_csv, index=False)
    cfg.paths.reports.mkdir(parents=True)
    pd.DataFrame({"id_code": ["train00"]}).to_csv(cfg.paths.leaked_ids_csv, index=False)
    manifest_mod.write(cfg.data_dir, cfg.manifest_fields(), n_images=len(labels))

    # CPU everywhere, and no ImageNet download: the weights are irrelevant here.
    monkeypatch.setattr(cv_mod, "pick_device", lambda: "cpu")
    monkeypatch.setattr(single, "pick_device", lambda: "cpu")
    real_build = loop.build_model
    stub = lambda cfg, device, pretrained=True: real_build(cfg, device, pretrained=False)  # noqa: E731
    monkeypatch.setattr(cv_mod, "build_model", stub)
    monkeypatch.setattr(single, "build_model", stub)
    return cfg


def test_single_split_run_writes_a_complete_record(workspace):
    cfg = workspace
    cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, exclude_leaked=True))
    result = single.train(cfg, cfg.data_dir, label="baseline", track=False)

    assert result["epochs_run"] == 2 and 1 <= result["best_epoch"] <= 2
    assert len(result["thresholds"]) == 4
    assert -1.0 <= result["test_qwk"] <= 1.0

    out = cfg.paths.models / "single" / single.run_name(cfg, "baseline")
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["exclude_leaked"] is True and len(metrics["history"]) == 2
    arrays = np.load(out / "predictions.npz")
    assert len(arrays["test_raw"]) == PER_GRADE["test"] * 5
    assert torch.load(out / "model.pt", weights_only=False)["thresholds"] == result["thresholds"]


def test_smoke_run_leaves_nothing_behind(workspace):
    cfg = workspace
    single.train(cfg, cfg.data_dir, label="baseline", limit=3, track=False)
    assert not (cfg.paths.models / "single").exists()


def test_a_mismatched_cache_stops_the_run(workspace):
    cfg = workspace
    manifest_mod.write(cfg.data_dir, {**cfg.manifest_fields(), "clahe": True}, n_images=50)
    with pytest.raises(manifest_mod.ManifestMismatch):
        single.train(cfg, cfg.data_dir, label="baseline", track=False)


def test_cv_resumes_only_the_missing_fold_and_rebuilds_the_same_ensemble(workspace, monkeypatch):
    cfg = workspace
    trained = []
    real_train = cv_mod.train_one_fold

    def counting(cfg_, fold, *args, **kwargs):
        trained.append(fold)
        return real_train(cfg_, fold, *args, **kwargs)

    monkeypatch.setattr(cv_mod, "train_one_fold", counting)

    first = cv_mod.run_cv(cfg, track=False)
    assert trained == [1, 2]
    directory = cv_mod.sweep_dir(cfg, first["run_id"])
    assert all(cv_mod.fold_is_complete(directory, f) for f in (1, 2))
    assert first["folds"] == 2
    fold2_before = np.load(directory / "fold2.npz")["test_raw"]

    # Complete sweep: a restart trains nothing and rebuilds from the arrays.
    trained.clear()
    again = cv_mod.run_cv(cfg, track=False)
    assert trained == []
    # Serialised, because an undefined metric is NaN and NaN != NaN.
    assert json.dumps(again["ensemble"], sort_keys=True) == json.dumps(first["ensemble"], sort_keys=True)

    # A sweep that died after fold 1: only fold 2 is retrained, and because each
    # fold is seeded on its own, it comes back as the uninterrupted run made it.
    for path in cv_mod.fold_paths(directory, 2).values():
        path.unlink(missing_ok=True)
    trained.clear()
    resumed = cv_mod.run_cv(cfg, track=False)
    assert trained == [2]
    np.testing.assert_allclose(np.load(directory / "fold2.npz")["test_raw"], fold2_before,
                               rtol=1e-4, atol=1e-5)
    assert resumed["ensemble"]["qwk"] == pytest.approx(first["ensemble"]["qwk"])


def test_resolution_stratified_folds_keep_both_strata_in_every_fold(workspace):
    """The confound-aware split must put confounded and other images in each
    fold, or a fold could learn a resolution-to-label mapping it is never
    scored against."""
    cfg = workspace
    w, h = cfg.confound.confounded_resolution
    labels = pd.read_csv(cfg.paths.labels_csv)
    confounded = labels["id_code"].str.endswith(("0", "1"))
    pd.DataFrame({"id_code": labels["id_code"],
                  "width": np.where(confounded, w, 800),
                  "height": np.where(confounded, h, 600)}).to_csv(cfg.paths.image_stats_csv,
                                                                  index=False)
    cfg = dataclasses.replace(cfg, cv=dataclasses.replace(cfg.cv, stratify_on="diagnosis+resolution"))

    from aptos.data import labels as labels_mod

    df = labels_mod.add_resolution_bucket(labels_mod.load_labels(cfg), cfg)
    pool, _ = labels_mod.pool_and_holdout(df, cfg)
    folds = cv_mod.build_folds(cfg, pool)
    for _, valid_idx in folds:
        assert set(pool.iloc[valid_idx]["resolution_bucket"]) == {f"{w}x{h}", "other"}
