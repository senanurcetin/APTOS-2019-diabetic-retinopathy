"""Evaluation and label helpers that run on numpy and pandas alone.

These produce the numbers the reports quote - the stratified confound table, the
metadata shortcut, the Grad-CAM concentration ratio, the leak filter - so each
is checked on a small synthetic case whose answer is known by construction.
Nothing here needs torch or the dataset, so it runs in the fast CI job.
"""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from aptos.config import Config, Paths
from aptos.data import labels as labels_mod
from aptos.evaluation import confound, external, gradcam

pytestmark = pytest.mark.pure


def _cfg(root, **overrides):
    return dataclasses.replace(Config.load(**overrides), paths=Paths(root=root))


def _write(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


# ---------------------------------------------------------------------- labels

def test_referable_starts_at_moderate():
    df = labels_mod.add_referable(pd.DataFrame({"diagnosis": [0, 1, 2, 3, 4]}))
    assert df["is_referable"].tolist() == [False, False, True, True, True]


def test_resolution_bucket_tags_only_the_confounded_size(tmp_path):
    cfg = _cfg(tmp_path)
    w, h = cfg.confound.confounded_resolution
    stats = pd.DataFrame({"id_code": ["a", "b", "c"], "width": [w, w, 640],
                          "height": [h, h + 1, 480]})
    out = labels_mod.add_resolution_bucket(pd.DataFrame({"id_code": ["a", "b", "c"]}),
                                           cfg, stats=stats)
    assert out["resolution_bucket"].tolist() == [f"{w}x{h}", "other", "other"]


def test_resolution_bucket_refuses_images_the_scan_never_saw(tmp_path):
    cfg = _cfg(tmp_path)
    stats = pd.DataFrame({"id_code": ["a"], "width": [1], "height": [1]})
    with pytest.raises(ValueError, match="no entry"):
        labels_mod.add_resolution_bucket(pd.DataFrame({"id_code": ["a", "zzz"]}),
                                         cfg, stats=stats)


def test_resolution_bucket_names_the_missing_stage(tmp_path):
    with pytest.raises(FileNotFoundError, match="scan"):
        labels_mod.add_resolution_bucket(pd.DataFrame({"id_code": ["a"]}), _cfg(tmp_path))


def test_leak_filter_drops_only_training_copies(tmp_path):
    cfg = _cfg(tmp_path)
    _write(cfg.paths.leaked_ids_csv, pd.DataFrame({"id_code": ["x"]}))
    df = pd.DataFrame({"id_code": ["x", "x", "y"], "split": ["train", "test", "train"]})
    out, dropped = labels_mod.exclude_leaked(df, cfg)
    assert dropped == 1
    assert out.to_dict("records") == [{"id_code": "x", "split": "test"},
                                      {"id_code": "y", "split": "train"}]


def test_empty_leak_list_is_treated_as_the_ordering_bug(tmp_path):
    """An empty list is what running `quality` before `preprocess` produced."""
    cfg = _cfg(tmp_path)
    _write(cfg.paths.leaked_ids_csv, pd.DataFrame({"id_code": []}))
    with pytest.raises(ValueError, match="stage-ordering"):
        labels_mod.exclude_leaked(pd.DataFrame({"id_code": [], "split": []}), cfg)


def test_split_counts_reports_totals_and_imbalance():
    df = pd.DataFrame({"split": ["train"] * 6 + ["valid", "test"],
                       "diagnosis": [0, 0, 0, 0, 1, 2, 0, 3]})
    table = labels_mod.split_counts(df)
    assert list(table.index) == ["train", "valid", "test"]
    assert table.loc["train", "total"] == 6
    assert table.loc["train", "imbalance"] == 4.0   # 4 No DR against 1 of the rarest present


def test_pool_and_holdout_keep_the_disk_location(tmp_path):
    cfg = _cfg(tmp_path)
    df = pd.DataFrame({"id_code": ["a", "b", "c"], "split": ["train", "valid", "test"]})
    pool, holdout = labels_mod.pool_and_holdout(df, cfg)
    assert set(pool["orig_split"]) == set(cfg.cv.pool_splits)
    assert holdout["id_code"].tolist() == ["c"]
    path = labels_mod.image_path(cfg, "a", "train")
    assert path.parts[-2:] == ("train", "a.jpg")


# -------------------------------------------------------------------- confound

def test_stratified_report_has_every_stratum_and_the_total():
    df = pd.DataFrame({"stratum": ["s"] * 4 + ["o"] * 4,
                       "diagnosis": [0, 0, 0, 2, 0, 1, 2, 4],
                       "pred":      [0, 0, 0, 2, 0, 1, 2, 3]})
    table = confound.stratified_report(df).set_index("stratum")
    assert list(table.index) == ["o", "s", "ALL"]
    assert table.loc["s", "majority_acc"] == 0.75
    assert table.loc["s", "qwk"] == pytest.approx(1.0)
    assert table.loc["ALL", "n"] == 8
    text = confound.format_report(table.reset_index(), "demo")
    assert text.startswith("### demo") and "| ALL | 8 |" in text


def test_add_resolution_flags_the_confounded_size(tmp_path):
    cfg = _cfg(tmp_path)
    w, h = cfg.confound.confounded_resolution
    _write(cfg.paths.image_stats_csv,
           pd.DataFrame({"id_code": ["a", "b"], "width": [w, 800], "height": [h, 600]}))
    out = confound.add_resolution(pd.DataFrame({"id_code": ["a", "b"]}), cfg)
    assert out["stratum"].tolist() == [f"{w}x{h}", "other"]
    with pytest.raises(ValueError):
        confound.add_resolution(pd.DataFrame({"id_code": ["a", "missing"]}), cfg)


def _synthetic_scan(cfg, n=120, seed=0):
    """A scan in which resolution predicts the label, as it does in APTOS."""
    rng = np.random.default_rng(seed)
    w, h = cfg.confound.confounded_resolution
    confounded = rng.random(n) < 0.5
    diagnosis = np.where(confounded, 0, 2)
    stats = pd.DataFrame({
        "id_code": [f"i{i}" for i in range(n)],
        "width": np.where(confounded, w, 2000), "height": np.where(confounded, h, 1500),
    })
    for col in cfg.confound.meta_cols:
        if col not in stats:
            stats[col] = rng.random(n)
    stats["aspect_ratio"] = stats["width"] / stats["height"]
    labels = pd.DataFrame({"id_code": stats["id_code"], "diagnosis": diagnosis,
                           "split": rng.choice(["train", "valid", "test"], n)})
    _write(cfg.paths.image_stats_csv, stats)
    _write(cfg.paths.labels_csv, labels)


# The disease-only stratum holds one grade, so sklearn warns that QWK is
# undefined there. That is the synthetic design, not a failure.
@pytest.mark.filterwarnings("ignore::UserWarning", "ignore::RuntimeWarning")
def test_metadata_shortcut_is_found_when_it_exists(tmp_path):
    cfg = _cfg(tmp_path, confound={"n_estimators": 20})
    _synthetic_scan(cfg)
    table = confound.shortcut_by_stratum(cfg).set_index("stratum")
    assert {"ALL", "other"} <= set(table.index)
    # Resolution alone decides the grade here, so the metadata model should be
    # near perfect while always guessing the majority grade manages about half.
    assert table.loc["ALL", "accuracy"] > 0.95
    assert table.loc["ALL", "majority_acc"] < 0.7


def test_caveats_are_computed_from_the_data(tmp_path):
    cfg = _cfg(tmp_path)
    _synthetic_scan(cfg, n=400)
    text = "\n".join(confound._caveats(cfg))
    # All confounded images are No DR by construction, so zero are diseased.
    assert "only **0** of them are diseased" in text


# --------------------------------------------------------------------- gradcam

def test_retina_mask_ignores_the_black_frame():
    img = np.zeros((10, 10, 3), np.uint8)
    img[2:8, 2:8] = 120
    img[0, 0] = 8   # JPEG ringing on the bars stays below the tolerance
    mask = gradcam.retina_mask(img)
    assert mask.sum() == 36 and not mask[0, 0]


def test_concentration_is_one_for_uniform_attention():
    mask = np.zeros((8, 8), bool)
    mask[:, :4] = True
    out = gradcam.attention_concentration(np.ones((8, 8)), mask)
    assert out["area_retina"] == 0.5
    assert out["concentration"] == pytest.approx(1.0)


def test_concentration_doubles_when_all_attention_is_on_half_the_image():
    mask = np.zeros((8, 8), bool)
    mask[:, :4] = True
    cam = mask.astype(float)
    assert gradcam.attention_concentration(cam, mask)["concentration"] == pytest.approx(2.0)


def test_concentration_is_undefined_without_attention_or_contrast():
    mask = np.ones((4, 4), bool)
    assert np.isnan(gradcam.attention_concentration(np.ones((4, 4)), mask)["concentration"])
    assert np.isnan(gradcam.attention_concentration(np.zeros((4, 4)), ~mask)["concentration"])


def test_concentration_resizes_a_mask_to_the_heatmap():
    mask = np.zeros((16, 16), bool)
    mask[:, :8] = True
    out = gradcam.attention_concentration(np.ones((4, 4)), mask)
    assert out["area_retina"] == 0.5


def test_overlay_keeps_the_image_size():
    img = np.full((32, 48, 3), 100, np.uint8)
    out = gradcam.overlay(img, np.random.default_rng(0).random((8, 8)))
    assert out.shape == img.shape and out.dtype == np.uint8


# -------------------------------------------------------------------- external

def test_idrid_labels_ignore_the_trailing_empty_columns(tmp_path):
    cfg = _cfg(tmp_path)
    path = cfg.paths.external / external.IDRID_LABELS
    path.parent.mkdir(parents=True)
    path.write_text("id_code,diagnosis,Risk of macular edema ,,,\n"
                    "IDRiD_001,3,2,,,\nIDRiD_002,0,0,,,\n,,,,,\n", encoding="utf-8")
    df = external.load_idrid_labels(cfg)
    assert df.to_dict("records") == [{"id_code": "IDRiD_001", "diagnosis": 3},
                                     {"id_code": "IDRiD_002", "diagnosis": 0}]


def test_idrid_labels_reject_grades_off_the_scale(tmp_path):
    cfg = _cfg(tmp_path)
    path = cfg.paths.external / external.IDRID_LABELS
    path.parent.mkdir(parents=True)
    path.write_text("id_code,diagnosis\nIDRiD_001,7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        external.load_idrid_labels(cfg)


def test_idrid_labels_explain_how_to_get_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle datasets download"):
        external.load_idrid_labels(_cfg(tmp_path))


def test_external_report_shows_the_comparison_when_given_one():
    report = {"dataset": "IDRiD", "n": 10, "variant": "baseline", "sweep": "s",
              "thresholds": [0.5, 1.5, 2.5, 3.5], "qwk": 0.8, "accuracy": 0.6,
              "macro_f1": 0.5, "ref_sensitivity": 0.9, "ref_specificity": 0.95,
              "healthy_n": 4, "healthy_called_healthy": 0.75,
              "healthy_called_referable": 0.0, "severe_missed": 1, "severe_total": 3}
    alone = external.format_report(report)
    both = external.format_report(report, {"qwk": 0.9})
    assert "APTOS test" not in alone
    assert "| QWK | 0.8000 | 0.9000 |" in both
    assert "Severe cases missed: 1/3." in both
