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


@pytest.mark.pure
def test_missing_grade_recall_is_json_null_not_nan():
    import json

    eyes = [f"e{i}" for i in range(20)]
    labels = [0, 2] * 10
    frame = pd.DataFrame({"id_code": eyes, "diagnosis": labels, "raw": labels})
    result = compare_frames(frame, frame, [0.5, 1.5, 2.5, 3.5],
                            [0.5, 1.5, 2.5, 3.5], n_resamples=200)
    assert result["baseline"]["metrics"]["recall_grade_1"] is None
    assert result["candidate"]["metrics"]["recall_grade_4"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.pure
def test_kaggle_experiment_notebook_is_valid_and_off_by_default():
    import ast
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "notebooks" / "ordinal_experiment_matrix.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    combined = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert "RUN_EXPERIMENTS = False" in combined
    for cell in code_cells:
        assert cell["execution_count"] is None and cell["outputs"] == []
        ast.parse("".join(cell["source"]))



@pytest.mark.pure
def test_external_comparison_cache_checks_checkpoints_pixels_labels_and_csv(tmp_path):
    import json

    from aptos.config import Config, Paths
    from aptos.data import manifest as manifest_mod
    from aptos.evaluation.compare_sweeps import verified_external_comparison_frame
    from aptos.evaluation.external import (
        _sha256,
        dataset,
        external_source_signature,
        score_input_signature,
        source_provenance_path,
    )

    cfg = Config.load("configs/cv.yaml")
    cfg.paths = Paths(root=tmp_path)
    cfg.cv.folds = 2
    folder = cfg.paths.external
    folder.mkdir(parents=True)
    cache = folder / "idrid_baseline"
    cache.mkdir()
    labels = pd.DataFrame({
        "id_code": ["eye-a", "eye-b"],
        "diagnosis": [0, 2],
    })
    for name in labels["id_code"]:
        (cache / f"{name}.jpg").write_bytes(f"processed-{name}".encode())
    manifest_mod.write(cache, cfg.manifest_fields(), n_images=2)
    original_images = folder / "Imagenes" / "Imagenes"
    original_images.mkdir(parents=True)
    for name in labels["id_code"]:
        (original_images / f"{name}.jpg").write_bytes(f"raw-{name}".encode())
    source_provenance_path(cache).write_text(
        json.dumps(external_source_signature(cfg, dataset("idrid"), labels))
    )
    sweep = tmp_path / "models" / "cv" / "baseline-test"
    sweep.mkdir(parents=True)
    (sweep / "config.json").write_text(json.dumps(cfg.to_dict()))
    for fold in (1, 2):
        (sweep / f"fold{fold}.pt").write_bytes(f"weights-{fold}".encode())

    output = folder / f"idrid_baseline_{sweep.name}_scores.csv"
    scores = labels.copy()
    scores["raw"] = [0.1, 2.1]
    scores.to_csv(output, index=False)
    meta = output.with_suffix(".meta.json")

    def sign():
        meta.write_text(json.dumps({
            "input_signature": score_input_signature(cfg, sweep, cache, labels),
            "score_csv_sha256": _sha256(output),
        }))

    sign()
    matched, path = verified_external_comparison_frame(sweep, folder, cfg, labels, "idrid")
    assert path == output and matched["raw"].tolist() == pytest.approx([0.1, 2.1])

    for changed_file in (output, cache / "eye-a.jpg", sweep / "fold1.pt"):
        before = changed_file.read_bytes()
        changed_file.write_bytes(before + b"-tampered")
        with pytest.raises(ValueError, match="missing or stale"):
            verified_external_comparison_frame(sweep, folder, cfg, labels, "idrid")
        changed_file.write_bytes(before)

    tampered_labels = labels.copy()
    tampered_labels.loc[0, "diagnosis"] = 4
    with pytest.raises(ValueError, match="missing or stale"):
        verified_external_comparison_frame(
            sweep, folder, cfg, tampered_labels, "idrid",
        )

    (original_images / "eye-b.jpg").write_bytes(b"new-upstream-retina")
    with pytest.raises(manifest_mod.ManifestMismatch, match="source provenance"):
        verified_external_comparison_frame(sweep, folder, cfg, labels, "idrid")
    (original_images / "eye-b.jpg").write_bytes(b"raw-eye-b")

    (sweep / "fold2.pt").unlink()
    with pytest.raises(ValueError, match="every CV checkpoint"):
        verified_external_comparison_frame(sweep, folder, cfg, labels, "idrid")


@pytest.mark.pure
def test_external_comparison_rejects_unsigned_historical_csv(tmp_path):
    import json

    from aptos.config import Config, Paths
    from aptos.data import manifest as manifest_mod
    from aptos.evaluation.compare_sweeps import verified_external_comparison_frame
    from aptos.evaluation.external import (
        dataset,
        external_source_signature,
        source_provenance_path,
    )

    cfg = Config.load("configs/cv.yaml")
    cfg.paths = Paths(root=tmp_path)
    cfg.cv.folds = 1
    folder = cfg.paths.external
    folder.mkdir(parents=True)
    cache = folder / "idrid_baseline"
    cache.mkdir()
    (cache / "retina.jpg").write_bytes(b"synthetic-cache")
    manifest_mod.write(cache, cfg.manifest_fields(), n_images=1)
    originals = folder / "Imagenes" / "Imagenes"
    originals.mkdir(parents=True)
    (originals / "retina.jpg").write_bytes(b"real-raw-image")
    labels = pd.DataFrame({"id_code": ["retina"], "diagnosis": [2]})
    source_provenance_path(cache).write_text(
        json.dumps(external_source_signature(cfg, dataset("idrid"), labels))
    )
    sweep = tmp_path / "baseline-test"
    sweep.mkdir()
    (sweep / "config.json").write_text(json.dumps(cfg.to_dict()))
    (sweep / "fold1.pt").write_bytes(b"synthetic-weights")
    output = folder / "idrid_baseline_baseline-test_scores.csv"
    pd.DataFrame({"id_code": ["retina"], "diagnosis": [2], "raw": [2.2]}).to_csv(
        output, index=False,
    )
    with pytest.raises(ValueError, match="missing or stale"):
        verified_external_comparison_frame(sweep, folder, cfg, labels, "idrid")


@pytest.mark.pure
def test_no_predicted_referrals_is_json_serializable_without_fake_precision(sample):
    import json

    baseline, candidate = sample
    baseline = baseline.copy()
    candidate = candidate.copy()
    baseline["raw"] = -100.0
    candidate["raw"] = -100.0
    result = compare_frames(
        baseline, candidate, [0.5, 1.5, 2.5, 3.5],
        [0.5, 1.5, 2.5, 3.5], n_resamples=120,
    )
    assert result["baseline"]["referable"]["tp"] == 0
    assert result["baseline"]["referable"]["precision"] is None
    assert result["candidate"]["referable"]["precision"] is None
    assert result["baseline"]["referable"]["sensitivity"] == 0.0
    assert result["baseline"]["referable"]["specificity"] == 1.0
    json.dumps(result, allow_nan=False)


@pytest.mark.pure
def test_compare_frames_exposes_paired_grading_intervals(sample):
    baseline, candidate = sample
    result = compare_frames(
        baseline, candidate, [0.5, 1.5, 2.5, 3.5],
        [0.5, 1.5, 2.5, 3.5], n_resamples=150, seed=7,
    )
    grades = result["paired_grading_uncertainty"]
    assert grades["analysis_unit"] == "eye"
    assert grades["n_eyes"] == len(baseline)
    assert grades["metrics"]["qwk"]["difference"] == pytest.approx(
        result["ordinal_minus_baseline"]["qwk"]
    )
    assert grades["metrics"]["macro_f1"]["difference"] == pytest.approx(
        result["ordinal_minus_baseline"]["macro_f1"]
    )


@pytest.mark.pure
def test_paired_report_decomposes_grade2_missed_referrals(sample):
    import json

    baseline, candidate = sample
    report = compare_frames(
        baseline, candidate,
        [0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5, 3.5],
        n_resamples=150,
    )
    baseline_g2 = report["baseline"]["grade_referral_breakdown"]["2"]
    ordinal_g2 = report["candidate"]["grade_referral_breakdown"]["2"]
    discordance = report["paired_referral_discordance"]["grade_2"]
    assert baseline_g2["n"] == ordinal_g2["n"] == discordance["n"] == 3
    assert baseline_g2["missed_referable"] == 3
    assert ordinal_g2["missed_referable"] == 0
    assert discordance["baseline_only_missed"] == 3
    assert discordance["candidate_only_missed"] == 0
    assert report["schema_version"] == 3
    json.dumps(report, allow_nan=False)
