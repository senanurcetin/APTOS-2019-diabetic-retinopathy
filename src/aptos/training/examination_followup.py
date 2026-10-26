"""Separate five-model Messidor follow-up; frozen split, paired arms, local outputs.

The baseline is the audited seed-42 scalar sweep, not the historical deployed
ensemble. Outputs cannot overwrite an existing experiment. No paid fallback.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from aptos.config import Config, Paths
from aptos.evaluation import external
from aptos.evaluation.examination_split import validate_examination_split
from aptos.evaluation.paired_uncertainty import paired_bootstrap
from aptos.modeling.thresholds import apply_thresholds, grade_metrics, referable_confusion
from aptos.training.finetune import FineTuneConfig, fine_tune_fold

ARMS = ("baseline", "adjudicated", "control")
CONTRASTS = {
    "adjudicated_minus_control": ("control", "adjudicated"),
    "adjudicated_minus_baseline": ("baseline", "adjudicated"),
    "control_minus_baseline": ("baseline", "control"),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_baseline(sweep: Path, audit: dict) -> list[Path]:
    """Metadata identity and exactly five complete checkpoints before tensor loading."""
    if sweep.name != audit.get("sweep"):
        raise ValueError("baseline sweep identity differs from the audited seed-42 run")
    for name, expected in audit["inputs_sha256"].items():
        if not (sweep / name).is_file() or digest(sweep / name) != expected:
            raise ValueError(f"audited baseline artifact missing or changed: {name}")
    files = external.complete_sweep_checkpoints(sweep)
    if len(files) != 5:
        raise ValueError("the follow-up requires all five base models")
    return files


def align_source_filenames(grades: pd.DataFrame, images: Path) -> pd.DataFrame:
    """Resolve verified image stems to exact mirror casing on case-sensitive hosts."""
    lookup = {}
    for path in images.iterdir():
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            if path.stem in lookup:
                raise ValueError("ambiguous image stems in source mirror")
            lookup[path.stem] = path.name
    frame = grades.copy()
    names = []
    for name in frame.image_id:
        if not isinstance(name, str) or "/" in name or "\\" in name or not name.strip():
            raise ValueError("invalid reference image filename")
        actual = lookup.get(Path(name).stem)
        if actual is None:
            raise ValueError("source mirror is missing a graded image")
        names.append(actual)
    frame["image_id"] = names
    return frame


def paired_auc_intervals(frame: pd.DataFrame, *, draws: int = 2000, seed: int = 42) -> dict:
    """Same examination draws for all three fixed-model AUC contrasts."""
    if not isinstance(draws, int) or isinstance(draws, bool) or draws < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    required = {"id_code", "exam_id", "diagnosis", *(f"{arm}_raw" for arm in ARMS)}
    if not required <= set(frame.columns) or frame.empty:
        raise ValueError("aligned image IDs, examination keys, grades and raw scores required")
    if frame.id_code.isna().any() or frame.id_code.duplicated().any():
        raise ValueError("image IDs must be unique and present")
    if frame.exam_id.isna().any() or frame.exam_id.astype(str).str.strip().eq("").any():
        raise ValueError("supplied examination IDs are required")
    truth = frame.diagnosis.to_numpy(dtype=float)
    if (not np.isfinite(truth).all() or np.any(truth != np.floor(truth))
            or np.any((truth < 0) | (truth > 4))):
        raise ValueError("diagnoses must be whole grades 0..4")
    truth = truth >= 2
    if np.unique(truth).size != 2:
        raise ValueError("both referable classes are required")
    raw = {arm: frame[f"{arm}_raw"].to_numpy(dtype=float) for arm in ARMS}
    if any(not np.isfinite(values).all() for values in raw.values()):
        raise ValueError("raw predictions must be finite")
    _, inverse = np.unique(frame.exam_id.astype(str), return_inverse=True)
    clusters = [np.flatnonzero(inverse == i) for i in range(int(inverse.max()) + 1)]
    if len(clusters) < 2:
        raise ValueError("at least two supplied examinations required")
    point = {arm: float(roc_auc_score(truth, scores)) for arm, scores in raw.items()}
    rng = np.random.default_rng(seed)
    values = {name: [] for name in CONTRASTS}
    rejected = 0
    for _ in range(draws):
        selected = np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))])
        if np.unique(truth[selected]).size != 2:
            rejected += 1
            continue
        auc = {arm: roc_auc_score(truth[selected], scores[selected]) for arm, scores in raw.items()}
        for name, (first, second) in CONTRASTS.items():
            values[name].append(float(auc[second] - auc[first]))
    if draws - rejected < max(50, int(draws * 0.8)):
        raise ValueError("too many single-class bootstrap draws")
    return {
        "unit": "examination", "n_eyes": len(frame), "n_examinations": len(clusters),
        "draws": draws, "seed": seed, "single_class_draws_rejected": rejected,
        "method": "paired percentile examination bootstrap, 95% intervals",
        "auc": point,
        "contrasts": {
            name: {"difference": point[second] - point[first],
                   "ci_lower": float(np.quantile(values[name], 0.025)),
                   "ci_upper": float(np.quantile(values[name], 0.975))}
            for name, (first, second) in CONTRASTS.items()
        },
    }


def metric_block(truth, raw, cuts) -> dict:
    grade = apply_thresholds(np.asarray(raw), cuts)
    return {
        "auc": float(roc_auc_score(np.asarray(truth) >= 2, raw)),
        "thresholds": [float(cut) for cut in cuts],
        **grade_metrics(truth, grade),
        "referable": referable_confusion(truth, grade),
    }


def training_frames(split: pd.DataFrame, pseudo_grades, arm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Only training targets vary; adjudicated validation is identical in both arms."""
    if arm not in ("adjudicated", "control") or len(pseudo_grades) != len(split):
        raise ValueError("unknown arm or unaligned pseudo-label array")
    train = split.loc[split.part.eq("tune-train"), ["id_code", "diagnosis"]].copy()
    valid = split.loc[split.part.eq("tune-valid"), ["id_code", "diagnosis"]].copy()
    if arm == "control":
        pseudo = pd.Series(np.asarray(pseudo_grades), index=split.index)
        train["diagnosis"] = pseudo.loc[train.index].to_numpy()
    return train, valid


def run(sweep: Path, split_path: Path, plan_path: Path, pairs_path: Path,
        runtime_root: Path, output: Path, repository: Path) -> dict:
    """Run only on CUDA after validating frozen data/sweep contracts."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if digest(split_path) != plan["split_csv_sha256"] or digest(pairs_path) != plan["inputs_sha256"]["pairs"]:
        raise ValueError("split or examination pairing bytes differ from the frozen plan")
    split = pd.read_csv(split_path)
    pairs = pd.read_csv(pairs_path, sep=";")
    cfg = Config.load(repository / "configs/baseline.yaml")
    cfg.paths = Paths(root=runtime_root)
    cfg.train = dataclasses.replace(cfg.train, workers=2)
    cfg.preprocess = dataclasses.replace(cfg.preprocess, workers=4)
    labels = external.load_messidor2_labels(cfg)
    validate_examination_split(split, labels, pairs)
    base_audit = json.loads((repository / "reports/research/kaggle_reg42_audit.json").read_text())
    checkpoints = verify_baseline(sweep, base_audit)
    external.validate_sweep_contract(cfg, sweep)
    # The attached audited run contains the exact 366 cached APTOS test eyes.
    retention_cache = sweep.parents[2] / "data/processed/test"
    with np.load(sweep / "fold1.npz", allow_pickle=False) as saved:
        retention = pd.DataFrame({"id_code": saved["test_ids"], "diagnosis": saved["test_true"]})
    if len(retention) != 366 or any(not (retention_cache / f"{ident}.jpg").is_file()
                                    for ident in retention.id_code):
        raise ValueError("the audited APTOS retention cache is incomplete")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU marathon or paid accelerator fallback")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "RUNNING", "baseline_sweep": sweep.name,
        "split_sha256": digest(split_path), "plan_sha256": digest(plan_path),
        "checkpoint_sha256": {path.name: digest(path) for path in checkpoints},
        "fine_tuned_checkpoint_sha256": {},
        "configuration": cfg.to_dict(), "fine_tuning": dataclasses.asdict(FineTuneConfig()),
        "validation_targets": "adjudicated in both arms; only training targets vary",
        "gpu": torch.cuda.get_device_name(0), "torch_version": str(torch.__version__),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
        ).strip(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    states, original_cuts = external.load_ensemble(sweep)
    cache = external.build_cache(cfg, dataset_key="messidor2")
    manifest["processed_cache_source_provenance_sha256"] = digest(external.source_provenance_path(cache))
    print("Computing unchanged baseline scores and fixed control labels", flush=True)
    baseline_raw = external.predict(cfg, cache, split, states, "cuda")
    retention["baseline_raw"] = external.predict(cfg, retention_cache, retention, states, "cuda")
    original_raw = []
    for fold in range(1, 6):
        with np.load(sweep / f"fold{fold}.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["test_ids"], retention.id_code.to_numpy()):
                raise ValueError("saved APTOS retention predictions have changed row order")
            original_raw.append(saved["test_raw"])
    saved_ensemble_raw = np.mean(original_raw, axis=0)
    pseudo = apply_thresholds(baseline_raw, original_cuts)
    split.assign(baseline_raw=baseline_raw, control_grade=pseudo).to_csv(
        output / "baseline_all_predictions.csv", index=False, lineterminator="\n",
    )
    test_mask = split.part.eq("test")
    predictions = split.loc[test_mask, ["id_code", "exam_id", "diagnosis"]].reset_index(drop=True)
    predictions["baseline_raw"] = baseline_raw[test_mask.to_numpy()]
    cuts = {"baseline": original_cuts}
    # Baseline predictions are not used to change the allocation or hyperparameters.
    for arm in ("adjudicated", "control"):
        train, valid = training_frames(split, pseudo, arm)
        arm_states, arm_cuts = [], []
        folder = output / arm
        folder.mkdir()
        for fold, state in enumerate(states, start=1):
            print(f"{arm}: fold {fold}/5, train={len(train)}, valid={len(valid)}", flush=True)
            best, history, summary = fine_tune_fold(
                cfg, state, train, valid, cache, FineTuneConfig(), "cuda",
            )
            torch.save({"state_dict": best["state"], "thresholds": best["thresholds"],
                        "fold": fold, "arm": arm, "split_sha256": manifest["split_sha256"],
                        "history": history, **summary}, folder / f"fold{fold}.pt")
            manifest["fine_tuned_checkpoint_sha256"][f"{arm}/fold{fold}.pt"] = digest(folder / f"fold{fold}.pt")
            (folder / f"fold{fold}.json").write_text(json.dumps({
                "fold": fold, "thresholds": best["thresholds"], "history": history, **summary,
            }, indent=2) + "\n")
            arm_states.append(best["state"])
            arm_cuts.append(best["thresholds"])
            gc.collect()
            torch.cuda.empty_cache()
        cuts[arm] = np.mean(arm_cuts, axis=0)
        predictions[f"{arm}_raw"] = external.predict(cfg, cache, predictions, arm_states, "cuda")
        retention[f"{arm}_raw"] = external.predict(cfg, retention_cache, retention, arm_states, "cuda")
        del arm_states
    truth = predictions.diagnosis.to_numpy()
    report = {
        "experiment": "messidor2-examination-followup-v2", "baseline_sweep": sweep.name,
        "n_test_eyes": len(predictions), "n_test_examinations": predictions.exam_id.nunique(),
        "primary_contrast": "adjudicated_minus_control referable AUC",
        "fixed_original_cuts": {}, "validation_fitted_cuts": {},
        "aptos_retention": {"n": len(retention), "fixed_original_cuts": {},
                            "messidor_validation_cuts": {},
                            "baseline_rescore_max_abs_difference": float(np.max(np.abs(
                                retention.baseline_raw.to_numpy() - saved_ensemble_raw))),
                            "baseline_rescore_grade_mismatches": int(np.sum(
                                apply_thresholds(retention.baseline_raw.to_numpy(), original_cuts)
                                != apply_thresholds(saved_ensemble_raw, original_cuts)))},
    }
    for arm in ARMS:
        raw = predictions[f"{arm}_raw"].to_numpy()
        report["fixed_original_cuts"][arm] = metric_block(truth, raw, original_cuts)
        report["validation_fitted_cuts"][arm] = metric_block(truth, raw, cuts[arm])
        predictions[f"{arm}_grade_frozen"] = apply_thresholds(raw, original_cuts)
        predictions[f"{arm}_grade_validation"] = apply_thresholds(raw, cuts[arm])
        retention_raw = retention[f"{arm}_raw"].to_numpy()
        report["aptos_retention"]["fixed_original_cuts"][arm] = metric_block(
            retention.diagnosis.to_numpy(), retention_raw, original_cuts,
        )
        report["aptos_retention"]["messidor_validation_cuts"][arm] = metric_block(
            retention.diagnosis.to_numpy(), retention_raw, cuts[arm],
        )
    retention.to_csv(output / "aptos_retention_predictions.csv", index=False, lineterminator="\n")
    predictions.to_csv(output / "test_predictions.csv", index=False, lineterminator="\n")
    report["paired_auc"] = paired_auc_intervals(predictions)
    report["paired_fixed_cut_referral"] = {}
    for name, (first, second) in CONTRASTS.items():
        comparison = predictions[["id_code", "exam_id", "diagnosis"]].copy()
        comparison["baseline_grade"] = predictions[f"{first}_grade_frozen"]
        comparison["candidate_grade"] = predictions[f"{second}_grade_frozen"]
        comparison.to_csv(output / f"paired_{name}.csv", index=False, lineterminator="\n")
        report["paired_fixed_cut_referral"][name] = paired_bootstrap(
            truth, comparison.baseline_grade, comparison.candidate_grade,
            unit="examination", examination_ids=comparison.exam_id, n_resamples=2000, seed=42,
        )
    report["limitations"] = [
        "Exploratory follow-up on a previously inspected cohort; not fresh external confirmation.",
        "Third-party examination mapping; original ADCIS parity and patient identity unverified.",
        "New baseline sweep differs from the historical 0.925 AUC experiment.",
        "Grade-4 support: two validation eyes and 24 test eyes; rare-grade metrics are unstable.",
        "Secondary contrasts are descriptive; bootstrap conditions on these fitted models.",
        "APTOS retention reuses the inspected 366-eye holdout; no fresh IDRiD or prospective assessment.",
        "No automatic model promotion or clinical-use claim.",
    ]
    report["prediction_sha256"] = digest(output / "test_predictions.csv")
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest["status"] = "COMPLETE"
    manifest["result_sha256"] = digest(output / "results.json")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report["paired_auc"], indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sweep", "split", "plan", "pairs", "runtime-root", "output", "repository"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.sweep, args.split, args.plan, args.pairs, args.runtime_root, args.output, args.repository)


if __name__ == "__main__":
    main()
