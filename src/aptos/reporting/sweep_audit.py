"""Recompute five-fold test metrics without loading images or pickle checkpoints."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from aptos.modeling.thresholds import (
    apply_thresholds,
    clinically_costly_errors,
    grade_metrics,
    referable_confusion,
)
from aptos.reporting.clinical_audit import wilson_interval
from aptos.reporting.research_evidence import file_sha256


def _matches(actual: dict, recorded: dict, context: str) -> None:
    for key, value in actual.items():
        if key not in recorded or not math.isfinite(float(recorded[key])):
            raise ValueError(f"{context}: missing/nonfinite metric {key}")
        if not math.isclose(float(value), float(recorded[key]), rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError(f"{context}: inconsistent metric {key}")


def _thresholds(value) -> np.ndarray:
    cuts = np.asarray(value, dtype=float)
    if cuts.shape != (4,) or not np.isfinite(cuts).all() or not (np.diff(cuts) > 0).all():
        raise ValueError("four finite, strictly ordered thresholds required")
    return cuts


def audit_sweep(directory: Path, *, expected_n: int | None = None) -> dict:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if summary["folds"] != 5 or config["cv.folds"] != 5:
        raise ValueError("this audit requires five folds")
    records, raws, cuts, hashes = [], [], [], {}
    for name in ("summary.json", "config.json"):
        hashes[name] = file_sha256(directory / name)
    reference_ids = reference_truth = None
    for fold in range(1, 6):
        metric_path = directory / f"fold{fold}.json"
        prediction_path = directory / f"fold{fold}.npz"
        record = json.loads(metric_path.read_text(encoding="utf-8"))
        if record["fold"] != fold:
            raise ValueError("fold identity differs from filename")
        with np.load(prediction_path, allow_pickle=False) as arrays:
            if not {"test_raw", "test_true", "test_ids"}.issubset(arrays.files):
                raise ValueError("raw scores, truth and authentic held-out image IDs required")
            raw = arrays["test_raw"].astype(float)
            truth = arrays["test_true"].astype(float)
            ids = arrays["test_ids"].astype(str)
        if (raw.ndim != 1 or raw.size == 0 or truth.shape != raw.shape or ids.shape != raw.shape
                or not np.isfinite(raw).all() or not np.isfinite(truth).all()
                or np.any(truth != np.floor(truth)) or np.any((truth < 0) | (truth > 4))
                or any(not ident.strip() for ident in ids) or len(set(ids)) != len(ids)):
            raise ValueError(f"fold {fold}: invalid arrays or nonunique image IDs")
        if expected_n is not None and len(raw) != expected_n:
            raise ValueError("held-out row count differs from expected cohort")
        if reference_ids is None:
            reference_ids, reference_truth = ids, truth
        elif not np.array_equal(ids, reference_ids) or not np.array_equal(truth, reference_truth):
            raise ValueError("fold image IDs or labels are not aligned")
        thresholds = _thresholds(record["thresholds"])
        pred = apply_thresholds(raw, thresholds)
        _matches({f"test_{key}": value for key, value in grade_metrics(truth, pred).items()},
                 record, f"fold {fold}")
        _matches({f"test_{key}": value for key, value in referable_confusion(truth, pred).items()},
                 record, f"fold {fold}")
        _matches(clinically_costly_errors(truth, pred), record, f"fold {fold}")
        history = record["history"]
        if ([entry["epoch"] for entry in history] != list(range(1, record["epochs_run"] + 1))
                or not 1 <= record["best_epoch"] <= record["epochs_run"] <= config["train.epochs"]):
            raise ValueError("incomplete or invalid epoch history")
        _matches({"valid_qwk": history[record["best_epoch"] - 1]["valid_qwk"]}, record,
                 f"fold {fold} selected epoch")
        records.append(record)
        raws.append(raw)
        cuts.append(thresholds)
        hashes[metric_path.name] = file_sha256(metric_path)
        hashes[prediction_path.name] = file_sha256(prediction_path)
    for key, summary_key in (("valid_qwk", "valid_qwk"), ("test_qwk", "test_qwk"),
                             ("test_accuracy", "test_acc"), ("test_macro_f1", "test_macro_f1")):
        values = [record[key] for record in records]
        _matches({f"{summary_key}_mean": np.mean(values),
                  f"{summary_key}_std": np.std(values, ddof=1)}, summary, "fold aggregation")
    mean_cuts = np.mean(cuts, axis=0)
    if not np.allclose(mean_cuts, _thresholds(summary["ensemble"]["thresholds"]), atol=1e-8):
        raise ValueError("ensemble thresholds differ from fold mean")
    ensemble_pred = apply_thresholds(np.mean(raws, axis=0), mean_cuts)
    metrics = {**grade_metrics(reference_truth, ensemble_pred),
               **{f"referable_{k}": v for k, v in referable_confusion(reference_truth, ensemble_pred).items()},
               **clinically_costly_errors(reference_truth, ensemble_pred)}
    metrics["gain_over_fold_mean"] = metrics["qwk"] - summary["test_qwk_mean"]
    _matches(metrics, summary["ensemble"], "ensemble")
    provenance_path = directory / "provenance.json"
    provenance = None
    if provenance_path.exists():
        raw_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance = {key: raw_provenance[key] for key in (
            "cache_settings", "identity_version", "labels_sha256", "run_id", "run_kind"
        ) if key in raw_provenance}
        hashes[provenance_path.name] = file_sha256(provenance_path)
    return {
        "schema_version": 1, "status": "PASS_RECORDED_PREDICTION_AUDIT",
        "sweep": directory.name, "n_test_images": len(reference_truth), "folds": 5,
        "config": {key: value for key, value in config.items()
                   if key == "variant" or key.startswith(("preprocess.", "quality.", "augment.",
                                                          "thresholds.", "train.", "cv.", "confound."))},
        "recomputed_ensemble": metrics,
        "folds_checked": [{k: record[k] for k in ("fold", "best_epoch", "epochs_run", "test_qwk")}
                          for record in records],
        "inputs_sha256": hashes,
        "recorded_provenance": provenance,
        "wilson95_independent_eye_assumption": {
            "sensitivity": wilson_interval(metrics["referable_tp"],
                                           metrics["referable_tp"] + metrics["referable_fn"]),
            "specificity": wilson_interval(metrics["referable_tn"],
                                           metrics["referable_tn"] + metrics["referable_fp"]),
        },
        "limitations": [
            "Recomputes recorded predictions; does not rerun checkpoint inference or training.",
            "Checkpoint contents, source image hashes and patient identities are not audited here.",
            "Shared test images do not constitute five independent evaluation cohorts.",
            "No external evaluation or ordinal-model comparison is established for this sweep.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--expected-n", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(audit_sweep(args.sweep, expected_n=args.expected_n), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
