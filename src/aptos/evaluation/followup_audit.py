"""Recompute the v2 follow-up from downloaded per-eye scores, without model tensors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aptos.evaluation.paired_uncertainty import paired_bootstrap
from aptos.modeling.thresholds import apply_thresholds
from aptos.training.examination_followup import (
    ARMS,
    CONTRASTS,
    digest,
    metric_block,
    paired_auc_intervals,
)


def assert_same(actual, expected, location="result") -> None:
    """Compare numeric trees; reject missing keys, nonfinite values and optimistic edits."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"changed keys at {location}")
        for key in expected:
            assert_same(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"changed list at {location}")
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            assert_same(a, b, f"{location}[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if (isinstance(actual, bool) or not isinstance(actual, (int, float)) or not np.isfinite(actual)
                or not np.isclose(actual, expected, atol=1e-7, rtol=1e-7)):
            raise ValueError(f"numeric disagreement at {location}")
    elif actual != expected:
        raise ValueError(f"disagreement at {location}")


def aligned(frame: pd.DataFrame, reference: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not set(columns) <= set(frame) or frame.id_code.duplicated().any():
        raise ValueError("missing or duplicated prediction identifiers")
    if set(frame.id_code) != set(reference.id_code):
        raise ValueError("prediction cohort differs from frozen reference")
    indexed = frame.set_index("id_code").loc[reference.id_code].reset_index()
    for name in columns:
        if not np.array_equal(indexed[name].to_numpy(), reference[name].to_numpy()):
            raise ValueError(f"prediction reference mismatch: {name}")
    return indexed


def audit(output: Path, split_path: Path, base_metadata: Path, *, expected_source: str) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    report = json.loads((output / "results.json").read_text())
    if manifest["status"] != "COMPLETE" or manifest["source_commit"] != expected_source:
        raise ValueError("execution is incomplete or source differs from pinned launch")
    if digest(output / "results.json") != manifest["result_sha256"]:
        raise ValueError("downloaded result hash differs from completed manifest")
    if digest(split_path) != manifest["split_sha256"]:
        raise ValueError("frozen split bytes differ from execution")
    if manifest["validation_targets"] != "adjudicated in both arms; only training targets vary":
        raise ValueError("validation reference differs between arms")
    assert_same(manifest["fine_tuning"], {"lr": 1e-4, "weight_decay": 1e-4, "epochs": 8, "seed": 42})
    split = pd.read_csv(split_path)
    reference = split.loc[split.part.eq("test")].reset_index(drop=True)
    if len(reference) != 872 or reference.exam_id.nunique() != 437:
        raise ValueError("unexpected follow-up test cohort size")
    raw = pd.read_csv(output / "test_predictions.csv")
    # The pinned inference routine returns float32. Restore that dtype from
    # its shortest CSV representation before checking threshold-boundary grades.
    for arm in ARMS:
        raw[f"{arm}_raw"] = raw[f"{arm}_raw"].astype(np.float32)
    aligned(raw, reference, ["id_code", "diagnosis", "exam_id"])
    if digest(output / "test_predictions.csv") != report["prediction_sha256"]:
        raise ValueError("prediction bytes differ from result fingerprint")
    source_summary = json.loads((base_metadata / "summary.json").read_text())
    original_cuts = source_summary["ensemble"]["thresholds"]
    all_baseline = aligned(pd.read_csv(output / "baseline_all_predictions.csv"), split,
                           ["id_code", "diagnosis", "exam_id", "part"])
    if not np.array_equal(all_baseline.control_grade,
                          apply_thresholds(all_baseline.baseline_raw.to_numpy(dtype=np.float32), original_cuts)):
        raise ValueError("control labels differ from frozen original baseline cuts")
    for name, (first, second) in CONTRASTS.items():
        comparison = aligned(pd.read_csv(output / f"paired_{name}.csv"), raw,
                             ["id_code", "diagnosis", "exam_id"])
        if (not np.array_equal(comparison.baseline_grade, raw[f"{first}_grade_frozen"])
                or not np.array_equal(comparison.candidate_grade, raw[f"{second}_grade_frozen"])):
            raise ValueError("paired comparison file differs from wide prediction table")
    cuts = {"baseline": original_cuts}
    for arm in ("adjudicated", "control"):
        values = []
        for fold in range(1, 6):
            record = json.loads((output / arm / f"fold{fold}.json").read_text())
            history = record["history"]
            if len(history) != 8 or record["fold"] != fold:
                raise ValueError("incomplete eight-epoch fold history")
            best = max(history, key=lambda row: row["valid_qwk"])
            if record["best_epoch"] != best["epoch"]:
                raise ValueError("checkpoint epoch disagrees with validation history")
            assert_same(record["valid_qwk"], best["valid_qwk"])
            values.append(record["thresholds"])
        cuts[arm] = np.mean(values, axis=0)
    expected = {f"{arm}/fold{fold}.pt" for arm in ("adjudicated", "control") for fold in range(1, 6)}
    if set(manifest["fine_tuned_checkpoint_sha256"]) != expected:
        raise ValueError("ten new fine-tuned checkpoint fingerprints required")
    for arm in ARMS:
        scores = raw[f"{arm}_raw"].to_numpy()
        for section, thresholds in (("fixed_original_cuts", original_cuts),
                                    ("validation_fitted_cuts", cuts[arm])):
            assert_same(report[section][arm], metric_block(raw.diagnosis, scores, thresholds))
        for suffix, thresholds in (("frozen", original_cuts), ("validation", cuts[arm])):
            if not np.array_equal(raw[f"{arm}_grade_{suffix}"], apply_thresholds(scores, thresholds)):
                raise ValueError("saved grades disagree with saved raw scores and cuts")
    assert_same(report["paired_auc"], paired_auc_intervals(raw))
    for name, (first, second) in CONTRASTS.items():
        result = paired_bootstrap(
            raw.diagnosis, raw[f"{first}_grade_frozen"], raw[f"{second}_grade_frozen"],
            unit="examination", examination_ids=raw.exam_id, n_resamples=2000, seed=42,
        )
        assert_same(report["paired_fixed_cut_referral"][name], result)
    with np.load(base_metadata / "fold1.npz", allow_pickle=False) as saved:
        aptos_reference = pd.DataFrame({"id_code": saved["test_ids"], "diagnosis": saved["test_true"]})
    retention = aligned(pd.read_csv(output / "aptos_retention_predictions.csv"), aptos_reference,
                        ["id_code", "diagnosis"])
    saved_raw = []
    for fold in range(1, 6):
        with np.load(base_metadata / f"fold{fold}.npz", allow_pickle=False) as saved:
            saved_raw.append(saved["test_raw"])
    saved_raw = np.mean(saved_raw, axis=0)
    for arm in ARMS:
        retention[f"{arm}_raw"] = retention[f"{arm}_raw"].astype(np.float32)
    assert_same(report["aptos_retention"]["baseline_rescore_max_abs_difference"],
                float(np.max(np.abs(retention.baseline_raw.to_numpy() - saved_raw))))
    assert_same(report["aptos_retention"]["baseline_rescore_grade_mismatches"], int(np.sum(
        apply_thresholds(retention.baseline_raw.to_numpy(), original_cuts)
        != apply_thresholds(saved_raw, original_cuts),
    )))
    for arm in ARMS:
        scores = retention[f"{arm}_raw"].to_numpy()
        assert_same(report["aptos_retention"]["fixed_original_cuts"][arm],
                    metric_block(retention.diagnosis, scores, original_cuts))
        assert_same(report["aptos_retention"]["messidor_validation_cuts"][arm],
                    metric_block(retention.diagnosis, scores, cuts[arm]))
    return {
        "status": "PASS_PER_EYE_RECOMPUTATION", "source_commit": expected_source,
        "n_messidor_test_eyes": len(raw), "n_test_examinations": raw.exam_id.nunique(),
        "n_aptos_retention_eyes": len(retention), "new_fold_histories_checked": 10,
        "paired_auc": report["paired_auc"],
        "inputs_sha256": {name: digest(output / name) for name in (
            "manifest.json", "results.json", "test_predictions.csv", "aptos_retention_predictions.csv",
        )},
        "limitations": ["Audits saved predictions and histories, not prospective model validity.",
                        "Weight fingerprints are recorded by the run; tensors are not loaded here.",
                        "Reused cohorts and examination-map provenance limitations remain."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "split", "base-metadata", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    result = audit(args.run, args.split, args.base_metadata, expected_source=args.source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
