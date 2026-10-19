"""Research-only, paired MSE-vs-ordinal benchmark on identical retinal eyes.

Inputs are genuine saved sweep outputs. It never trains models, creates
per-image predictions from aggregate reports, or uses a held-out cohort for
threshold fitting. Ordinal and scalar regression thresholds come from their
respective APTOS fold-validation results, not the external cohort.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aptos.evaluation.paired_uncertainty import paired_bootstrap
from aptos.modeling.thresholds import (
    apply_thresholds,
    grade_metrics,
    referable_confusion,
)
from aptos.reporting.research_evidence import file_sha256

ESSENTIAL_MATCH = (
    "variant", "train.model", "train.seed", "train.size", "train.epochs",
    "train.lr", "train.weight_decay", "train.batch", "train.exclude_leaked",
    "cv.folds", "cv.stratify_on", "cv.pool_splits",
    "preprocess.size", "preprocess.clahe", "preprocess.square_mode",
    "preprocess.clip_limit",
)
# A reporting backend should not make scientific model experiments
# incomparable. Every *scientific* config key must match except training mode.
IGNORED_PREFIX = "tracking."


def compare_sweep_configs(baseline: dict, ordinal: dict) -> dict:
    """Refuse cross-cohort or non-controlled comparisons."""
    if baseline.get("train.mode") != "reg" or ordinal.get("train.mode") != "ordinal":
        raise ValueError("expected reg baseline and ordinal candidate sweep configs")
    for key in ESSENTIAL_MATCH:
        if key not in baseline or key not in ordinal:
            raise ValueError(f"missing required experimental parameter: {key}")
    keys = (set(baseline) | set(ordinal)) - {"train.mode"}
    mismatches = sorted(
        key for key in keys if not key.startswith(IGNORED_PREFIX)
        and baseline.get(key) != ordinal.get(key)
    )
    if mismatches:
        raise ValueError(f"non-comparable sweep settings: {', '.join(mismatches)}")
    return {key: baseline[key] for key in ESSENTIAL_MATCH}


def validated_thresholds(summary: dict) -> np.ndarray:
    """Use only the APTOS-validation-fitted ensemble thresholds."""
    try:
        cuts = np.asarray(summary["ensemble"]["thresholds"], dtype=float)
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("missing four APTOS ensemble thresholds") from exc
    if (cuts.ndim != 1 or len(cuts) != 4 or not np.all(np.isfinite(cuts))
            or not np.all(np.diff(cuts) > 0)):
        raise ValueError("four ordered finite ensemble thresholds required")
    return cuts


def _clean_eye_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"id_code", "diagnosis", "raw"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{name}: missing id_code, diagnosis or raw")
    df = frame.loc[:, ["id_code", "diagnosis", "raw"]].copy()
    ids = df["id_code"].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError(f"{name}: image IDs must be present and unique")
    df["id_code"] = ids.astype(str)
    try:
        grades = pd.to_numeric(df["diagnosis"], errors="raise").to_numpy(dtype=float)
        raw = pd.to_numeric(df["raw"], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: grades and scores must be numeric") from exc
    if (not np.all(np.isfinite(grades)) or not np.all((grades >= 0) & (grades <= 4))
            or not np.all(grades == np.floor(grades))):
        raise ValueError(f"{name}: diagnosis must be an integer grade 0–4")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{name}: raw severity scores must be finite")
    df["diagnosis"] = grades.astype(int)
    df["raw"] = raw
    if len(df) == 0:
        raise ValueError(f"{name}: no eyes in the cohort")
    return df


def compare_frames(
    baseline_frame: pd.DataFrame,
    ordinal_frame: pd.DataFrame,
    baseline_cuts,
    ordinal_cuts,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
    patient_map: dict[str, str] | None = None,
) -> dict:
    """Compare two models after a strict one-to-one identity and label join."""
    a = _clean_eye_frame(baseline_frame, "baseline")
    b = _clean_eye_frame(ordinal_frame, "ordinal")
    if set(a["id_code"]) != set(b["id_code"]):
        raise ValueError("models scored different sets of eyes")
    joined = a.merge(b, on="id_code", validate="one_to_one", suffixes=("_baseline", "_ordinal"))
    joined = joined.sort_values("id_code").reset_index(drop=True)
    if not np.array_equal(joined["diagnosis_baseline"], joined["diagnosis_ordinal"]):
        raise ValueError("same eye has inconsistent ground-truth diagnoses")
    actual = joined["diagnosis_baseline"].to_numpy(dtype=int)
    a_pred = apply_thresholds(joined["raw_baseline"].to_numpy(), baseline_cuts)
    b_pred = apply_thresholds(joined["raw_ordinal"].to_numpy(), ordinal_cuts)
    if np.any((a_pred < 0) | (a_pred > 4) | (b_pred < 0) | (b_pred > 4)):
        raise ValueError("predicted grades must lie in 0..4")

    if patient_map is not None:
        if set(patient_map) != set(joined["id_code"]):
            raise ValueError("patient linkage must match exactly the evaluated image IDs")
        patients = [patient_map[ident] for ident in joined["id_code"]]
        if any(not isinstance(p, str) or not p.strip() for p in patients):
            raise ValueError("patient IDs must be nonempty real identifiers")
        unit = "patient"
    else:
        patients, unit = None, "eye"

    paired = paired_bootstrap(
        actual, a_pred, b_pred, unit=unit, patient_ids=patients,
        n_resamples=n_resamples, seed=seed,
    )
    # Some cohorts lack rare classes; absent-grade recall is undefined, not
    # zero. JSON null preserves that distinction without writing NaN tokens.
    metric_a = grade_metrics(actual, a_pred)
    metric_b = grade_metrics(actual, b_pred)
    metric_a = {k: (float(v) if np.isfinite(v) else None) for k, v in metric_a.items()}
    metric_b = {k: (float(v) if np.isfinite(v) else None) for k, v in metric_b.items()}
    return {
        "schema_version": 1,
        "research_only": True,
        "n_eyes": len(joined),
        "n_referable": int(np.sum(actual >= 2)),
        "method": "fixed validation thresholds; same-eye paired comparison",
        "baseline": {
            "model": "reg", "metrics": metric_a,
            "referable": referable_confusion(actual, a_pred),
        },
        "candidate": {
            "model": "ordinal", "metrics": metric_b,
            "referable": referable_confusion(actual, b_pred),
        },
        "ordinal_minus_baseline": {
            "qwk": (metric_b["qwk"] - metric_a["qwk"]
                    if metric_b["qwk"] is not None and metric_a["qwk"] is not None else None),
            "macro_f1": (metric_b["macro_f1"] - metric_a["macro_f1"]
                         if metric_b["macro_f1"] is not None
                         and metric_a["macro_f1"] is not None else None),
        },
        "paired_referral_uncertainty": paired,
        "limitations": [
            "Descriptive fixed-score comparison, not an independent model selection procedure.",
            "No statistical interval is claimed for QWK or macro F1 differences.",
            "Eye-level bootstrap assumes independent eyes unless true patient IDs are supplied.",
            "A holdout cohort must not be reused to select training hyperparameters.",
            "External model generalization cannot be claimed without running external inference.",
        ],
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verified_external_comparison_frame(
    sweep: Path, directory: Path, cfg, labels: pd.DataFrame, prefix: str,
) -> tuple[pd.DataFrame, Path]:
    """Accept cached cross-site predictions only if their provenance is live.

    The inference pipeline signs the checkpoint bytes, input JPEG bytes,
    current labels, preprocessing settings and CSV bytes. The comparison
    pipeline must verify the *same* signature instead of just reading an
    existing filename. Never run model inference as a side effect here.
    """
    from aptos.data import manifest as manifest_mod
    from aptos.evaluation.external import (
        read_verified_score_cache,
        score_input_signature,
        validate_sweep_contract,
        verify_external_source_provenance,
    )

    validate_sweep_contract(cfg, sweep)
    checkpoints = sorted(path.name for path in sweep.glob("fold*.pt"))
    expected = sorted(f"fold{fold}.pt" for fold in range(1, cfg.cv.folds + 1))
    if checkpoints != expected:
        raise ValueError("paired external comparison requires every CV checkpoint")
    cache = directory / f"{prefix}_{cfg.variant}"
    manifest_mod.check(cache, cfg.manifest_fields(), count_images=True)
    from aptos.evaluation.external import dataset

    # Do not trust processed JPEGs if the original retinal source has changed.
    verify_external_source_provenance(cfg, dataset(prefix), labels, cache)
    existing_ids = {path.stem for path in cache.glob("*.jpg")}
    if not {"id_code", "diagnosis"}.issubset(labels.columns):
        raise ValueError("external cohort is missing image IDs or ground-truth grades")
    filtered = labels[labels["id_code"].isin(existing_ids)].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("external cohort has no accepted images")
    if filtered["id_code"].duplicated().any():
        raise ValueError("external cohort has duplicate image IDs")
    signature = score_input_signature(cfg, sweep, cache, filtered)
    output = directory / f"{prefix}_{cfg.variant}_{sweep.name}_scores.csv"
    frame = read_verified_score_cache(output, signature, filtered)
    if frame is None:
        raise ValueError(
            f"{output} is missing or stale: regenerate external scores with "
            "aptos.evaluation.external before a paired comparison"
        )
    return frame, output


def _load_frame(sweep: Path, cohort: str, directory: Path, variant: str) -> tuple[pd.DataFrame, Path | None]:
    if cohort == "aptos":
        from aptos.config import Config
        from aptos.evaluation.confound import load_sweep_predictions

        config = Config.load("configs/cv.yaml")
        original = _read_json(sweep / "config.json")
        config.variant = variant
        config.train.exclude_leaked = original["train.exclude_leaked"]
        config.cv.folds = original["cv.folds"]
        config.cv.stratify_on = original["cv.stratify_on"]
        config.cv.pool_splits = tuple(original["cv.pool_splits"])
        frame = load_sweep_predictions(config, sweep, strict_identity=True)
        return frame, None

    from aptos.config import Config
    from aptos.evaluation.external import dataset

    original = _read_json(sweep / "config.json")
    cfg = Config.load("configs/cv.yaml")
    cfg.variant = variant
    for name in ("mode", "model", "size"):
        setattr(cfg.train, name, original[f"train.{name}"])
    cfg.cv.folds = original["cv.folds"]
    for name in ("size", "clahe", "clip_limit", "square_mode", "jpeg_quality"):
        setattr(cfg.preprocess, name, original[f"preprocess.{name}"])

    # The source loader resolves its labels under the configured research root.
    # A second arbitrary directory could reference different images/labels.
    if directory.resolve() != cfg.paths.external.resolve():
        raise ValueError(
            "--external-dir must match the configured research data/external "
            "directory; use APTOS_ROOT to change the research root"
        )
    spec = dataset(cohort)
    labels = spec.load_labels(cfg)
    return verified_external_comparison_frame(sweep, directory, cfg, labels, spec.prefix)


def _patient_map(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, dtype={"id_code": "string", "patient_id": "string"})
    if (not {"id_code", "patient_id"}.issubset(df.columns)
            or df["id_code"].isna().any() or df["patient_id"].isna().any()
            or df["id_code"].duplicated().any()
            or df["id_code"].eq("").any() or df["patient_id"].eq("").any()):
        raise ValueError("patient mapping requires unique IDs and real nonempty patient IDs")
    return dict(zip(df["id_code"].astype(str), df["patient_id"].astype(str), strict=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline-sweep", type=Path, required=True)
    parser.add_argument("--ordinal-sweep", type=Path, required=True)
    parser.add_argument("--cohort", choices=("aptos", "idrid", "messidor2"), required=True)
    parser.add_argument("--external-dir", type=Path, default=Path("data/external"))
    parser.add_argument("--patient-map", type=Path, help="optional real patient linkage; never guessed")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="write aggregate JSON (default stdout)")
    args = parser.parse_args(argv)

    base, ordinal = args.baseline_sweep, args.ordinal_sweep
    config_a, config_b = _read_json(base / "config.json"), _read_json(ordinal / "config.json")
    matched = compare_sweep_configs(config_a, config_b)
    thresholds_a = validated_thresholds(_read_json(base / "summary.json"))
    thresholds_b = validated_thresholds(_read_json(ordinal / "summary.json"))
    a, source_a = _load_frame(base, args.cohort, args.external_dir, matched["variant"])
    b, source_b = _load_frame(ordinal, args.cohort, args.external_dir, matched["variant"])
    report = compare_frames(
        a, b, thresholds_a, thresholds_b, n_resamples=args.draws,
        seed=args.seed, patient_map=_patient_map(args.patient_map) if args.patient_map else None,
    )
    report.update({
        "cohort": args.cohort,
        "controlled_settings": matched,
        "sweep_paths": {"baseline": base.name, "ordinal": ordinal.name},
        "source_sha256": {
            name: file_sha256(path) for name, path in (
                ("baseline_config", base / "config.json"),
                ("ordinal_config", ordinal / "config.json"),
                ("baseline_summary", base / "summary.json"),
                ("ordinal_summary", ordinal / "summary.json"),
                ("baseline_scores", source_a),
                ("ordinal_scores", source_b),
            ) if path is not None
        },
    })
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"aggregate paired benchmark written to {args.output}")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
