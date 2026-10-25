"""Prepare a separate, deterministic examination-disjoint follow-up split.

This reuses an already inspected cohort; it is not a new independent dataset.
Only a local CSV and aggregate plan are written. No model or GPU is launched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from aptos.evaluation.examination_audit import audit_examinations, examination_mapping
from aptos.evaluation.external_label_integrity import validate_external_labels

PARTITIONS = {"tune-train", "tune-valid", "test"}


def make_examination_split(
    labels: pd.DataFrame, pairs: pd.DataFrame, *, seed: int = 42,
) -> pd.DataFrame:
    """One fixed 2-fold then 5-fold allocation; never search seeds by performance."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    frame = validate_external_labels(labels, source="examination follow-up")
    frame = frame.sort_values("id_code").reset_index(drop=True)
    frame["exam_id"] = frame.id_code.map(examination_mapping(pairs))
    if frame.exam_id.isna().any():
        raise ValueError("pairing coverage incomplete for labeled images")
    # Enough independent groups to make the fixed nested allocation meaningful.
    if frame.exam_id.nunique() < 20:
        raise ValueError("at least 20 supplied examinations are required")
    target = frame.diagnosis.ge(2).astype(int)
    if target.nunique() != 2:
        raise ValueError("both referable classes are required")
    outer = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    _, development = next(iter(outer.split(frame, target, frame.exam_id)))
    frame["part"] = "test"
    frame.loc[development, "part"] = "tune-train"
    inner_frame = frame.loc[development]
    if inner_frame.exam_id.nunique() < 5:
        raise ValueError("too few development examinations for five-fold allocation")
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, validation = next(iter(inner.split(
        inner_frame, target.loc[development], inner_frame.exam_id,
    )))
    frame.loc[inner_frame.index[validation], "part"] = "tune-valid"
    validate_examination_split(frame, labels, pairs)
    return frame


def validate_examination_split(
    split: pd.DataFrame, labels: pd.DataFrame, pairs: pd.DataFrame,
) -> dict:
    """Reject changed grades, omitted eyes, forged keys and cross-partition pairs."""
    expected = validate_external_labels(labels, source="reference labels").set_index("id_code")
    actual = validate_external_labels(split, source="split labels").set_index("id_code")
    if set(actual.index) != set(expected.index):
        raise ValueError("split must contain exactly the reference labeled eyes")
    if not actual.diagnosis.sort_index().equals(expected.diagnosis.sort_index()):
        raise ValueError("split grades differ from reference labels")
    split = split.copy()
    split["diagnosis"] = split.id_code.map(actual.diagnosis)
    if not {"part", "exam_id"} <= set(split.columns) or set(split.part) != PARTITIONS:
        raise ValueError("split must contain exam_id and exactly the three named partitions")
    mapping = examination_mapping(pairs)
    if not split.exam_id.equals(split.id_code.map(mapping)):
        raise ValueError("split examination keys disagree with supplied pairing table")
    audit = audit_examinations(pairs, split)
    if not audit["examination_disjoint"]:
        raise ValueError("an examination crosses split partitions")
    for part, rows in split.groupby("part"):
        if rows.diagnosis.ge(2).nunique() != 2:
            raise ValueError(f"{part} must contain both referable classes; do not silently resample")
    return audit


def split_plan(split: pd.DataFrame, pairs: pd.DataFrame, *, seed: int) -> dict:
    """Aggregate release record without image IDs or per-examination hashes."""
    audit = audit_examinations(pairs, split)
    return {
        "schema_version": 1,
        "experiment": "messidor2-examination-followup-v2",
        "seed": seed,
        "algorithm": "first SGKF(2) split, then first SGKF(5) development split",
        "stratification_target": "eye-level diagnosis >= 2, grouped by supplied examination",
        "audit": audit,
        "partition_summary": {
            name: {"eyes": len(rows), "examinations": rows.exam_id.nunique(),
                   "referable_eyes": int(rows.diagnosis.ge(2).sum()),
                   "grade_counts": {str(k): int(v) for k, v in
                                    rows.diagnosis.value_counts().sort_index().items()}}
            for name, rows in split.groupby("part")
        },
        "split_csv_sha256": hashlib.sha256(
            split.to_csv(index=False, lineterminator="\n").encode()
        ).hexdigest(),
        "model_training_status": "not started; split preparation only",
        "interpretation": "Examination-disjoint under supplied third-party transcription. "
        "Original ADCIS parity and longitudinal patient identity remain unverified. "
        "This reuses an already inspected cohort and is not independent confirmatory data. "
        "Historical AUC values must not be assigned to this new split.",
    }


def compare_historical_split(historical: pd.DataFrame, followup: pd.DataFrame) -> dict:
    """Quantify prior development exposure; old fine-tuned weights cannot be reused."""
    old = validate_external_labels(historical, source="historical split").set_index("id_code")
    new = validate_external_labels(followup, source="follow-up split").set_index("id_code")
    if not old.sort_index().equals(new.sort_index()):
        raise ValueError("historical and follow-up labels/IDs must match exactly")
    for frame in (historical, followup):
        if "part" not in frame or not frame.part.isin(PARTITIONS).all():
            raise ValueError("historical comparison requires valid partitions for every eye")
    merged = historical[["id_code", "part"]].merge(
        followup[["id_code", "part"]], on="id_code", suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    transitions = pd.crosstab(merged.part_old, merged.part_new)
    return {
        "partition_transitions": {
            str(old_part): {str(new_part): int(count) for new_part, count in row.items()}
            for old_part, row in transitions.to_dict(orient="index").items()
        },
        "new_test_eyes_previously_in_training": int(
            (merged.part_new.eq("test") & merged.part_old.eq("tune-train")).sum()
        ),
        "new_test_eyes_previously_in_validation": int(
            (merged.part_new.eq("test") & merged.part_old.eq("tune-valid")).sum()
        ),
        "old_finetuned_checkpoint_reuse_allowed": False,
        "interpretation": "Restart both arms from the frozen APTOS-only base checkpoints. "
        "Historical fine-tuned checkpoints have development exposure to new test eyes.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="local CSV; never overwritten")
    parser.add_argument("--plan", type=Path, required=True, help="aggregate JSON; never overwritten")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--historical-split", type=Path,
                        help="optional historical CSV to quantify prior development exposure")
    args = parser.parse_args(argv)
    if args.output.resolve() == args.plan.resolve() or args.output.exists() or args.plan.exists():
        parser.error("output paths must be distinct and new; historical artifacts are never replaced")
    pairs = pd.read_csv(args.pairs, sep=";")
    labels = pd.read_csv(args.labels)
    if "adjudicated_gradable" in labels:
        labels = labels[labels.adjudicated_gradable.eq(1)]
    labels = labels.rename(columns={"image_id": "id_code", "adjudicated_dr_grade": "diagnosis"})
    labels = validate_external_labels(labels, source="source grades", filenames=True)
    split = make_examination_split(labels, pairs, seed=args.seed)
    plan = split_plan(split, pairs, seed=args.seed)
    plan["inputs_sha256"] = {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in (("pairs", args.pairs), ("labels", args.labels))
    }
    if args.historical_split:
        plan["historical_comparison"] = compare_historical_split(
            pd.read_csv(args.historical_split), split,
        )
        plan["inputs_sha256"]["historical_split"] = hashlib.sha256(
            args.historical_split.read_bytes()
        ).hexdigest()
    plan["code_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(examination_mapping.__code__.co_filename))
    }
    import sklearn

    plan["software"] = {"pandas": pd.__version__, "scikit_learn": sklearn.__version__}
    for path in (args.output, args.plan):
        path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against races after the initial existence check.
    with args.output.open("x", encoding="utf-8", newline="") as output:
        output.write(split.to_csv(index=False, lineterminator="\n"))
    with args.plan.open("x", encoding="utf-8") as output:
        json.dump(plan, output, indent=2)
        output.write("\n")
    print(json.dumps(plan["partition_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
