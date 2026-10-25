"""Audit a historical split against supplied bilateral examination pairs.

Pair rows identify examinations, not longitudinal patients. No identity is
inferred from filename proximity. Output contains aggregates and input hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def audit_examinations(pairs: pd.DataFrame, split: pd.DataFrame) -> dict:
    """Require a complete one-to-one image mapping; count cross-partition exams."""
    pairs = pairs.copy()
    pairs.columns = pairs.columns.str.strip()
    if set(pairs.columns) != {"left", "right"} or pairs.empty:
        raise ValueError("pairs must have exactly left/right columns and nonempty rows")
    if pairs.isna().any().any():
        raise ValueError("pair members must not be missing")
    mapping = {}
    for exam, row in enumerate(pairs.itertuples(index=False, name=None)):
        for value in row:
            name = str(value).strip()
            if not name or "/" in name or "\\" in name:
                raise ValueError("pair members must be nonempty filenames, not paths")
            ident = Path(name).stem
            if not ident or ident in mapping:
                raise ValueError("each image must occur exactly once in pairing table")
            mapping[ident] = exam
    if not {"id_code", "part"} <= set(split.columns) or split.empty:
        raise ValueError("split needs nonempty id_code/part columns")
    split = split.copy()
    if split[["id_code", "part"]].isna().any().any():
        raise ValueError("split IDs and partitions must not be missing")
    for column in ("id_code", "part"):
        split[column] = split[column].astype(str).str.strip()
        if split[column].eq("").any():
            raise ValueError("split IDs and partitions must not be empty")
    if split.id_code.duplicated().any():
        raise ValueError("split image IDs must be unique")
    split["exam"] = split.id_code.map(mapping)
    if split.exam.isna().any():
        raise ValueError("pairing coverage incomplete for split images")
    groups = split.groupby("exam")
    parts = groups.part.agg(lambda values: tuple(sorted(set(values))))
    crossing = parts[parts.map(len) > 1]
    test_crossing = crossing[crossing.map(lambda values: "test" in values)]
    return {
        "unit": "examination; not authenticated longitudinal patient identity",
        "pair_table_examinations": len(pairs),
        "pair_table_images": len(mapping),
        "split_images": len(split),
        "covered_split_images": int(split.exam.notna().sum()),
        "represented_examinations": len(groups),
        "complete_pairs": int(groups.size().eq(2).sum()),
        "single_gradable_eye_examinations": int(groups.size().eq(1).sum()),
        "partition_images": split.part.value_counts().sort_index().to_dict(),
        "cross_partition_examinations": len(crossing),
        "partition_combinations": {
            " + ".join(key): int(value) for key, value in parts.value_counts().items()
        },
        "test_development_overlap_examinations": len(test_crossing),
        "test_images_with_development_partner": int(
            (split.exam.isin(test_crossing.index) & split.part.eq("test")).sum()
        ),
        "examination_disjoint": len(crossing) == 0,
        "interpretation": (
            "Historical split is not examination-disjoint. Fine-tuning results remain "
            "exploratory; removing test eyes after training does not restore a fresh "
            "confirmatory evaluation. Retraining requires a new prespecified grouped split."
            if len(crossing) else
            "No examination crosses partitions in the supplied mapping; this does not "
            "establish longitudinal patient independence or clinical validity."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_examinations(pd.read_csv(args.pairs, sep=";"), pd.read_csv(args.split))
    result["inputs_sha256"] = {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in (("pairs", args.pairs), ("split", args.split))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
