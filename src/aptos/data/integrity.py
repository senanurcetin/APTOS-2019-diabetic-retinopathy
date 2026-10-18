"""Local label-integrity and provenance manifest for Kaggle-derived CSVs.

Runs without imaging packages, GCP credentials or an image download. ID
deduplication is NOT perceptual-image deduplication; see quality_report.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

VALID_SPLITS = ("train", "valid", "test")
GRADES = tuple(range(5))


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(data)
    return sha.hexdigest()


def inspect_labels(path: Path, *, expected_rows: int | None = None) -> dict:
    """Fail on invalid labels and return a deterministic manifest."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"id_code", "diagnosis", "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"label CSV requires columns {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("label CSV has no rows")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} labels, found {len(rows)}")
    ids: set[str] = set()
    counts: Counter[tuple[str, int]] = Counter()
    for index, row in enumerate(rows, start=2):
        identifier = row["id_code"].strip()
        if not identifier or "/" in identifier or "\\" in identifier:
            raise ValueError(f"invalid image ID at line {index}")
        if identifier in ids:
            raise ValueError(f"duplicate image ID {identifier!r} at line {index}")
        ids.add(identifier)
        split = row["split"].strip()
        if split not in VALID_SPLITS:
            raise ValueError(f"invalid split {split!r} at line {index}")
        try:
            diagnosis = int(row["diagnosis"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-integer grade at line {index}") from exc
        if str(diagnosis) != row["diagnosis"].strip() or diagnosis not in GRADES:
            raise ValueError(f"invalid grade at line {index}")
        if "is_referable" in row:
            flag = row["is_referable"].strip().lower()
            if flag not in ("true", "false") or (flag == "true") != (diagnosis >= 2):
                raise ValueError(f"incorrect is_referable at line {index}")
        counts[(split, diagnosis)] += 1

    by_split = {
        split: {
            "total": sum(counts[(split, grade)] for grade in GRADES),
            "grades": {str(grade): counts[(split, grade)] for grade in GRADES},
            "referable": sum(counts[(split, grade)] for grade in (2, 3, 4)),
        }
        for split in VALID_SPLITS
    }
    return {
        "schema_version": 1,
        "source_file": path.name,  # portable: never include user-specific absolute paths
        "sha256": sha256_file(path),
        "n": len(rows),
        "unique_ids": len(ids),
        "by_split": by_split,
        "id_deduplication_scope": "identifier only; not pixel/perceptual duplicates",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=Path("data/bq/aptos_labels.csv"))
    parser.add_argument("--expected-rows", type=int, default=3662)
    parser.add_argument("--out", type=Path, help="optional JSON output; default stdout")
    args = parser.parse_args(argv)
    report = inspect_labels(args.csv, expected_rows=args.expected_rows)
    document = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(document, encoding="utf-8")
        print(f"written to {args.out}")
    else:
        print(document, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
