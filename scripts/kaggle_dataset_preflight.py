"""Stage an authorized mounted APTOS Kaggle dataset without copying retinal pixels.

No Kaggle token is read or stored. Symlink the original image directories as
read-only inputs; write only local research labels and a provenance receipt.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

SPLITS = {"train": ("train_1.csv", "train_images"),
          "valid": ("valid.csv", "val_images"),
          "test": ("test.csv", "test_images")}
VALID_GRADE = re.compile(r"[0-4](?:\.0+)?\Z")


def find_unique(root: Path, filename: str) -> Path:
    found = sorted(p for p in root.rglob(filename) if p.is_file())
    if len(found) != 1:
        raise ValueError(f"{filename}: expected exactly one mounted file, found {len(found)}")
    return found[0]


def read_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"id_code", "diagnosis"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path.name}: id_code and diagnosis columns required")
        values = []
        for line, row in enumerate(reader, start=2):
            ident = (row.get("id_code") or "").strip()
            grade = (row.get("diagnosis") or "").strip()
            if not ident and not grade:
                continue  # historical blank padding at the end of test.csv
            if (not ident or ident in (".", "..") or "/" in ident or "\\" in ident
                    or not VALID_GRADE.fullmatch(grade)):
                raise ValueError(f"{path.name}: invalid eye ID or grade at CSV row {line}")
            values.append(ident)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{path.name}: labels are empty or contain duplicate image IDs")
    return values


def stage_dataset(
    mounted: Path, repository: Path, *, expected_rows: int = 3662,
    dry_run: bool = False,
) -> dict:
    """Validate all splits before creating any symlink or metadata files."""
    if not mounted.is_dir() or not (repository / "configs" / "cv.yaml").is_file():
        raise FileNotFoundError("mounted APTOS dataset or repository configs/cv.yaml absent")
    csvs = {part: find_unique(mounted, filename)
            for part, (filename, _) in SPLITS.items()}
    ids_by_split = {part: read_ids(path) for part, path in csvs.items()}
    all_ids = [identifier for ids in ids_by_split.values() for identifier in ids]
    if len(all_ids) != expected_rows or len(set(all_ids)) != len(all_ids):
        raise ValueError(f"expected {expected_rows} unique APTOS eyes, found {len(all_ids)} rows")

    # Map only PNGs, not similarly named JPEG derivatives from other datasets.
    png_by_id = {}
    for path in mounted.rglob("*.png"):
        png_by_id.setdefault(path.stem, []).append(path)
    image_dirs = {}
    for part, ids in ids_by_split.items():
        parents = set()
        for ident in ids:
            matching = png_by_id.get(ident, [])
            if len(matching) != 1:
                raise ValueError(f"{part}: expected one PNG for {ident}, found {len(matching)}")
            parents.add(matching[0].parent)
        if len(parents) != 1:
            raise ValueError(f"{part}: source images do not occupy one directory")
        image_dirs[part] = parents.pop()

    planned = []
    for part, (name, image_folder) in SPLITS.items():
        csv_dst = repository / "data" / "raw" / name
        img_dst = (repository / "data" / "images" / image_folder / image_folder)
        if csv_dst.exists() and csv_dst.read_bytes() != csvs[part].read_bytes():
            raise ValueError(f"refusing to overwrite different staged labels: {csv_dst}")
        if img_dst.exists() or img_dst.is_symlink():
            if not img_dst.is_symlink() or img_dst.resolve() != image_dirs[part].resolve():
                raise ValueError(f"refusing to overwrite a different image directory: {img_dst}")
        planned.append((csvs[part], csv_dst, image_dirs[part], img_dst))

    manifest = {
        "schema_version": 1,
        "dataset": "mariaherrerot/aptos2019",
        "expected_rows": expected_rows,
        "per_split_counts": {key: len(value) for key, value in ids_by_split.items()},
        "label_csv_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                             for key, path in csvs.items()},
        "image_validation": "exactly one PNG per ID, split directories unique",
        "source_pixels_sha256": None,
        "source_pixels_sha256_note": "Not hashed; source image contents require separate provenance.",
        "staging": "read-only image-directory symlinks",
    }
    if not dry_run:
        for csv_src, csv_dst, _, img_dst in planned:
            csv_dst.parent.mkdir(parents=True, exist_ok=True)
            if not csv_dst.exists():
                shutil.copy2(csv_src, csv_dst)
        for _, _, src, img_dst in planned:
            if not img_dst.is_symlink():
                img_dst.parent.mkdir(parents=True, exist_ok=True)
                img_dst.symlink_to(src.resolve(), target_is_directory=True)
        receipt = repository / "outputs" / "kaggle_dataset_preflight.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    receipt = stage_dataset(args.input_root, args.repo_root, dry_run=args.dry_run)
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
