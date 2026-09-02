"""Turn the raw Kaggle APTOS-2019 CSVs into BigQuery-ready tables.

Input : data/raw/{train_1,valid,test}.csv
        (kaggle datasets download mariaherrerot/aptos2019)
Output: data/bq/aptos_labels.csv         single combined table (recommended)
        data/bq/{train,valid,test}.csv   per-split tables

The image_uri column carries the full gs:// path, so consumers never have to
reconstruct the bucket prefix themselves.
"""
import os
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "bq"
OUT.mkdir(parents=True, exist_ok=True)

# ICDRSS grades (International Clinical Diabetic Retinopathy Disease Severity
# Scale).
GRADES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

# Actual folder layout inside the GCS bucket (verified by listing it).
GCS_BUCKET = os.environ.get("APTOS_GCS_BUCKET", "aptos2019-retina-images")
GCS_PREFIXES = {
    "train": "aptos_train_images",
    "valid": "aptos_valid_images",
    "test": "aptos_test_images",
}

SPLIT_FILES = {"train": "train_1.csv", "valid": "valid.csv", "test": "test.csv"}


def load(split: str) -> pd.DataFrame:
    # test.csv ends with roughly 500 blank lines; skip_blank_lines drops them.
    # Loading it raw with schema autodetection produces a broken single-column
    # table.
    df = pd.read_csv(RAW / SPLIT_FILES[split], skip_blank_lines=True)
    df = df.dropna(subset=["id_code", "diagnosis"])
    df["id_code"] = df["id_code"].astype(str).str.strip()
    df["diagnosis"] = df["diagnosis"].astype(int)
    df["split"] = split
    df["diagnosis_label"] = df["diagnosis"].map(GRADES)
    df["is_referable"] = df["diagnosis"] >= 2  # referable DR, a common binary target
    df["image_file"] = df["id_code"] + ".png"
    df["image_uri"] = (f"gs://{GCS_BUCKET}/{GCS_PREFIXES[split]}/"
                       + df["image_file"])
    return df[["id_code", "diagnosis", "diagnosis_label", "is_referable",
               "split", "image_file", "image_uri"]]


def main() -> None:
    parts = {s: load(s) for s in SPLIT_FILES}
    combined = pd.concat(parts.values(), ignore_index=True)

    assert combined["diagnosis"].between(0, 4).all(), "unexpected diagnosis value"
    assert not combined["id_code"].duplicated().any(), "id_code repeats across splits"
    assert len(combined) == 3662, f"expected 3662 rows, got {len(combined)}"

    combined.to_csv(OUT / "aptos_labels.csv", index=False)
    for split, df in parts.items():
        df.to_csv(OUT / f"{split}.csv", index=False)

    print(f"{len(combined)} rows -> {OUT}")
    print(combined.pivot_table(index="diagnosis_label", columns="split",
                               values="id_code", aggfunc="count", fill_value=0))


if __name__ == "__main__":
    main()
