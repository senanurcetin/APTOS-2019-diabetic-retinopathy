"""Pure unit tests for Kaggle/BigQuery label-export provenance."""
import csv

import pytest

from aptos.data.integrity import inspect_labels


def _save(tmp_path, rows):
    path = tmp_path / "aptos_labels.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id_code", "diagnosis", "split", "is_referable"])
        writer.writeheader()
        writer.writerows(rows)
    return path


VALID = [
    {"id_code": "a", "diagnosis": "0", "split": "train", "is_referable": "False"},
    {"id_code": "b", "diagnosis": "2", "split": "valid", "is_referable": "True"},
    {"id_code": "c", "diagnosis": "4", "split": "test", "is_referable": "True"},
]


@pytest.mark.pure
def test_manifest_has_split_proportions_and_hash(tmp_path):
    result = inspect_labels(_save(tmp_path, VALID), expected_rows=3)
    assert result["n"] == result["unique_ids"] == 3
    assert result["by_split"]["train"]["grades"]["0"] == 1
    assert result["by_split"]["valid"]["referable"] == 1
    assert len(result["sha256"]) == 64
    assert result["source_file"] == "aptos_labels.csv"


@pytest.mark.pure
@pytest.mark.parametrize("index,change,error", [
    (1, {"id_code": "a"}, "duplicate image ID"),
    (1, {"diagnosis": "5"}, "invalid grade"),
    (1, {"diagnosis": "2.0"}, "non-integer grade"),
    (1, {"split": "holdout"}, "invalid split"),
    (1, {"is_referable": "False"}, "incorrect is_referable"),
    (1, {"id_code": "../escape"}, "invalid image ID"),
])
def test_rejects_broken_labels(tmp_path, index, change, error):
    records = [dict(row) for row in VALID]
    records[index].update(change)
    with pytest.raises(ValueError, match=error):
        inspect_labels(_save(tmp_path, records))


@pytest.mark.pure
def test_rejects_wrong_dataset_size(tmp_path):
    with pytest.raises(ValueError, match="expected 3662"):
        inspect_labels(_save(tmp_path, VALID), expected_rows=3662)
