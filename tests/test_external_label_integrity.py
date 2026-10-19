"""Offline validation of external grading metadata, no GPU or real images."""
import pandas as pd
import pytest

from aptos.evaluation.external_label_integrity import validate_external_labels

pytestmark = pytest.mark.pure


def _labels(names, grades):
    return pd.DataFrame({"id_code": names, "diagnosis": grades})


def test_ignores_only_fully_blank_padding_rows():
    result = validate_external_labels(_labels(["a", None], [2, None]), source="IDRiD")
    assert result.to_dict("records") == [{"id_code": "a", "diagnosis": 2}]


def test_accepts_integral_numeric_grade():
    assert validate_external_labels(_labels(["a"], [2.0]), source="IDRiD")[
        "diagnosis"
    ].tolist() == [2]


@pytest.mark.parametrize("value", [2.7, 4.9, -0.1, 5, float("inf"), "oops", "nan"])
def test_refuses_invalid_grade_instead_of_truncation(value):
    with pytest.raises(ValueError, match="grade"):
        validate_external_labels(_labels(["a"], [value]), source="IDRiD")


def test_refuses_missing_grade_for_real_image():
    with pytest.raises(ValueError, match="incomplete"):
        validate_external_labels(_labels(["a"], [None]), source="IDRiD")


@pytest.mark.parametrize("ident", ["../escape", "part/x", "x\\y", " a", "", ".", "..", "x\x00x"])
def test_refuses_unsafe_cache_identifier(ident):
    with pytest.raises(ValueError, match="image IDs"):
        validate_external_labels(_labels([ident], [2]), source="IDRiD")


def test_refuses_duplicate_eye_id():
    with pytest.raises(ValueError, match="duplicate"):
        validate_external_labels(_labels(["a", "a"], [0, 2]), source="IDRiD")


def test_refuses_distinct_messidor_files_with_same_cache_stem():
    with pytest.raises(ValueError, match="collision"):
        validate_external_labels(_labels(["a.png", "a.JPG"], [0, 2]),
                                 source="Messidor-2", filenames=True)


def test_preserves_original_messidor_filename_and_integer_grade():
    result = validate_external_labels(
        _labels(["IM000123.JPG", "a.png"], [2, 0]),
        source="Messidor-2", filenames=True,
    )
    assert result.to_dict("records") == [
        {"id_code": "IM000123", "diagnosis": 2, "source_name": "IM000123.JPG"},
        {"id_code": "a", "diagnosis": 0, "source_name": "a.png"},
    ]
