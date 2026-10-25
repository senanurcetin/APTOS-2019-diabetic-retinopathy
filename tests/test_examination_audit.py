"""Examination leakage must not be hidden by filename proxy groups."""
import pandas as pd
import pytest

from aptos.evaluation.examination_audit import audit_examinations

pytestmark = pytest.mark.pure


def test_crossing_pair_is_detected_and_single_eye_kept():
    pairs = pd.DataFrame({"left": ["a.png", "c.png"], "right ": ["b.png", "d.png"]})
    split = pd.DataFrame({"id_code": ["a", "b", "c"],
                          "part": ["test", "tune-train", "test"]})
    result = audit_examinations(pairs, split)
    assert result["test_development_overlap_examinations"] == 1
    assert result["test_images_with_development_partner"] == 1
    assert result["single_gradable_eye_examinations"] == 1
    assert not result["examination_disjoint"]
    assert "a.png" not in str(result)


def test_disjoint_examinations():
    pairs = pd.DataFrame({"left": ["a.png"], "right": ["b.png"]})
    split = pd.DataFrame({"id_code": ["a", "b"], "part": ["test", "test"]})
    assert audit_examinations(pairs, split)["examination_disjoint"]


@pytest.mark.parametrize("left,right,ids,parts", [
    (["a.png"], ["a.jpg"], ["a"], ["test"]),
    ([None], ["b.png"], ["b"], ["test"]),
    (["a.png"], ["b.png"], ["unknown"], ["test"]),
    (["a.png"], ["b.png"], ["a", "a"], ["test", "tune-train"]),
    (["a.png"], ["b.png"], ["a"], [None]),
    (["../a.png"], ["b.png"], ["a"], ["test"]),
])
def test_ambiguous_or_incomplete_mapping_rejected(left, right, ids, parts):
    with pytest.raises(ValueError):
        audit_examinations(pd.DataFrame({"left": left, "right": right}),
                           pd.DataFrame({"id_code": ids, "part": parts}))
