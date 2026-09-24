"""The data and loss pieces of the training loop."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from aptos.config import Config  # noqa: E402
from aptos.training.loop import FoldDataset, build_criterion  # noqa: E402

pytestmark = pytest.mark.torch


def _image(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((8, 8, 3), value, np.uint8))


def test_fold_dataset_reads_from_the_original_split_directory(tmp_path):
    """Under cross-validation an image moves between training and validation
    roles, but its file stays where it was. Reading from the fold role instead
    of `orig_split` would load the wrong image - or none."""
    _image(tmp_path / "train" / "a.jpg", 10)
    _image(tmp_path / "valid" / "b.jpg", 200)

    frame = pd.DataFrame({
        "id_code": ["a", "b"], "diagnosis": [0, 3],
        # b is in this fold's *training* role but lives in valid/ on disk
        "orig_split": ["train", "valid"], "split": ["train", "train"],
    })
    ds = FoldDataset(frame, transform=lambda im: torch.tensor(np.asarray(im).mean()),
                     root=tmp_path)

    (a, ya), (b, yb) = ds[0], ds[1]
    assert float(a) == pytest.approx(10, abs=2)
    assert float(b) == pytest.approx(200, abs=2)
    assert (int(ya), int(yb)) == (0, 3)


def test_fold_dataset_insists_on_the_orig_split_column(tmp_path):
    frame = pd.DataFrame({"id_code": ["a"], "diagnosis": [0], "split": ["train"]})
    with pytest.raises(KeyError, match="orig_split"):
        FoldDataset(frame, transform=None, root=tmp_path)


def test_class_weights_survive_a_missing_grade():
    """The original built weights from value_counts(), which produced a vector
    shorter than five when a grade was absent - and crashed inside the loss."""
    cfg = Config.load(train={"mode": "cls"})
    frame = pd.DataFrame({"diagnosis": [0, 0, 0, 1, 2, 2]})  # no grade 3 or 4
    criterion = build_criterion(cfg, frame, "cpu")

    assert criterion.weight.shape == (5,)
    assert criterion.weight[3] == 0 and criterion.weight[4] == 0
    logits = torch.randn(6, 5)
    loss = criterion(logits, torch.tensor([0, 0, 0, 1, 2, 2]))
    assert torch.isfinite(loss)


def test_class_weights_upweight_the_rare_grades():
    cfg = Config.load(train={"mode": "cls"})
    frame = pd.DataFrame({"diagnosis": [0] * 50 + [1] * 5 + [2] * 20 + [3] * 3 + [4] * 4})
    w = build_criterion(cfg, frame, "cpu").weight
    assert w[3] > w[2] > w[0]


def test_regression_mode_uses_mse():
    cfg = Config.load(train={"mode": "reg"})
    assert isinstance(build_criterion(cfg, pd.DataFrame({"diagnosis": [0]}), "cpu"),
                      torch.nn.MSELoss)
