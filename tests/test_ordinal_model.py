"""Toy-tensor experiments for cumulative-link ordinal training."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from aptos.config import Config  # noqa: E402
from aptos.training.loop import build_criterion, run_epoch  # noqa: E402
from aptos.training.ordinal import (  # noqa: E402
    CumulativeOrdinalHead,
    cumulative_targets,
    ordinal_expected_grade,
)

pytestmark = pytest.mark.torch


class ToyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, images):
        return self.linear(images)


def test_rank_consistency_and_probability_bounds():
    head = CumulativeOrdinalHead(ToyBackbone())
    logits = head(torch.randn(6, 2))
    assert logits.shape == (6, 4)
    assert torch.all(head.cutpoints()[1:] > head.cutpoints()[:-1])
    prob = logits.sigmoid()
    assert torch.all(prob[:, :-1] > prob[:, 1:])
    raw = ordinal_expected_grade(logits)
    assert raw.shape == (6,)
    assert torch.all((raw >= 0) & (raw <= 4))


def test_cumulative_labels():
    result = cumulative_targets(torch.arange(5))
    np.testing.assert_array_equal(result.numpy(), [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ])


@pytest.mark.parametrize("bad", [
    torch.tensor([]), torch.tensor([5]), torch.tensor([-1]),
    torch.tensor([1.5]), torch.tensor([[1]]),
])
def test_reject_invalid_cumulative_labels(bad):
    with pytest.raises(ValueError, match="ordinal labels"):
        cumulative_targets(bad)


def test_gradients_and_loss():
    cfg = Config.load(train={"mode": "ordinal"})
    criterion = build_criterion(cfg, pd.DataFrame({"diagnosis": [0, 1, 2, 4]}), "cpu")
    assert isinstance(criterion, torch.nn.BCEWithLogitsLoss)
    head = CumulativeOrdinalHead(ToyBackbone())
    output = head(torch.randn(4, 2))
    loss = criterion(output, cumulative_targets(torch.tensor([0, 1, 2, 4])))
    loss.backward()
    assert torch.isfinite(loss)
    assert head.base_cut.grad is not None
    assert head.cut_gaps.grad is not None
    assert head.backbone.linear.weight.grad is not None


def test_epoch_training_and_scoring():
    cfg = Config.load(train={"mode": "ordinal"})
    head = CumulativeOrdinalHead(ToyBackbone())
    x = torch.randn(10, 2)
    y = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=5,
    )
    loss, predicted, observed = run_epoch(
        head, loader, build_criterion(cfg, pd.DataFrame({"diagnosis": y.numpy()}), "cpu"),
        "cpu", "ordinal", torch.optim.Adam(head.parameters(), lr=0.01),
    )
    assert np.isfinite(loss)
    assert predicted.shape == observed.shape == (10,)
    assert np.all((predicted >= 0) & (predicted <= 4))
    assert set(observed) == {0, 1, 2, 3, 4}


def test_legacy_regression_remains_the_default():
    assert Config.load().train.mode == "reg"
    assert Config.load(train={"mode": "ordinal"}).train.mode == "ordinal"
    with pytest.raises(ValueError, match="unknown train mode"):
        Config.load(train={"mode": "wrong"})


def test_timm_backbone_is_wrapped_with_four_logits():
    from aptos.training.loop import build_model

    cfg = Config.load(train={"mode": "ordinal", "model": "resnet18"})
    model = build_model(cfg, "cpu", pretrained=False)
    assert isinstance(model, CumulativeOrdinalHead)
    with torch.no_grad():
        output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 4)
    assert torch.all(output[:, :-1] > output[:, 1:])
