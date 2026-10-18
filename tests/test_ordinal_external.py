"""Research-only external score decoding and checkpoint contract tests."""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from aptos.config import Config, Paths
from aptos.evaluation.external import validate_sweep_contract


@pytest.mark.pure
def test_ordinal_sweep_contract_requires_config(tmp_path):
    cfg = Config.load(train={"mode": "ordinal"})
    with pytest.raises(ValueError, match="requires the training sweep"):
        validate_sweep_contract(cfg, tmp_path)


@pytest.mark.pure
def test_legacy_missing_config_is_regression_only(tmp_path):
    validate_sweep_contract(Config.load(train={"mode": "reg"}), tmp_path)


@pytest.mark.pure
def test_contract_rejects_wrong_model_mode_or_image_preprocessing(tmp_path):
    original = Config.load("configs/cv.yaml", train={"mode": "ordinal"})
    (tmp_path / "config.json").write_text(json.dumps(original.to_dict()))
    validate_sweep_contract(original, tmp_path)
    for field, value in (("mode", "reg"), ("model", "resnet18"), ("size", 224)):
        cfg = dataclasses.replace(original, train=dataclasses.replace(
            original.train, **{field: value}
        ))
        with pytest.raises(ValueError, match=f"train.{field}"):
            validate_sweep_contract(cfg, tmp_path)
    cfg = dataclasses.replace(original, preprocess=dataclasses.replace(
        original.preprocess, square_mode="squash"
    ))
    with pytest.raises(ValueError, match="preprocess.square_mode"):
        validate_sweep_contract(cfg, tmp_path)


@pytest.mark.torch
def test_external_predict_decodes_ordinal_four_logits_not_argmax(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pil = pytest.importorskip("PIL.Image")
    from aptos.evaluation.external import predict
    from aptos.training import loop

    cache = tmp_path / "cache"
    cache.mkdir()
    pil.new("RGB", (12, 12), color=(100, 70, 90)).save(cache / "eye-a.jpg")
    pil.new("RGB", (12, 12), color=(110, 80, 80)).save(cache / "eye-b.jpg")
    labels = pd.DataFrame({"id_code": ["eye-a", "eye-b"], "diagnosis": [0, 3]})

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("four_logits", torch.zeros(4))

        def forward(self, images):
            return self.four_logits.unsqueeze(0).expand(images.shape[0], -1)

    monkeypatch.setattr(loop, "build_model",
                        lambda cfg, device, *, pretrained: Model().to(device))
    cfg = dataclasses.replace(
        Config.load(train={"mode": "ordinal", "workers": 0, "batch": 2}),
        paths=Paths(root=tmp_path),
    )
    state1 = {"four_logits": torch.tensor([3., 2., 1., -1.])}
    state2 = {"four_logits": torch.tensor([2., 1., -1., -3.])}
    output = predict(cfg, cache, labels, [state1, state2], "cpu")
    expected = 0.5 * (torch.sigmoid(state1["four_logits"]).sum().item()
                      + torch.sigmoid(state2["four_logits"]).sum().item())
    assert output.shape == (2,)
    assert np.allclose(output, expected)
    assert expected != 0  # An argmax or a single-logit squeeze would be invalid.


@pytest.mark.torch
def test_external_predict_preserves_regression_scalar(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pil = pytest.importorskip("PIL.Image")
    from aptos.evaluation.external import predict
    from aptos.training import loop

    cache = tmp_path / "cache"
    cache.mkdir()
    pil.new("RGB", (12, 12), color=(90, 70, 90)).save(cache / "eye-a.jpg")
    labels = pd.DataFrame({"id_code": ["eye-a"], "diagnosis": [0]})

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("value", torch.tensor([1.25]))

        def forward(self, images):
            return self.value.reshape(1, 1).expand(images.shape[0], 1)

    monkeypatch.setattr(loop, "build_model",
                        lambda cfg, device, *, pretrained: Model().to(device))
    cfg = dataclasses.replace(Config.load(train={"mode": "reg", "workers": 0}),
                              paths=Paths(root=tmp_path))
    raw = predict(cfg, cache, labels, [{"value": torch.tensor([1.25])}], "cpu")
    assert raw.tolist() == pytest.approx([1.25])
