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


@pytest.mark.pure
def test_external_inference_rejects_partial_cv_sweep(tmp_path):
    from aptos.evaluation.external import complete_sweep_checkpoints

    cfg = Config.load("configs/cv.yaml", cv={"folds": 2})
    (tmp_path / "config.json").write_text(json.dumps(cfg.to_dict()))
    with pytest.raises(ValueError, match="summary.json is missing"):
        complete_sweep_checkpoints(tmp_path)

    (tmp_path / "summary.json").write_text(json.dumps({
        "folds": 2, "ensemble": {"thresholds": [0.5, 1.5, 2.5, 3.5]},
    }))
    for i in (1, 2):
        (tmp_path / f"fold{i}.pt").write_bytes(b"weights")
        (tmp_path / f"fold{i}.json").write_text('{"fold":1}')
        (tmp_path / f"fold{i}.npz").write_bytes(b"predictions")
    assert len(complete_sweep_checkpoints(tmp_path)) == 2
    (tmp_path / "fold2.npz").unlink()
    with pytest.raises(ValueError, match="incomplete fold artifacts"):
        complete_sweep_checkpoints(tmp_path)
    (tmp_path / "fold2.npz").write_bytes(b"predictions")
    (tmp_path / "fold3.pt").write_bytes(b"partial-fold")
    with pytest.raises(ValueError, match="missing, extra"):
        complete_sweep_checkpoints(tmp_path)


@pytest.mark.pure
def test_external_inference_rejects_mismatched_cv_summary(tmp_path):
    from aptos.evaluation.external import complete_sweep_checkpoints

    cfg = Config.load("configs/cv.yaml", cv={"folds": 5})
    (tmp_path / "config.json").write_text(json.dumps(cfg.to_dict()))
    (tmp_path / "summary.json").write_text(json.dumps({"folds": 2}))
    with pytest.raises(ValueError, match="fold count differs"):
        complete_sweep_checkpoints(tmp_path)


@pytest.mark.torch
def test_ensemble_rejects_tampered_saved_thresholds_and_fold_numbers(tmp_path):
    torch = pytest.importorskip("torch")
    from aptos.evaluation.external import load_ensemble

    cfg = Config.load("configs/cv.yaml", cv={"folds": 2})
    (tmp_path / "config.json").write_text(json.dumps(cfg.to_dict()))
    (tmp_path / "summary.json").write_text(json.dumps({
        "folds": 2, "ensemble": {"thresholds": [0.5, 1.5, 2.5, 3.5]},
    }))
    state = {"weight": torch.tensor([1.])}
    for fold in (1, 2):
        torch.save({"fold": fold, "state_dict": state,
                    "thresholds": [0.5, 1.5, 2.5, 3.5]},
                   tmp_path / f"fold{fold}.pt")
        (tmp_path / f"fold{fold}.npz").write_bytes(b"predictions")
        (tmp_path / f"fold{fold}.json").write_text(json.dumps({"fold": fold}))
    loaded, cuts = load_ensemble(tmp_path)
    assert len(loaded) == 2
    assert cuts.tolist() == pytest.approx([0.5, 1.5, 2.5, 3.5])

    torch.save({"fold": 99, "state_dict": state,
                "thresholds": [0.5, 1.5, 2.5, 3.5]},
               tmp_path / "fold2.pt")
    with pytest.raises(ValueError, match="fold identity"):
        load_ensemble(tmp_path)
    torch.save({"fold": 2, "state_dict": state,
                "thresholds": [0.5, 1.5, 2.5, 3.8]},
               tmp_path / "fold2.pt")
    with pytest.raises(ValueError, match="disagree with saved APTOS summary"):
        load_ensemble(tmp_path)
