"""The deployed path: the numpy input transform and the weight fetcher.

Both are second implementations of something, which is exactly where this
project has found its bugs. The transform replaces torchvision's; the fetcher
replaces "the files happen to be there".
"""
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- fetch_weights

@pytest.fixture
def fetch(monkeypatch, tmp_path):
    """fetch_weights with the network replaced by a local directory."""
    module = _load("fetch_weights", "serving/fetch_weights.py")
    source = tmp_path / "remote"
    source.mkdir()

    def fake_retrieve(url, path):
        name = url.rsplit("/", 1)[1]
        pathlib.Path(path).write_bytes((source / name).read_bytes())

    monkeypatch.setattr(module.urllib.request, "urlretrieve", fake_retrieve)
    return module, source


def _publish(source, *, sweep="baseline-x", mismatches=0):
    (source / "export.json").write_text(json.dumps({
        "sweep": sweep,
        "parity": {"max_abs_diff_raw": 3e-5, "grade_mismatches": mismatches},
    }))
    for i in range(1, 6):
        (source / f"fold{i}.onnx").write_bytes(b"onnx")


@pytest.mark.pure
def test_fetch_downloads_every_file(fetch, tmp_path):
    module, source = fetch
    _publish(source)
    target = module.fetch("r", "abc", "baseline-x", tmp_path / "out")
    assert sorted(p.name for p in target.iterdir()) == sorted(module.FILES)


@pytest.mark.pure
def test_fetch_refuses_an_export_that_failed_parity(fetch, tmp_path):
    """An export that did not reproduce the torch model must never be served,
    even if someone uploaded it anyway."""
    module, source = fetch
    _publish(source, mismatches=2)
    with pytest.raises(RuntimeError, match="parity"):
        module.fetch("r", "abc", "baseline-x", tmp_path / "out")


@pytest.mark.pure
def test_fetch_refuses_the_wrong_sweep(fetch, tmp_path):
    module, source = fetch
    _publish(source, sweep="clahe-y")
    with pytest.raises(RuntimeError, match="expected"):
        module.fetch("r", "abc", "baseline-x", tmp_path / "out")


# ------------------------------------------------------------ input transform

@pytest.mark.torch
def test_numpy_transform_matches_torchvision_exactly():
    """The ONNX backend feeds the model through numpy and PIL instead of
    torchvision. If the two disagree by even a pixel, the export's parity check
    - done on torchvision inputs - would not cover what is actually served."""
    pytest.importorskip("torchvision")
    pytest.importorskip("fastapi")
    from PIL import Image

    from aptos.training.loop import build_transforms

    app = _load("serving_app", "serving/app.py")
    grader = app.OnnxGrader.__new__(app.OnnxGrader)
    grader.size = 384
    _, torchvision_eval = build_transforms(384)

    rng = np.random.default_rng(0)
    for shape in [(512, 512, 3), (400, 640, 3), (600, 450, 3)]:
        bgr = rng.integers(0, 256, shape, dtype=np.uint8)
        rgb = bgr[..., ::-1].copy()
        expected = torchvision_eval(Image.fromarray(rgb)).unsqueeze(0).numpy()
        actual = grader._to_input(bgr)
        assert actual.shape == expected.shape == (1, 3, 384, 384)
        assert np.array_equal(actual, expected)
