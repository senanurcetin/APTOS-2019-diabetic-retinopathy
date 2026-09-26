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


# ------------------------------------------------------------------------ page

@pytest.fixture
def app_module():
    pytest.importorskip("fastapi")
    return _load("serving_app_page", "serving/app.py")


@pytest.mark.torch
def test_page_has_every_placeholder_filled(app_module):
    """The page reads its numbers from MODEL_CARD. A placeholder left in would
    ship as a JavaScript syntax error and a blank page."""
    html = app_module.index()
    for placeholder in ("__CARD__", "__REPORTED__", "__SPREAD__"):
        assert placeholder not in html


@pytest.mark.torch
def test_every_metric_the_page_names_exists_in_the_model_card(app_module):
    """The comparison skips a row whose key is missing, so a renamed key would
    silently drop a metric from the page rather than fail."""
    import re

    page = (ROOT / "serving" / "static" / "index.html").read_text(encoding="utf-8")
    named = set(re.findall(r"\b((?:aptos|idrid|messidor2)_[a-z_]+)\b", page))
    assert named, "the page no longer names any metric"
    assert named <= set(app_module.MODEL_CARD["reported"])


@pytest.mark.torch
def test_fold_spread_reference_is_a_usable_percentile_table(app_module):
    ref = app_module.MODEL_CARD["fold_spread_reference"]
    values = ref["values"]
    assert len(values) == 100 // ref["percentiles_step"] + 1
    assert all(a <= b for a, b in zip(values[:-1], values[1:], strict=True))


@pytest.mark.torch
def test_preview_is_a_small_jpeg_data_url(app_module):
    import base64

    import cv2

    url = app_module.preview_data_url(np.full((512, 512, 3), 90, np.uint8))
    prefix = "data:image/jpeg;base64,"
    assert url.startswith(prefix)
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(url[len(prefix):]), np.uint8),
                           cv2.IMREAD_COLOR)
    assert decoded.shape == (320, 320, 3)
