"""The provenance record of a processed cache.

The manifest existed before this rebuild but was only ever printed: training at
384px against a cache built at 512 was never flagged, and nothing compared the
manifest with what was actually on disk. These tests cover the check that
replaced the printing.
"""
import json

import pytest

from aptos.data import manifest as m

pytestmark = pytest.mark.pure

SETTINGS = {"size": 512, "clahe": False, "clip_limit": None, "square_mode": "pad"}


def _cache(tmp_path, n=4, settings=SETTINGS, recorded=None):
    for i in range(n):
        (tmp_path / "train").mkdir(exist_ok=True)
        (tmp_path / "train" / f"img{i}.jpg").write_bytes(b"")
    m.write(tmp_path, settings, recorded if recorded is not None else n)
    return tmp_path


def test_round_trip(tmp_path):
    _cache(tmp_path)
    back = m.read(tmp_path)
    assert {k: back[k] for k in SETTINGS} == SETTINGS
    assert back["n_images"] == 4


def test_write_records_only_the_settings_that_define_a_cache(tmp_path):
    m.write(tmp_path, {**SETTINGS, "workers": 8, "jpeg_quality": 95}, 1)
    assert set(json.loads(m.path_for(tmp_path).read_text())) == set(SETTINGS) | {"n_images"}


def test_a_matching_cache_passes(tmp_path):
    found = m.check(_cache(tmp_path), SETTINGS)
    assert found["problems"] == []
    assert found["n_images_on_disk"] == 4


@pytest.mark.parametrize("key,value", [
    ("size", 384), ("clahe", True), ("square_mode", "squash"), ("clip_limit", 2.0),
])
def test_every_setting_mismatch_is_caught(tmp_path, key, value):
    _cache(tmp_path)
    with pytest.raises(m.ManifestMismatch, match=key):
        m.check(tmp_path, {**SETTINGS, key: value})


def test_a_cache_interrupted_partway_is_caught(tmp_path):
    """Right settings, fewer files than recorded: the settings alone cannot
    reveal this, which is why the files are counted."""
    _cache(tmp_path, n=4, recorded=3662)
    with pytest.raises(m.ManifestMismatch, match="n_images"):
        m.check(tmp_path, SETTINGS)


def test_non_strict_reports_instead_of_raising(tmp_path):
    _cache(tmp_path)
    found = m.check(tmp_path, {**SETTINGS, "size": 384}, strict=False)
    assert len(found["problems"]) == 1


def test_a_cache_without_a_manifest_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not record how it was built"):
        m.read(tmp_path)


def test_describe_is_readable():
    assert m.describe({**SETTINGS, "n_images": 3662}) == "512px, no CLAHE, square=pad, n=3662"
    assert "CLAHE clip=2.0" in m.describe({**SETTINGS, "clahe": True, "clip_limit": 2.0})
