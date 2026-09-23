"""Tests for scripts/preprocessing.py.

The whole pipeline rests on this module; a function breaking silently would
corrupt every result downstream. These tests use synthetic images, so they run
on a machine that has never downloaded the dataset.

Usage:
    pytest tests/                        (if pytest is installed)
    python tests/test_preprocessing.py   (if it is not)
"""
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from preprocessing import (  # noqa: E402
    HARD_BRIGHT,
    HARD_DARK,
    apply_clahe,
    auto_crop,
    brightness_outliers,
    dhash,
    image_quality,
    pad_to_square,
    preprocess,
    to_square,
)


def fake_fundus(h=400, w=600, radius=180, brightness=120, seed=0):
    """A textured circular 'retina' on a black background."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= radius ** 2
    texture = rng.integers(brightness - 30, brightness + 30,
                           size=(h, w, 3), dtype=np.int16)
    img[mask] = np.clip(texture, 0, 255).astype(np.uint8)[mask]
    return img


# ------------------------------------------------------------------ auto_crop

def test_auto_crop_removes_black_border():
    img = fake_fundus(400, 600, radius=150)
    c = auto_crop(img)
    # should hug the circle: roughly 2 * radius
    assert 290 <= c.shape[0] <= 310, c.shape
    assert 290 <= c.shape[1] <= 310, c.shape


def test_auto_crop_loses_no_tissue():
    img = fake_fundus()
    before = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 7).sum()
    after = (cv2.cvtColor(auto_crop(img), cv2.COLOR_BGR2GRAY) > 7).sum()
    assert after == before, f"tissue lost: {before} -> {after}"


def test_auto_crop_survives_all_black_image():
    # Nothing to crop: must return the original, not an empty array.
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    assert auto_crop(img).shape == img.shape


def test_auto_crop_leaves_borderless_image_alone():
    img = np.full((80, 90, 3), 100, dtype=np.uint8)
    assert auto_crop(img).shape == img.shape


def test_auto_crop_accepts_grayscale():
    gray = cv2.cvtColor(fake_fundus(), cv2.COLOR_BGR2GRAY)
    c = auto_crop(gray)
    assert c.ndim == 2 and c.shape[0] < gray.shape[0]


# ------------------------------------------------------------ square handling

def test_pad_to_square_returns_square():
    for shape in [(100, 200, 3), (200, 100, 3), (150, 150, 3)]:
        out = pad_to_square(np.full(shape, 90, dtype=np.uint8))
        assert out.shape[0] == out.shape[1] == max(shape[:2])


def test_pad_preserves_content():
    img = np.full((100, 200, 3), 90, dtype=np.uint8)
    out = pad_to_square(img)
    assert (out > 7).sum() == (img > 7).sum()


def test_to_square_both_modes_hit_target_size():
    img = fake_fundus(300, 400)
    for mode in ("pad", "squash"):
        assert to_square(img, 256, mode=mode).shape == (256, 256, 3)


def test_squash_leaves_less_black_than_pad():
    # Mimic a real fundus image: a disc CLIPPED VERTICALLY by the sensor frame
    # (radius > height/2). Cropping then leaves a wide rectangle, where pad adds
    # black bars and squash does not.
    img = fake_fundus(300, 500, radius=200)
    c = auto_crop(img)
    black = {m: (cv2.cvtColor(to_square(c, 256, mode=m), cv2.COLOR_BGR2GRAY) <= 7).mean()
             for m in ("pad", "squash")}
    assert black["squash"] < black["pad"], black


def test_to_square_rejects_unknown_mode():
    try:
        to_square(fake_fundus(), 128, mode="nonsense")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown mode")


# ---------------------------------------------------------------------- CLAHE

def test_clahe_preserves_shape_and_dtype():
    img = fake_fundus()
    out = apply_clahe(img)
    assert out.shape == img.shape and out.dtype == img.dtype


def test_clahe_increases_contrast():
    rng = np.random.default_rng(1)
    img = rng.integers(100, 130, size=(300, 300, 3), dtype=np.int16).astype(np.uint8)
    before = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std()
    after = cv2.cvtColor(apply_clahe(img, clip_limit=3.0), cv2.COLOR_BGR2GRAY).std()
    assert after > before, f"contrast did not increase: {before:.2f} -> {after:.2f}"


def test_higher_clip_limit_is_more_aggressive():
    img = fake_fundus(seed=2)
    stds = [cv2.cvtColor(apply_clahe(img, clip_limit=cl), cv2.COLOR_BGR2GRAY).std()
            for cl in (1.0, 4.0)]
    assert stds[1] > stds[0], stds


def test_clahe_accepts_grayscale():
    gray = cv2.cvtColor(fake_fundus(), cv2.COLOR_BGR2GRAY)
    out = apply_clahe(gray)
    assert out.ndim == 2 and out.shape == gray.shape


# ------------------------------------------------------------ quality control

def test_image_quality_passes_a_normal_image():
    ok, m = image_quality(fake_fundus(brightness=120))
    assert ok and not m["too_dark"] and not m["too_bright"]


def test_image_quality_catches_near_black():
    ok, m = image_quality(np.full((100, 100, 3), 3, dtype=np.uint8))
    assert not ok and m["too_dark"]


def test_image_quality_catches_near_white():
    ok, m = image_quality(np.full((100, 100, 3), 253, dtype=np.uint8))
    assert not ok and m["too_bright"]


def test_hard_thresholds_are_sane():
    assert 0 < HARD_DARK < HARD_BRIGHT < 256


# ---------------------------------------------------------- outlier detection

def test_outliers_flags_almost_nothing_on_clean_data():
    # A clean normal distribution will occasionally exceed k=3.5. Expecting
    # exactly zero would be statistically wrong; we expect under 1%.
    rng = np.random.default_rng(0)
    rate = brightness_outliers(rng.normal(70, 8, 500)).mean()
    assert rate < 0.01, f"flagged {rate * 100:.1f}% of clean data"


def test_outliers_handles_zero_mad():
    # Constant input: must not divide by zero.
    assert brightness_outliers([50.0] * 100).sum() == 0


def test_outliers_catches_real_deviations():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(70, 5, 200), [5.0, 250.0]])
    m = brightness_outliers(v)
    assert m[-2] and m[-1], "genuine outliers were not flagged"
    assert m[:-2].sum() == 0, "normal values were flagged by mistake"


def test_smaller_k_flags_more():
    rng = np.random.default_rng(3)
    v = rng.normal(70, 8, 400)
    assert brightness_outliers(v, k=1.0).sum() >= brightness_outliers(v, k=3.5).sum()


# ----------------------------------------------------------------------- dhash

def test_dhash_is_stable():
    img = fake_fundus()
    assert dhash(img) == dhash(img.copy())


def test_dhash_separates_different_images():
    assert dhash(fake_fundus(seed=1)) != dhash(fake_fundus(seed=2))


def test_dhash_survives_resizing():
    # This is the point of a perceptual hash: catching a downscaled copy.
    img = fake_fundus(400, 400, radius=150)
    small = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
    h1, h2 = dhash(img), dhash(small)
    distance = bin(int(h1, 16) ^ int(h2, 16)).count("1")
    assert distance <= 8, f"hamming distance too large: {distance}"


def test_dhash_has_fixed_length():
    assert len(dhash(fake_fundus())) == 16


# -------------------------------------------------------------------- pipeline

def test_preprocess_end_to_end():
    path = pathlib.Path("_test_fundus.png")
    cv2.imwrite(str(path), fake_fundus())
    try:
        img, info = preprocess(path, size=128, use_clahe=True)
        assert img is not None and img.shape == (128, 128, 3)
        assert info["orig_width"] == 600 and info["orig_height"] == 400
        assert info["crop_width"] < info["orig_width"]
    finally:
        path.unlink(missing_ok=True)


def test_preprocess_normalize_mode():
    path = pathlib.Path("_test_fundus_norm.png")
    cv2.imwrite(str(path), fake_fundus())
    try:
        arr, _ = preprocess(path, size=64, normalize=True)
        assert arr.dtype == np.float32 and arr.shape == (64, 64, 3)
        # After ImageNet normalisation values land roughly in [-2.5, 2.5].
        assert -3.5 < arr.min() and arr.max() < 3.5
    finally:
        path.unlink(missing_ok=True)


def test_preprocess_returns_none_for_missing_file():
    img, info = preprocess("no_such_file.png")
    assert img is None and info["error"] == "unreadable"


def test_preprocess_rejects_unusable_image():
    path = pathlib.Path("_test_black.png")
    cv2.imwrite(str(path), np.full((100, 100, 3), 2, dtype=np.uint8))
    try:
        img, info = preprocess(path, quality_check=True)
        assert img is None and "dark" in info["error"]
    finally:
        path.unlink(missing_ok=True)


def test_quality_check_can_be_disabled():
    path = pathlib.Path("_test_black2.png")
    cv2.imwrite(str(path), np.full((100, 100, 3), 2, dtype=np.uint8))
    try:
        img, _ = preprocess(path, size=32, quality_check=False)
        assert img is not None
    finally:
        path.unlink(missing_ok=True)


# --------------------------------------------------------- run without pytest

def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    passed, failures = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))

    for name, err in failures:
        print(f"  FAILED  {name}\n          {err}")
    print(f"\n{passed}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
