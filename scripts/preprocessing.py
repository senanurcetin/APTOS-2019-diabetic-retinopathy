"""Backwards-compatible shim.

The Colab notebook clones this repository at run time and imports
`preprocessing` by path, so the module has to stay reachable here even though it
now lives in the installable package at `src/aptos/preprocessing.py`.

Import from `aptos.preprocessing` in new code.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aptos.preprocessing import (  # noqa: E402,F401
    HARD_BRIGHT,
    HARD_DARK,
    IMAGENET_MEAN,
    IMAGENET_STD,
    apply_clahe,
    auto_crop,
    brightness_outliers,
    dhash,
    image_quality,
    pad_to_square,
    preprocess,
    to_square,
)

__all__ = [
    "HARD_BRIGHT", "HARD_DARK", "IMAGENET_MEAN", "IMAGENET_STD",
    "apply_clahe", "auto_crop", "brightness_outliers", "dhash",
    "image_quality", "pad_to_square", "preprocess", "to_square",
]
