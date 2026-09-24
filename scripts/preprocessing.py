"""Backwards-compatible shim.

The scripts still living under scripts/ import `preprocessing` by path, so
the module has to stay reachable here even though it lives in the installable
package at `src/aptos/preprocessing.py`. (The Colab notebook used to be the
reason; it now installs the package instead.)

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
