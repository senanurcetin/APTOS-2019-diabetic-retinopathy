"""Strict, data-free validation of independent retinal grading tables.

Only entirely blank CSV padding rows are ignored. Real image/grade pairs
must not be changed by lossy integer coercion or filename-stem collisions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def validate_external_labels(
    frame: pd.DataFrame, *, source: str, filenames: bool = False,
) -> pd.DataFrame:
    """Validate grade 0..4; optionally preserve source filename and derive cache ID."""
    if not {"id_code", "diagnosis"}.issubset(frame.columns):
        raise ValueError(f"{source}: missing id_code or diagnosis column")
    clean = frame.dropna(how="all", subset=["id_code", "diagnosis"]).copy()
    if clean.empty:
        raise ValueError(f"{source}: no gradable labeled images")
    if clean[["id_code", "diagnosis"]].isna().any().any():
        raise ValueError(f"{source}: incomplete image/diagnosis pairs")
    names = clean["id_code"].astype("string").tolist()
    if any(
        not name or name != name.strip() or name in (".", "..")
        or "/" in name or "\\" in name or "\x00" in name
        for name in names
    ):
        raise ValueError(f"{source}: image IDs must be nonempty, path-safe and unmodified")

    if filenames:
        clean["source_name"] = names
        keys = [Path(name).stem for name in names]
    else:
        keys = names
    if any(not key or key in (".", "..") for key in keys):
        raise ValueError(f"{source}: invalid derived image ID")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{source}: duplicate image ID or filename-stem collision")

    try:
        values = pd.to_numeric(clean["diagnosis"], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: diagnosis grades must be numeric integers 0..4") from exc
    if (not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 4)
            or not np.equal(values, np.floor(values)).all()):
        raise ValueError(f"unexpected {source} grades: expected whole integers 0..4")
    clean["id_code"] = keys
    clean["diagnosis"] = values.astype(int)
    columns = ["id_code", "diagnosis", "source_name"] if filenames else ["id_code", "diagnosis"]
    return clean[columns].reset_index(drop=True)
