"""Provenance for a processed image cache.

Every `data/processed*/` directory carries a `_manifest.json` recording the
settings that produced it, so a training run can always state what it trained
on.

What is new here is `check`. Previously the manifest was printed and then
ignored: training at `--size 384` against a cache built at 512 was not flagged,
and nothing ever compared the manifest to the directory's actual contents. A
provenance record nobody verifies is decoration.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

MANIFEST_NAME = "_manifest.json"

# The fields that define a cache. `n_images` is recorded too but is a result,
# not a setting, so it is compared separately.
SETTING_FIELDS = ("size", "clahe", "clip_limit", "square_mode")


class ManifestMismatch(RuntimeError):
    """The cache on disk was not built the way the caller asked for."""


def path_for(directory: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(directory) / MANIFEST_NAME


def write(directory: str | pathlib.Path, settings: dict[str, Any], n_images: int) -> pathlib.Path:
    """Record how this cache was produced."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {k: settings[k] for k in SETTING_FIELDS if k in settings}
    payload["n_images"] = int(n_images)
    target = path_for(directory)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def read(directory: str | pathlib.Path) -> dict[str, Any]:
    """Read a manifest, or raise if the cache does not declare its provenance."""
    target = path_for(directory)
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found - this cache does not record how it was built. "
            f"Re-run the preprocess stage for this variant."
        )
    return json.loads(target.read_text(encoding="utf-8"))


def check(
    directory: str | pathlib.Path,
    expected: dict[str, Any],
    *,
    count_images: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    """Verify a cache matches the settings the caller expects.

    Checks two separate things, because they fail for different reasons:

      1. the recorded settings against `expected` - a config pointing at the
         wrong directory, or a cache rebuilt with different settings;
      2. the recorded `n_images` against what is actually on disk - a cache that
         was interrupted partway through, which the settings alone cannot reveal.

    `strict=False` downgrades both to returned findings instead of an exception,
    which is what the report stages want: they should describe a broken cache,
    not refuse to run.
    """
    directory = pathlib.Path(directory)
    manifest = read(directory)
    problems: list[str] = []

    for key in SETTING_FIELDS:
        if key not in expected:
            continue
        want, got = expected[key], manifest.get(key)
        if want != got:
            problems.append(f"{key}: cache has {got!r}, run asked for {want!r}")

    if count_images:
        on_disk = sum(1 for _ in directory.rglob("*.jpg"))
        recorded = manifest.get("n_images")
        if recorded is not None and on_disk != recorded:
            problems.append(
                f"n_images: manifest says {recorded}, found {on_disk} .jpg files on disk"
            )
        manifest["n_images_on_disk"] = on_disk

    manifest["problems"] = problems
    if problems and strict:
        raise ManifestMismatch(
            f"{directory} does not match the requested settings:\n  - "
            + "\n  - ".join(problems)
        )
    return manifest


def describe(manifest: dict[str, Any]) -> str:
    """One line for a run log, so every run states its preprocessing."""
    clahe = manifest.get("clahe")
    clahe_text = f"CLAHE clip={manifest.get('clip_limit')}" if clahe else "no CLAHE"
    return (
        f"{manifest.get('size', '?')}px, {clahe_text}, "
        f"square={manifest.get('square_mode', '?')}, "
        f"n={manifest.get('n_images', '?')}"
    )
