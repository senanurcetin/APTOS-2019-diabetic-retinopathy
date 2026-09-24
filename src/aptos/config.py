"""Every constant in the project, in one place.

Before this module the same values lived in several files at once: `SOURCE_DIRS`
was copy-pasted verbatim into three scripts, `GRADES` into four, the ImageNet
statistics into three, and the augmentation pipeline into two. Duplication like
that does not stay in sync — the repository already carried a live example, with
`HARD_BRIGHT = 250.0` in code and "240" written in two places of prose.

Design choice: plain dataclasses plus a YAML loader, deliberately not Hydra.
Hydra brings its own CLI grammar and rewrites the working directory, which
fights the argparse interface the pipeline already exposes; the configuration
surface here is small enough that a dataclass tree is easier to read, easier to
test, and adds no dependency.

Usage:
    from aptos.config import Config
    cfg = Config.load()                       # defaults
    cfg = Config.load("configs/squash.yaml")  # a variant
    cfg = Config.load("configs/cv.yaml", train={"seed": 43})
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any

import yaml

# Resolved from this file's location: src/aptos/config.py -> repository root.
# APTOS_ROOT overrides it, which is what makes the package work when it is
# installed somewhere other than the checkout (a container, a Space).
_DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[2]
ROOT = pathlib.Path(os.environ.get("APTOS_ROOT", _DEFAULT_ROOT))

# ICDRSS grades. The ordering is the severity ramp and is relied upon by the
# ordinal loss, the threshold search, and every per-class table.
GRADES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}
N_GRADES = len(GRADES)

# A referral decision, not a grade. Screening asks "does this person need to see
# an ophthalmologist?", which is grade 2 and above.
REFERABLE_FROM = 2


@dataclass
class Paths:
    """Where things live. Every path is derived from `root`."""

    root: pathlib.Path = ROOT

    @property
    def data(self) -> pathlib.Path:
        return self.root / "data"

    @property
    def raw(self) -> pathlib.Path:
        return self.data / "raw"

    @property
    def images(self) -> pathlib.Path:
        return self.data / "images"

    @property
    def bq(self) -> pathlib.Path:
        return self.data / "bq"

    @property
    def models(self) -> pathlib.Path:
        return self.root / "models"

    @property
    def reports(self) -> pathlib.Path:
        return self.root / "reports"

    @property
    def figures(self) -> pathlib.Path:
        return self.reports / "figures"

    @property
    def external(self) -> pathlib.Path:
        return self.data / "external"

    @property
    def labels_csv(self) -> pathlib.Path:
        return self.bq / "aptos_labels.csv"

    @property
    def image_stats_csv(self) -> pathlib.Path:
        return self.bq / "image_stats.csv"

    @property
    def problem_images_csv(self) -> pathlib.Path:
        return self.reports / "problem_images.csv"

    @property
    def leaked_ids_csv(self) -> pathlib.Path:
        return self.reports / "leaked_train_ids.csv"

    def processed(self, variant: str) -> pathlib.Path:
        """Directory holding one preprocessing variant's cached JPEGs."""
        return self.data / PROCESSED_DIRS[variant]

    # The Kaggle archive nests each split inside a directory of the same name,
    # and uses `val_images` where the label CSVs say `valid`. Both quirks are
    # the archive's, not ours; they are encoded here once so no script has to
    # remember them.
    @property
    def source_dirs(self) -> dict[str, pathlib.Path]:
        return {
            "train": self.images / "train_images" / "train_images",
            "valid": self.images / "val_images" / "val_images",
            "test": self.images / "test_images" / "test_images",
        }


# Variant name -> directory name. These are the three caches already built on
# disk; their real settings are recorded in each directory's `_manifest.json`
# and are asserted against this table at load time.
PROCESSED_DIRS = {
    "baseline": "processed",
    "clahe": "processed_clahe",
    "squash": "processed_squash",
}


@dataclass
class PreprocessConfig:
    """Settings that define a processed image cache.

    These are exactly the fields written to `_manifest.json`, so a cache can
    always state how it was produced, and a training run can refuse a cache that
    does not match what it asked for.
    """

    size: int = 512
    clahe: bool = False
    clip_limit: float | None = None
    # `to_square`'s own default is "squash"; the three caches on disk were built
    # as pad / pad / squash. Variant configs set this explicitly rather than
    # inheriting a default, because that mismatch is exactly how the published
    # results came to describe padded images as if they were squashed.
    square_mode: str = "pad"
    crop_tol: int = 7
    jpeg_quality: int = 95
    workers: int = 8


@dataclass
class QualityConfig:
    """Thresholds for the data-quality pass."""

    # Usability gate: "does this frame carry any information at all?" Measured
    # brightness in this dataset spans 15.0-129.6, so these never fire. That is
    # a clean-data signal, not dead code.
    hard_dark: float = 8.0
    hard_bright: float = 250.0

    # Distribution-relative outliers. MAD flags nothing when nothing genuinely
    # deviates, so a zero here is informative in a way a percentile's zero
    # never is.
    mad_k: float = 3.5

    # Duplicate verification. dHash is a candidate generator only: 181 of 312
    # candidate groups were false positives, because every fundus image is a
    # bright disc on black. These two thresholds are the pixel-level check that
    # turns a candidate into a duplicate.
    dup_min_corr: float = 0.995
    dup_max_mae: float = 3.0
    thumb_size: int = 128


@dataclass
class AugmentationConfig:
    """Training-time augmentation. Kept here so the figure that illustrates it
    and the transform that applies it cannot drift apart."""

    resized_crop_scale: tuple[float, float] = (0.85, 1.0)
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    rotation_degrees: int = 20
    color_jitter_brightness: float = 0.15
    color_jitter_contrast: float = 0.15


@dataclass
class ThresholdConfig:
    """Coordinate search that turns a regression output into ordinal grades.

    Fitted on validation only. Worth remembering when reading validation QWK:
    part of it is a quantity we fitted, which is why the CLAHE effect that won
    on validation in all three seeds did not survive on test.
    """

    start: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5)
    rounds: int = 60
    deltas: tuple[float, ...] = (-0.12, -0.04, 0.04, 0.12)
    min_gap: float = 0.05


@dataclass
class TrainConfig:
    model: str = "efficientnet_b0"
    mode: str = "reg"  # "reg" (ordinal regression) or "cls"
    size: int = 384
    batch: int = 16
    grad_accum: int = 1
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 5
    min_delta: float = 0.0
    seed: int = 42
    # Two, not four. Windows dataloader workers re-import the main module by
    # path, and four of them alongside the 6 GB GPU is what exhausted the
    # commit limit and killed earlier runs.
    workers: int = 2
    persistent_workers: bool = True
    amp: bool = True
    exclude_leaked: bool = True


@dataclass
class CVConfig:
    folds: int = 5
    # "diagnosis" reproduces the original design. "diagnosis+resolution" is the
    # confound-aware variant: stratifying jointly means no fold can exploit a
    # resolution-to-label mapping that is absent from the others.
    stratify_on: str = "diagnosis"
    # The test split stays out of the pool entirely, in both modes.
    pool_splits: tuple[str, ...] = ("train", "valid")


@dataclass
class ConfoundConfig:
    """The metadata shortcut baseline.

    A RandomForest on these columns alone reaches QWK 0.652 without seeing a
    retinal pixel, because 92.5% of the 1050x1050 images are No DR against
    33.6% at every other resolution.
    """

    meta_cols: tuple[str, ...] = (
        "width", "height", "aspect_ratio", "megapixels",
        "brightness", "contrast_std", "file_kb",
    )
    n_estimators: int = 200
    cv_folds: int = 5
    random_state: int = 0
    # The resolution that carries the confound, kept as data rather than as a
    # sentence hardcoded inside a generated report.
    confounded_resolution: tuple[int, int] = (1050, 1050)


@dataclass
class TrackingConfig:
    """MLflow is the system of record. BigQuery is an optional extra sink.

    The project logged to BigQuery once and became unreproducible the day that
    access was withdrawn. Local-first is not a downgrade here; it is the fix.
    """

    experiment: str = "aptos-2019"
    # Empty resolves to a SQLite store at <root>/mlflow.db. MLflow 3 put the
    # filesystem backend into maintenance mode and refuses it outright, and
    # SQLite is the better choice here anyway: the leaderboard, confusion and
    # per-class views that used to be BigQuery queries become SQL again,
    # against a file that travels with the repository.
    uri: str = ""
    bigquery_enabled: bool = False
    bigquery_project: str = ""
    bigquery_dataset: str = ""

    def resolved_uri(self, paths: Paths) -> str:
        if self.uri:
            return self.uri
        # SQLAlchemy wants forward slashes, including on Windows.
        return "sqlite:///" + str(paths.root / "mlflow.db").replace("\\", "/")


@dataclass
class Config:
    variant: str = "baseline"
    paths: Paths = field(default_factory=Paths)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    augment: AugmentationConfig = field(default_factory=AugmentationConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    confound: ConfoundConfig = field(default_factory=ConfoundConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, path: str | os.PathLike | None = None, **overrides: Any) -> Config:
        """Build a config from defaults, optionally a YAML file, then overrides.

        `overrides` is keyed by section: `Config.load(train={"seed": 43})`.
        Unknown sections and unknown keys raise rather than being ignored - a
        silently dropped `--seed` is how you get two runs you cannot tell apart.
        """
        raw: dict[str, Any] = {}
        if path is not None:
            p = pathlib.Path(path)
            if not p.is_absolute():
                p = ROOT / p
            if not p.exists():
                raise FileNotFoundError(f"config not found: {p}")
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

        for section, values in overrides.items():
            if values is None:
                continue
            raw.setdefault(section, {})
            if isinstance(raw[section], dict) and isinstance(values, dict):
                raw[section].update(values)
            else:
                raw[section] = values

        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> Config:
        kwargs: dict[str, Any] = {}
        fields = {f.name: f for f in dataclasses.fields(cls)}

        for key, value in raw.items():
            if key not in fields:
                known = ", ".join(sorted(fields))
                raise KeyError(f"unknown config section {key!r}; known sections: {known}")
            if key in ("variant",):
                kwargs[key] = value
                continue
            if key == "paths":
                kwargs[key] = Paths(root=pathlib.Path(value["root"])) if "root" in value else Paths()
                continue
            section_cls = fields[key].default_factory  # type: ignore[union-attr]
            kwargs[key] = _build_section(section_cls, value, key)

        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    # --------------------------------------------------------------- validation

    def validate(self) -> None:
        if self.variant not in PROCESSED_DIRS:
            known = ", ".join(sorted(PROCESSED_DIRS))
            raise ValueError(f"unknown variant {self.variant!r}; known: {known}")
        if self.preprocess.square_mode not in ("pad", "squash"):
            raise ValueError(f"unknown square_mode: {self.preprocess.square_mode!r}")
        if self.train.mode not in ("reg", "cls"):
            raise ValueError(f"unknown train mode: {self.train.mode!r}")
        if self.cv.stratify_on not in ("diagnosis", "diagnosis+resolution"):
            raise ValueError(f"unknown cv.stratify_on: {self.cv.stratify_on!r}")
        if self.preprocess.clahe and self.preprocess.clip_limit is None:
            raise ValueError("preprocess.clahe is on but clip_limit is None")
        if self.tracking.bigquery_enabled and not self.tracking.bigquery_project:
            raise ValueError("tracking.bigquery_enabled is on but bigquery_project is empty")

    # ------------------------------------------------------------------ helpers

    @property
    def data_dir(self) -> pathlib.Path:
        """The processed cache this config trains from."""
        return self.paths.processed(self.variant)

    def manifest_fields(self) -> dict[str, Any]:
        """Exactly what gets written to, and checked against, `_manifest.json`."""
        return {
            "size": self.preprocess.size,
            "clahe": self.preprocess.clahe,
            "clip_limit": self.preprocess.clip_limit,
            "square_mode": self.preprocess.square_mode,
        }

    def to_dict(self) -> dict[str, Any]:
        """Flat, MLflow-friendly view. Paths are excluded: they are machine
        state, not an experimental parameter."""
        out: dict[str, Any] = {"variant": self.variant}
        for f in dataclasses.fields(self):
            if f.name in ("variant", "paths"):
                continue
            section = getattr(self, f.name)
            for sf in dataclasses.fields(section):
                out[f"{f.name}.{sf.name}"] = getattr(section, sf.name)
        return out


def _build_section(section_cls: Any, value: Any, section_name: str) -> Any:
    """Instantiate one config section, rejecting unknown keys."""
    if not isinstance(value, dict):
        raise TypeError(f"config section {section_name!r} must be a mapping, got {type(value).__name__}")
    known = {f.name: f for f in dataclasses.fields(section_cls())}
    unknown = set(value) - set(known)
    if unknown:
        raise KeyError(
            f"unknown key(s) {sorted(unknown)} in config section {section_name!r}; "
            f"known keys: {', '.join(sorted(known))}"
        )
    coerced = {}
    for key, val in value.items():
        # YAML gives lists where the dataclasses declare tuples; normalise so
        # equality checks and hashing behave.
        if isinstance(val, list):
            val = tuple(val)
        coerced[key] = val
    return section_cls(**coerced)
