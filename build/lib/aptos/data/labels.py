"""Loading labels, and the two derived columns the rest of the project needs.

Changed from the original: the local CSV is now the primary source and BigQuery
is an optional extra, rather than the other way round. The old `load_labels`
tried BigQuery first and fell back on any exception, which meant a working run
and a run with dead credentials looked identical in the logs — and when the
BigQuery project was withdrawn, the project's provenance went with it.
"""
from __future__ import annotations

import pathlib

import pandas as pd

from aptos.config import GRADES, REFERABLE_FROM, Config

REQUIRED_COLUMNS = ("id_code", "diagnosis", "split")
SPLITS = ("train", "valid", "test")


def load_labels(cfg: Config | None = None, *, source: str = "csv") -> pd.DataFrame:
    """Return the label table with `id_code`, `diagnosis`, `split`.

    source="csv"      read data/bq/aptos_labels.csv (default, always available)
    source="bigquery" read the BigQuery table, and fail loudly if it cannot
    """
    cfg = cfg or Config.load()

    if source == "bigquery":
        df = _load_from_bigquery(cfg)
    elif source == "csv":
        path = cfg.paths.labels_csv
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - run the `prepare` stage first "
                f"(python -m aptos.pipeline run prepare)"
            )
        df = pd.read_csv(path)
    else:
        raise ValueError(f"unknown label source: {source!r}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"label table is missing column(s): {missing}")

    bad_split = set(df["split"]) - set(SPLITS)
    if bad_split:
        raise ValueError(f"unexpected split value(s): {sorted(bad_split)}")

    bad_grade = set(df["diagnosis"].unique()) - set(GRADES)
    if bad_grade:
        raise ValueError(f"unexpected diagnosis value(s): {sorted(bad_grade)}")

    if df["id_code"].duplicated().any():
        dupes = df.loc[df["id_code"].duplicated(), "id_code"].tolist()[:5]
        raise ValueError(f"duplicate id_code(s) in the label table, e.g. {dupes}")

    return df


def _load_from_bigquery(cfg: Config) -> pd.DataFrame:
    if not cfg.tracking.bigquery_project:
        raise RuntimeError(
            "source='bigquery' requires tracking.bigquery_project to be set in the config"
        )
    from google.cloud import bigquery  # imported here so the package works without it

    query = (
        f"SELECT id_code, diagnosis, split "
        f"FROM `{cfg.tracking.bigquery_project}.{cfg.tracking.bigquery_dataset}.aptos_labels`"
    )
    return bigquery.Client(project=cfg.tracking.bigquery_project).query(query).to_dataframe()


# ------------------------------------------------------------------- enrichment

def add_referable(df: pd.DataFrame) -> pd.DataFrame:
    """Add the screening decision: does this eye need an ophthalmologist?

    Grade 2 (Moderate) and above. This is the column the clinical evaluation
    works in, and it is a materially easier and more useful question than the
    five-way grade - minority-grade confusion between 3 and 4 does not change
    the referral.
    """
    out = df.copy()
    out["is_referable"] = out["diagnosis"] >= REFERABLE_FROM
    return out


def add_resolution_bucket(
    df: pd.DataFrame, cfg: Config | None = None, stats: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Tag each image as sitting in the confounded resolution or not.

    92.5% of the 1050x1050 images carry the label No DR, against 33.6% at every
    other resolution, which is why a classifier on file metadata alone reaches
    QWK 0.652. This column is what lets evaluation be stratified on the
    shortcut, and what lets cross-validation split so the shortcut cannot be
    exploited across folds.
    """
    cfg = cfg or Config.load()
    if stats is None:
        path = cfg.paths.image_stats_csv
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - run the `scan` stage first "
                f"(python -m aptos.pipeline run scan)"
            )
        stats = pd.read_csv(path, usecols=["id_code", "width", "height"])

    w, h = cfg.confound.confounded_resolution
    stats = stats.copy()
    stats["resolution_bucket"] = (
        ((stats["width"] == w) & (stats["height"] == h))
        .map({True: f"{w}x{h}", False: "other"})
    )
    merged = df.merge(
        stats[["id_code", "resolution_bucket"]], on="id_code", how="left", validate="one_to_one"
    )
    missing = int(merged["resolution_bucket"].isna().sum())
    if missing:
        raise ValueError(
            f"{missing} image(s) have no entry in image_stats.csv; the scan stage is "
            f"out of date with the label table"
        )
    return merged


# ---------------------------------------------------------------------- leakage

def exclude_leaked(df: pd.DataFrame, cfg: Config | None = None) -> tuple[pd.DataFrame, int]:
    """Drop training images that also appear in valid or test.

    Returns (filtered, n_dropped).

    The leak was measured before it was removed: excluding these moved test QWK
    from 0.8960 to 0.8983, so it had not inflated anything. It is excluded
    regardless - measuring that a flaw is harmless is not the same as fixing it.
    """
    cfg = cfg or Config.load()
    path = cfg.paths.leaked_ids_csv
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run the `quality` stage first "
            f"(python -m aptos.pipeline run quality)"
        )
    leaked = set(pd.read_csv(path)["id_code"])
    if not leaked:
        raise ValueError(
            f"{path} is empty. That is the signature of the stage-ordering bug: "
            f"the quality stage verifies duplicates from the processed cache, so "
            f"running it before `preprocess` silently finds nothing."
        )
    before = len(df)
    out = df[~((df["split"] == "train") & (df["id_code"].isin(leaked)))].reset_index(drop=True)
    return out, before - len(out)


# ------------------------------------------------------------------------ views

def split_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Grade distribution per split, with the imbalance ratio that motivates
    every metric choice in this project."""
    table = (
        df.groupby(["split", "diagnosis"]).size().unstack(fill_value=0).reindex(columns=list(GRADES))
    )
    table.columns = [GRADES[c] for c in table.columns]
    table["total"] = table.sum(axis=1)
    table["imbalance"] = (table.iloc[:, :-1].max(axis=1) / table.iloc[:, :-1].replace(0, pd.NA).min(axis=1)).round(2)
    return table.reindex(list(SPLITS))


def pool_and_holdout(
    df: pd.DataFrame, cfg: Config | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into the cross-validation pool and the untouched test set.

    `orig_split` is carried through because an image's *role* changes from fold
    to fold but its *location on disk* does not.
    """
    cfg = cfg or Config.load()
    out = df.copy()
    out["orig_split"] = out["split"]
    pool = out[out["split"].isin(cfg.cv.pool_splits)].reset_index(drop=True)
    holdout = out[~out["split"].isin(cfg.cv.pool_splits)].reset_index(drop=True)
    return pool, holdout


def image_path(cfg: Config, id_code: str, orig_split: str) -> pathlib.Path:
    """Where one processed image lives for the configured variant."""
    return cfg.data_dir / orig_split / f"{id_code}.jpg"
