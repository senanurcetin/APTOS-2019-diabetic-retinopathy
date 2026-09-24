"""Experiment tracking.

This replaces a BigQuery layer that no longer exists. The project logged every
run to `datascientis.APTOS_2019` and, when that project was withdrawn, its
entire experimental record went with it - the numbers survive only because they
were transcribed into RESULTS.md by hand.

So the system of record is local and travels with the repository. BigQuery
remains available as an extra sink for anyone who wants it, and is off by
default. Local-first is not a downgrade here; it is the fix for the specific
way this project broke.

MLflow is imported lazily so the package still works - and CI still runs - when
it is not installed.
"""
from __future__ import annotations

import contextlib
import pathlib
import subprocess
from collections.abc import Iterator
from typing import Any

from aptos.config import Config


def _mlflow():
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "mlflow is not installed. Install the training extras:\n"
            "    pip install -e .[train]"
        ) from exc
    return mlflow


def git_commit(root: pathlib.Path | None = None) -> str:
    """The commit a run was produced from, or 'unknown'.

    Recorded as a parameter because a metric you cannot tie to a revision is a
    number without a method.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def setup(cfg: Config):
    """Point MLflow at this repository's own store."""
    mlflow = _mlflow()
    mlflow.set_tracking_uri(cfg.tracking.resolved_uri(cfg.paths))
    mlflow.set_experiment(cfg.tracking.experiment)
    return mlflow


@contextlib.contextmanager
def run(cfg: Config, name: str, *, tags: dict[str, Any] | None = None,
        nested: bool = False) -> Iterator[Any]:
    """One tracked run.

    Logs the fully resolved config as parameters, so a run always states how it
    was produced - the same guarantee `_manifest.json` gives a processed cache.
    """
    mlflow = setup(cfg)
    all_tags = {"variant": cfg.variant, "git_commit": git_commit(cfg.paths.root)}
    all_tags.update(tags or {})

    with mlflow.start_run(run_name=name, nested=nested, tags=all_tags) as active:
        params = {k: v for k, v in cfg.to_dict().items() if v is not None}
        # MLflow rejects parameters over 500 characters; nothing here is close,
        # but tuples are stringified so they stay readable.
        mlflow.log_params({k: (str(v) if isinstance(v, tuple) else v) for k, v in params.items()})
        yield active


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    mlflow = _mlflow()
    clean = {
        k: float(v) for k, v in metrics.items()
        if isinstance(v, (int, float)) and not _is_nan(v)
    }
    if clean:
        mlflow.log_metrics(clean, step=step)


def log_dict(payload: dict[str, Any], filename: str) -> None:
    """Attach a JSON artefact - thresholds, per-class tables, fold summaries."""
    _mlflow().log_dict(payload, filename)


def log_artifact(path: str | pathlib.Path, artifact_path: str | None = None) -> None:
    _mlflow().log_artifact(str(path), artifact_path=artifact_path)


def _is_nan(value: float) -> bool:
    return value != value


# ----------------------------------------------------------------- backfilling

# The six single-split runs recorded in RESULTS.md, transcribed once so the
# history and anything new live in the same place and can be compared directly.
# These predate this tracking layer; `source: results_md` marks them as
# transcribed rather than observed, because that distinction matters.
HISTORICAL_RUNS: tuple[dict[str, Any], ...] = (
    {"variant": "baseline", "seed": 42, "valid_qwk": 0.8980, "test_qwk": 0.8960,
     "test_acc": 0.7732, "test_macro_f1": 0.5680},
    {"variant": "baseline", "seed": 43, "valid_qwk": 0.8968, "test_qwk": 0.9024,
     "test_acc": 0.7760, "test_macro_f1": 0.5777},
    {"variant": "baseline", "seed": 44, "valid_qwk": 0.8927, "test_qwk": 0.8976,
     "test_acc": 0.8005, "test_macro_f1": 0.5919},
    {"variant": "clahe", "seed": 42, "valid_qwk": 0.9059, "test_qwk": 0.9112,
     "test_acc": 0.8197, "test_macro_f1": 0.6060},
    {"variant": "clahe", "seed": 43, "valid_qwk": 0.9020, "test_qwk": 0.8823,
     "test_acc": 0.7787, "test_macro_f1": 0.5439},
    {"variant": "clahe", "seed": 44, "valid_qwk": 0.9054, "test_qwk": 0.8927,
     "test_acc": 0.7787, "test_macro_f1": 0.4850},
)


def backfill_history(cfg: Config) -> int:
    """Record the six pre-existing single-split runs as MLflow runs.

    Idempotent: a second call finds them already there and does nothing.
    """
    mlflow = setup(cfg)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(cfg.tracking.experiment)

    existing = set()
    if experiment is not None:
        for r in client.search_runs([experiment.experiment_id], max_results=500):
            existing.add(r.data.tags.get("mlflow.runName", ""))

    written = 0
    for record in HISTORICAL_RUNS:
        name = f"historical-{record['variant']}-seed{record['seed']}"
        if name in existing:
            continue
        with mlflow.start_run(
            run_name=name,
            tags={
                "variant": record["variant"],
                "source": "results_md",
                "split": "single",
                "note": "transcribed from RESULTS.md, predates this tracking layer",
            },
        ):
            mlflow.log_params({
                "variant": record["variant"],
                "train.seed": record["seed"],
                "train.model": "efficientnet_b0",
                "train.mode": "reg",
                "train.size": 384,
                "train.batch": 16,
                "train.epochs": 15,
                "preprocess.size": 512,
                "preprocess.clahe": record["variant"] == "clahe",
                # Both recorded variants were built with pad, despite the
                # documentation implying squash. See configs/base.yaml.
                "preprocess.square_mode": "pad",
            })
            mlflow.log_metrics({
                k: v for k, v in record.items() if k not in ("variant", "seed")
            })
            written += 1
    return written
