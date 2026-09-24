"""Tests for the stage graph and its preconditions.

The central one is `test_quality_is_blocked_without_the_processed_cache`. That
is the regression for the defect this module exists to prevent: running the
quality stage before preprocessing used to produce an empty answer instead of an
error, which is how the repository came to document a run order that could not
reproduce its own published results.
"""
import pytest

from aptos.config import Config
from aptos.pipeline import (
    DEFAULT_ORDER,
    STAGES,
    STAGES_BY_NAME,
    PreconditionError,
    Requirement,
    resolve_order,
    run_stages,
)

pytestmark = pytest.mark.pure


# ------------------------------------------------------------------- fixtures

def _make_tree(root, *, processed: bool = True, labels: bool = True,
               stats: bool = True, leaks: bool = True, raw_images: int = 120):
    """Build a skeleton repository tree. Files are empty - the preconditions
    check existence and count, which is exactly what we want to exercise."""
    (root / "data" / "raw").mkdir(parents=True)
    for name in ("train_1.csv", "valid.csv", "test.csv"):
        (root / "data" / "raw" / name).write_text("id_code,diagnosis\n")

    images = root / "data" / "images" / "train_images" / "train_images"
    images.mkdir(parents=True)
    for i in range(raw_images):
        (images / f"img{i:04d}.png").write_bytes(b"")

    (root / "data" / "bq").mkdir(parents=True)
    if labels:
        (root / "data" / "bq" / "aptos_labels.csv").write_text("id_code,diagnosis,split\n")
    if stats:
        (root / "data" / "bq" / "image_stats.csv").write_text("id_code,width,height\n")

    (root / "reports").mkdir(parents=True)
    (root / "reports" / "problem_images.csv").write_text("id_code,split,issue,value\n")
    if leaks:
        (root / "reports" / "leaked_train_ids.csv").write_text("id_code\n")

    if processed:
        for split in ("train", "valid", "test"):
            d = root / "data" / "processed" / split
            d.mkdir(parents=True)
            for i in range(120):
                (d / f"img{i:04d}.jpg").write_bytes(b"")

    (root / "scripts").mkdir(parents=True)
    return root


def _cfg(root):
    return Config.load(paths={"root": str(root)})


class _Args:
    dry_run = False
    extra: list[str] = []


# --------------------------------------------------------- the ordering defect

def test_quality_is_blocked_without_the_processed_cache(tmp_path, monkeypatch):
    """The regression for the defect that motivated this module.

    `quality` verifies duplicate candidates at pixel level by reading thumbnails
    out of the processed cache. Without that cache every comparison silently
    returns False and the stage writes zero duplicates and an empty leak list.
    It must refuse to run instead.

    The upstream stages are stubbed out so the runner actually reaches `quality`
    - what is under test is the refusal, not the scripts in front of it.
    """
    root = _make_tree(tmp_path / "repo", processed=False)
    cfg = _cfg(root)

    problems = STAGES_BY_NAME["quality"].check(cfg)

    assert problems, "quality reported no problem despite the processed cache being absent"
    assert any("processed cache" in p for p in problems)
    assert any("preprocess" in p for p in problems), "the message should name the skipped stage"

    executed = []
    for name in ("prepare", "scan", "preprocess"):
        monkeypatch.setattr(
            STAGES_BY_NAME[name], "run",
            lambda cfg, args, _n=name: executed.append(_n),
        )
    # `quality` must never be entered at all.
    monkeypatch.setattr(
        STAGES_BY_NAME["quality"], "run",
        lambda cfg, args: pytest.fail("quality ran despite its inputs being absent"),
    )

    with pytest.raises(PreconditionError, match="quality"):
        run_stages(cfg, ["quality"], _Args())

    assert executed == ["prepare", "scan", "preprocess"]


def test_quality_is_ready_once_the_cache_exists(tmp_path):
    root = _make_tree(tmp_path / "repo", processed=True)
    assert STAGES_BY_NAME["quality"].check(_cfg(root)) == []


def test_a_half_written_cache_is_also_blocked(tmp_path):
    """An interrupted preprocessing run leaves a directory that exists but is
    nearly empty. That is as damaging as a missing one and must not pass."""
    root = _make_tree(tmp_path / "repo", processed=False)
    partial = root / "data" / "processed" / "train"
    partial.mkdir(parents=True)
    for i in range(5):
        (partial / f"img{i}.jpg").write_bytes(b"")

    problems = STAGES_BY_NAME["quality"].check(_cfg(root))
    assert problems and "found 5" in problems[0]


def test_quality_declares_preprocess_as_a_dependency():
    """The graph itself has to encode the ordering, not just the file checks."""
    assert "preprocess" in STAGES_BY_NAME["quality"].depends_on
    assert DEFAULT_ORDER.index("preprocess") < DEFAULT_ORDER.index("quality")


# ------------------------------------------------------------ dependency graph

def test_resolve_order_pulls_in_dependencies():
    assert resolve_order(["quality"]) == ["prepare", "scan", "preprocess", "quality"]


def test_resolve_order_is_topological_and_deduplicated():
    order = resolve_order(["cv", "confound"])
    assert order.count("preprocess") == 1
    for name in order:
        for dep in STAGES_BY_NAME[name].depends_on:
            assert order.index(dep) < order.index(name), f"{dep} must precede {name}"


def test_resolve_order_rejects_an_unknown_stage():
    with pytest.raises(KeyError, match="unknown stage"):
        resolve_order(["preprocesss"])


def test_every_declared_dependency_exists():
    for stage in STAGES:
        for dep in stage.depends_on:
            assert dep in STAGES_BY_NAME, f"{stage.name} depends on unknown stage {dep!r}"


def test_the_gpu_stages_are_the_training_ones():
    """Only one GPU stage may run at a time on this hardware; the flag is what
    the runner uses to warn about that."""
    assert {s.name for s in STAGES if s.gpu} == {"train", "cv"}


# ------------------------------------------------------------------ dry run

def test_dry_run_reports_failures_without_raising(tmp_path):
    """A dry run should survey everything, not stop at the first problem."""
    root = _make_tree(tmp_path / "repo", processed=False)
    args = _Args()
    args.dry_run = True
    assert run_stages(_cfg(root), ["quality"], args) == 0


# ---------------------------------------------------------------- requirements

def test_requirement_counts_glob_matches(tmp_path):
    (tmp_path / "data").mkdir()
    for i in range(3):
        (tmp_path / "data" / f"f{i}.csv").write_text("")
    cfg = _cfg(tmp_path)

    assert Requirement("csvs", "data/*.csv", min_count=3).check(cfg)[0] is True
    assert Requirement("csvs", "data/*.csv", min_count=4).check(cfg)[0] is False


def test_requirement_handles_a_plain_path(tmp_path):
    (tmp_path / "file.txt").write_text("")
    cfg = _cfg(tmp_path)
    assert Requirement("f", "file.txt").check(cfg)[0] is True
    assert Requirement("g", "missing.txt").check(cfg)[0] is False


def test_requirement_message_names_the_producing_stage(tmp_path):
    cfg = _cfg(tmp_path)
    ok, message = Requirement("label table", "data/bq/x.csv", produced_by="prepare").check(cfg)
    assert ok is False
    assert "`prepare`" in message
