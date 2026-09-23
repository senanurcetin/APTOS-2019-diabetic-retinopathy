"""The pipeline: one entry point, an explicit stage graph, and preconditions.

Why this exists. The published run order could not reproduce the published
results. `quality_report.py` verifies duplicate candidates by comparing
thumbnails read out of `data/processed/`, but the README ran it at step 4 and
`preprocess_images.py` at step 5. Run in the documented order every thumbnail
read returned None, every pixel comparison returned False, and the stage wrote
0 verified duplicate groups and an empty leak list - without an error. The
committed 131 groups and 49 leaked ids are real, but they were produced in a
different order than the repository told you to use.

A comment saying "run preprocess first" would not have prevented that; the
original scripts each carried exactly such a comment. So the ordering is data
here: every stage declares what must already exist, and the runner refuses to
start a stage whose inputs are absent. The failure mode changes from a silently
empty result to a message naming the stage you skipped.

    python -m aptos.pipeline list
    python -m aptos.pipeline run quality
    python -m aptos.pipeline run --from preprocess
    python -m aptos.pipeline run all --dry-run
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import subprocess
import sys
import time
from collections.abc import Callable, Sequence

from aptos.config import PROCESSED_DIRS, Config


class PreconditionError(RuntimeError):
    """A stage was asked to run before something it reads exists."""


@dataclasses.dataclass(frozen=True)
class Requirement:
    """Something that must exist before a stage may run.

    `min_count` is what catches the interesting failures. A directory that
    exists but holds three files is the signature of an interrupted
    preprocessing run, and it is exactly as damaging as one that is missing.
    """

    describe: str
    pattern: str            # path relative to the repo root; may contain globs
    min_count: int = 1
    produced_by: str = ""   # the stage to point at when this is missing

    def check(self, cfg: Config) -> tuple[bool, str]:
        root = cfg.paths.root
        matches = list(root.glob(self.pattern)) if _has_glob(self.pattern) else (
            [root / self.pattern] if (root / self.pattern).exists() else []
        )
        found = len(matches)
        if found >= self.min_count:
            return True, f"{self.describe}: {found} found"
        hint = f" - produced by the `{self.produced_by}` stage" if self.produced_by else ""
        return False, (
            f"{self.describe}: found {found}, need at least {self.min_count} "
            f"(looked for {self.pattern!r}){hint}"
        )


def _has_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[")


@dataclasses.dataclass
class Stage:
    name: str
    description: str
    run: Callable[[Config, argparse.Namespace], None]
    requires: tuple[Requirement, ...] = ()
    depends_on: tuple[str, ...] = ()
    # Stages that touch the GPU. Only one of these should ever run at a time on
    # this hardware: the Windows commit limit, not VRAM, is what killed three
    # earlier cross-validation sweeps.
    gpu: bool = False

    def check(self, cfg: Config) -> list[str]:
        problems = []
        for req in self.requires:
            ok, message = req.check(cfg)
            if not ok:
                problems.append(message)
        return problems


# --------------------------------------------------------------------- helpers

def _script(name: str) -> Callable[[Config, argparse.Namespace], None]:
    """Run one of the not-yet-ported scripts in `scripts/`.

    Ports are landing module by module; until a stage moves into the package it
    keeps running through its original script. The precondition checking above
    applies either way, which is the part that actually matters.
    """

    def runner(cfg: Config, args: argparse.Namespace) -> None:
        script_path = cfg.paths.root / "scripts" / name
        if not script_path.exists():
            raise FileNotFoundError(f"{script_path} not found")
        cmd = [sys.executable, "-u", str(script_path), *getattr(args, "extra", [])]
        print(f"  $ {' '.join(cmd[1:])}")
        result = subprocess.run(cmd, cwd=cfg.paths.root)
        if result.returncode != 0:
            raise RuntimeError(f"{name} exited with code {result.returncode}")

    runner.__name__ = f"run_{name.replace('.py', '')}"
    return runner


# ---------------------------------------------------------------------- stages

def _processed_requirements() -> tuple[Requirement, ...]:
    """Every processed cache the duplicate verification reads from."""
    return tuple(
        Requirement(
            describe=f"processed cache '{variant}' ({directory}/train)",
            pattern=f"data/{directory}/train/*.jpg",
            min_count=100,
            produced_by="preprocess",
        )
        for variant, directory in [("baseline", PROCESSED_DIRS["baseline"])]
    )


STAGES: tuple[Stage, ...] = (
    Stage(
        name="prepare",
        description="Turn the raw Kaggle CSVs into the label table.",
        run=_script("prepare_bq_csv.py"),
        requires=(
            Requirement("raw label CSVs", "data/raw/*.csv", min_count=3),
        ),
    ),
    Stage(
        name="scan",
        description="Measure every raw image: size, brightness, contrast, dHash.",
        run=_script("scan_images.py"),
        depends_on=("prepare",),
        requires=(
            Requirement("label table", "data/bq/aptos_labels.csv", produced_by="prepare"),
            Requirement("raw training images", "data/images/train_images/train_images/*.png",
                        min_count=100),
        ),
    ),
    Stage(
        name="preprocess",
        description="Build the cached 512px JPEGs for one variant.",
        run=_script("preprocess_images.py"),
        depends_on=("prepare",),
        requires=(
            Requirement("raw training images", "data/images/train_images/train_images/*.png",
                        min_count=100),
        ),
    ),
    Stage(
        name="quality",
        description="Data-quality report, verified duplicates, and the leak list.",
        run=_script("quality_report.py"),
        # This is the ordering fix. The stage reads thumbnails out of the
        # processed cache to verify duplicate candidates at pixel level, so the
        # cache has to exist first. Previously this dependency was real but
        # undeclared, and violating it produced an empty answer instead of an
        # error.
        depends_on=("scan", "preprocess"),
        requires=(
            Requirement("image statistics", "data/bq/image_stats.csv", produced_by="scan"),
            Requirement("label table", "data/bq/aptos_labels.csv", produced_by="prepare"),
            *_processed_requirements(),
        ),
    ),
    Stage(
        name="confound",
        description="Metadata shortcut baseline and the label-noise ceiling.",
        run=_script("confound_analysis.py"),
        depends_on=("quality",),
        requires=(
            Requirement("image statistics", "data/bq/image_stats.csv", produced_by="scan"),
            Requirement("problem images", "reports/problem_images.csv", produced_by="quality"),
        ),
    ),
    Stage(
        name="images",
        description="Per-image property report.",
        run=_script("image_report.py"),
        depends_on=("scan",),
        requires=(
            Requirement("image statistics", "data/bq/image_stats.csv", produced_by="scan"),
        ),
    ),
    Stage(
        name="figures",
        description="Generate the report figures.",
        run=_script("make_figures.py"),
        depends_on=("scan",),
        requires=(
            Requirement("label table", "data/bq/aptos_labels.csv", produced_by="prepare"),
        ),
    ),
    Stage(
        name="train",
        description="Single-split training run.",
        run=_script("train.py"),
        depends_on=("preprocess", "quality"),
        gpu=True,
        requires=(
            Requirement("label table", "data/bq/aptos_labels.csv", produced_by="prepare"),
            Requirement("leak list", "reports/leaked_train_ids.csv", produced_by="quality"),
            *_processed_requirements(),
        ),
    ),
    Stage(
        name="cv",
        description="K-fold cross-validation over train+valid, test held out.",
        run=_script("train_cv.py"),
        depends_on=("preprocess", "quality"),
        gpu=True,
        requires=(
            Requirement("label table", "data/bq/aptos_labels.csv", produced_by="prepare"),
            Requirement("leak list", "reports/leaked_train_ids.csv", produced_by="quality"),
            *_processed_requirements(),
        ),
    ),
)

STAGES_BY_NAME = {s.name: s for s in STAGES}
DEFAULT_ORDER = tuple(s.name for s in STAGES)


# ----------------------------------------------------------------------- runner

def resolve_order(names: Sequence[str]) -> list[str]:
    """Expand the requested stages to include their dependencies, in order."""
    wanted: list[str] = []

    def visit(name: str) -> None:
        if name in wanted:
            return
        stage = STAGES_BY_NAME.get(name)
        if stage is None:
            known = ", ".join(DEFAULT_ORDER)
            raise KeyError(f"unknown stage {name!r}; known stages: {known}")
        for dep in stage.depends_on:
            visit(dep)
        wanted.append(name)

    for name in names:
        visit(name)
    return sorted(wanted, key=DEFAULT_ORDER.index)


def run_stages(cfg: Config, names: Sequence[str], args: argparse.Namespace) -> int:
    order = resolve_order(names)
    print(f"stages: {' -> '.join(order)}\n")

    gpu_stages = [n for n in order if STAGES_BY_NAME[n].gpu]
    if len(gpu_stages) > 1:
        print(
            f"note: {len(gpu_stages)} GPU stages queued ({', '.join(gpu_stages)}). "
            f"They run one at a time - concurrent GPU jobs exhaust the Windows "
            f"commit limit and kill dataloader workers.\n"
        )

    for name in order:
        stage = STAGES_BY_NAME[name]
        print(f"=== {name} - {stage.description}")

        problems = stage.check(cfg)
        if problems:
            message = (
                f"stage '{name}' cannot run, its inputs are not there:\n  - "
                + "\n  - ".join(problems)
            )
            if args.dry_run:
                print(f"  WOULD FAIL\n  {message}\n")
                continue
            raise PreconditionError(message)

        if args.dry_run:
            print("  ok - preconditions satisfied (dry run, not executing)\n")
            continue

        started = time.time()
        stage.run(cfg, args)
        print(f"  done in {time.time() - started:.1f}s\n")

    return 0


# -------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aptos", description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show the stage graph")

    run = sub.add_parser("run", help="run one or more stages, with their dependencies")
    run.add_argument("stages", nargs="*", default=["all"],
                     help="stage names, or 'all' (default)")
    run.add_argument("--config", default=None, help="path to a YAML config")
    run.add_argument("--from", dest="from_stage", default=None,
                     help="run this stage and everything after it")
    run.add_argument("--only", action="store_true",
                     help="run exactly the named stages, skipping dependency expansion")
    run.add_argument("--dry-run", action="store_true",
                     help="check preconditions and report, without executing")
    run.add_argument("extra", nargs=argparse.REMAINDER,
                     help="arguments passed through to the underlying script")

    check = sub.add_parser("check", help="report which stages could run right now")
    check.add_argument("--config", default=None)
    return parser


def cmd_list() -> int:
    print("stage        depends on                 description")
    print("-" * 92)
    for stage in STAGES:
        deps = ", ".join(stage.depends_on) or "-"
        gpu = " [gpu]" if stage.gpu else ""
        print(f"{stage.name:<12} {deps:<26} {stage.description}{gpu}")
    return 0


def cmd_check(cfg: Config) -> int:
    print(f"root: {cfg.paths.root}")
    print(f"variant: {cfg.variant}  ->  {cfg.data_dir.name}\n")
    print("stage        status")
    print("-" * 92)
    blocked = 0
    for stage in STAGES:
        problems = stage.check(cfg)
        if problems:
            blocked += 1
            print(f"{stage.name:<12} BLOCKED")
            for p in problems:
                print(f"             - {p}")
        else:
            print(f"{stage.name:<12} ready")
    print(f"\n{len(STAGES) - blocked}/{len(STAGES)} stages ready")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "list":
        return cmd_list()

    cfg = Config.load(args.config)

    if args.command == "check":
        return cmd_check(cfg)

    names = list(args.stages)
    if args.from_stage:
        start = DEFAULT_ORDER.index(args.from_stage)
        names = list(DEFAULT_ORDER[start:])
    elif names == ["all"]:
        names = list(DEFAULT_ORDER)

    if args.only:
        order = names
        print(f"stages: {' -> '.join(order)} (--only: dependencies not expanded)\n")
        for name in order:
            stage = STAGES_BY_NAME[name]
            problems = stage.check(cfg)
            if problems and not args.dry_run:
                raise PreconditionError(
                    f"stage '{name}' cannot run, its inputs are not there:\n  - "
                    + "\n  - ".join(problems)
                )
            if not args.dry_run:
                stage.run(cfg, args)
        return 0

    try:
        return run_stages(cfg, names, args)
    except PreconditionError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
