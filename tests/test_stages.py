"""What each pipeline stage actually runs.

`test_pipeline.py` covers the graph and the preconditions. These cover the
commands, because two stages were wired wrong in a way preconditions could not
catch: `preprocess` called its script with no arguments and would have
overwritten the CLAHE cache with differently squared images, and `cv` ran the
old non-resumable script instead of the package's sweep.
"""
import argparse
import json

import pytest

from aptos import pipeline
from aptos.config import Config
from aptos.data import manifest as manifest_mod

pytestmark = pytest.mark.pure


@pytest.fixture
def captured(monkeypatch):
    """Record commands instead of running them."""
    calls = []
    monkeypatch.setattr(pipeline, "_run", lambda cmd, cfg: calls.append([str(c) for c in cmd]))
    return calls


def _args(**kw):
    base = {"extra": [], "force": False, "dry_run": False, "config": None}
    base.update(kw)
    return argparse.Namespace(**base)


def _cfg(root, **overrides):
    return Config.load(paths={"root": str(root)}, **overrides)


def _write_cache(cfg, settings, n=3):
    target = cfg.data_dir
    for split in ("train", "valid", "test"):
        (target / split).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (target / "train" / f"img{i}.jpg").write_bytes(b"")
    manifest_mod.write(target, settings, n)
    return target


# ------------------------------------------------------------------ preprocess

def test_preprocess_passes_the_variants_own_settings(tmp_path, captured):
    """Baseline is no-CLAHE, pad, written to data/processed - the script's own
    defaults are CLAHE, squash, data/processed_clahe, the opposite on every axis."""
    cfg = _cfg(tmp_path)  # defaults: baseline
    pipeline._run_preprocess(cfg, _args())

    cmd = captured[0]
    assert "--no-clahe" in cmd
    assert cmd[cmd.index("--square-mode") + 1] == "pad"
    assert cmd[cmd.index("--out") + 1].endswith("processed")
    assert cmd[cmd.index("--size") + 1] == "512"


def test_preprocess_passes_clahe_settings_for_the_clahe_variant(tmp_path, captured):
    cfg = _cfg(tmp_path, variant="clahe",
               preprocess={"clahe": True, "clip_limit": 2.0, "square_mode": "pad"})
    pipeline._run_preprocess(cfg, _args())

    cmd = captured[0]
    assert "--no-clahe" not in cmd
    assert cmd[cmd.index("--clip-limit") + 1] == "2.0"
    assert cmd[cmd.index("--square-mode") + 1] == "pad"
    assert cmd[cmd.index("--out") + 1].endswith("processed_clahe")


def test_preprocess_skips_a_cache_that_already_matches(tmp_path, captured):
    cfg = _cfg(tmp_path)
    _write_cache(cfg, cfg.manifest_fields())
    pipeline._run_preprocess(cfg, _args())
    assert captured == []


def test_preprocess_refuses_to_overwrite_a_cache_built_differently(tmp_path, captured):
    """The failure this guards against: a squash rebuild landing on top of a pad
    cache, silently changing the pixels behind results already published."""
    cfg = _cfg(tmp_path)
    _write_cache(cfg, {**cfg.manifest_fields(), "square_mode": "squash"})

    with pytest.raises(pipeline.PreconditionError, match="square_mode"):
        pipeline._run_preprocess(cfg, _args())
    assert captured == [], "nothing may run once the mismatch is found"


def test_force_rebuilds_a_mismatched_cache(tmp_path, captured):
    cfg = _cfg(tmp_path)
    _write_cache(cfg, {**cfg.manifest_fields(), "square_mode": "squash"})
    pipeline._run_preprocess(cfg, _args(force=True))
    assert len(captured) == 1


def test_an_interrupted_cache_is_not_mistaken_for_a_finished_one(tmp_path, captured):
    """Right settings, wrong image count - a run that died partway."""
    cfg = _cfg(tmp_path)
    target = _write_cache(cfg, cfg.manifest_fields(), n=3)
    manifest = json.loads(manifest_mod.path_for(target).read_text())
    manifest["n_images"] = 3662
    manifest_mod.path_for(target).write_text(json.dumps(manifest))

    with pytest.raises(pipeline.PreconditionError, match="n_images"):
        pipeline._run_preprocess(cfg, _args())


# ------------------------------------------------------------ other runners

def test_scan_does_not_try_bigquery_by_default(tmp_path, captured):
    pipeline._run_scan(_cfg(tmp_path), _args())
    assert "--no-bq" in captured[0]


def test_scan_writes_to_bigquery_only_when_asked(tmp_path, captured):
    cfg = _cfg(tmp_path, tracking={"bigquery_enabled": True, "bigquery_project": "p"})
    pipeline._run_scan(cfg, _args())
    assert "--no-bq" not in captured[0]


def test_cv_runs_the_resumable_package_sweep_not_the_old_script(tmp_path, captured):
    pipeline._run_cv(_cfg(tmp_path), _args())
    cmd = captured[0]
    assert "aptos.training.cv" in cmd
    assert not any(part.endswith("train_cv.py") for part in cmd)


def test_cv_carries_the_stratification(tmp_path, captured):
    cfg = _cfg(tmp_path, cv={"stratify_on": "diagnosis+resolution"})
    pipeline._run_cv(cfg, _args())
    cmd = captured[0]
    assert cmd[cmd.index("--stratify") + 1] == "diagnosis+resolution"


def test_train_uses_the_variant_cache_and_excludes_leaks(tmp_path, captured):
    pipeline._run_train(_cfg(tmp_path), _args())
    cmd = captured[0]
    assert cmd[cmd.index("--data-dir") + 1].endswith("processed")
    assert "--exclude-leaked" in cmd
    assert "--no-bq" in cmd


# ------------------------------------------------------------------------- CLI
# These go through main(), i.e. through argparse, because that is where the
# dry-run bug lived: every test above called run_stages() directly and never
# saw how the command line was parsed.

@pytest.fixture
def no_execution(monkeypatch):
    """Fail loudly if any stage body runs."""
    for stage in pipeline.STAGES:
        monkeypatch.setattr(stage, "run",
                            lambda cfg, args, _n=stage.name: pytest.fail(f"{_n} executed"))


def _preconditions(monkeypatch, problems):
    """Pin every stage's precondition result, so the test does not depend on
    whether this machine happens to have the dataset. The first version of the
    test below read the real repository and passed only where data/ existed."""
    for stage in pipeline.STAGES:
        monkeypatch.setattr(stage, "check", lambda cfg, _p=problems: list(_p))


def test_dry_run_after_the_stage_names_does_not_execute(no_execution, monkeypatch, capsys):
    """`run all --dry-run` - the form the module's own docstring recommends -
    used to execute the prepare stage for real."""
    _preconditions(monkeypatch, [])
    assert pipeline.main(["run", "quality", "--dry-run"]) == 0
    assert "dry run" in capsys.readouterr().out


def test_dry_run_reports_blocked_stages_without_executing(no_execution, monkeypatch, capsys):
    """With inputs missing, a dry run should survey and report, not raise."""
    _preconditions(monkeypatch, ["label table: found 0, need at least 1"])
    assert pipeline.main(["run", "quality", "--dry-run"]) == 0
    assert "WOULD FAIL" in capsys.readouterr().out


def test_dry_run_before_the_stage_names_does_not_execute(no_execution, monkeypatch):
    _preconditions(monkeypatch, [])
    assert pipeline.main(["run", "--dry-run", "quality"]) == 0


def test_arguments_after_a_double_dash_reach_the_stage(monkeypatch):
    seen = {}
    for stage in pipeline.STAGES:
        monkeypatch.setattr(stage, "run", lambda cfg, args: seen.setdefault("extra", args.extra))
        monkeypatch.setattr(stage, "check", lambda cfg: [])
    pipeline.main(["run", "prepare", "--only", "--", "--workers", "2"])
    assert seen["extra"] == ["--workers", "2"]


def test_pipeline_flags_are_not_passed_through_to_scripts(monkeypatch):
    seen = {}
    for stage in pipeline.STAGES:
        monkeypatch.setattr(stage, "run", lambda cfg, args: seen.setdefault("extra", args.extra))
        monkeypatch.setattr(stage, "check", lambda cfg: [])
    pipeline.main(["run", "prepare", "--only", "--force"])
    assert seen["extra"] == []
