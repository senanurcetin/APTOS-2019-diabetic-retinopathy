"""Resumable K-fold cross-validation.

Three earlier attempts at this sweep died and each one lost everything. Two
causes were environmental and are documented in the README - a concurrent GPU
job exhausting the Windows commit limit, and editing the running script, which
kills dataloader workers because Windows workers re-import the main module by
path. The third cause was in the code, and it is the reason those failures were
total rather than partial:

  - the run id was `uuid4()`, generated fresh on every invocation, so a restart
    could not recognise its own earlier work;
  - per-fold results lived in an in-memory list;
  - the per-fold checkpoint saved weights and thresholds but not `test_raw`,
    which the ensemble needs.

So a run that died on fold 4 lost folds 1-3 for ensembling even though their
weights were sitting on disk.

Here the run id is derived from what actually defines the sweep, every fold
writes its predictions and metrics next to its checkpoint before the next fold
starts, and a restart skips folds that already finished. A died fold now costs
one fold.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from aptos.config import Config
from aptos.data import labels as labels_mod
from aptos.data import manifest as manifest_mod
from aptos.modeling.thresholds import (
    apply_thresholds,
    clinically_costly_errors,
    grade_metrics,
    optimize_thresholds,
    qwk,
    referable_confusion,
)
from aptos.training.loop import (
    build_criterion,
    build_model,
    build_optimizer,
    build_transforms,
    make_loader,
    pick_device,
    run_epoch,
    set_seed,
)


def run_id_for(cfg: Config, manifest: dict) -> str:
    """A stable identifier for one sweep.

    Derived from everything that defines the experiment, so re-invoking the same
    sweep lands in the same directory and finds its own completed folds. Change
    any of these and you get a different sweep, which is correct - the results
    would not be comparable.
    """
    payload = {
        "variant": cfg.variant,
        "folds": cfg.cv.folds,
        "stratify_on": cfg.cv.stratify_on,
        "seed": cfg.train.seed,
        "model": cfg.train.model,
        "mode": cfg.train.mode,
        "size": cfg.train.size,
        "batch": cfg.train.batch,
        "epochs": cfg.train.epochs,
        "lr": cfg.train.lr,
        "exclude_leaked": cfg.train.exclude_leaked,
        "cache": {k: manifest.get(k) for k in manifest_mod.SETTING_FIELDS},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:10]


def sweep_dir(cfg: Config, run_id: str) -> pathlib.Path:
    return cfg.paths.models / "cv" / f"{cfg.variant}-{run_id}"


# --------------------------------------------------------------------- folding

def build_folds(cfg: Config, pool: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified folds over the pool.

    `diagnosis` reproduces the original design. `diagnosis+resolution` is the
    confound-aware variant: the acquisition shortcut reaches QWK 0.652 on file
    metadata alone, so stratifying jointly means no fold can exploit a
    resolution-to-label mapping that is absent from the folds it is scored
    against. Comparing the two is an estimate of what the shortcut was carrying.
    """
    if cfg.cv.stratify_on == "diagnosis":
        strata = pool["diagnosis"].astype(str)
    elif cfg.cv.stratify_on == "diagnosis+resolution":
        if "resolution_bucket" not in pool.columns:
            raise KeyError(
                "stratify_on='diagnosis+resolution' needs a resolution_bucket column - "
                "call labels.add_resolution_bucket() first"
            )
        strata = pool["diagnosis"].astype(str) + "|" + pool["resolution_bucket"].astype(str)
    else:
        raise ValueError(f"unknown cv.stratify_on: {cfg.cv.stratify_on!r}")

    # A stratum smaller than the fold count cannot be split evenly; scikit-learn
    # warns and degrades silently, so say it out loud instead.
    tiny = strata.value_counts()
    tiny = tiny[tiny < cfg.cv.folds]
    if len(tiny):
        print(f"  note: {len(tiny)} stratum/strata smaller than {cfg.cv.folds} folds: "
              f"{dict(tiny)} - those folds will be uneven")

    skf = StratifiedKFold(n_splits=cfg.cv.folds, shuffle=True, random_state=cfg.train.seed)
    return list(skf.split(pool, strata))


# ------------------------------------------------------------------- one fold

def train_one_fold(cfg: Config, fold: int, tr_df, va_df, te_df, device: str) -> dict:
    train_tf, eval_tf = build_transforms(cfg.train.size, cfg.augment)
    root = cfg.data_dir
    loaders = {
        "train": make_loader(tr_df, train_tf, root, cfg, shuffle=True),
        "valid": make_loader(va_df, eval_tf, root, cfg, shuffle=False),
        "test": make_loader(te_df, eval_tf, root, cfg, shuffle=False),
    }

    model = build_model(cfg, device)
    criterion = build_criterion(cfg, tr_df, device)
    optimizer = build_optimizer(cfg, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    scaler = torch.amp.GradScaler("cuda") if (device == "cuda" and cfg.train.amp) else None

    best_qwk, best_state, best_thr = -1.0, None, None
    best_epoch, no_improve, last_epoch = 0, 0, 0
    history = []

    for epoch in range(1, cfg.train.epochs + 1):
        last_epoch = epoch
        started = time.time()
        tr_loss, _, _ = run_epoch(model, loaders["train"], criterion, device,
                                  cfg.train.mode, optimizer, scaler)
        va_loss, va_raw, va_true = run_epoch(model, loaders["valid"], criterion,
                                             device, cfg.train.mode)
        thr, va_qwk = optimize_thresholds(va_true, va_raw, cfg.thresholds)
        scheduler.step()

        flag = ""
        if va_qwk > best_qwk + cfg.train.min_delta:
            best_qwk, best_thr, best_epoch = va_qwk, thr, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve, flag = 0, "  <- best"
        else:
            no_improve += 1
            if cfg.train.patience:
                flag = f"  ({no_improve}/{cfg.train.patience})"

        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "valid_loss": va_loss, "valid_qwk": va_qwk})
        print(f"  fold {fold} epoch {epoch:>2}/{cfg.train.epochs}  "
              f"train={tr_loss:.4f}  valid={va_loss:.4f}  QWK={va_qwk:.4f}  "
              f"{time.time() - started:.0f}s{flag}", flush=True)

        if cfg.train.patience and no_improve >= cfg.train.patience:
            print(f"  fold {fold} early stop (best epoch {best_epoch})", flush=True)
            break

    model.load_state_dict(best_state)
    _, te_raw, te_true = run_epoch(model, loaders["test"], criterion, device, cfg.train.mode)
    te_pred = apply_thresholds(te_raw, best_thr)

    result = {
        "fold": fold,
        "best_epoch": best_epoch,
        "epochs_run": last_epoch,
        "valid_qwk": float(best_qwk),
        "valid_n": int(len(va_df)),
        "train_n": int(len(tr_df)),
        "thresholds": [float(t) for t in best_thr],
        "history": history,
    }
    result.update({f"test_{k}": v for k, v in grade_metrics(te_true, te_pred).items()})
    result.update({f"test_{k}": v for k, v in referable_confusion(te_true, te_pred).items()})
    result.update(clinically_costly_errors(te_true, te_pred))
    return result, best_state, te_raw, te_true


# ------------------------------------------------------------------ persistence

def fold_paths(directory: pathlib.Path, fold: int) -> dict[str, pathlib.Path]:
    return {
        "metrics": directory / f"fold{fold}.json",
        "preds": directory / f"fold{fold}.npz",
        "checkpoint": directory / f"fold{fold}.pt",
    }


def fold_is_complete(directory: pathlib.Path, fold: int) -> bool:
    """A fold counts as done only when its metrics AND its predictions exist.

    The predictions matter as much as the weights: without `test_raw` the
    ensemble cannot be rebuilt, which is exactly how the earlier design turned
    a partial failure into a total one.
    """
    paths = fold_paths(directory, fold)
    return paths["metrics"].exists() and paths["preds"].exists()


def save_fold(directory: pathlib.Path, fold: int, result: dict, state, te_raw, te_true,
              *, save_weights: bool = True) -> None:
    paths = fold_paths(directory, fold)
    np.savez_compressed(paths["preds"], test_raw=te_raw, test_true=te_true)
    paths["metrics"].write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    if save_weights:
        torch.save({"state_dict": state, "thresholds": result["thresholds"],
                    "fold": fold}, paths["checkpoint"])


def load_fold(directory: pathlib.Path, fold: int) -> tuple[dict, np.ndarray, np.ndarray]:
    paths = fold_paths(directory, fold)
    result = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    arrays = np.load(paths["preds"])
    return result, arrays["test_raw"], arrays["test_true"]


# ----------------------------------------------------------------------- sweep

def run_cv(cfg: Config, *, resume: bool = True, track: bool = True,
           limit: int | None = None) -> dict:
    """Run (or resume) the full sweep for one variant."""
    device = pick_device()
    print(f"device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    # Refuse a cache that is not what the config asked for. This is the check the
    # original printed and then ignored.
    mf = manifest_mod.check(cfg.data_dir, cfg.manifest_fields())
    print(f"data: {cfg.data_dir.name}  ({manifest_mod.describe(mf)})")

    df = labels_mod.load_labels(cfg)
    if cfg.train.exclude_leaked:
        df, dropped = labels_mod.exclude_leaked(df, cfg)
        print(f"leak cleanup: {dropped} training images excluded")
    if cfg.cv.stratify_on == "diagnosis+resolution":
        df = labels_mod.add_resolution_bucket(df, cfg)

    pool, holdout = labels_mod.pool_and_holdout(df, cfg)
    if limit:
        pool = pool.groupby("diagnosis", group_keys=False).head(max(2, limit // 5))
        holdout = holdout.head(limit)
        print(f"--limit: pool cut to {len(pool)}, test to {len(holdout)} (SMOKE TEST)")

    print(f"pool: {len(pool)} images, {cfg.cv.folds} folds, "
          f"stratified on {cfg.cv.stratify_on}")
    print(f"test: {len(holdout)} held out entirely")

    run_id = run_id_for(cfg, mf)
    directory = sweep_dir(cfg, run_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    print(f"sweep: {directory.name}\n")

    set_seed(cfg.train.seed)
    splits = build_folds(cfg, pool)

    results, raws, trues = [], [], []
    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        if resume and fold_is_complete(directory, fold):
            result, te_raw, te_true = load_fold(directory, fold)
            print(f"--- fold {fold}/{cfg.cv.folds}  already done "
                  f"(valid QWK={result['valid_qwk']:.4f}, "
                  f"test QWK={result['test_qwk']:.4f}) - skipping")
            results.append(result)
            raws.append(te_raw)
            trues.append(te_true)
            continue

        tr_df, va_df = pool.iloc[tr_idx], pool.iloc[va_idx]
        print(f"--- fold {fold}/{cfg.cv.folds}  train={len(tr_df)}  valid={len(va_df)}",
              flush=True)

        # Re-seed per fold so a resumed sweep produces the same fold as a
        # continuous one would have.
        set_seed(cfg.train.seed + fold)
        result, state, te_raw, te_true = train_one_fold(
            cfg, fold, tr_df, va_df, holdout, device
        )
        save_fold(directory, fold, result, state, te_raw, te_true,
                  save_weights=limit is None)
        print(f"  fold {fold} done: valid QWK={result['valid_qwk']:.4f}  "
              f"test QWK={result['test_qwk']:.4f}\n", flush=True)

        results.append(result)
        raws.append(te_raw)
        trues.append(te_true)

        if track and limit is None:
            _log_fold(cfg, run_id, result)

    summary = summarise(cfg, run_id, results, raws, trues)
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    print_summary(cfg, summary, results)

    if track and limit is None:
        _log_summary(cfg, run_id, summary)
    return summary


def summarise(cfg: Config, run_id: str, results: list[dict],
              raws: list[np.ndarray], trues: list[np.ndarray]) -> dict:
    """Fold aggregates plus the ensemble, rebuilt from the persisted arrays."""
    vq = np.array([r["valid_qwk"] for r in results], dtype=float)
    tq = np.array([r["test_qwk"] for r in results], dtype=float)
    ta = np.array([r["test_accuracy"] for r in results], dtype=float)
    tf = np.array([r["test_macro_f1"] for r in results], dtype=float)

    summary = {
        "run_id": run_id,
        "variant": cfg.variant,
        "folds": len(results),
        "stratify_on": cfg.cv.stratify_on,
        "valid_qwk_mean": float(vq.mean()), "valid_qwk_std": _std(vq),
        "test_qwk_mean": float(tq.mean()), "test_qwk_std": _std(tq),
        "test_acc_mean": float(ta.mean()), "test_acc_std": _std(ta),
        "test_macro_f1_mean": float(tf.mean()), "test_macro_f1_std": _std(tf),
    }

    if raws:
        te_true = trues[0]
        for other in trues[1:]:
            if not np.array_equal(te_true, other):
                raise RuntimeError(
                    "folds disagree on the test labels - the held-out set changed "
                    "between folds, so the ensemble would be meaningless"
                )
        ens_raw = np.mean(raws, axis=0)
        ens_thr = np.mean([r["thresholds"] for r in results], axis=0)
        ens_pred = apply_thresholds(ens_raw, ens_thr)
        summary["ensemble"] = {
            "thresholds": [float(t) for t in ens_thr],
            **{k: v for k, v in grade_metrics(te_true, ens_pred).items()},
            **{f"referable_{k}": v for k, v in referable_confusion(te_true, ens_pred).items()},
            **clinically_costly_errors(te_true, ens_pred),
            "gain_over_fold_mean": float(qwk(te_true, ens_pred) - tq.mean()),
        }
    return summary


def _std(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if values.size > 1 else 0.0


def print_summary(cfg: Config, summary: dict, results: list[dict]) -> None:
    line = "=" * 68
    print(line)
    print(f"{summary['folds']}-FOLD CROSS-VALIDATION - {cfg.variant} "
          f"(stratified on {summary['stratify_on']})")
    print(line)
    print(f"{'fold':>5}{'valid QWK':>12}{'test QWK':>11}{'test acc':>11}"
          f"{'macro F1':>11}{'best ep':>9}")
    for r in results:
        print(f"{r['fold']:>5}{r['valid_qwk']:>12.4f}{r['test_qwk']:>11.4f}"
              f"{r['test_accuracy']:>11.4f}{r['test_macro_f1']:>11.4f}{r['best_epoch']:>9}")
    print(f"\n  CV valid QWK : {summary['valid_qwk_mean']:.4f} +/- {summary['valid_qwk_std']:.4f}")
    print(f"  test QWK     : {summary['test_qwk_mean']:.4f} +/- {summary['test_qwk_std']:.4f}")
    print(f"  test acc     : {summary['test_acc_mean']:.4f} +/- {summary['test_acc_std']:.4f}")
    print(f"  test macro F1: {summary['test_macro_f1_mean']:.4f} +/- {summary['test_macro_f1_std']:.4f}")
    if "ensemble" in summary:
        ens = summary["ensemble"]
        print(f"\n  ENSEMBLE test QWK: {ens['qwk']:.4f}  acc: {ens['accuracy']:.4f}  "
              f"macro F1: {ens['macro_f1']:.4f}")
        print(f"  ({ens['gain_over_fold_mean']:+.4f} against the single-fold mean)")
        print(f"  referable sensitivity {ens['referable_sensitivity']:.3f}  "
              f"specificity {ens['referable_specificity']:.3f}")
        print(f"  severe cases missed: {ens['severe_missed']}/{ens['severe_total']}")
    print(line)


# -------------------------------------------------------------------- tracking

def _log_fold(cfg: Config, run_id: str, result: dict) -> None:
    from aptos import tracking

    name = f"cv-{cfg.variant}-{run_id}-fold{result['fold']}"
    with tracking.run(cfg, name, tags={"split": "cv-fold", "fold": str(result["fold"]),
                                       "sweep": run_id, "stratify": cfg.cv.stratify_on}):
        tracking.log_metrics({k: v for k, v in result.items()
                              if isinstance(v, (int, float))})
        for step in result.get("history", []):
            tracking.log_metrics(
                {f"epoch_{k}": v for k, v in step.items() if k != "epoch"},
                step=step["epoch"],
            )
        tracking.log_dict({"thresholds": result["thresholds"]}, "thresholds.json")


def _log_summary(cfg: Config, run_id: str, summary: dict) -> None:
    from aptos import tracking

    name = f"cv-{cfg.variant}-{run_id}-summary"
    with tracking.run(cfg, name, tags={"split": "cv-summary", "sweep": run_id,
                                       "stratify": cfg.cv.stratify_on}):
        flat = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
        for k, v in summary.get("ensemble", {}).items():
            if isinstance(v, (int, float)):
                flat[f"ensemble_{k}"] = v
        tracking.log_metrics(flat)
        tracking.log_dict(summary, "summary.json")


# ------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/cv.yaml")
    parser.add_argument("--variant", default=None, choices=["baseline", "clahe", "squash"])
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--stratify", default=None,
                        choices=["diagnosis", "diagnosis+resolution"])
    parser.add_argument("--no-resume", action="store_true",
                        help="retrain every fold even if results exist")
    parser.add_argument("--no-track", action="store_true", help="skip MLflow logging")
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke test on a handful of images; never tracked, "
                             "never saves weights")
    args = parser.parse_args(argv)

    overrides: dict = {}
    if args.variant:
        overrides["variant"] = args.variant
    train_over = {k: v for k, v in
                  [("seed", args.seed), ("epochs", args.epochs)] if v is not None}
    if train_over:
        overrides["train"] = train_over
    cv_over = {k: v for k, v in
               [("folds", args.folds), ("stratify_on", args.stratify)] if v is not None}
    if cv_over:
        overrides["cv"] = cv_over

    cfg = Config.load(args.config, **overrides)
    # The variant selects the cache, so its preprocessing settings have to come
    # with it; otherwise the manifest check would compare against base defaults.
    variant_cfg = Config.load(f"configs/{cfg.variant}.yaml")
    cfg.preprocess = dataclasses.replace(variant_cfg.preprocess)

    run_cv(cfg, resume=not args.no_resume, track=not args.no_track,
           limit=args.limit or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
