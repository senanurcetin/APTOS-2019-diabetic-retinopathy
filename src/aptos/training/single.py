"""Single-split training: fit on train, select on valid, report on test.

Ported from scripts/train.py, which now forwards here. The behaviour that
produced the six recorded single-split runs is kept - same model, loss,
optimiser, schedule, threshold search and early stopping - so a rerun is
comparable with them. What changed:

  * results go to MLflow instead of a BigQuery project that no longer exists;
  * the cache is checked against its manifest instead of the manifest being
    printed and ignored;
  * class weights are built over all five grades (the original crashed the loss
    when a grade was missing from the split);
  * `--limit` smoke runs are never tracked and never leave weights behind.

The command line is the old one, flag for flag, because scripts/run_seeds.sh
and the pipeline's `train` stage drive it.

    python -m aptos.training.single --data-dir data/processed --seed 42
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib

import numpy as np
import torch

from aptos.config import PROCESSED_DIRS, Config
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


def variant_for(data_dir: pathlib.Path) -> str | None:
    """Which variant a cache directory belongs to, if it is one of the known ones."""
    by_dir = {name: variant for variant, name in PROCESSED_DIRS.items()}
    return by_dir.get(data_dir.name)


def build_config(args: argparse.Namespace) -> tuple[Config, pathlib.Path]:
    """The config a run should use, from the old flags.

    The cache directory decides the variant, and the variant decides the
    preprocessing settings the manifest is checked against. `--variant` is a
    label only, exactly as in the original script.
    """
    base = Config.load(args.config) if args.config else Config.load("configs/base.yaml")
    data_dir = pathlib.Path(args.data_dir) if args.data_dir else base.data_dir
    if not data_dir.is_absolute():
        data_dir = base.paths.root / data_dir

    variant = variant_for(data_dir) or base.variant
    preprocess = Config.load(f"configs/{variant}.yaml").preprocess

    train = dataclasses.replace(
        base.train,
        mode=args.mode, model=args.model, size=args.size, batch=args.batch,
        epochs=args.epochs, lr=args.lr, workers=args.workers,
        patience=args.patience, min_delta=args.min_delta, seed=args.seed,
        exclude_leaked=args.exclude_leaked,
    )
    cfg = dataclasses.replace(base, variant=variant, preprocess=preprocess, train=train)
    cfg.validate()
    return cfg, data_dir


def run_name(cfg: Config, label: str) -> str:
    """Stable, readable, and different whenever the experiment differs."""
    payload = json.dumps(cfg.to_dict(), sort_keys=True, default=str).encode()
    return f"single-{label}-seed{cfg.train.seed}-{hashlib.sha256(payload).hexdigest()[:8]}"


def train(cfg: Config, data_dir: pathlib.Path, *, label: str, limit: int = 0,
          track: bool = True) -> dict:
    device = pick_device()
    print(f"device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU (slow)'}")

    # A cache that is not what the variant says it is stops the run here.
    known = variant_for(data_dir) is not None
    mf = manifest_mod.check(data_dir, cfg.manifest_fields(), strict=known)
    print(f"data: {data_dir.name}  ({manifest_mod.describe(mf)})  label={label}")
    if not known:
        print("  note: not a known variant directory; manifest printed, not enforced")

    set_seed(cfg.train.seed)
    df = labels_mod.load_labels(cfg)
    if cfg.train.exclude_leaked:
        df, dropped = labels_mod.exclude_leaked(df, cfg)
        print(f"leak cleanup: {dropped} training images excluded")
    df["orig_split"] = df["split"]

    if limit:
        df = (df.sample(frac=1, random_state=cfg.train.seed)
                .groupby("split", group_keys=False).head(limit).reset_index(drop=True))
        print(f"--limit {limit}: SMOKE TEST - not tracked, no weights saved")

    train_tf, eval_tf = build_transforms(cfg.train.size, cfg.augment)
    frames = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test")}
    loaders = {
        "train": make_loader(frames["train"], train_tf, data_dir, cfg, shuffle=True),
        "valid": make_loader(frames["valid"], eval_tf, data_dir, cfg, shuffle=False),
        "test": make_loader(frames["test"], eval_tf, data_dir, cfg, shuffle=False),
    }
    for split, frame in frames.items():
        print(f"  {split}: {len(frame)} images")

    model = build_model(cfg, device)
    criterion = build_criterion(cfg, frames["train"], device)
    optimizer = build_optimizer(cfg, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    scaler = torch.amp.GradScaler("cuda") if (device == "cuda" and cfg.train.amp) else None
    mode = cfg.train.mode

    best_qwk, best_state, best_thr = -1.0, None, None
    best_epoch, no_improve, last_epoch = 0, 0, 0
    history = []

    for epoch in range(1, cfg.train.epochs + 1):
        last_epoch = epoch
        tr_loss, _, _ = run_epoch(model, loaders["train"], criterion, device, mode,
                                  optimizer, scaler)
        va_loss, va_raw, va_true = run_epoch(model, loaders["valid"], criterion, device, mode)
        if mode == "reg":
            thr, va_qwk = optimize_thresholds(va_true, va_raw, cfg.thresholds)
        else:
            thr, va_qwk = None, qwk(va_true, va_raw.astype(int))
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
        print(f"epoch {epoch:>2}/{cfg.train.epochs}  train={tr_loss:.4f}  "
              f"valid={va_loss:.4f}  QWK={va_qwk:.4f}{flag}", flush=True)

        if cfg.train.patience and no_improve >= cfg.train.patience:
            print(f"early stop (best epoch {best_epoch}, QWK {best_qwk:.4f})")
            break

    model.load_state_dict(best_state)

    result = {"label": label, "variant": cfg.variant, "seed": cfg.train.seed,
              "best_epoch": best_epoch, "epochs_run": last_epoch,
              "exclude_leaked": cfg.train.exclude_leaked,
              "thresholds": None if best_thr is None else [float(t) for t in best_thr]}
    predictions = {}
    for split in ("valid", "test"):
        _, raw, true = run_epoch(model, loaders[split], criterion, device, mode)
        pred = apply_thresholds(raw, best_thr) if mode == "reg" else raw.astype(int)
        predictions[split] = (raw, true)
        for k, v in grade_metrics(true, pred).items():
            result[f"{split}_{k}"] = v
        for k, v in referable_confusion(true, pred).items():
            result[f"{split}_referable_{k}"] = v
        result.update({f"{split}_{k}": v for k, v in clinically_costly_errors(true, pred).items()})
        print(f"{split.upper():5s} QWK={result[f'{split}_qwk']:.4f}  "
              f"acc={result[f'{split}_accuracy']:.4f}  "
              f"macro_F1={result[f'{split}_macro_f1']:.4f}")

    if limit:
        return result

    name = run_name(cfg, label)
    out = save_run(cfg.paths.models / "single" / name, cfg, best_state, result,
                   predictions, history)
    print(f"saved to {out}")

    if track:
        from aptos import tracking

        with tracking.run(cfg, name, tags={"split": "single", "source": "observed",
                                           "label": label}):
            tracking.log_metrics({k: v for k, v in result.items()
                                  if isinstance(v, (int, float)) and not isinstance(v, bool)})
            for step in history:
                tracking.log_metrics({f"epoch_{k}": v for k, v in step.items() if k != "epoch"},
                                     step=step["epoch"])
    return result


def save_run(out: pathlib.Path, cfg: Config, state: dict, result: dict,
             predictions: dict, history: list) -> pathlib.Path:
    """Write weights, per-image predictions and metrics for one run.

    Separate from `train` so it can be tested without a GPU: a smoke run never
    reaches this code, so without a test the first real run would be the first
    time it executed.
    """
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "thresholds": result["thresholds"],
                "config": cfg.to_dict()}, out / "model.pt")
    np.savez_compressed(out / "predictions.npz",
                        **{f"{split}_{kind}": array
                           for split, (raw, true) in predictions.items()
                           for kind, array in (("raw", raw), ("true", true))})
    (out / "metrics.json").write_text(json.dumps({**result, "history": history},
                                                 indent=2, default=float), encoding="utf-8")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # The original script's flags, with its defaults, so existing callers work.
    ap.add_argument("--mode", choices=["cls", "reg"], default="reg")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--exclude-leaked", action="store_true")
    ap.add_argument("--variant", default=None, help="run label, as in the original")
    ap.add_argument("--author", default=None, help="accepted for compatibility; unused")
    ap.add_argument("--no-bq", action="store_true",
                    help="accepted for compatibility; results never go to BigQuery now")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-track", action="store_true", help="skip MLflow")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg, data_dir = build_config(args)
    train(cfg, data_dir, label=args.variant or cfg.variant, limit=args.limit,
          track=not args.no_track)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
