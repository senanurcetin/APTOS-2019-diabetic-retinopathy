"""K-fold cross-validated training.

Why this exists: single-split measurements turned out to be fragile. Across
three seeds the CLAHE difference swung between +0.015 and -0.020, and the
validation set held only 366 images. Cross-validation uses the whole
train+valid pool for evaluation, which markedly reduces the variance of the
estimate.

Design:
  - the test set is held out completely
  - the pool is split into K folds stratified on diagnosis
  - within each fold, thresholds and the best epoch are chosen on that fold's
    own validation slice
  - the mean of the folds' raw outputs is also scored as an ensemble
  - per-fold results go to BigQuery with a `fold` column (0 = ensemble)

Reuses the functions in train.py rather than copying them.

Usage:
    python scripts/train_cv.py --folds 5 --variant baseline
    python scripts/train_cv.py --folds 5 --data-dir data/processed_clahe --variant clahe
"""
import argparse
import datetime as dt
import json
import pathlib
import sys
import uuid

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train import (DEFAULT_PROCESSED, MODELS, build_transforms,  # noqa: E402
                   load_labels, optimize_thresholds, qwk, run_epoch,
                   set_seed, write_to_bigquery)

ROOT = pathlib.Path(__file__).resolve().parent.parent


class FoldDataset(Dataset):
    """Read each image from its ORIGINAL split directory.

    Under cross-validation an image's role (training or validation) changes
    from fold to fold, but its location on disk does not. So the directory
    comes from the orig_split column, not from the fold assignment.
    """

    def __init__(self, df, transform, root):
        self.df = df.reset_index(drop=True)
        self.root = pathlib.Path(root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.root / row["orig_split"] / f"{row['id_code']}.jpg"
        return self.transform(Image.open(path).convert("RGB")), \
            torch.tensor(int(row["diagnosis"]))


def make_loader(df, transform, root, batch, workers, shuffle):
    return DataLoader(FoldDataset(df, transform, root), batch_size=batch,
                      shuffle=shuffle, num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0)


def train_one_fold(fold, tr_df, va_df, te_df, args, data_root, device):
    """Train and evaluate a single fold."""
    train_tf, eval_tf = build_transforms(args.size)
    loaders = {
        "train": make_loader(tr_df, train_tf, data_root, args.batch, args.workers, True),
        "valid": make_loader(va_df, eval_tf, data_root, args.batch, args.workers, False),
        "test": make_loader(te_df, eval_tf, data_root, args.batch, args.workers, False),
    }

    model = timm.create_model(args.model, pretrained=True, num_classes=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    best_qwk, best_state, best_thr = -1.0, None, None
    best_epoch, no_improve, last_epoch = 0, 0, 0

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        tr_loss, _, _ = run_epoch(model, loaders["train"], criterion, device,
                                  "reg", optimizer, scaler)
        va_loss, va_raw, va_true = run_epoch(model, loaders["valid"], criterion,
                                             device, "reg")
        thr, va_qwk = optimize_thresholds(va_true, va_raw)
        scheduler.step()

        flag = ""
        if va_qwk > best_qwk:
            best_qwk, best_thr, best_epoch = va_qwk, thr, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve, flag = 0, "  <- best"
        else:
            no_improve += 1
            if args.patience:
                flag = f"  ({no_improve}/{args.patience})"

        print(f"  fold {fold} epoch {epoch:>2}/{args.epochs}  train={tr_loss:.4f}  "
              f"valid={va_loss:.4f}  QWK={va_qwk:.4f}{flag}")

        if args.patience and no_improve >= args.patience:
            print(f"  fold {fold} early stop (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    _, te_raw, te_true = run_epoch(model, loaders["test"], criterion, device, "reg")
    te_pred = np.digitize(te_raw, best_thr)

    return {
        "fold": fold,
        "best_epoch": best_epoch,
        "epochs": last_epoch,
        "valid_qwk": float(best_qwk),
        "valid_n": len(va_df),
        "test_qwk": float(qwk(te_true, te_pred)),
        "test_acc": float((te_true == te_pred).mean()),
        "test_f1": float(f1_score(te_true, te_pred, average="macro", zero_division=0)),
        "test_raw": te_raw,
        "test_true": te_true,
        "thresholds": best_thr,
        "state": best_state,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--exclude-leaked", action="store_true")
    ap.add_argument("--author", default="senanur")
    ap.add_argument("--no-bq", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    data_root = pathlib.Path(args.data_dir) if args.data_dir else DEFAULT_PROCESSED
    if not data_root.exists():
        raise SystemExit(f"{data_root} not found - run preprocess_images.py first")
    variant = args.variant or ("clahe" if "clahe" in data_root.name else "baseline")

    manifest = data_root / "_manifest.json"
    if manifest.exists():
        mf = json.loads(manifest.read_text(encoding="utf-8"))
        print(f"data: {data_root}  ({mf['size']}px, CLAHE={mf['clahe']}, "
              f"square={mf.get('square_mode', '?')})")
    else:
        print(f"data: {data_root}  (WARNING: no _manifest.json)")

    df = load_labels()
    df["orig_split"] = df["split"]

    if args.exclude_leaked:
        leak_file = ROOT / "reports" / "leaked_train_ids.csv"
        if not leak_file.exists():
            raise SystemExit(f"{leak_file} not found - run quality_report.py first")
        leaked = set(pd.read_csv(leak_file).id_code)
        before = len(df)
        df = df[~((df.split == "train") & (df.id_code.isin(leaked)))].reset_index(drop=True)
        print(f"leak cleanup: {before - len(df)} training images excluded")

    # The test set stays out entirely; the pool is train + valid.
    te_df = df[df.split == "test"].reset_index(drop=True)
    pool = df[df.split != "test"].reset_index(drop=True)
    print(f"pool: {len(pool)} images ({args.folds} folds), test: {len(te_df)} (untouched)")

    run_id = uuid.uuid4().hex[:8]
    print(f"\nrun_id={run_id}  variant={variant}  {args.folds} folds  seed={args.seed}\n")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    MODELS.mkdir(exist_ok=True)
    results = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(pool, pool.diagnosis), start=1):
        tr_df, va_df = pool.iloc[tr_idx], pool.iloc[va_idx]
        print(f"--- fold {fold}/{args.folds}  train={len(tr_df)}  valid={len(va_df)} ---")
        r = train_one_fold(fold, tr_df, va_df, te_df, args, data_root, device)
        torch.save({"state_dict": r.pop("state"), "thresholds": r["thresholds"],
                    "fold": fold, "args": vars(args)},
                   MODELS / f"{run_id}_fold{fold}.pt")
        print(f"  fold {fold} done: valid QWK={r['valid_qwk']:.4f}  "
              f"test QWK={r['test_qwk']:.4f}\n")
        results.append(r)

    # ------------------------------------------------------------------ summary
    vq = np.array([r["valid_qwk"] for r in results])
    tq = np.array([r["test_qwk"] for r in results])
    ta = np.array([r["test_acc"] for r in results])

    print("=" * 62)
    print(f"{args.folds}-FOLD CROSS-VALIDATION - {variant}")
    print("=" * 62)
    print(f"{'fold':>5}{'valid QWK':>12}{'test QWK':>11}{'test acc':>10}{'best ep':>10}")
    for r in results:
        print(f"{r['fold']:>5}{r['valid_qwk']:>12.4f}{r['test_qwk']:>11.4f}"
              f"{r['test_acc']:>10.4f}{r['best_epoch']:>10}")
    print(f"\n  CV valid QWK : {vq.mean():.4f} +/- {vq.std(ddof=1):.4f}")
    print(f"  test QWK     : {tq.mean():.4f} +/- {tq.std(ddof=1):.4f}")
    print(f"  test acc     : {ta.mean():.4f} +/- {ta.std(ddof=1):.4f}")

    # Ensemble: average the folds' raw regression outputs.
    ens_raw = np.mean([r["test_raw"] for r in results], axis=0)
    ens_thr = np.mean([r["thresholds"] for r in results], axis=0)
    te_true = results[0]["test_true"]
    ens_pred = np.digitize(ens_raw, ens_thr)
    ens_qwk = float(qwk(te_true, ens_pred))
    ens_acc = float((te_true == ens_pred).mean())
    print(f"\n  ENSEMBLE test QWK: {ens_qwk:.4f}   acc: {ens_acc:.4f}")
    print(f"  ({ens_qwk - tq.mean():+.4f} against the single-fold mean)")

    if args.no_bq:
        return

    created = dt.datetime.now(dt.timezone.utc)
    rows = []
    for r in results:
        for split, q_, a_, f_, n_ in (
                ("valid", r["valid_qwk"], None, None, r["valid_n"]),
                ("test", r["test_qwk"], r["test_acc"], r["test_f1"], len(te_df))):
            rows.append({
                "run_id": run_id, "author": args.author, "model": args.model,
                "mode": "reg", "split": split, "qwk": q_,
                "accuracy": a_, "macro_f1": f_, "n": n_,
                "epochs": r["epochs"], "best_epoch": r["best_epoch"],
                "img_size": args.size, "seed": args.seed, "variant": variant,
                "leak_excluded": bool(args.exclude_leaked),
                "fold": r["fold"], "created_at": created,
            })
    rows.append({
        "run_id": run_id, "author": args.author, "model": args.model,
        "mode": "reg", "split": "test", "qwk": ens_qwk, "accuracy": ens_acc,
        "macro_f1": float(f1_score(te_true, ens_pred, average="macro", zero_division=0)),
        "n": len(te_df), "epochs": 0, "best_epoch": 0, "img_size": args.size,
        "seed": args.seed, "variant": f"{variant}-ensemble",
        "leak_excluded": bool(args.exclude_leaked), "fold": 0, "created_at": created,
    })

    preds = pd.DataFrame({
        "run_id": run_id, "author": args.author, "model": args.model, "mode": "reg",
        "split": "test", "id_code": te_df.id_code,
        "y_true": te_true.astype(int), "y_pred": ens_pred.astype(int),
        "raw_score": ens_raw.astype(float), "created_at": created,
    })

    print("\nwriting to BigQuery...")
    try:
        write_to_bigquery(run_id, preds, pd.DataFrame(rows))
    except Exception as e:
        print(f"  write failed: {type(e).__name__}: {str(e)[:160]}")


if __name__ == "__main__":
    main()
