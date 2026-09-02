"""Diabetic retinopathy grading on APTOS-2019.

Labels come from BigQuery, images from the 512px JPEGs under data/processed*
(run scripts/preprocess_images.py first).

Two modes:
  cls  five-way softmax with class-weighted cross-entropy
  reg  single-output regression plus thresholds tuned on validation

Why QWK: 49% of the dataset is "No DR". A model that learns nothing and always
predicts 0 scores 49% accuracy but 0 on QWK. It was also the competition's
official metric.

Usage:
    python scripts/train.py --mode reg --epochs 30 --patience 5
    python scripts/train.py --data-dir data/processed_clahe --variant clahe
    python scripts/train.py --mode cls --model resnet50 --no-bq
"""
import argparse
import datetime as dt
import json
import os
import pathlib
import random
import uuid

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

# Cloud identifiers are read from the environment so the project runs against
# any BigQuery project and bucket. The defaults are the ones this work used.
PROJECT_ID = os.environ.get("APTOS_GCP_PROJECT", "datascientis")
BQ_DATASET = os.environ.get("APTOS_BQ_DATASET", "APTOS_2019")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    """Make a run reproducible: same seed, same result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------------ data

def load_labels():
    """Read labels from BigQuery, falling back to the local CSV."""
    try:
        from google.cloud import bigquery
        q = f"""SELECT id_code, diagnosis, split
                FROM `{PROJECT_ID}.{BQ_DATASET}.aptos_labels`"""
        df = bigquery.Client(project=PROJECT_ID).query(q).to_dataframe()
        print(f"labels read from BigQuery: {len(df)} rows")
    except Exception as e:
        print(f"BigQuery unavailable ({type(e).__name__}), using the local CSV")
        df = pd.read_csv(ROOT / "data" / "bq" / "aptos_labels.csv")
    return df


class APTOSDataset(Dataset):
    def __init__(self, df, split, transform, root=DEFAULT_PROCESSED):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.dir = pathlib.Path(root) / split
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.dir / f"{row['id_code']}.jpg").convert("RGB")
        return self.transform(img), torch.tensor(int(row["diagnosis"]))


def build_transforms(size):
    """Training augmentations and the plain evaluation transform.

    Vertical flip is included because a fundus photograph has no meaningful
    up/down orientation. Rotation and scale jitter cover variation in how the
    camera was aimed. Colour jitter is kept mild: large shifts would fight
    CLAHE, which normalises local contrast on purpose.
    """
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.85, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    evaluate = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, evaluate


# --------------------------------------------------------------------- metrics

def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def optimize_thresholds(y_true, raw):
    """Search the four cut points that map a regression output onto 0-4."""
    best = np.array([0.5, 1.5, 2.5, 3.5])
    best_score = qwk(y_true, np.digitize(raw, best))
    for _ in range(60):
        improved = False
        for i in range(4):
            for delta in (-0.12, -0.04, 0.04, 0.12):
                cand = best.copy()
                cand[i] += delta
                if not np.all(np.diff(cand) > 0.05):
                    continue
                score = qwk(y_true, np.digitize(raw, cand))
                if score > best_score:
                    best, best_score, improved = cand, score, True
        if not improved:
            break
    return best, best_score


# -------------------------------------------------------------------- training

def run_epoch(model, loader, criterion, device, mode, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, trues = 0.0, [], []

    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)
            target = labels.float().unsqueeze(1) if mode == "reg" else labels

            with torch.autocast("cuda", enabled=scaler is not None):
                out = model(imgs)
                loss = criterion(out, target)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            raw = out.squeeze(1) if mode == "reg" else out.argmax(1)
            preds.append(raw.float().detach().cpu().numpy())
            trues.append(labels.cpu().numpy())

    return total_loss / len(loader.dataset), np.concatenate(preds), np.concatenate(trues)


# -------------------------------------------------------------------- BigQuery

def write_to_bigquery(run, preds_df, metrics_df):
    from google.cloud import bigquery
    client = bigquery.Client(project=PROJECT_ID)
    cfg = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")

    for name, df in [("aptos_predictions", preds_df), ("aptos_metrics", metrics_df)]:
        table_id = f"{PROJECT_ID}.{BQ_DATASET}.{name}"
        client.load_table_from_dataframe(df, table_id, job_config=cfg).result()
        print(f"  {table_id} <- {len(df)} rows  (run_id={run})")


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cls", "reg"], default="reg")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=5,
                    help="stop if validation QWK does not improve for N epochs; 0 disables")
    ap.add_argument("--min-delta", type=float, default=0.0,
                    help="smallest QWK gain that counts as an improvement")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap images per split - for quick smoke tests only")
    ap.add_argument("--data-dir", default=None,
                    help="processed image directory; defaults to data/processed")
    ap.add_argument("--exclude-leaked", action="store_true",
                    help="drop training images that have a copy in valid or test "
                         "(reports/leaked_train_ids.csv)")
    ap.add_argument("--variant", default=None,
                    help="run label, e.g. clahe / baseline")
    ap.add_argument("--author", default="senanur")
    ap.add_argument("--no-bq", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU found, running on CPU (very slow)")
    else:
        print(f"device: {torch.cuda.get_device_name(0)}")

    data_root = pathlib.Path(args.data_dir) if args.data_dir else DEFAULT_PROCESSED
    if not data_root.exists():
        raise SystemExit(f"{data_root} not found - run scripts/preprocess_images.py first")
    variant = args.variant or ("clahe" if "clahe" in data_root.name else "baseline")

    manifest = data_root / "_manifest.json"
    print(f"data: {data_root}  (variant={variant})")
    if manifest.exists():
        mf = json.loads(manifest.read_text(encoding="utf-8"))
        print(f"  preprocessing: {mf['size']}px, CLAHE={mf['clahe']}, "
              f"square={mf.get('square_mode', '?')}, {mf['n_images']} images")
    else:
        print("  WARNING: no _manifest.json - the preprocessing used to build this "
              "directory is unknown. Re-run preprocess_images.py.")

    df = load_labels()

    # Cross-split duplicates make the test result optimistic. We drop the
    # training side so the evaluation sets stay intact and runs remain
    # comparable with each other.
    if args.exclude_leaked:
        leak_file = ROOT / "reports" / "leaked_train_ids.csv"
        if not leak_file.exists():
            raise SystemExit(f"{leak_file} not found - run scripts/quality_report.py first")
        leaked = set(pd.read_csv(leak_file).id_code)
        before = len(df)
        df = df[~((df.split == "train") & (df.id_code.isin(leaked)))].reset_index(drop=True)
        print(f"leak cleanup: {before - len(df)} training images excluded")

    if args.limit:
        df = (df.sample(frac=1, random_state=args.seed)
                .groupby("split", group_keys=False).head(args.limit).reset_index(drop=True))
        print(f"WARNING: --limit {args.limit} is on; this is a smoke test, not a result")

    train_tf, eval_tf = build_transforms(args.size)
    loaders = {}
    for split, tf, shuffle in [("train", train_tf, True),
                               ("valid", eval_tf, False),
                               ("test", eval_tf, False)]:
        ds = APTOSDataset(df, split, tf, root=data_root)
        loaders[split] = DataLoader(ds, batch_size=args.batch, shuffle=shuffle,
                                    num_workers=args.workers, pin_memory=True,
                                    persistent_workers=args.workers > 0)
        print(f"  {split}: {len(ds)} images")

    n_out = 1 if args.mode == "reg" else 5
    model = timm.create_model(args.model, pretrained=True, num_classes=n_out).to(device)

    if args.mode == "reg":
        criterion = nn.MSELoss()
    else:
        counts = df[df.split == "train"].diagnosis.value_counts().sort_index().values
        weights = torch.tensor(counts.sum() / (5 * counts), dtype=torch.float32).to(device)
        print("class weights:", np.round(weights.cpu().numpy(), 2))
        criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    MODELS.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    best_qwk, best_state, best_thr = -1.0, None, None
    best_epoch, no_improve, last_epoch = 0, 0, 0

    print(f"\nrun_id={run_id}  mode={args.mode}  model={args.model}  {args.size}px"
          f"  early_stopping={args.patience or 'off'}\n")

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        tr_loss, _, _ = run_epoch(model, loaders["train"], criterion, device,
                                  args.mode, optimizer, scaler)
        va_loss, va_raw, va_true = run_epoch(model, loaders["valid"], criterion,
                                             device, args.mode)

        if args.mode == "reg":
            thr, va_qwk = optimize_thresholds(va_true, va_raw)
        else:
            thr, va_qwk = None, qwk(va_true, va_raw.astype(int))

        scheduler.step()
        flag = ""
        if va_qwk > best_qwk + args.min_delta:
            best_qwk, best_thr, best_epoch = va_qwk, thr, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve, flag = 0, "  <- best"
        else:
            no_improve += 1
            if args.patience:
                flag = f"  ({no_improve}/{args.patience})"

        print(f"epoch {epoch:>2}/{args.epochs}  train_loss={tr_loss:.4f}  "
              f"valid_loss={va_loss:.4f}  valid_QWK={va_qwk:.4f}{flag}")

        # Once validation QWK stops improving the remaining epochs go into
        # memorisation, and the best checkpoint is already saved.
        if args.patience and no_improve >= args.patience:
            print(f"\nearly stop: validation QWK flat for {args.patience} epochs "
                  f"(best epoch {best_epoch}, QWK {best_qwk:.4f})")
            break

    model.load_state_dict(best_state)
    ckpt = MODELS / f"{run_id}_{args.model}_{args.mode}.pt"
    torch.save({"state_dict": best_state, "thresholds": best_thr, "args": vars(args)}, ckpt)
    print(f"\nbest valid QWK: {best_qwk:.4f} (epoch {best_epoch}/{last_epoch})  -> {ckpt.name}")

    # ---- final evaluation on valid and test
    created = dt.datetime.now(dt.timezone.utc)
    pred_rows, metric_rows = [], []

    for split in ("valid", "test"):
        _, raw, true = run_epoch(model, loaders[split], criterion, device, args.mode)
        pred = np.digitize(raw, best_thr) if args.mode == "reg" else raw.astype(int)

        ids = df[df.split == split].reset_index(drop=True)["id_code"]
        pred_rows.append(pd.DataFrame({
            "run_id": run_id, "author": args.author, "model": args.model,
            "mode": args.mode, "split": split, "id_code": ids,
            "y_true": true.astype(int), "y_pred": pred.astype(int),
            "raw_score": raw.astype(float), "created_at": created,
        }))

        s_qwk = qwk(true, pred)
        metric_rows.append({
            "run_id": run_id, "author": args.author, "model": args.model,
            "mode": args.mode, "split": split, "qwk": s_qwk,
            "accuracy": float((true == pred).mean()),
            "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
            "n": int(len(true)), "epochs": last_epoch, "best_epoch": best_epoch,
            "variant": variant, "leak_excluded": bool(args.exclude_leaked),
            "img_size": args.size, "seed": args.seed, "created_at": created,
        })
        print(f"\n{split.upper()}  QWK={s_qwk:.4f}  "
              f"acc={(true == pred).mean():.4f}  "
              f"macro_F1={f1_score(true, pred, average='macro', zero_division=0):.4f}")
        print(confusion_matrix(true, pred, labels=range(5)))

    # --limit is a smoke test; writing those results next to real runs would
    # pollute every comparison. Dropping --no-bq is not enough - you also have
    # to drop --limit.
    if args.limit:
        print("\n--limit is on: results were NOT written to BigQuery "
              "(smoke tests must not land in the results table).")
    elif not args.no_bq:
        print("\nwriting to BigQuery...")
        try:
            write_to_bigquery(run_id, pd.concat(pred_rows, ignore_index=True),
                              pd.DataFrame(metric_rows))
        except Exception as e:
            print(f"  write failed: {type(e).__name__}: {str(e)[:160]}")


if __name__ == "__main__":
    main()
