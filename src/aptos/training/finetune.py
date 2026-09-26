"""Fine-tune the deployed ensemble on adjudicated Messidor-2 labels.

Messidor-2 showed the APTOS model under-grading Moderate disease: 83% of
Moderate eyes graded below 2, referable AUC 0.819. The post-hoc reading was that
the model learned where APTOS's single graders draw the Mild/Moderate line. This
tests that reading, with the design and predictions fixed first in
docs/finetune-prediction.md.

Two arms are fine-tuned on the same Messidor-2 images:

  * adjudicated - the specialist panel's grades;
  * control     - the unchanged ensemble's own grades for those images, i.e. the
                  APTOS boundary on Messidor-2's cameras.

If the gain is the labels, only the adjudicated arm improves. If it is the
camera, both do.

    python -m aptos.training.finetune split       # write the patient-grouped split
    python -m aptos.training.finetune run         # both arms, then the report
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import pandas as pd

from aptos.config import REFERABLE_FROM, Config
from aptos.evaluation import external
from aptos.modeling.thresholds import (
    apply_thresholds,
    grade_metrics,
    optimize_thresholds,
    qwk,
)

SWEEP = "baseline-6e66147526"
SPLIT_FILE = "messidor2_split.csv"


# ------------------------------------------------------------------------ split

def patient_groups(id_codes: pd.Series) -> pd.Series:
    """A group key that never separates a patient's two eyes.

    Original Messidor images start with the exam date, and both eyes of a
    patient are photographed on the same day, so the date is the group. The
    later Brest images number a patient's two eyes consecutively, so every
    chain of consecutive IM numbers is one group. Both are coarser than a
    patient - a group can hold several - which costs balance, never leakage.
    """
    keys = pd.Series(index=id_codes.index, dtype=object)
    im = id_codes.str.extract(r"^IM(\d+)")[0]
    is_im = im.notna()
    keys[~is_im] = "date:" + id_codes[~is_im].str.split("_").str[0]

    numbers = im[is_im].astype(int).sort_values()
    chain, previous = 0, None
    for idx, n in numbers.items():
        if previous is not None and n != previous + 1:
            chain += 1
        keys[idx] = f"im:{chain}"
        previous = n
    return keys


def make_split(labels: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Assign every image to test, tune-train or tune-valid, by patient group."""
    from sklearn.model_selection import StratifiedGroupKFold

    df = labels[["id_code", "diagnosis"]].copy().reset_index(drop=True)
    df["group"] = patient_groups(df["id_code"])
    referable = (df["diagnosis"] >= REFERABLE_FROM).astype(int)

    halves = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    _, tune_idx = next(iter(halves.split(df, referable, df["group"])))
    df["part"] = "test"
    df.loc[tune_idx, "part"] = "tune-train"

    tune = df.loc[tune_idx]
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, valid_local = next(iter(inner.split(tune, referable.loc[tune_idx], tune["group"])))
    df.loc[tune.index[valid_local], "part"] = "tune-valid"
    return df


def load_split(cfg: Config) -> pd.DataFrame:
    path = cfg.paths.reports / SPLIT_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m aptos.training.finetune split`")
    return pd.read_csv(path)


# ------------------------------------------------------------------- training

@dataclasses.dataclass(frozen=True)
class FineTuneConfig:
    """Fixed in docs/finetune-prediction.md before the first run."""

    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 8
    seed: int = 42


def _loader(frame, cache, transform, cfg: Config, *, shuffle: bool):
    from torch.utils.data import DataLoader

    return DataLoader(external.FlatDataset(frame, cache, transform),
                      batch_size=cfg.train.batch, shuffle=shuffle,
                      num_workers=cfg.train.workers, pin_memory=True)


def fine_tune_fold(cfg: Config, state: dict, train_df: pd.DataFrame, valid_df: pd.DataFrame,
                   cache: pathlib.Path, ft: FineTuneConfig, device: str) -> tuple[dict, list, dict]:
    """One fold model, fine-tuned; returns the best state, thresholds and history."""
    import torch

    from aptos.training.loop import build_model, build_transforms, run_epoch, set_seed

    set_seed(ft.seed)
    train_tf, eval_tf = build_transforms(cfg.train.size, cfg.augment)
    train_loader = _loader(train_df, cache, train_tf, cfg, shuffle=True)
    valid_loader = _loader(valid_df, cache, eval_tf, cfg, shuffle=False)

    model = build_model(cfg, device, pretrained=False)
    model.load_state_dict(state)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=ft.lr, weight_decay=ft.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft.epochs)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    best = {"qwk": -1.0, "state": None, "thresholds": None, "epoch": 0}
    history = []
    for epoch in range(1, ft.epochs + 1):
        tr_loss, _, _ = run_epoch(model, train_loader, criterion, device, "reg", optimizer, scaler)
        _, va_raw, va_true = run_epoch(model, valid_loader, criterion, device, "reg")
        thr, va_qwk = optimize_thresholds(va_true, va_raw, cfg.thresholds)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "valid_qwk": va_qwk})
        flag = ""
        if va_qwk > best["qwk"]:
            best = {"qwk": va_qwk, "thresholds": [float(t) for t in thr], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            flag = "  <- best"
        print(f"    epoch {epoch}/{ft.epochs}  train={tr_loss:.4f}  valid QWK={va_qwk:.4f}{flag}",
              flush=True)
    return best, history, {"best_epoch": best["epoch"], "valid_qwk": best["qwk"]}


def run_arm(cfg: Config, arm: str, split: pd.DataFrame, labels_for_training: pd.Series,
            ft: FineTuneConfig, device: str) -> pathlib.Path:
    """Fine-tune all five fold models for one arm; resumable per fold."""
    import torch

    out = cfg.paths.models / "finetune" / arm
    out.mkdir(parents=True, exist_ok=True)
    cache = external.build_cache(cfg, dataset_key="messidor2")

    frame = split.copy()
    frame["diagnosis"] = labels_for_training.to_numpy()
    train_df = frame[frame["part"] == "tune-train"]
    valid_df = frame[frame["part"] == "tune-valid"]

    checkpoints = sorted((cfg.paths.models / "cv" / SWEEP).glob("fold*.pt"),
                         key=lambda p: int(p.stem[4:]))
    for path in checkpoints:
        target = out / path.name
        if target.exists():
            print(f"  {arm} {path.stem}: already done - skipping")
            continue
        print(f"  {arm} {path.stem}: train={len(train_df)} valid={len(valid_df)}", flush=True)
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        best, history, summary = fine_tune_fold(cfg, state, train_df, valid_df, cache, ft, device)
        torch.save({"state_dict": best["state"], "thresholds": best["thresholds"],
                    "history": history, **summary}, target)
    return out


# ------------------------------------------------------------------ evaluation

def score_sets(cfg: Config, fold_dir: pathlib.Path, device: str) -> dict[str, pd.DataFrame]:
    """Ensemble scores for the Messidor-2 test half, IDRiD and APTOS test."""
    from aptos.data import labels as labels_mod

    states, thresholds = external.load_ensemble(fold_dir)
    split = load_split(cfg)
    messidor = split[split["part"] == "test"][["id_code", "diagnosis"]]
    idrid = external.load_idrid_labels(cfg)
    aptos = labels_mod.load_labels(cfg)
    aptos = aptos[aptos["split"] == "test"][["id_code", "diagnosis"]]

    caches = {
        "Messidor-2 test half": (messidor, external.build_cache(cfg, dataset_key="messidor2")),
        "IDRiD": (idrid, external.build_cache(cfg, dataset_key="idrid")),
        "APTOS test": (aptos, cfg.paths.processed(cfg.variant) / "test"),
    }
    out = {}
    for name, (frame, cache) in caches.items():
        frame = frame.reset_index(drop=True)
        raw = external.predict(cfg, cache, frame, states, device)
        out[name] = frame.assign(raw=raw, pred=apply_thresholds(raw, thresholds))
    return out


def summarise(scored: dict[str, pd.DataFrame], dme: pd.Series) -> dict:
    """The measures the predictions are stated in."""
    from sklearn.metrics import roc_auc_score

    result = {}
    for name, f in scored.items():
        true, raw, pred = f["diagnosis"].to_numpy(), f["raw"].to_numpy(), f["pred"].to_numpy()
        row = {"n": int(len(f)),
               "referable_auc": float(roc_auc_score(true >= REFERABLE_FROM, raw)),
               "qwk": float(qwk(true, pred))}
        moderate = true == 2
        if moderate.any():
            row["moderate_below_2"] = float((pred[moderate] < 2).mean())
            row["moderate_n"] = int(moderate.sum())
        if name.startswith("Messidor-2"):
            flags = f["id_code"].map(dme)
            for flag, label in ((0, "without_dme"), (1, "with_dme")):
                sel = moderate & (flags == flag).to_numpy()
                row[f"moderate_{label}_median_raw"] = float(np.median(raw[sel])) if sel.any() else None
        row.update({k: float(v) for k, v in grade_metrics(true, pred).items()
                    if k in ("accuracy", "macro_f1")})
        result[name] = row
    return result


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["split", "run"])
    args = parser.parse_args(argv)

    cfg = Config.load("configs/baseline.yaml")
    base = Config.load("configs/cv.yaml")
    cfg.train = dataclasses.replace(base.train, workers=0)
    labels = external.load_messidor2_labels(cfg)

    if args.command == "split":
        split = make_split(labels)
        out = cfg.paths.reports / SPLIT_FILE
        split.to_csv(out, index=False, lineterminator="\n")
        print(split.groupby("part").agg(n=("id_code", "size"), groups=("group", "nunique"),
                                        referable=("diagnosis", lambda d: float((d >= 2).mean()))))
        print(f"written to {out}")
        return 0

    from aptos.training.loop import pick_device

    device = pick_device()
    ft = FineTuneConfig()
    split = load_split(cfg)

    # The control's labels: what the unchanged ensemble says about these images.
    scores = external.score(cfg, cfg.paths.models / "cv" / SWEEP, "messidor2")
    _, thresholds = external.load_ensemble(cfg.paths.models / "cv" / SWEEP)
    own = dict(zip(scores["id_code"], apply_thresholds(scores["raw"].to_numpy(), thresholds),
                   strict=True))
    arms = {
        "adjudicated": split["diagnosis"],
        "control": split["id_code"].map(own).astype(int),
    }
    print(f"control labels agree with the panel on "
          f"{(arms['control'] == split['diagnosis']).mean():.1%} of images")

    raw_dme = pd.read_csv(cfg.paths.external / external.MESSIDOR2_DIR / external.MESSIDOR2_LABELS)
    raw_dme["id_code"] = raw_dme["id_code"].map(lambda n: pathlib.Path(n).stem)
    dme = raw_dme.set_index("id_code")["adjudicated_dme"]

    report = {"design": "docs/finetune-prediction.md", "finetune": dataclasses.asdict(ft)}
    print("scoring the unchanged ensemble")
    report["none"] = summarise(score_sets(cfg, cfg.paths.models / "cv" / SWEEP, device), dme)
    for arm, training_labels in arms.items():
        print(f"arm: {arm}")
        fold_dir = run_arm(cfg, arm, split, training_labels, ft, device)
        report[arm] = summarise(score_sets(cfg, fold_dir, device), dme)

    out = cfg.paths.reports / "finetune_messidor2.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k in ("none", "adjudicated", "control")},
                     indent=2))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

