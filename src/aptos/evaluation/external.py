"""External validation: train on APTOS, test somewhere else entirely.

Within APTOS the acquisition shortcut cannot be removed, only split around, and
splitting around it does not work here: the stratum where the confounded
1050x1050 resolution is excluded still contains sixteen other resolutions whose
label priors range from 0% No DR to 100% No DR. Every within-dataset stratum is
still a mixture of camera priors.

IDRiD removes it structurally. All 455 images are 4288x2848 - one resolution,
so geometry carries exactly zero label information - and the prior attached to
that geometry is inverted relative to APTOS, where all 52 images at 4288x2848
are diseased against IDRiD's 28.4% healthy.

Messidor-2 is the second external set: France rather than India, graded by an
adjudicating panel of three retina specialists (Krause et al. 2018) rather than
by single readers. It is not a second shortcut test - it was captured at more
than one resolution and the available mirror is already cropped - but a second
population, with the best labels of any set here. Its predictions were written
down first, in docs/second-external-validation-prediction.md.

Nothing here refits. The thresholds stay exactly as fitted on APTOS validation,
because refitting them on an external set would answer a much easier question
than the one being asked.
"""
from __future__ import annotations

import concurrent.futures as futures
import dataclasses
import json
import pathlib
from collections.abc import Callable

import cv2
import numpy as np
import pandas as pd

from aptos.config import Config
from aptos.data import manifest as manifest_mod
from aptos.modeling.thresholds import (
    apply_thresholds,
    clinically_costly_errors,
    grade_metrics,
    per_class_recall,
    referable_confusion,
)
from aptos.preprocessing import preprocess

IDRID_IMAGES = "Imagenes/Imagenes"
IDRID_LABELS = "idrid_labels.csv"


def load_idrid_labels(cfg: Config) -> pd.DataFrame:
    """Read the IDRiD grading table.

    The CSV carries trailing empty columns from stray commas - the same defect
    class as APTOS's own test.csv, which ended in ~500 blank lines - so only the
    two columns that matter are taken.
    """
    path = cfg.paths.external / IDRID_LABELS
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it first:\n"
            f"    kaggle datasets download mariaherrerot/idrid-dataset "
            f"-p data/external --unzip"
        )
    df = pd.read_csv(path, usecols=["id_code", "diagnosis"])
    df = df.dropna(subset=["id_code", "diagnosis"])
    df["diagnosis"] = df["diagnosis"].astype(int)

    bad = set(df["diagnosis"]) - set(range(5))
    if bad:
        raise ValueError(f"unexpected IDRiD grades: {sorted(bad)}")
    return df.reset_index(drop=True)


MESSIDOR2_DIR = "messidor2"
MESSIDOR2_LABELS = "messidor_data.csv"
MESSIDOR2_IMAGES = "messidor-2/messidor-2/preprocess"


def load_messidor2_labels(cfg: Config) -> pd.DataFrame:
    """Read Google's adjudicated Messidor-2 grades.

    Only images the panel marked gradable are kept - the one exclusion fixed in
    advance. `id_code` is the file stem, so the processed cache can use the same
    `<id_code>.jpg` naming as every other set.
    """
    path = cfg.paths.external / MESSIDOR2_DIR / MESSIDOR2_LABELS
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it first:\n"
            f"    kaggle datasets download google-brain/messidor2-dr-grades "
            f"-p data/external/messidor2 --unzip\n"
            f"    kaggle datasets download mariaherrerot/messidor2preprocess "
            f"-p data/external/messidor2 --unzip"
        )
    df = pd.read_csv(path)
    # The Kaggle copy renames Google's image_id/adjudicated_dr_grade columns.
    df = df.rename(columns={"image_id": "id_code", "adjudicated_dr_grade": "diagnosis"})
    if "adjudicated_gradable" in df:
        df = df[df["adjudicated_gradable"] == 1]
    df = df.dropna(subset=["id_code", "diagnosis"]).copy()
    df["source_name"] = df["id_code"].astype(str)
    df["id_code"] = df["source_name"].map(lambda name: pathlib.Path(name).stem)
    df["diagnosis"] = df["diagnosis"].astype(int)

    bad = set(df["diagnosis"]) - set(range(5))
    if bad:
        raise ValueError(f"unexpected Messidor-2 grades: {sorted(bad)}")
    return df[["id_code", "diagnosis", "source_name"]].reset_index(drop=True)


@dataclasses.dataclass(frozen=True)
class ExternalSet:
    """Where one external dataset lives and how its labels are read."""

    name: str
    prefix: str
    load_labels: Callable[[Config], pd.DataFrame]
    source: Callable[[Config, pd.Series], pathlib.Path]


DATASETS = {
    "idrid": ExternalSet(
        name="IDRiD", prefix="idrid", load_labels=load_idrid_labels,
        source=lambda cfg, row: cfg.paths.external / IDRID_IMAGES / f"{row['id_code']}.jpg",
    ),
    "messidor2": ExternalSet(
        name="Messidor-2", prefix="messidor2", load_labels=load_messidor2_labels,
        source=lambda cfg, row: (cfg.paths.external / MESSIDOR2_DIR / MESSIDOR2_IMAGES
                                 / row["source_name"]),
    ),
}


def dataset(key: str) -> ExternalSet:
    try:
        return DATASETS[key]
    except KeyError:
        raise ValueError(f"unknown external dataset {key!r}; "
                         f"known: {sorted(DATASETS)}") from None


def build_cache(cfg: Config, *, dataset_key: str = "idrid", force: bool = False) -> pathlib.Path:
    """Put an external set through the identical preprocessing pipeline.

    The whole comparison rests on only the images changing, so this uses the
    same `preprocess()` and the same settings recorded in the APTOS cache's
    manifest - not a reimplementation that happens to look similar.
    """
    ds = dataset(dataset_key)
    labels = ds.load_labels(cfg)
    rows = {row["id_code"]: row for _, row in labels.iterrows()}
    target = cfg.paths.external / f"{ds.prefix}_{cfg.variant}"

    settings = cfg.manifest_fields()
    if target.exists() and not force:
        try:
            manifest_mod.check(target, settings, count_images=True)
            return target
        except (FileNotFoundError, manifest_mod.ManifestMismatch):
            pass  # rebuild below

    target.mkdir(parents=True, exist_ok=True)
    print(f"preprocessing {len(labels)} {ds.name} images -> {target.name} "
          f"({manifest_mod.describe({**settings, 'n_images': len(labels)})})")

    def one(id_code: str) -> tuple[str, bool]:
        out_path = target / f"{id_code}.jpg"
        if out_path.exists() and not force:
            return id_code, True
        image, _info = preprocess(
            ds.source(cfg, rows[id_code]),
            size=settings["size"],
            use_clahe=settings["clahe"],
            clip_limit=settings["clip_limit"] or 2.0,
            square_mode=settings["square_mode"],
            normalize=False,
        )
        if image is None:
            return id_code, False
        cv2.imwrite(str(out_path), image,
                    [cv2.IMWRITE_JPEG_QUALITY, cfg.preprocess.jpeg_quality])
        return id_code, True

    failures = []
    with futures.ThreadPoolExecutor(max_workers=cfg.preprocess.workers) as pool:
        for id_code, ok in pool.map(one, labels["id_code"]):
            if not ok:
                failures.append(id_code)

    if failures:
        print(f"  {len(failures)} image(s) failed the quality gate: {failures[:5]}")
    manifest_mod.write(target, settings, len(labels) - len(failures))
    return target


# ---------------------------------------------------------------------- scoring

def load_ensemble(sweep: pathlib.Path) -> tuple[list[dict], np.ndarray]:
    """Per-fold weights and the APTOS-fitted threshold set.

    Thresholds are the mean of the folds', exactly as the cross-validation
    ensemble computed them. They are carried over untouched.
    """
    checkpoints = sorted(sweep.glob("fold*.pt"), key=lambda p: int(p.stem[4:]))
    if not checkpoints:
        raise FileNotFoundError(f"no fold checkpoints under {sweep}")

    import torch

    states, thresholds = [], []
    for path in checkpoints:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        states.append(blob["state_dict"])
        thresholds.append(blob["thresholds"])
    return states, np.mean(thresholds, axis=0)


class FlatDataset:
    """A single directory of processed images, no split subdirectories.

    Defined at module level on purpose. Windows spawns dataloader workers and
    pickles the dataset to reach them, and a class defined inside a function
    cannot be pickled - the worker dies with "Can't pickle local object".
    """

    def __init__(self, frame, cache: pathlib.Path, transform):
        self.frame = frame.reset_index(drop=True)
        self.cache = cache
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx):
        from PIL import Image

        row = self.frame.iloc[idx]
        image = Image.open(self.cache / f"{row['id_code']}.jpg").convert("RGB")
        return self.transform(image), int(row["diagnosis"])


def predict(cfg: Config, cache: pathlib.Path, labels: pd.DataFrame,
            states: list[dict], device: str) -> np.ndarray:
    """Average the folds' raw outputs, the same way the APTOS ensemble does."""
    import torch
    from torch.utils.data import DataLoader

    from aptos.training.loop import build_model, build_transforms

    _, eval_tf = build_transforms(cfg.train.size, cfg.augment)

    loader = DataLoader(FlatDataset(labels, cache, eval_tf),
                        batch_size=cfg.train.batch,
                        shuffle=False, num_workers=cfg.train.workers,
                        pin_memory=True)

    model = build_model(cfg, device)
    model.eval()

    fold_raws = []
    for i, state in enumerate(states, start=1):
        model.load_state_dict(state)
        outputs = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device, non_blocking=True)
                with torch.autocast("cuda", enabled=device == "cuda"):
                    outputs.append(model(images).squeeze(1).float().cpu().numpy())
        fold_raws.append(np.concatenate(outputs))
        print(f"  fold {i}/{len(states)} scored")
    return np.mean(fold_raws, axis=0)


def score(cfg: Config, sweep: pathlib.Path, dataset_key: str = "idrid", *,
          device: str | None = None, force: bool = False) -> pd.DataFrame:
    """Per-image raw ensemble scores for one external set, computed once.

    Saved next to the cache as `<prefix>_<variant>_<sweep>_scores.csv`, so the
    external report and the calibration experiments read the same numbers
    instead of each re-running five models.
    """
    ds = dataset(dataset_key)
    out = cfg.paths.external / f"{ds.prefix}_{cfg.variant}_{sweep.name}_scores.csv"
    if out.exists() and not force:
        return pd.read_csv(out)

    from aptos.training.loop import pick_device

    labels = ds.load_labels(cfg)
    cache = build_cache(cfg, dataset_key=dataset_key)
    kept = {p.stem for p in cache.glob("*.jpg")}
    dropped = int((~labels["id_code"].isin(kept)).sum())
    if dropped:
        print(f"  {dropped} image(s) failed the quality gate and are not scored")
    labels = labels[labels["id_code"].isin(kept)].reset_index(drop=True)

    states, _ = load_ensemble(sweep)
    print(f"scoring {len(labels)} {ds.name} images with {len(states)} folds")
    raw = predict(cfg, cache, labels, states, device or pick_device())
    frame = pd.DataFrame({"id_code": labels["id_code"], "diagnosis": labels["diagnosis"],
                          "raw": raw})
    frame.to_csv(out, index=False)
    return frame


def evaluate(cfg: Config, sweep: pathlib.Path, *, dataset_key: str = "idrid",
             device: str | None = None) -> dict:
    """Run the whole external test and return the report payload."""
    ds = dataset(dataset_key)
    scores = score(cfg, sweep, dataset_key, device=device)
    _, thresholds = load_ensemble(sweep)
    print(f"APTOS thresholds {np.round(thresholds, 3).tolist()} (not refitted)")

    raw = scores["raw"].to_numpy()
    pred = apply_thresholds(raw, thresholds)
    true = scores["diagnosis"].to_numpy()

    report = {
        "dataset": ds.name,
        "n": int(len(scores)),
        "variant": cfg.variant,
        "sweep": sweep.name,
        "thresholds": [float(t) for t in thresholds],
        "refitted": False,
        **grade_metrics(true, pred),
        **{f"ref_{k}": v for k, v in referable_confusion(true, pred).items()},
        **clinically_costly_errors(true, pred),
        "per_class_recall": {str(k): v for k, v in per_class_recall(true, pred).items()},
        "predicted_distribution": {str(g): int((pred == g).sum()) for g in range(5)},
        "true_distribution": {str(g): int((true == g).sum()) for g in range(5)},
    }

    # The discriminating measurement, called in advance: specificity on the
    # healthy eyes. A model leaning on acquisition cues should over-call disease
    # here, because in APTOS this geometry always meant disease.
    healthy = true == 0
    report["healthy_n"] = int(healthy.sum())
    report["healthy_called_healthy"] = float((pred[healthy] == 0).mean())
    report["healthy_called_referable"] = float((pred[healthy] >= 2).mean())
    return report


def format_report(report: dict, aptos_reference: dict | None = None) -> str:
    lines = [
        f"## External validation on {report['dataset']}",
        "",
        f"- {report['n']} images, scored by the {report['variant']} "
        f"{len(report['thresholds'])}-threshold fold ensemble from sweep `{report['sweep']}`",
        "- no fine-tuning, and the APTOS validation thresholds were **not refitted**",
        "",
        f"| metric | {report['dataset']} |" + (" APTOS test |" if aptos_reference else ""),
        "|---|---|" + ("---|" if aptos_reference else ""),
    ]
    rows = [("QWK", "qwk"), ("accuracy", "accuracy"), ("macro F1", "macro_f1"),
            ("referable sensitivity", "ref_sensitivity"),
            ("referable specificity", "ref_specificity")]
    for label, key in rows:
        line = f"| {label} | {report[key]:.4f} |"
        if aptos_reference:
            line += f" {aptos_reference.get(key, float('nan')):.4f} |"
        lines.append(line)
    lines += [
        "",
        f"Healthy eyes (grade 0): {report['healthy_n']}. "
        f"{report['healthy_called_healthy']:.1%} were called healthy; "
        f"{report['healthy_called_referable']:.1%} were called referable.",
        "",
        f"Severe cases missed: {report['severe_missed']}/{report['severe_total']}.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sweep", required=True, help="models/cv/<sweep-dir>")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--dataset", default="idrid", choices=sorted(DATASETS))
    parser.add_argument("--out", default=None,
                        help="default: reports/external_validation.md for IDRiD, "
                             "reports/external_validation_<dataset>.md otherwise")
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = ("reports/external_validation.md" if args.dataset == "idrid"
                    else f"reports/external_validation_{args.dataset}.md")

    sweep = pathlib.Path(args.sweep)
    variant = args.variant or sweep.name.split("-")[0]
    cfg = Config.load(f"configs/{variant}.yaml")
    cfg.train.size = Config.load("configs/cv.yaml").train.size

    report = evaluate(cfg, sweep, dataset_key=args.dataset)
    out = cfg.paths.root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(format_report(report), encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + format_report(report))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
