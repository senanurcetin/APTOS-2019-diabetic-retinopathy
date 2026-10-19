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
import hashlib
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


SOURCE_PROVENANCE = "_source_provenance.json"


def external_source_signature(cfg: Config, ds: ExternalSet, labels: pd.DataFrame) -> dict:
    """Fingerprint actual raw fundus bytes, JPEG encoding and preprocessing code.

    Reading source bytes is intentional. Size, mtime and filename alone are not
    sufficient to verify the images from a mutable Kaggle mirror.
    """
    if labels.empty or "id_code" not in labels:
        raise ValueError("source fingerprint requires nonempty image IDs")
    hashes = {}
    for _, row in labels.iterrows():
        ident = str(row["id_code"])
        if not ident or ident in hashes:
            raise ValueError("source image IDs must be nonempty and unique")
        original = ds.source(cfg, row)
        if not original.is_file():
            raise FileNotFoundError(f"source retinal image missing: {original}")
        hashes[ident] = _sha256(original)
    return {
        "signature_version": 1,
        "dataset": ds.prefix,
        "source_sha256": dict(sorted(hashes.items())),
        "preprocessing_code_sha256": _sha256(pathlib.Path(preprocess.__code__.co_filename)),
        "preprocess": cfg.manifest_fields(),
        "jpeg_quality": cfg.preprocess.jpeg_quality,
    }


def source_provenance_path(cache: pathlib.Path) -> pathlib.Path:
    return cache / SOURCE_PROVENANCE


def verify_external_source_provenance(
    cfg: Config, ds: ExternalSet, labels: pd.DataFrame, cache: pathlib.Path,
) -> None:
    """Refuse old derived JPEGs when raw retinal source bytes changed."""
    signature = external_source_signature(cfg, ds, labels)
    try:
        stored = json.loads(source_provenance_path(cache).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise manifest_mod.ManifestMismatch(
            "external source provenance missing or corrupt; rebuild cache"
        ) from exc
    if stored != signature:
        raise manifest_mod.ManifestMismatch(
            "external source provenance changed; regenerate derived images"
        )


def build_cache(cfg: Config, *, dataset_key: str = "idrid", force: bool = False) -> pathlib.Path:
    """Regenerate derived external JPEGs whenever provenance or IDs disagree.

    A mismatched manifest previously triggered a rebuild but skipped existing
    JPEGs, then certified stale pixels with the *new* settings. That made
    cross-site model comparisons scientifically invalid.
    """
    ds = dataset(dataset_key)
    labels = ds.load_labels(cfg)
    if labels.empty or not {"id_code", "diagnosis"}.issubset(labels.columns):
        raise ValueError("external labels need id_code, diagnosis and at least one row")
    if labels["id_code"].isna().any():
        raise ValueError("external image IDs must not be null")
    ids = labels["id_code"].astype(str).tolist()
    if (len(ids) != len(set(ids)) or any(
            not identifier or identifier in (".", "..")
            or "/" in identifier or "\\" in identifier for identifier in ids)):
        raise ValueError("external image IDs must be unique and path-safe")
    rows = {row["id_code"]: row for _, row in labels.iterrows()}
    target = cfg.paths.external / f"{ds.prefix}_{cfg.variant}"
    settings = cfg.manifest_fields()
    signature = external_source_signature(cfg, ds, labels)

    if target.exists() and not force:
        try:
            manifest_mod.check(target, settings, count_images=True)
            saved_signature = json.loads(
                source_provenance_path(target).read_text(encoding="utf-8")
            )
            if saved_signature != signature:
                raise manifest_mod.ManifestMismatch("external source provenance changed")
            cached_ids = {path.stem for path in target.glob("*.jpg")}
            if cached_ids == set(ids):
                return target
        except (FileNotFoundError, manifest_mod.ManifestMismatch, json.JSONDecodeError):
            pass

    target.mkdir(parents=True, exist_ok=True)
    # Invalidate BEFORE touching files. An interrupted rebuild cannot be
    # mistaken for a complete, verified cache on a later run.
    manifest_mod.path_for(target).unlink(missing_ok=True)
    source_provenance_path(target).unlink(missing_ok=True)
    for image_path in target.glob("*.jpg"):
        image_path.unlink()
    for temporary in target.glob(".*.tmp.jpg"):
        temporary.unlink()

    print(f"preprocessing {len(labels)} {ds.name} images -> {target.name} "
          f"({manifest_mod.describe({**settings, 'n_images': len(labels)})})")

    def one(id_code: str) -> tuple[str, bool]:
        output = target / f"{id_code}.jpg"
        temp = target / f".{id_code}.tmp.jpg"
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
        if not cv2.imwrite(
            str(temp), image, [cv2.IMWRITE_JPEG_QUALITY, cfg.preprocess.jpeg_quality]
        ):
            raise OSError(f"could not write processed external image {id_code}")
        temp.replace(output)
        return id_code, True

    failures = []
    with futures.ThreadPoolExecutor(max_workers=cfg.preprocess.workers) as pool:
        for id_code, ok in pool.map(one, ids):
            if not ok:
                failures.append(id_code)

    if failures:
        print(f"  {len(failures)} image(s) failed the quality gate: {failures[:5]}")
    manifest_mod.write(target, settings, len(labels) - len(failures))
    # Verify what is on disk rather than only trusting the manifest we wrote.
    manifest_mod.check(target, settings, count_images=True)
    source_path = source_provenance_path(target)
    temporary = source_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(signature, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(source_path)
    return target


# ---------------------------------------------------------------------- scoring

def validate_sweep_contract(cfg: Config, sweep: pathlib.Path) -> None:
    """Fail closed on ordinal checkpoint/config mismatches and wrong input sizes.

    The versioned flat config is authoritative for new sweeps. Historical
    scalar sweeps lacking config.json are supported only in regression mode.
    """
    manifest = sweep / "config.json"
    if not manifest.is_file():
        if cfg.train.mode != "reg":
            raise ValueError("ordinal evaluation requires the training sweep config.json")
        return
    data = json.loads(manifest.read_text(encoding="utf-8"))
    fields = {
        "variant": cfg.variant,
        "train.mode": cfg.train.mode,
        "train.model": cfg.train.model,
        "train.size": cfg.train.size,
        "preprocess.square_mode": cfg.preprocess.square_mode,
        "preprocess.clahe": cfg.preprocess.clahe,
        "preprocess.size": cfg.preprocess.size,
        "preprocess.clip_limit": cfg.preprocess.clip_limit,
        "preprocess.jpeg_quality": cfg.preprocess.jpeg_quality,
    }
    for field, value in fields.items():
        if field not in data or data[field] != value:
            raise ValueError(f"sweep config mismatch for {field}: "
                             f"stored={data.get(field)!r}, requested={value!r}")
    if cfg.train.mode not in ("reg", "ordinal"):
        raise ValueError("external research comparison supports reg and ordinal modes only")


def complete_sweep_checkpoints(sweep: pathlib.Path) -> list[pathlib.Path]:
    """A partial or unrecorded CV sweep is not an external-validation ensemble.

    Require complete saved fold metadata and a full-run summary. Filename
    counting alone is insufficient: a failed training run can leave some
    checkpoints but never publish the completed CV summary.
    """
    summary_path = sweep / "summary.json"
    if not summary_path.is_file():
        raise ValueError("CV sweep is incomplete: summary.json is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    count = summary.get("folds")
    config_path = sweep / "config.json"
    if config_path.is_file():
        settings = json.loads(config_path.read_text(encoding="utf-8"))
        configured = settings.get("cv.folds")
        if configured != count:
            raise ValueError("CV sweep summary fold count differs from training config")
        if settings.get("train.mode") not in ("reg", "ordinal"):
            raise ValueError("CV sweep has unknown training mode")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("full external CV requires at least two recorded folds")
    expected = [sweep / f"fold{i}.pt" for i in range(1, count + 1)]
    available = set(sweep.glob("fold*.pt"))
    if available != set(expected):
        raise ValueError("CV ensemble has missing, extra or misnumbered checkpoints")
    for i, path in enumerate(expected, start=1):
        for name in (f"fold{i}.pt", f"fold{i}.npz", f"fold{i}.json"):
            candidate = sweep / name
            if not candidate.is_file() or candidate.stat().st_size == 0:
                raise ValueError(f"CV ensemble has incomplete fold artifacts: {name}")
    return expected


def load_ensemble(sweep: pathlib.Path) -> tuple[list[dict], np.ndarray]:
    """Load a complete sweep with frozen APTOS thresholds, without pickle code."""
    checkpoints = complete_sweep_checkpoints(sweep)
    import torch

    states, thresholds = [], []
    for fold, path in enumerate(checkpoints, start=1):
        # State dictionaries and primitive metadata are all this project
        # saves; arbitrary Python objects must not be deserialized.
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(blob, dict) or blob.get("fold") != fold:
            raise ValueError(f"{path.name} has a mismatched fold identity")
        if not isinstance(blob.get("state_dict"), dict) or not blob["state_dict"]:
            raise ValueError(f"{path.name} has no model parameters")
        try:
            cuts = np.asarray(blob["thresholds"], dtype=float)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"{path.name} missing saved validation thresholds") from exc
        if (cuts.shape != (4,) or not np.isfinite(cuts).all()
                or np.any(np.diff(cuts) <= 0)):
            raise ValueError(f"{path.name} has invalid validation thresholds")
        states.append(blob["state_dict"])
        thresholds.append(cuts)
    mean_cuts = np.mean(thresholds, axis=0)
    summary = json.loads((sweep / "summary.json").read_text(encoding="utf-8"))
    try:
        expected = np.asarray(summary["ensemble"]["thresholds"], dtype=float)
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("CV summary lacks ensemble thresholds") from exc
    if (expected.shape != (4,) or not np.isfinite(expected).all()
            or not np.allclose(mean_cuts, expected, atol=1e-7, rtol=0)):
        raise ValueError("CV ensemble thresholds disagree with saved APTOS summary")
    return states, mean_cuts


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

    # The saved fold contains full trained weights: avoid downloading ImageNet
    # parameters only to overwrite them, including when using Kaggle offline.
    model = build_model(cfg, device, pretrained=False)
    model.eval()

    fold_raws = []
    for i, state in enumerate(states, start=1):
        model.load_state_dict(state)
        outputs = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device, non_blocking=True)
                with torch.autocast("cuda", enabled=device == "cuda"):
                    logits = model(images)
                    if cfg.train.mode == "ordinal":
                        from aptos.training.ordinal import ordinal_expected_grade

                        raw = ordinal_expected_grade(logits)
                    else:
                        raw = logits.squeeze(1)
                    if raw.ndim != 1 or raw.shape[0] != images.shape[0]:
                        raise ValueError("external model produced invalid raw-score shape")
                    outputs.append(raw.float().cpu().numpy())
        fold_raws.append(np.concatenate(outputs))
        print(f"  fold {i}/{len(states)} scored")
    return np.mean(fold_raws, axis=0)


def _sha256(path: pathlib.Path) -> str:
    """Fingerprint bytes; never trust filenames or mtimes as model provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_input_signature(cfg: Config, sweep: pathlib.Path,
                          cache: pathlib.Path, labels: pd.DataFrame) -> dict:
    """Content-address external predictions by model, labels and derived pixels."""
    checkpoints = sorted(sweep.glob("fold*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no fold checkpoints under {sweep}")
    if labels.empty or labels["id_code"].isna().any():
        raise ValueError("no valid external images for scoring")
    ids = labels["id_code"].astype(str)
    if ids.duplicated().any():
        raise ValueError("external score labels contain duplicate image IDs")
    digest = hashlib.sha256()
    for identifier in sorted(ids):
        path = cache / f"{identifier}.jpg"
        if not path.is_file():
            raise FileNotFoundError(f"missing processed external image: {identifier}")
        digest.update(identifier.encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    records = labels[["id_code", "diagnosis"]].copy()
    records["id_code"] = records["id_code"].astype(str)
    records = records.sort_values("id_code", kind="stable")
    label_hash = hashlib.sha256(
        records.to_json(orient="records").encode("utf-8")
    ).hexdigest()
    settings = cfg.manifest_fields()
    return {
        "signature_version": 1,
        "model_checkpoints": {f.name: _sha256(f) for f in checkpoints},
        "sweep_config": (_sha256(sweep / "config.json")
                         if (sweep / "config.json").is_file() else None),
        "input_image_sha256": digest.hexdigest(),
        "labels_sha256": label_hash,
        "preprocess": settings,
        "n_scored": len(labels),
    }


def read_verified_score_cache(output: pathlib.Path, signature: dict,
                              labels: pd.DataFrame) -> pd.DataFrame | None:
    """Reuse scores only when bytes, content and current labels are identical."""
    meta = output.with_suffix(".meta.json")
    if not output.is_file() or not meta.is_file():
        return None
    try:
        record = json.loads(meta.read_text(encoding="utf-8"))
        if record.get("input_signature") != signature:
            return None
        if record.get("score_csv_sha256") != _sha256(output):
            return None
        frame = pd.read_csv(output, dtype={"id_code": "string"})
        if list(frame.columns) != ["id_code", "diagnosis", "raw"]:
            return None
        if not frame["id_code"].astype(str).equals(
            labels["id_code"].astype(str).reset_index(drop=True)
        ):
            return None
        actual = pd.to_numeric(frame["diagnosis"], errors="raise").to_numpy()
        if not np.array_equal(actual, labels["diagnosis"].to_numpy()):
            return None
        raw = pd.to_numeric(frame["raw"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(raw).all():
            return None
        return frame
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def score(cfg: Config, sweep: pathlib.Path, dataset_key: str = "idrid", *,
          device: str | None = None, force: bool = False) -> pd.DataFrame:
    """Score or reuse a content-verified, immutable-model external score cache.

    Prior CSV-only caching silently reused scores when checkpoints, external
    labels, or processed retinal images changed. Existing CSVs without
    provenance sidecars are treated as unverified and must be recomputed.
    """
    validate_sweep_contract(cfg, sweep)
    # Do not reuse even previously signed CSVs when the originating CV run
    # lacks complete fold artifacts and an end-of-sweep summary.
    complete_sweep_checkpoints(sweep)
    ds = dataset(dataset_key)
    out = cfg.paths.external / f"{ds.prefix}_{cfg.variant}_{sweep.name}_scores.csv"
    labels = ds.load_labels(cfg)
    cache = build_cache(cfg, dataset_key=dataset_key)
    kept = {p.stem for p in cache.glob("*.jpg")}
    dropped = int((~labels["id_code"].isin(kept)).sum())
    if dropped:
        print(f"  {dropped} image(s) failed the quality gate and are not scored")
    labels = labels[labels["id_code"].isin(kept)].reset_index(drop=True)
    signature = score_input_signature(cfg, sweep, cache, labels)

    if not force:
        existing = read_verified_score_cache(out, signature, labels)
        if existing is not None:
            return existing

    from aptos.training.loop import pick_device

    states, _ = load_ensemble(sweep)
    print(f"scoring {len(labels)} {ds.name} images with {len(states)} folds")
    raw = predict(cfg, cache, labels, states, device or pick_device())
    if len(raw) != len(labels) or not np.isfinite(raw).all():
        raise ValueError("external scores must have one finite value per image")
    frame = pd.DataFrame({
        "id_code": labels["id_code"],
        "diagnosis": labels["diagnosis"],
        "raw": raw,
    })
    # Publish CSV, then its provenance, only after both are computed.
    tmp = out.with_name(out.stem + ".tmp.csv")
    frame.to_csv(tmp, index=False)
    tmp.replace(out)
    meta = out.with_suffix(".meta.json")
    meta_tmp = meta.with_name(meta.stem + ".tmp.json")
    meta_tmp.write_text(json.dumps({
        "input_signature": signature,
        "score_csv_sha256": _sha256(out),
    }, indent=2, sort_keys=True), encoding="utf-8")
    meta_tmp.replace(meta)
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
        "training_mode": cfg.train.mode,
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
    parser.add_argument("--mode", choices=("reg", "ordinal"), default=None,
                        help="assert the stored sweep's training mode")
    parser.add_argument("--dataset", default="idrid", choices=sorted(DATASETS))
    parser.add_argument("--out", default=None,
                        help="default: reports/external_validation.md for IDRiD, "
                             "reports/external_validation_<dataset>.md otherwise")
    args = parser.parse_args(argv)
    sweep = pathlib.Path(args.sweep)
    variant = args.variant or sweep.name.split("-")[0]
    cfg = Config.load(f"configs/{variant}.yaml")
    saved = sweep / "config.json"
    params = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() else {}
    cfg.train.size = params.get("train.size", Config.load("configs/cv.yaml").train.size)
    cfg.train.model = params.get("train.model", cfg.train.model)
    cfg.train.mode = params.get("train.mode", "reg")
    cfg.cv.folds = params.get("cv.folds", cfg.cv.folds)
    for key in ("size", "clahe", "clip_limit", "square_mode", "jpeg_quality"):
        setattr(cfg.preprocess, key, params.get(f"preprocess.{key}",
                                               getattr(cfg.preprocess, key)))
    if args.mode is not None and args.mode != cfg.train.mode:
        raise ValueError(f"requested mode {args.mode} disagrees with stored {cfg.train.mode}")
    validate_sweep_contract(cfg, sweep)

    if args.out is None:
        if cfg.train.mode == "ordinal":
            args.out = f"reports/research/ordinal_{sweep.name}_{args.dataset}.md"
        else:
            args.out = ("reports/external_validation.md" if args.dataset == "idrid"
                        else f"reports/external_validation_{args.dataset}.md")
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
