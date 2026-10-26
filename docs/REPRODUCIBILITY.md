# Reproduction guide

Run commands from the repository root. Python 3.10–3.12 is covered by CI.
The package separates evidence inspection, GPU training and serving dependencies.

## Installation and offline checks

```bash
python -m pip install -e ".[dev]"
pytest -m pure -q
ruff check .
python -m aptos.reporting.research_evidence --reports-dir reports --format json
python -m aptos.reporting.operating_point_audit --reports-dir reports --format json
python -m aptos.reporting.clinical_audit --format json
python -m aptos.reporting.portfolio --check
```

These checks do not require retinal images or model weights. For the full local
suite, install `.[train,serve,dev]` and ONNX Runtime, then run
`pytest -q --cov --cov-report=term`. CI uses CPU Torch for the full suite and
requires at least 65% coverage. Dataset-dependent tests need their own inputs;
a software test pass is not a model replication.

The explorer embeds hashes of committed evidence bytes. Use an LF checkout for
byte-for-byte artifact reproduction across Windows and Linux; changing line
endings changes a hash even when parsed JSON is identical. If the explorer is
intentionally updated, regenerate it with `python -m aptos.reporting.portfolio`.
Do not hand-edit its embedded evidence.

## Data preparation

Configure authorized Kaggle credentials outside the repository. Dataset access
and redistribution follow each source's terms, not the code license.

```bash
python -m pip install kaggle
kaggle datasets download mariaherrerot/aptos2019 -p data/images --unzip
python -m aptos.pipeline list
python -m aptos.pipeline check
python -m aptos.pipeline run prepare
# Before training: build the cache and run duplicate verification.
python -m aptos.pipeline run quality
```

Inspect the stage graph before `python -m aptos.pipeline run all`: it executes
all available stages, rather than just loading labels. Preprocessing must
precede duplicate verification. The mirror uses `val_images/`; its label CSVs
may contain trailing blank lines. Prepared caches have settings manifests;
training rejects incompatible caches.

The data-quality, image-property and duplicate reports remain in `reports/`.
Perceptual hashes identify candidates; pixel verification distinguishes real
repeated images from similar fundus geometry. Unique label IDs alone do not
establish image or patient independence.

## APTOS training and evaluation

```bash
python -m pip install -e ".[train,dev]"
python -m aptos.training.cv --config configs/cv.yaml --variant baseline --mode reg --seed 42
```

Read `configs/cv.yaml` before allocating compute. Scalar regression uses MSE;
validation determines the four grade thresholds and selected checkpoint.
CV saves configuration, predictions and metrics per fold and supports resuming
completed folds. Five models evaluated on the same holdout are dependent.

The shared image path is auto-crop, optional CLAHE, pad or squash, resize and
training-time normalization. Do not describe all experiments as padded: the
`squash` variant explicitly changes that operation. Preprocessing and geometric
arguments are documented alongside measured results in [RESULTS.md](../RESULTS.md).

For the uncompleted ordinal comparison, use the
[paired benchmark protocol](ordinal-benchmark.md). A smoke run or synthetic test
does not constitute a trained research result.

## External cohorts and completed follow-up

```bash
kaggle datasets download mariaherrerot/idrid-dataset -p data/external --unzip
python -m aptos.evaluation.external --sweep models/cv/baseline-RUN_ID --dataset idrid
```

Replace `RUN_ID` with the actual saved sweep. Keep its preprocessing, architecture,
mode and validation thresholds fixed. For Messidor-2, use the exact source and
pairing checks in [acquisition](MESSIDOR2_ACQUISITION.md), then the
[frozen allocation](MESSIDOR2_FOLLOWUP.md) and [v2 protocol](MESSIDOR2_V2_PROTOCOL.md).
The historical `finetune` CLI requires `--legacy-replay` because its proxy split
has known examination overlap. It must not be used as the corrected split.

The v2 run starts from the untouched scalar seed-42 baseline, not a previously
fine-tuned checkpoint. Its 698/174/872 train/validation/test eyes have no
cross-partition examinations under the supplied pairing table. Both arms share
adjudicated validation targets. Test predictions and APTOS retention predictions
are audited separately; see [audited results](../reports/research/messidor2_v2_results.md).

The [GPU runbook](kaggle-gpu-launch.md) covers generic packaging. The v2-specific
packager is `scripts/package_examination_followup.py`; inspect `--help` and the
frozen protocol before use. Re-running is not needed to inspect existing reports.

## Prediction audit and report generation

The initial scalar audit uses `scripts/download_kaggle_audit.py` and
`aptos.reporting.sweep_audit`; see [scientific review](SCIENTIFIC_REVIEW.md).
The completed follow-up uses `aptos.evaluation.followup_audit` with the matching
run directory, frozen split, baseline metadata and pinned source commit. Its
per-eye recomputation requires authorized local execution files; aggregate
reports alone cannot replace them.

```bash
python -m aptos.evaluation.followup_audit --help
python -m aptos.reporting.followup_report \
  --results reports/research/messidor2_v2_results.json \
  --audit reports/research/messidor2_v2_audit.json \
  --output outputs/messidor2_v2_results.md
```

The renderer rejects results without a matching successful audit. Write local
reproduction outputs under ignored `outputs/` before comparing them with
committed evidence. Keep historic reports and frozen plans intact.

## Optional inference illustration

```bash
python -m pip install -e ".[train,serve]"
uvicorn serving.app:app --reload
```

Model artifacts are separate downloads. See [serving/MODEL_CARD.md](../serving/MODEL_CARD.md)
and `serving/fetch_weights.py` for the historical export and its checks. The
scalar ONNX demonstration does not use the newer v2 models. `POST /predict`,
`GET /model-card` and `GET /health` expose research outputs only. Fold-score
spread measures disagreement, not calibrated confidence or diagnostic safety.

The static explorer needs no running inference service; see [PORTFOLIO.md](PORTFOLIO.md).
Do not submit identifiable patient images to a public demonstration.

## Storage and compute boundaries

- `data/`: licensed local datasets and prepared caches.
- `models/`: local training checkpoints and sweep outputs.
- `mlruns/`: local MLflow tracking; reviewable historical tables are in `reports/runs.csv`.
- `outputs/`: execution downloads, temporary packaging and audit reproductions.

These directories are ignored. Do not delete them as documentation cleanup:
they can contain the only local reproduction artifacts. GPU jobs use available
free Kaggle quota, one experiment at a time, with no paid fallback. The project
needs no database or paid API; BigQuery export is optional.
