# Reproducible ordinal-vs-MSE benchmark (Kaggle-ready)

The research goal is **not** to select the highest APTOS test score. It is
to compare a frozen scalar MSE baseline with a cumulative ordinal model using
identical training partitions, then measure where each model fails on external
retinal populations. Keep all results, including negative results.

## Prerequisites

This repository must be available inside a Python environment with its
training extras installed (`pip install -e ".[train,dev]"`). For Kaggle,
attach your authorized APTOS, IDRiD and adjudicated Messidor-2 datasets as
read-only inputs and build the matching preprocessed cache. Do not add retina
photographs or credentials to this public repository.

This ChatGPT session has **no Kaggle execution connector** and **no BigQuery
access**. Commands below are **run instructions, not completed experiments**.
Do not record benchmark metrics until GPU inference/training has actually run.

## Fixed paired-training protocol

```sh
# Preflight: the prepared cache manifest, labels and dataset versions are fixed.
python -m aptos.pipeline --help

# Same original preprocessing, CV folds, and seed; two different model families.
python -m aptos.training.cv --config configs/cv.yaml --mode reg --seed 42
python -m aptos.training.cv --config configs/cv.yaml --mode ordinal --seed 42
```

Repeat with seeds **42, 43 and 44**, retaining all sweeps and their
`models/cv/baseline-RUN_ID/config.json`, `summary.json`, fold checkpoints and
prediction arrays. Do **not** adapt model parameters to the held-out APTOS test
set. Investigate incomplete folds rather than treating the partial run as a
successful replication.

## External inference (both models, independent datasets)

```sh
python -m aptos.evaluation.external --sweep models/cv/baseline-REG_ID \
  --dataset idrid --mode reg
python -m aptos.evaluation.external --sweep models/cv/baseline-ORDINAL_ID \
  --dataset idrid --mode ordinal
python -m aptos.evaluation.external --sweep models/cv/baseline-REG_ID \
  --dataset messidor2 --mode reg
python -m aptos.evaluation.external --sweep models/cv/baseline-ORDINAL_ID \
  --dataset messidor2 --mode ordinal
```

The command reads saved sweep metadata and validates model mode, architecture,
input size and preprocessing. The original APTOS thresholds remain frozen.
Ordinal research outputs go to `reports/research/`; the public baseline
reports are not overwritten. Existing ONNX serving stays scalar MSE only.

## Generate paired analyses (without retraining)

```sh
python -m aptos.evaluation.compare_sweeps \
  --baseline-sweep models/cv/baseline-REG_ID \
  --ordinal-sweep models/cv/baseline-ORDINAL_ID --cohort aptos \
  --output reports/research/paired_aptos_seed42.json

python -m aptos.evaluation.compare_sweeps \
  --baseline-sweep models/cv/baseline-REG_ID \
  --ordinal-sweep models/cv/baseline-ORDINAL_ID --cohort idrid \
  --output reports/research/paired_idrid_seed42.json

python -m aptos.evaluation.compare_sweeps \
  --baseline-sweep models/cv/baseline-REG_ID \
  --ordinal-sweep models/cv/baseline-ORDINAL_ID --cohort messidor2 \
  --output reports/research/paired_messidor2_seed42.json
```

Comparisons **require the exact same eye IDs and labels**, even if row order
differs. Configuration differences other than training mode or tracking
configuration are rejected, including seed, model architecture, training
budget, split, preprocessing and augmentation. Severity thresholds are taken
from stored APTOS fold validation, **never retuned** on IDRiD/Messidor-2.

The paired analysis provides descriptive QWK and macro F1 differences and
95% percentile-bootstrap intervals for *sensitivity, specificity and referral
rate*. Eye-level resampling assumes independent eyes; use `--patient-map`
only with authentic patient-to-eye linkage (CSV columns id_code,patient_id).
No individual image or patient identifiers appear in the aggregate output.
See `docs/paired-uncertainty.md` for the inferential limitations.

**Acceptance is not an uplift target:** Report external false negatives,
sensitivity/specificity tradeoffs, calibration and any subgroup degradation,
even if the ordinal model scores worse. These experiments are open-source
research, not clinical validation or productization.


## CV provenance v2 and smoke isolation

New sweep IDs capture scientific training/augmentation/threshold/CV/preprocessing
configuration plus a SHA-256 fingerprint of effective labels and resolution
strata. Changes in learning settings or data labels will not silently resume
old folds. Earlier published run IDs remain in committed reports, but
**new sweeps do not reuse old local fold checkpoints**; this is intentional.

Full resumes require aligned test predictions, split metadata, and nonempty
checkpoint files. Incomplete results fail closed; clean only the affected
fold's partial files or explicitly run `--no-resume` to retrain.
`--limit` writes diagnostics only under `models/cv_smoke/` and never
contributes results to full-model `models/cv/`.

This fingerprint verifies labels and manifest settings, **not retinal pixel
bytes**. Data/version provenance remains a separate prerequisite.



## External preprocessing-cache integrity

The IDRiD/Messidor-2 processed-image cache now verifies both preprocessing
settings and the exact expected image ID set before it can be reused. If
either differs, generated JPEGs are invalidated and regenerated **before**
a new manifest is written. Failed-quality-gate images cannot silently retain
previous processed files, and interrupted rebuilds leave no valid manifest.

The cache directory contains **derived files only**; source fundus images are
not deleted. A full rebuild can take time, but reusing old pixels with a new
manifest could invalidate the comparison. Changing source pixel bytes while
retaining the same IDs/settings is not yet content-detected; record and audit
upstream dataset versions/hashes independently.



## Content-addressed external prediction CSVs

`aptos.evaluation.external.score` now stores a `.meta.json` sidecar
alongside its raw score CSV. The signature includes **checkpoint byte hashes**,
saved sweep config, current external labels, the processed image byte hashes,
preprocessing settings and the score CSV's own digest. A missing or stale
sidecar invalidates the score cache and triggers fresh inference; a CSV
cannot silently describe a different model or retinal input.

**Migration:** historical score CSVs without a matching sidecar are
unverified and will be recomputed the first time a compatible trained model
and authorized IDRiD/Messidor-2 images are available. This does not overwrite
committed published aggregate reports until an analyst explicitly runs the
report command. No GPU computation was executed when authoring this change.



## Held-out APTOS eye identity (fold integrity v3)

Every newly trained full/smoke fold now persists the ordered `test_ids`
next to `test_true` and `test_raw`. Resume verifies those IDs one by one.
The paired MSE-versus-ordinal benchmark requires **both v2 provenance and
explicitly aligned image IDs in every APTOS fold**; earlier positional-only
NPZ results cannot be used for strict paired inference.

This matters because a same-grade image permutation can preserve every
ground-truth value while invalidating the paired model comparison. Legacy
aggregate reports remain untouched; regenerate full experimental sweeps
under the new code before comparing their per-image predictions.



## Paired external comparison must verify saved score provenance

The independent `compare_sweeps` tool now verifies **the same SHA-256 input
signature as external inference**, rather than directly trusting an existing
score CSV. Both external datasets require current labels, the exact derived
image cache, full CV fold checkpoints, compatible training/preprocessing
settings and a matching signed CSV sidecar. Changed retinal input pixels,
checkpoint weights or labels invalidate the comparison until the external
score command is rerun.

The benchmark stays **read-only**: unlike the inference command, it never
rebuilds images or reruns expensive model inference automatically. Old scores
with no matching provenance sidecar cannot support a new paired analysis.
Use the configured `APTOS_ROOT/data/external` directory (the default);
`--external-dir` must resolve to that same directory.



## Original external retinal image provenance

Derived IDRiD/Messidor-2 JPEGs additionally carry an
`_source_provenance.json` record with SHA-256 hashes of **original source
image bytes**, preprocessing implementation, JPEG encoding quality and
configuration. Reusing the cache verifies these values. Replacing a raw
fundus photograph without renaming it now forces a derived-image rebuild.
Source files are never deleted. The verification reads all original image
bytes, so expect extra I/O for large external cohorts.

The separate record is intentional: published APTOS processed-image
manifests retain their historical schema and are not retroactively changed.
Old external derived caches without this record must be rebuilt with the
authorized original datasets before further comparisons.



## External ensemble completeness and checkpoint integrity

External inference now refuses an **incomplete CV run**, including missing
summary.json, missing/misnumbered fold weights, or incomplete fold metadata
and prediction artifacts. The recorded fold count must agree with the
training config; two or more completed folds are required. Loading checks
each checkpoint's fold identity and four ordered validation thresholds
against the APTOS ensemble thresholds in the saved summary.

PyTorch checkpoint loading uses `weights_only=True`: a CV checkpoint needs
only tensor state and primitive metadata, not arbitrary Python deserialization.
Legacy folders without a completed CV summary cannot be silently treated as
complete ensembles; reconstruct documented provenance or rerun authorized
research training. This does not change historical committed benchmark numbers.


## Paired grading-metric uncertainty (QWK and macro F1)

The MSE-versus-ordinal comparison also reports descriptive **95% paired
percentile-bootstrap intervals for QWK and fixed-five-grade macro F1
differences**, in `paired_grading_uncertainty`. The intervals use the same
image alignment and explicitly declared eye/patient resampling unit as
referral-metric uncertainty. With authentic patient linkage, both eyes of
a patient are sampled as a unit; without it, eye-level intervals assume
independence and may understate uncertainty. Neither option synthesizes
missing patient IDs.

QWK is undefined on certain resamples with degenerate grade variation;
those QWK draws are excluded and their count is recorded separately.
If fewer than 80% yield defined QWK, its CI is returned as JSON null rather
than claiming false precision. Macro F1 uses a **fixed 0–4 grade label
space** across all bootstrap draws, including draws that omit rare classes.

These are descriptive uncertainty estimates for the *already observed*
test cohorts. They do not account for stochastic training variability,
dataset selection, multiple comparisons, site shift or repeated tuning on
the same held-out test set. They provide **no clinical safety guarantee**.
No new model or retinal dataset was trained/scored when implementing this.
