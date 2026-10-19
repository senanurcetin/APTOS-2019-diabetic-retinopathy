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
