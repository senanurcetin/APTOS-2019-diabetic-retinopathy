# Messidor-2 acquisition and examination audit

Retrieved on **25 October 2026**, using authenticated Kaggle CLI and public source
files only. No screen access, browser automation, new training or paid service.

## Sources and provenance

- [Google Brain adjudicated grades on Kaggle](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades):
  downloaded separately from the image mirror; all 1,744 gradable labels agree
  with the historical split, with zero missing IDs and zero grade mismatches.
- [Preprocessed image mirror on Kaggle](https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess):
  downloaded locally. Its similarly named label file must not overwrite the
  separately preserved Google source. This is already processed imagery, not
  original camera bytes.
- [RFM author-released pairing CSV](https://github.com/ok1zjf/RFM/blob/b8c9322863b5e2d79df964e524d74ad53d1d1b7a/datasets/messidor2_pairs.csv),
  pinned commit `b8c9322863b5e2d79df964e524d74ad53d1d1b7a`.
  Its [README](https://github.com/ok1zjf/RFM/blob/b8c9322863b5e2d79df964e524d74ad53d1d1b7a/README.md)
  attributes the table to the official ADCIS examination pairing spreadsheet.
  This is a third-party transcription; parity with the original spreadsheet
  has **not** been independently established. It contains 874 left/right pairs,
  with 1,748 unique image stems and complete coverage of the 1,744 gradable eyes.
- [ADCIS official provider](https://www.adcis.net/en/third-party/messidor2/):
  examinations are not authenticated longitudinal patient identities. Official
  source terms govern image use separately from this repository's code license.

Raw images and the pairing table stay in ignored `data/`. Only aggregate
counts, hashes and reproduction code are published. Mirror metadata does not
grant a broader image license. No personal-information form or legal agreement
was submitted on the researcher's behalf.

## Result: the historical split has bilateral overlap

| Audit | Result |
|---|---:|
| Historical split eyes | 1,744 |
| Examinations represented | 874 |
| Both eyes included | 870 |
| One gradable eye included | 4 |
| Cross-partition examinations | **7** |
| Train/test examinations | **5** |
| Validation/test examinations | **1** |
| Train/validation examinations | **1** |
| Test eyes with a development partner | **6 / 872** |

Filename proxy groups kept all their members together but did not preserve every
examination. This is direct evidence against an examination-disjoint historical
fine-tuning evaluation under the supplied mapping. Its effect on AUC cannot be
quantified from the aggregate reports. The reported AUC 0.925 remains a historical,
exploratory observation; it is not an independent confirmatory result. The frozen
APTOS model's earlier external failure analysis is separate from this later split.

A cluster bootstrap does not repair training/test overlap. Deleting six test eyes
after seeing outcomes would be a post-hoc sensitivity analysis, not a new unbiased
confirmation. A new examination-grouped split, fixed before training, would be a
new experiment. Longitudinal patient linkage and original-table verification
remain separate limitations.

## Reproduce without a browser

With Kaggle credentials already configured locally:

```bash
kaggle datasets download google-brain/messidor2-dr-grades -p data/external/messidor2/google_grades --unzip
kaggle datasets download mariaherrerot/messidor2preprocess -p data/external/messidor2 --unzip
python scripts/fetch_messidor2_pairs.py
python -m aptos.evaluation.examination_audit --pairs data/external/messidor2/messidor2_pairs.csv --split reports/messidor2_split.csv --output outputs/messidor2_examination_audit.json
```

The downloader verifies the pinned pairing bytes. The auditor rejects duplicate
members, missing mappings, ambiguous IDs and missing partitions; it publishes no
per-eye IDs. Compare [the committed aggregate audit](../reports/research/messidor2_examination_audit.json)
and [source fingerprints](../reports/research/messidor2_sources.json).

## Per-image prediction search

The authenticated Kaggle kernel listing contains two APTOS kernels. The completed
seed-42 training output inventory contains APTOS test predictions, not matching
Messidor-2 fine-tuned predictions. The pinned Hugging Face release contains the
original five ONNX exports and export metadata, not the two fine-tuned arms.
Thus no matched per-eye fine-tuning uncertainty estimate is claimed. These
downloaded images cannot recreate absent fine-tuned weights. No scores or patient
IDs were fabricated, and no new GPU experiment was launched to conceal this gap.
