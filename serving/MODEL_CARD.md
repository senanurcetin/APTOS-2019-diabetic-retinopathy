---
license: mit
library_name: onnx
pipeline_tag: image-classification
tags:
  - medical-imaging
  - diabetic-retinopathy
  - fundus
  - efficientnet
  - onnx
datasets:
  - aptos2019
---

# APTOS scalar ensemble: model card

This card describes the historical five-fold EfficientNet-B0 scalar regression
ensemble, `baseline-6e66147526`, exported to ONNX for a research illustration.
It does not describe the newer scalar seed-42 sweep or either v2 fine-tuned arm.
No fine-tuned model was promoted.

**Research only.** No prospective clinical utility, regulatory clearance or
clinical safety has been established. Predictions must not guide patient care.

## Intended use and model

The model maps color fundus photographs to an ordinal severity score and four
validation-fitted thresholds for grades 0–4. The demonstration also reports an
experimental referral flag for predicted grade ≥ 2. This flag differs from the
separately selected operating point in the calibration report.

Training uses APTOS-2019, ImageNet initialization and 384px inputs. Shared
preprocessing crops the retinal field, squares and resizes the image, then
normalizes it for the ensemble. The source and experiment record are in the
[repository](https://github.com/senanurcetin/APTOS-2019-diabetic-retinopathy).

## Recorded evaluation

The external columns use unchanged APTOS grade thresholds. These are historical
results on dataset mirrors, not prospective screening estimates.

| Metric | APTOS test | IDRiD | Messidor-2 |
|---|---:|---:|---:|
| QWK | 0.9091 | 0.8045 | 0.4928 |
| Referable ROC AUC | 0.983 | 0.984 | 0.819 |
| Referable sensitivity | 0.956 | 0.885 | 0.328 |
| Referable specificity | 0.917 | 0.987 | 0.990 |

APTOS has 366 test eyes; the external mirrors contain 455 IDRiD and 1,744
Messidor-2 images. Messidor-2 includes 457 referable eyes, of which 307 were
missed by the original grade decision. High IDRiD performance does not establish
transferability to another population or grading standard.

## Limitations

- **Under-grading:** 83% of historical Messidor grade-2 eyes were predicted below
  grade 2. The observed hard-exudate pattern supports a grading-target hypothesis;
  it is not a validated lesion detector or definitive causal mechanism.
- **Dataset shortcuts:** low-level file/image summaries reach historical QWK
  0.652. Brightness derives from pixels; this is not a pixel-free test or a
  mathematical performance floor.
- **Label disagreement:** 43 of 148 selected same-image pairs carry conflicting
  labels. This does not establish an 84% population accuracy ceiling.
- **Selection and identity:** the APTOS holdout has been inspected repeatedly.
  Demographic fairness and longitudinal patient independence are unverified.
- **Calibration:** thresholds and score calibration shift across populations.
  Site-specific fitting requires a separate evaluation cohort; it does not
  establish usefulness at another site.
- **Explanations:** the recorded Grad-CAM controls do not support reliable lesion
  localization. Nonsignificance does not prove equivalence between maps.
- **Preprocessing parity:** training used cached JPEGs; upload-time preprocessing
  can move scores near a grade threshold. Recorded export parity does not
  guarantee identical predictions for every decoding path.

The historical fine-tuning result (AUC 0.925) has six test/development
examination overlaps. The separate examination-disjoint v2 experiment reaches
AUC 0.9435 but uses a different baseline and loses APTOS accuracy (80.6% to
64.2%). Neither result replaces this model's evaluation. See
[audited v2 results](../reports/research/messidor2_v2_results.md) and the
[scientific review](../docs/SCIENTIFIC_REVIEW.md).

## Export and inference illustration

The historical ONNX export recorded a maximum score difference of 3.1e-05
against PyTorch on 40 images, with zero grade mismatches. Its `export.json`
contains thresholds and preprocessing settings. This limited check is specific
to that export; the newer v2 baseline re-score is a separate comparison.

```text
POST /predict       multipart image → grade, referral flag, score, fold spread
GET  /model-card    structured model metadata and limitations
GET  /health        service health
```

Fold spread is the standard deviation of model scores, not calibrated confidence.
The [reproduction guide](../docs/REPRODUCIBILITY.md) describes optional local
serving. The static [NEXUS explorer](https://senanurcetin.github.io/APTOS-2019-diabetic-retinopathy/)
needs no model or uploads. Do not send identifiable retinal images to a public
demonstration. Code licensing does not grant rights to the underlying datasets;
see [NOTICE](../NOTICE).
