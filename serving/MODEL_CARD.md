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

# APTOS retinopathy grader

A five-fold EfficientNet-B0 ensemble that grades diabetic retinopathy (ICDRSS
0-4) from colour fundus photographs and makes a referral decision. Trained on
APTOS-2019; exported to ONNX for CPU serving.

The export was checked against the original PyTorch models before publication:
maximum raw-score difference 3.1e-05 on 40 held-out images, zero grade
mismatches. `export.json` records that check alongside the thresholds and the
preprocessing settings the model expects.

> **Not a medical device.** A benchmark model trained on one public dataset,
> with no clinical validation and no regulatory clearance. Do not use it for
> anyone's care.

Source, analysis and every number below:
[senanurcetin/APTOS-2019-diabetic-retinopathy](https://github.com/senanurcetin/APTOS-2019-diabetic-retinopathy).

Live demo, serving these exact weights on a free CPU instance:
[aptos-2019-diabetic-retinopathy.onrender.com](https://aptos-2019-diabetic-retinopathy.onrender.com).
The instance sleeps when idle, so the first request after a pause takes
about a minute.

## What it gets right

| | APTOS test | IDRiD (external) | Messidor-2 (external) |
|---|---|---|---|
| QWK | 0.9091 | 0.8045 | 0.4928 |
| referable ROC AUC | 0.983 | 0.984 | 0.819 |
| referable sensitivity | 0.956 | 0.885 | 0.328 |
| referable specificity | 0.917 | 0.987 | 0.990 |

IDRiD is a different clinic, camera and grading team (both sets are from India),
and all of its images share one resolution - so the acquisition shortcut described below is
unavailable there. No fine-tuning, and the APTOS thresholds were not refitted.
Discrimination transferred essentially intact.

Messidor-2 (France, 1744 images graded by an adjudicating panel of three retina
specialists) is where it stops. A prediction written before the run expected
referable AUC >= 0.93; it came out 0.819. The next section says why.

## What it gets wrong

- **Moderate disease without hard exudates is under-graded.** On Messidor-2,
  83% of Moderate eyes were graded below 2. Moderate eyes with exudates were
  caught; those defined by subtler signs were scored like Mild. The model
  learned where APTOS's single graders draw that line, and a specialist panel
  draws it lower. Read the referral flag as "exudate-level disease or worse".
- **A shortcut exists in the training data.** A classifier given only file
  metadata - resolution, aspect ratio, brightness, file size - reaches QWK 0.652
  on APTOS without looking at the retina, because camera correlates with disease
  prevalence there.
- **The labels are noisy.** Duplicate images carry conflicting grades 29% of the
  time, which puts a single label's accuracy near 84%. The model cannot be more
  right than its labels.
- **The grade drifts on new populations.** On IDRiD the model issued 8 grade-4
  predictions where there were 64. It compresses the top of the scale rather
  than missing disease - 2 of 148 severe cases were called non-referable - but
  the five-way grade should not be trusted outside the training distribution.
  The referral flag held up on IDRiD, but not at the Moderate boundary above.
- **Its probabilities are under-confident elsewhere, and thresholds are local.**
  Calibration error rose from 0.03 on APTOS to 0.12 on IDRiD and 0.15 on
  Messidor-2. A threshold chosen for sensitivity 0.90 on APTOS delivered 0.82 on
  IDRiD, and a threshold refitted on one external set did not transfer to the
  other. Set it on labelled data from the site it will serve.
- **There is no attention map.** Grad-CAM was implemented and measured: five
  identically trained folds disagree about where the model looks as much as
  trained and untrained weights do. A heatmap here would look informative
  without being so, so none is offered.
- **Uploads are preprocessed from source**, while training read JPEG-cached
  images. Raw scores can differ slightly from the offline evaluation; on six
  held-out images every predicted grade still matched.

## API

```
POST /predict       multipart file -> grade, referable, raw score, fold spread
GET  /model-card    the limitations above, as JSON
GET  /health
```

The response includes `fold_spread`, the standard deviation of the five fold
models' scores for that image. A large spread means the folds disagree, and the
prediction deserves less weight.
