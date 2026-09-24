---
title: APTOS Retinopathy Grader
emoji: 👁️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Diabetic retinopathy grading, with its limits stated
---

# APTOS retinopathy grader

Upload a colour fundus photograph and get an ICDRSS grade (0-4) and a referral
decision, from a five-fold EfficientNet-B0 ensemble trained on APTOS-2019.

> **Not a medical device.** A benchmark model trained on one public dataset,
> with no clinical validation and no regulatory clearance. Do not use it for
> anyone's care.

Source, analysis and every number below:
[senanurcetin/APTOS-2019-diabetic-retinopathy](https://github.com/senanurcetin/APTOS-2019-diabetic-retinopathy).

## What it gets right

| | APTOS test | IDRiD (external) |
|---|---|---|
| QWK | 0.9091 | 0.8045 |
| referable ROC AUC | 0.983 | 0.984 |
| referable sensitivity | 0.956 | 0.885 |
| referable specificity | 0.917 | 0.987 |

IDRiD is a different camera, country and grading team, and all of its images
share one resolution - so the acquisition shortcut described below is
unavailable there. No fine-tuning, and the APTOS thresholds were not refitted.
Discrimination transferred essentially intact.

## What it gets wrong

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
  **The referral flag is the output that held up.**
- **Its probabilities are under-confident elsewhere.** Calibration error rose
  from 0.03 on APTOS to 0.12 on IDRiD. A threshold chosen for sensitivity 0.90 on
  APTOS delivered 0.82 on IDRiD.
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
