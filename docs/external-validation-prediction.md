# External validation on IDRiD: a prediction, written before the experiment

**Written 24 September 2026, before any IDRiD image has been passed through the
model.** The point of writing it now is that a prediction made afterwards
explains any outcome. This one can be wrong.

## Why this project needs an external test

The repository already measures a shortcut: a RandomForest trained only on file
metadata - width, height, aspect ratio, megapixels, brightness, contrast, file
size - reaches **QWK 0.652** on APTOS without seeing a retinal pixel. The cause
is that APTOS was collected across sites with different cameras, and the device
signature correlates with disease prevalence.

Measuring that is where the project currently stops. It reports the headline
QWK 0.90 against that 0.652 floor and says the floor exists. What it has never
done is ask the question the floor raises: **does the model still work when the
shortcut is taken away?**

IDRiD answers that, because of a property of the dataset that turns out to be
unusually clean.

## What makes IDRiD the right test

Measured from the downloaded data, not assumed:

- 455 images, 455 labels, **455/455 matching on `id_code`**.
- The same ICDRSS 0-4 grading scale and the same column names as APTOS.
- **Every single image is 4288x2848.** There is exactly one resolution in the
  whole dataset.

That last point is what matters. In IDRiD the acquisition shortcut is not merely
weaker - it is *structurally unavailable*. A metadata-only classifier cannot
exceed the majority-class baseline, because every image has identical geometry.

## The shortcut is not absent here. It is inverted.

This is the part that makes the test sharp rather than merely fair.

| | APTOS | IDRiD |
|---|---|---|
| images at 4288x2848 | 52 (1.4%) | **455 (100%)** |
| of those, share graded No DR | **0.0%** | **28.4%** |

In APTOS, all 52 images at this resolution are diseased - grades 1, 2, 3 and 4,
not one grade 0. Any model that picked up acquisition cues has learned, on this
geometry, *"this camera means disease"*.

In IDRiD, 129 of the 455 images at exactly that geometry are healthy eyes.

So the shortcut does not just stop helping. It actively points the wrong way.

Resizing does not dispose of this. Every image is squared and resized to 512px
before training, so the model never sees raw pixel dimensions - but the
confound analysis already found that aspect ratio and sharpness carry the
signal through resizing. A 4288x2848 frame has aspect 1.505 and a 1050x1050
frame has 1.000, so the two are compressed differently by the squaring step and
arrive at the model carrying different distortion signatures.

## The prediction

If the APTOS model is substantially exploiting acquisition cues, then on IDRiD:

1. **Specificity collapses.** The model calls healthy eyes diseased, because
   everything it sees has the geometry that meant disease in training. Expect
   referable specificity well below the APTOS figure, concentrated in the 129
   grade-0 cases.
2. **Sensitivity holds up or rises**, for the same wrong reason - over-calling
   disease is cheap when almost everything is being called diseased.
3. **QWK drops sharply**, more than distribution shift alone would explain.

If instead the model is largely reading the retina:

1. Specificity degrades modestly and in proportion to the distribution shift.
2. QWK drops - IDRiD is a different population and only 28% No DR against
   APTOS's 49%, so some drop is expected regardless - but stays far above the
   metadata floor.

**The discriminating measurement is specificity on the 129 grade-0 images, not
the headline QWK.** A QWK drop alone is ambiguous: it is consistent with both
stories. Grade-0 specificity is not.

## What this cannot settle

Stated now, so it is not quietly dropped later if the result is flattering.

- **IDRiD differs from APTOS in more than one way at once.** Camera, country,
  grading team and class balance all change together. A drop cannot be
  attributed to the confound alone; it can only be attributed to "everything
  that differs between these two datasets". The inverted-prior argument above
  makes the confound the *leading* candidate, not the proven one.
- **455 is not the full IDRiD.** This is a Kaggle mirror. The official IEEE
  DataPort distribution of the disease-grading subset is larger, and the
  relationship between the two should be checked before any count is published.
- **No fine-tuning is involved**, by design, but that also means nothing here
  speaks to how well the model would transfer *with* adaptation. It measures
  transfer, not adaptability.
- **129 grade-0 cases is a small denominator.** A specificity computed on it
  moves by 0.008 per image, so the figure should be reported with that in mind
  rather than to three decimal places.

## How it will be run

- The APTOS cross-validation fold ensemble, with **no fine-tuning and no
  threshold refitting** - the thresholds stay exactly as fitted on APTOS
  validation. Refitting them on IDRiD would answer a different and much easier
  question.
- IDRiD images go through the identical preprocessing pipeline and the same
  512px cache format, so the only thing that changes is the images.
- Reported: QWK, accuracy, macro F1, per-class recall, and the referable-DR
  sensitivity/specificity pair - against the APTOS test figures and against the
  0.652 metadata floor.

---

# Outcome, 24 September 2026

**The prediction was wrong.** Nothing above has been edited; this section was
appended after the measurement.

The baseline five-fold ensemble, no fine-tuning, APTOS thresholds unchanged:

| metric | APTOS test | IDRiD |
|---|---|---|
| QWK | 0.9091 | 0.8045 |
| accuracy | 0.8033 | 0.5868 |
| macro F1 | 0.5450 | 0.4609 |
| referable sensitivity | 0.956 | 0.885 |
| **referable specificity** | **0.917** | **0.987** |

The discriminating measurement was specificity on the 129 healthy eyes, and it
went the opposite way to the prediction. The shortcut story required the model
to over-call disease on IDRiD, because in APTOS every image at 4288x2848 was
diseased. Instead **1.6% of healthy eyes were called referable** and 82.9% were
graded healthy outright. Specificity is *higher* on IDRiD than on APTOS.

So the model did not carry the acquisition prior across. On a population where
geometry says nothing, where the prior attached to that geometry is inverted
relative to training, and with no adaptation of any kind, it holds QWK 0.8045.
That is well above the 0.652 metadata floor measured on APTOS - and on IDRiD
the equivalent floor is lower still, because a single resolution leaves file
geometry with no signal to give.

This is the strongest evidence in the project that the model reads the retina.

## What did degrade

Transfer is not free, and the failure has a clear shape.

| grade | n | recall | predicted |
|---|---|---|---|
| 0 No DR | 129 | 0.829 | 130 |
| 1 Mild | 22 | 0.364 | 54 |
| 2 Moderate | 156 | 0.705 | 179 |
| 3 Severe | 84 | 0.417 | 84 |
| 4 Proliferative | 64 | **0.109** | **8** |

The model issues 8 grade-4 predictions where there are 64 true cases. But
missed referrals - true grade >= 3 called <= 1 - are only **2 of 148**. It is
not failing to see the disease; it is compressing the top of the scale, calling
proliferative cases severe or moderate.

That is calibration drift under covariate shift, not a detection failure, and
it is exactly what fixed thresholds carried across datasets would be expected to
produce. It also argues that the referable-DR framing is the transferable one:
the binary decision survives the shift (specificity 0.987, sensitivity 0.885)
while the five-way grade does not.

## The squash ensemble, for comparison

QWK 0.7859, referable sensitivity 0.862, specificity 0.993, 3 of 148 severe
cases missed. Marginally worse than baseline on everything except specificity -
consistent with squash being the null result cross-validation found it to be.

## What still cannot be claimed

The caveats written before the experiment stand, and the favourable result does
not retire them:

- IDRiD differs from APTOS in camera, country, grading team and class balance
  simultaneously. The result shows transfer survived; it does not isolate
  *which* difference the model was robust to.
- 129 healthy eyes is a small denominator. Specificity moves by 0.008 per image.
- 455 images is a Kaggle mirror, not the full official IDRiD distribution.
- No fine-tuning was involved by design, so nothing here speaks to how well the
  model would adapt if it were allowed to.
