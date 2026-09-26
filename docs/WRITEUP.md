# Two predictions, one failure each way

*What a diabetic retinopathy model learned, what it didn't, and how the project
found out. September 2026.*

The APTOS 2019 dataset holds 3662 retinal photographs graded 0-4 for diabetic
retinopathy. A five-fold EfficientNet-B0 ensemble trained on it reaches a
quadratic weighted kappa of 0.909 on held-out test images. Public solutions
reach 0.93. That number is the least interesting thing in this project, and this
write-up is about why.

## 1. The model could score well without looking at the eye

Before training anything, a random forest was given only file metadata:
resolution, aspect ratio, brightness, file size. It never saw a retinal pixel,
and it reached **QWK 0.652**.

The cause is that APTOS was collected at several sites with different cameras,
and camera correlates with disease. 92.5% of the 1050x1050 images are graded
healthy, against 33.6% at every other resolution. Resizing does not remove the
signal; aspect ratio and sharpness survive it.

So a score of 0.9 has a floor of 0.65 that requires no medicine at all. That
reframed every later question from "how high is the score?" to "how much of it
is the retina?"

## 2. Two sensible ideas did nothing

Two preprocessing choices had strong arguments behind them: contrast-limited
histogram equalisation (CLAHE), a staple of retinal imaging, and squashing
images to square rather than padding them, which keeps 21% more retinal pixels.
Both were tested with five-fold cross-validation, paired across folds:

| comparison | test QWK difference | p |
|---|---|---|
| squash vs baseline | +0.0038 | 0.47 |
| CLAHE vs baseline | -0.0005 | 0.87 |

Both are null. The largest effect found anywhere in the project was not a
preprocessing choice but averaging the five fold models, worth +0.019 QWK - four
times the largest gap between preprocessing variants.

## 3. The first external test: a prediction that failed in the model's favour

The shortcut cannot be removed within APTOS; excluding one resolution leaves
sixteen others with their own label priors. It can be removed by changing
dataset. **IDRiD** (455 images from another Indian clinic) was captured at one
resolution only, so geometry carries no information there. Better still, it
inverts the shortcut: in APTOS every image at IDRiD's resolution is diseased; in
IDRiD 28% are healthy.

Before running it, a prediction was committed to the repository: if the model
leans on acquisition cues, it will call IDRiD's healthy eyes diseased and
specificity will collapse.

It did the opposite. With no fine-tuning and APTOS thresholds unchanged:

| | APTOS test | IDRiD |
|---|---|---|
| referable ROC AUC | 0.983 | 0.984 |
| referable specificity | 0.917 | **0.987** |

Specificity went *up*. On the one dataset built to expose a shortcut, the model
ranked patients exactly as well as at home. It reads the retina.

What did move was calibration: scores shifted down, so a cut chosen for 90%
sensitivity on APTOS gave 82% on IDRiD. The model was not confused about who
was sick; it was too cautious about saying so.

## 4. The second external test: a prediction that failed against the model

One external set can be luck. **Messidor-2** was added: 1744 images from France,
and - the part that turned out to matter - graded not by one reader but by a
panel of three retina specialists who adjudicated their disagreements.

Four predictions were pushed to GitHub before a single image was downloaded. The
main one said referable AUC would be at least 0.93. It came out **0.819**.

The failure has one address. Median ensemble score by true grade:

| true grade | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| APTOS test | 0.03 | 1.31 | 1.93 | 2.47 | 2.73 |
| IDRiD | 0.42 | 0.60 | 1.68 | 2.21 | 2.60 |
| Messidor-2 | 0.22 | 0.28 | **0.54** | 1.79 | 2.10 |

Severe and proliferative disease are still recognised. Moderate disease -
grade 2, the referral boundary - is scored like mild: 83% of Moderate eyes are
graded below 2, against 21% on IDRiD.

The adjudicated labels include a flag for macular oedema, marked by hard
exudates. Moderate eyes with that flag score 1.02; Moderate eyes without it
score 0.46. The model finds moderate disease when there are exudates to see and
misses it when the signs are subtler.

The most likely reading - and it is a reading, made after the fact - is the
labels. APTOS is graded by single readers, and its duplicate images disagree
with themselves 29% of the time. A panel of specialists catches subtle findings
single readers miss. A model trained on single-reader labels learns where a
single reader draws the line between Mild and Moderate, and the panel draws it
lower.

## 5. Recalibrating does not travel

Calibration was the documented gap: fixing it needs labelled data from the site
where the model is used. With two external sets, one can stand in for "a new
site with some labels" and the other for "the next site".

Refitting the probability calibration on one external set did lower calibration
error on the other, in both directions. The decision threshold did not travel.
A cut chosen to reach 90% sensitivity on Messidor-2 turned IDRiD's specificity
from 1.00 into 0.13. A deployment would have to set its threshold on its own
labelled data, site by site.

## What this adds up to

- **The shortcut was real, and the model does not depend on it.** The IDRiD
  test was designed to catch exactly that and found the opposite.
- **The model inherits its labels.** It reads the retina the way APTOS's single
  graders did, and an adjudicated reference standard exposes the difference at
  the one boundary that decides referral.
- **The ceiling on this project is label quality, not architecture.** The next
  step is not a bigger network; it is training on better grades.

Two pre-registered predictions, one falsified in the model's favour and one
against it. Both results were more informative than a higher kappa would have
been, and neither could have been claimed honestly without writing the
prediction down first.

## Method notes

- Every number above comes from reports generated by the pipeline
  (`python -m aptos.pipeline`), recorded in MLflow and exported to
  `reports/runs.csv`. Details and caveats: [RESULTS.md](../RESULTS.md).
- The predictions and their outcomes, in the order they were written:
  [IDRiD](external-validation-prediction.md) and
  [Messidor-2](second-external-validation-prediction.md).
- The model is served, with these limitations attached, at
  [aptos-2019-diabetic-retinopathy.onrender.com](https://aptos-2019-diabetic-retinopathy.onrender.com).
  It is not a medical device.
