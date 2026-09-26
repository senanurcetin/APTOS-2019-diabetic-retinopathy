# Results

Every number here was produced by the code in this repository. Runs are tracked
in MLflow and exported as text to `reports/runs.csv`; the six single-split runs
predate that layer and were transcribed from the original BigQuery tables.

Setup for all runs: EfficientNet-B0 (ImageNet weights), 384px input, batch 16,
15 epochs, AdamW at 3e-4, cosine schedule, mixed precision, early stopping on
validation QWK (patience 5), best checkpoint kept. Regression mode with four
thresholds tuned on validation.

---

## Headline

| | |
|---|---|
| Best test QWK | **0.9098** — 5-fold squash ensemble |
| Baseline, 5-fold CV | 0.8902 ± 0.0086 per fold; ensemble **0.9091** |
| Metadata-only floor | **QWK 0.652** — without looking at the retina |
| Estimated label ceiling | **~84%** single-label accuracy |
| Preprocessing ideas tested | **2 of 2 came back null** |
| External validation (IDRiD) | **QWK 0.8045**, referable specificity **0.987** |
| Second external set (Messidor-2, adjudicated labels) | referable AUC **0.819**; pre-registered prediction **failed** |
| Fine-tuned on adjudicated labels (Messidor-2 test half) | AUC 0.830 -> **0.925**; a same-image control gains nothing |

The score is not the interesting part of this project. Public APTOS solutions
reach 0.93+. What follows — the shortcut floor, the label ceiling, and two
preprocessing techniques that did not survive measurement — is.

The largest effect found anywhere in this project is not a preprocessing
choice. It is **ensembling the five folds: +0.019 QWK** on the baseline, about
four times the biggest gap between any two preprocessing variants (0.0044).
The gain is smaller elsewhere - +0.016 for squash, +0.007 for clahe - so this is
a statement about the baseline, not a constant.

---

## Six training runs

Identical settings; only the seed and the preprocessing variant differ.

| variant | seed | valid QWK | test QWK | test acc | test macro F1 |
|---|---|---|---|---|---|
| baseline | 42 | 0.8980 | 0.8960 | 0.7732 | 0.5680 |
| baseline | 43 | 0.8968 | 0.9024 | 0.7760 | 0.5777 |
| baseline | 44 | 0.8927 | 0.8976 | 0.8005 | 0.5919 |
| clahe | 42 | 0.9059 | 0.9112 | 0.8197 | 0.6060 |
| clahe | 43 | 0.9020 | 0.8823 | 0.7787 | 0.5439 |
| clahe | 44 | 0.9054 | 0.8927 | 0.7787 | 0.4850 |

Mean ± standard deviation:

| variant | valid QWK | test QWK | test accuracy |
|---|---|---|---|
| baseline | 0.8958 ± 0.0028 | 0.8986 ± 0.0033 | 0.7832 ± 0.0150 |
| clahe | 0.9045 ± 0.0021 | 0.8954 ± 0.0146 | 0.7923 ± 0.0237 |

### Rerunning seed 42, a year on

The trainer was moved into the package (`aptos.training.single`) with its
behaviour meant to be unchanged. On 25 September 2026 baseline seed 42 was rerun
with the same settings as the recorded run: no leak exclusion, 15 epochs, 384px,
batch 16. The one forced change was `--workers 0` instead of 2, because another
heavy job shared the machine and worker processes are what exhaust the Windows
commit limit.

| run | valid QWK | test QWK | test acc | test macro F1 |
|---|---|---|---|---|
| recorded (original `train.py`) | 0.8980 | 0.8960 | 0.7732 | 0.5680 |
| ported trainer, 2026-09-25 | 0.8919 | **0.8853** | 0.7978 | 0.5477 |
| original `train.py` from git, 2026-09-25 | 0.8919 | **0.8853** | 0.7978 | 0.5477 |

The rerun lands 0.011 below the recorded run, about three standard deviations
of the three-seed spread above. That could have been a porting error, so the
pre-port `train.py` was checked out of git history and run under the same
conditions. It matches the ported trainer **exactly**: every epoch's training
loss, validation loss and validation QWK agree to four decimals, and so do the
final metrics. The port is faithful.

The 0.011 therefore comes from the run environment, not the code: the worker
count changes the order in which augmentation draws random numbers, and the
library stack has moved on since the recorded runs. Two conclusions:

- **Three seeds understated the noise.** The recorded seed spread (±0.0033)
  implied test QWK was pinned to about ±0.003. The same code and seed moved by
  0.011 across environments, which is in line with the ±0.0086 spread between
  cross-validation folds on this same test set. Differences between single runs
  smaller than about 0.01 should not be read as effects.
- **An exact numeric reproduction needs the environment as well as the seed.**
  The code is reproducible - two implementations agree bit for bit - but the
  recorded numbers are tied to a worker count and library versions that were
  not pinned when they were produced.

### QWK 0.90 but macro F1 0.57

The gap is not noise. Errors land on neighbouring grades rather than far away,
and QWK penalises neighbouring mistakes lightly, so it stays high. Macro F1
weights every class equally and exposes the weakness on the minority classes.
Reporting QWK alone hides it.

Per-class recall on test (baseline seed 42 / CLAHE seed 42):

| grade | n | baseline | clahe |
|---|---|---|---|
| 0 No DR | 199 | 0.955 | 0.980 |
| 1 Mild | 30 | 0.233 | 0.333 |
| 2 Moderate | 87 | 0.724 | 0.862 |
| 3 Severe | 17 | 0.588 | 0.353 |
| 4 Proliferative DR | 33 | 0.394 | 0.424 |

`Severe` has 17 test images. A single case moves its recall by 0.06, so these
per-class test figures are indicative, not precise.

Clinically costly errors — true grade >= 3 predicted as <= 1 — were 2 of 39 for
baseline and 1 of 39 for CLAHE.

---

## Five-fold cross-validation

Completed 24 September 2026, on the fourth attempt. The pool is train + valid,
3247 images after excluding the 49 leaked ids; the 366-image test split is held
out of every fold. Stratified on diagnosis, seed 42, otherwise identical
settings to the single-split runs. The same fold boundaries are used for all
three variants, so the comparisons below are paired.

Per-fold **test** QWK:

| fold | baseline | squash | clahe |
|---|---|---|---|
| 1 | 0.8860 | 0.8805 | 0.8826 |
| 2 | 0.8882 | 0.8938 | 0.8937 |
| 3 | 0.8936 | 0.8920 | 0.8923 |
| 4 | 0.8802 | 0.9018 | 0.8866 |
| 5 | 0.9030 | 0.9021 | 0.8932 |

| variant | CV valid QWK | CV test QWK | test acc | test macro F1 |
|---|---|---|---|---|
| baseline | 0.8951 ± 0.0073 | 0.8902 ± 0.0086 | 0.7732 ± 0.0086 | 0.5230 ± 0.0056 |
| squash | 0.8914 ± 0.0085 | 0.8940 ± 0.0088 | 0.7672 ± 0.0089 | 0.5257 ± 0.0172 |
| clahe | 0.8945 ± 0.0101 | 0.8897 ± 0.0049 | 0.7814 ± 0.0256 | 0.5441 ± 0.0334 |

Fold ensembles — averaging the five folds' raw outputs and their thresholds:

| variant | QWK | accuracy | macro F1 | referable sens. | referable spec. | severe missed |
|---|---|---|---|---|---|---|
| baseline | 0.9091 | 0.8033 | 0.5450 | 0.956 | 0.917 | 1/50 |
| **squash** | **0.9098** | 0.8033 | 0.5697 | 0.971 | 0.917 | **0/50** |
| clahe | 0.8965 | 0.8142 | 0.5808 | 0.971 | 0.917 | 1/50 |

### What the folds say

**Ensembling is worth more than any preprocessing decision.** Baseline goes from
a 0.8902 fold mean to 0.9091 as an ensemble: **+0.019**. The largest difference
between any two preprocessing variants is 0.0044. The preprocessing argument
this project spent most of its effort on was being conducted inside the noise of
a much larger effect sitting untouched next to it.

**Three seeds on one split understated the variance.** The baseline spread was
±0.0033 across seeds and is ±0.0086 across folds — 2.6x wider. Part of that is
real fold-to-fold variation the single split could not see, and part is that
each CV model trains on 2597 images against the single split's 2881, so the
folds are slightly weaker models. Both push the same way: the single-split
figure looked more precise than it was.

**The clinical numbers are strong and stable.** Referable sensitivity 0.956-0.971
at specificity 0.917, identical across all three variants. The squash ensemble
misses **none** of the 50 referable-severe cases in the test split. That is the
number a screening programme would care about, and it is considerably more
reassuring than macro F1 0.55.

### Statistical comparison, paired across folds

| comparison | mean Δ QWK | paired t | Wilcoxon | folds won |
|---|---|---|---|---|
| squash − baseline | **+0.0038** ± 0.0107 | p = 0.468 | p = 0.812 | 2/5 |
| clahe − baseline | **−0.0005** ± 0.0067 | p = 0.865 | p = 1.000 | 2/5 |
| clahe − squash | −0.0044 ± 0.0074 | p = 0.259 | p = 0.625 | 2/5 |

Neither preprocessing variant beats the control. Both are null results.

---

## squash: a sound argument that produced nothing

This is the gap the repository previously listed as unmeasured, and it now has
an answer.

The geometric case for `squash` was measured, not assumed: padding leaves 28.5%
of a 512px output black against squash's 13.4%, so squash carries **21% more
effective retina pixels**. Nothing about that measurement was wrong.

It bought **+0.0038 QWK, p = 0.468, winning 2 of 5 folds** — indistinguishable
from noise.

The README already warned, about CLAHE, that a sound argument is not a result.
squash is the second instance, and a cleaner one: the argument here was
quantitative and correct about the pixels, and the pixels turned out not to be
the binding constraint. Whatever limits this model, it is not how much of the
frame is black.

One thing does separate the two on a metric nobody was optimising: the squash
ensemble misses 0 of 50 severe cases against baseline's 1. On a denominator of
50 that is one image, and no conclusion should be hung on it.

---

## CLAHE did not replicate

The first run suggested CLAHE improved test QWK by +0.0152. Repeating across
three seeds did not confirm it.

| seed | valid QWK | test QWK | test accuracy |
|---|---|---|---|
| 42 | +0.0079 | **+0.0152** | +0.0464 |
| 43 | +0.0053 | **−0.0201** | +0.0027 |
| 44 | +0.0127 | **−0.0049** | −0.0219 |

| measure | mean difference | paired t-test | direction |
|---|---|---|---|
| valid QWK | +0.0086 ± 0.0038 | p = 0.058 | CLAHE 3/3 |
| test QWK | −0.0033 ± 0.0177 | p = 0.780 | mixed, 1/3 |
| test accuracy | +0.0091 ± 0.0346 | p = 0.693 | mixed, 2/3 |

**No benefit on test could be demonstrated.** The first result was that seed's
luck. CLAHE also made results less stable: its test QWK spread is 4.4x the
baseline's (0.0146 vs 0.0033).

CLAHE wins on validation in all three seeds, but does not carry to test. In
regression mode the thresholds are tuned *on validation*, so validation QWK is
partly a number we fitted to. CLAHE letting the model fit validation better,
without generalising, is the most likely reading — and a concrete illustration
of why model selection belongs on validation and test gets looked at once.

A parameter sweep over `clip_limit` (1-4) and channel choice (LAB-L vs green)
using training-free proxy measures was **inconclusive**: the
visibility-to-noise ratio sat at ~1.0 for every setting, because CLAHE
amplifies edge signal and high-frequency noise proportionally. The only usable
finding was calibration: LAB-L is markedly more aggressive than the green
channel at the same clip limit (2.32x vs 1.81x gain at clip=2).

---

### Cross-validation confirms it

The single-split result rested on three seeds with a test spread of ±0.0177.
Cross-validation is an independent design on a larger evaluation pool, and it
agrees:

| design | mean Δ test QWK (clahe − baseline) | spread | p |
|---|---|---|---|
| 3 seeds, single split | −0.0033 | ± 0.0177 | 0.780 |
| 5 folds, paired | **−0.0005** | ± 0.0067 | 0.865 |

The effect estimate moves towards zero and its spread narrows by 2.6x. Two
designs that could have disagreed did not. CLAHE at clip=2.0 on the LAB
lightness channel does nothing measurable for this task, and that is now about
as settled as this dataset can make it.

The validation-side advantage vanished as well. In the single-split runs CLAHE
won on validation in all three seeds, which was the strongest thing that could
be said for it. Under cross-validation it does not even do that: validation QWK
0.8945 against baseline's 0.8951. Thresholds are fitted on validation, so a
validation win is partly a quantity being optimised rather than a result — and
with a larger evaluation pool, this one stopped appearing at all.

---

## The metadata shortcut

A RandomForest trained **only on file properties** — resolution, aspect ratio,
brightness, contrast, file size — never seeing a single pixel of retina.
Five-fold cross-validation.

| measure | metadata only | always predict 0 | the model (5-fold ensemble) |
|---|---|---|---|
| QWK | **0.652** | 0.000 | 0.9091 |
| Accuracy | 0.708 | 0.493 | 0.8033 |

Where it comes from:

| grade | 1050x1050 (n=974) | every other resolution (n=2688) |
|---|---|---|
| 0 No DR | 901 (**92.5%**) | 904 (33.6%) |
| 1 Mild | 19 (2.0%) | 351 (13.1%) |
| 2 Moderate | 39 (4.0%) | 960 (35.7%) |
| 3 Severe | 2 (0.2%) | 191 (7.1%) |
| 4 Proliferative DR | 13 (1.3%) | 282 (10.5%) |

APTOS data was collected across several sites with different cameras.
Resolution is the device's signature, and device correlates with disease
prevalence. Even after resizing to 512px, aspect ratio, sharpness and edge
geometry still carry it. Kruskal-Wallis confirms the classes differ
significantly on brightness (p=7e-5), contrast (p<1e-5) and megapixels
(p<1e-5).

This does not invalidate the results, but **QWK 0.652 is reachable with no
diagnosis at all**, and any reported score should be read against that floor.

---

## The shortcut, tested rather than reported

A metadata-only classifier reaches QWK 0.652 on APTOS. Until now that floor was
reported alongside the headline and left there. Two measurements now ask whether
the score survives without it.

### Within APTOS: narrowed, not removed

| stratum | n | metadata QWK | model QWK | model acc | majority acc |
|---|---|---|---|---|---|
| ALL | 3662 / 366 test | 0.6519 | 0.9091 | 0.8033 | 0.5437 |
| 1050x1050 | 974 / 109 test | 0.3454 | 0.9357 | 0.9817 | 0.9633 |
| other | 2688 / 257 test | 0.5669 | 0.8766 | 0.7276 | 0.3658 |

In both strata the model beats file properties by a wide margin, so it is not
simply re-deriving the shortcut. But this analysis cannot do more than that, for
two reasons that are worth stating plainly:

- The confounded test stratum holds 109 images of which **4 are diseased**. Its
  QWK rests on four positive cases and should not be interpreted.
- `other` is not shortcut-free. It contains sixteen further resolutions whose
  No DR share runs from **0% to 100%** - 3388x2588 has none, 2048x1536 has
  nothing else. The metadata baseline still scores 0.5669 inside it.

Removing the shortcut needs a test set with one acquisition. That is IDRiD.

### External validation on IDRiD

All 455 IDRiD images are 4288x2848, so geometry carries no label information at
all. The prior attached to that geometry is also inverted: in APTOS every one of
the 52 images at that resolution is diseased; in IDRiD 129 of 455 are healthy.

A prediction was written down before the run (`docs/external-validation-prediction.md`):
if the model leaned on acquisition cues it would over-call disease, and
specificity on the healthy eyes would collapse.

**It did not.** Baseline fold ensemble, no fine-tuning, APTOS thresholds not
refitted:

| metric | APTOS test | IDRiD |
|---|---|---|
| QWK | 0.9091 | 0.8045 |
| accuracy | 0.8033 | 0.5868 |
| macro F1 | 0.5450 | 0.4609 |
| referable sensitivity | 0.956 | 0.885 |
| **referable specificity** | **0.917** | **0.987** |

1.6% of healthy eyes were called referable; 82.9% were graded healthy outright.
Specificity is *higher* on IDRiD than on APTOS. The prediction was falsified in
the direction that favours the model, and it is the strongest evidence here that
the model reads the retina rather than the camera.

Transfer is not free, and the damage has a shape: the model issues 8 grade-4
predictions where 64 exist, with per-class recall 0.829 / 0.364 / 0.705 / 0.417
/ 0.109. Yet missed referrals - true grade >= 3 called <= 1 - number **2 of
148**. It is compressing the top of the scale, not failing to see disease. That
is calibration drift under covariate shift with fixed thresholds, and it argues
the referable-DR framing is the part that transfers.

squash, for comparison: QWK 0.7859, sensitivity 0.862, specificity 0.993, 3 of
148 severe missed - marginally worse throughout, consistent with the null result.

### Across folds: confound-aware splitting changes nothing

The last check asks whether cross-validation itself was flattered. Folds
stratified on diagnosis alone could, in principle, give some folds a
resolution-to-label mix the others lack, and let a model profit from a mapping
it is never tested against. So the baseline sweep was rerun with folds
stratified jointly on (diagnosis, resolution bucket) - `configs/cv_resolution.yaml`,
sweep `baseline-d665175a1a`, 26 September 2026. Everything else is identical,
including the pool of 3247 leak-cleaned images and the 366-image test set.

| | diagnosis-stratified | diagnosis + resolution |
|---|---|---|
| fold test QWK | 0.8902 ± 0.0086 | 0.8898 ± 0.0055 |
| fold valid QWK | 0.8951 ± 0.0073 | 0.8954 ± 0.0075 |
| ensemble test QWK | 0.9091 | 0.9071 |
| ensemble accuracy / macro F1 | 0.8033 / 0.5450 | 0.8060 / 0.5697 |
| referable sensitivity / specificity | 0.956 / 0.917 | 0.964 / 0.917 |

The fold means differ by 0.0004 (Welch's t-test p = 0.93; the folds are
different splits, so the comparison is unpaired). The ensembles score the same
366 test images, so they can be compared image by image: a paired bootstrap puts
the difference at -0.002 with a 95% interval of [-0.017, +0.013], and the two
ensembles give the same grade to 93.7% of test images.

This is a null result, and the expected one after IDRiD. Stratifying folds on
resolution makes them comparable to one another; it cannot remove a shortcut
that every fold contains, and the external set had already shown the model
does not lean on it. What this run adds is narrower but still worth having:
the cross-validation estimates above were not inflated by uneven folds.

One caveat on the comparison: this sweep ran with `workers: 0` because another
job shared the machine, where the original used 2. The seed-42 rerun above
showed that change alone moves a single run by about 0.01, which is larger than
any difference in this table.

Generated reports: `reports/confound_evaluation.md`, `reports/external_validation.md`;
fold arrays under `models/cv/baseline-d665175a1a/`, summary rows in `reports/runs.csv`.

### A second external set: where the transfer stops

IDRiD was one dataset of 455 images. To test whether its result generalises,
Messidor-2 was added: France rather than India, 1744 gradable images, and a
different kind of label - each image graded by a panel of three retina
specialists who adjudicated disagreements (Krause et al. 2018), against APTOS's
single graders. Four predictions were committed and pushed before any image was
downloaded (`docs/second-external-validation-prediction.md`, commit `ffd6ce9`).

Same ensemble, no fine-tuning, APTOS thresholds and operating point unchanged:

| | APTOS test | IDRiD | Messidor-2 |
|---|---|---|---|
| referable ROC AUC | 0.983 | 0.984 | **0.819** |
| sensitivity at the APTOS operating point | 0.920 | 0.816 | **0.260** |
| specificity at the APTOS operating point | 0.934 | 1.000 | 0.991 |
| five-way QWK | 0.9091 | 0.8045 | 0.4928 |
| ECE (APTOS calibrator) | 0.032 | 0.117 | 0.152 |

| prediction | outcome |
|---|---|
| P1 - referable AUC >= 0.93 | **failed** (0.819) |
| P2 - sensitivity < 0.90 at the APTOS cut, under-confident, ECE > 0.05 | held (0.260, +0.149, 0.152) |
| P3 - grade 4 under-called | held (3 predicted, 35 true) |
| P4 - a calibrator refitted at one new site improves the other | held as written; not useful in practice (below) |

The failure is in one place. **83% of the 347 Moderate eyes are graded below
2**, against 21% on IDRiD. Median ensemble score by true grade:

| true grade | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| APTOS test | 0.03 | 1.31 | 1.93 | 2.47 | 2.73 |
| IDRiD | 0.42 | 0.60 | 1.68 | 2.21 | 2.60 |
| Messidor-2 | 0.22 | 0.28 | **0.54** | 1.79 | 2.10 |

Grades 3 and 4 are still recognised as disease; grade 2 is scored like grade 1.
What follows is analysis after the fact, not prediction. The pattern holds in
both parts of Messidor-2 (original Messidor AUC 0.848, the later Brest images
0.765), so it is not one sub-source. The adjudicated DME flag separates it:
Moderate eyes also marked for referable macular oedema - hard exudates near the
fovea - have a median score of 1.02 (n = 86); Moderate eyes without it, 0.46
(n = 261). The model recognises moderate disease when there are exudates to see
and misses the moderate disease defined by subtler signs.

The likeliest reading is the reference standard. Adjudication by a specialist
panel is known to catch subtle findings single graders miss, and APTOS's
single-grader labels disagree with themselves on 29% of duplicates. A model
trained on them learns where single graders put the Mild/Moderate boundary;
Messidor-2's panel puts it lower. This is a different failure from the one
IDRiD tested for. The model does read the retina - IDRiD settled that - but it
reads it through the labels it was trained on.

So the claim this project can make is narrower than before: the referral
decision transferred to IDRiD, and does not transfer to an adjudicated
reference standard, because Moderate disease without exudates is under-graded.

Generated reports: `reports/external_validation_messidor2.md`, `reports/calibration.md`.

### Fine-tuning on adjudicated labels: the cause, tested

The labels reading above was made after seeing the data. It was then tested,
with the design and five predictions committed first
(`docs/finetune-prediction.md`, commit `238587b`).

Messidor-2 was split in half by patient group - a patient's two eyes never
separated - and each of the five deployed fold models was fine-tuned on the tune
half for 8 epochs. The confound is that fine-tuning changes the labels *and*
the camera. So a control arm was fine-tuned on the same images with the
unchanged ensemble's own grades as labels: Messidor-2's cameras, APTOS's
boundary. Those labels agree with the panel on 59.4% of images.

| | unchanged | fine-tuned, adjudicated labels | fine-tuned, control labels |
|---|---|---|---|
| Messidor-2 test half, referable AUC | 0.830 | **0.925** | 0.834 |
| Messidor-2 test half, Moderate graded below 2 | 78.4% | **45.0%** | 82.5% |
| IDRiD, referable AUC | 0.984 | 0.960 | 0.981 |
| APTOS test, referable AUC | 0.983 | 0.965 | 0.976 |
| APTOS test, accuracy | 0.803 | 0.686 | 0.776 |

| prediction | outcome |
|---|---|
| F1 - Messidor-2 test AUC >= 0.90 | held (0.925) |
| F2 - Moderate below 2 <= 50% | held (45.0%) |
| F3 - IDRiD and APTOS AUC >= 0.95 | held (0.960, 0.965), with a grade-level cost |
| F4 - control does not reach it (AUC < 0.87, Moderate below 2 > 70%) | held (0.834, 82.5%) |
| F5 - the gain concentrates in Moderate eyes without exudates | **failed** |

The control settles the main question. Trained on exactly the same images, it
gains nothing: domain adaptation to the camera does not explain the
improvement, and the labels do. The Messidor-2 failure was the training labels.

F5 limits how far that goes. Median scores of Moderate eyes rose by 0.84 without
exudates and by 1.26 with them: the whole grade moved up, not the subtle cases
in particular. There is no evidence the model learned to see something it had
missed; it moved its boundary to where the panel draws it.

And F3's letter hides a cost. Referral AUC stays above 0.95 elsewhere, but APTOS
test accuracy falls from 0.803 to 0.686 and QWK from 0.909 to 0.867: the
fine-tuned model scores everything higher, and with thresholds fitted on
Messidor-2 it over-grades APTOS. It is not a drop-in replacement, and the
deployed model is unchanged. A model meant for both populations would need
labels of one standard across both.

Generated report: `reports/finetune_messidor2.md`; split: `reports/messidor2_split.csv`.

---

## Calibration and the clinical operating point

A grade is a benchmark output. A screening programme asks whether to refer, and
picks its point on the curve deliberately: a missed referral risks sight, a false
positive costs an appointment.

The operating point is chosen at **sensitivity >= 0.90 on out-of-fold
predictions** - each pool image scored by the fold model that held it out - and
then applied unchanged to the held-out test split and to both external sets. Choosing it on
test and reporting test performance at that choice would measure the selection.

Chosen cut: **1.336** on the raw ordinal score.

| set | n | prevalence | sensitivity | specificity | PPV | referral rate | ROC AUC |
|---|---|---|---|---|---|---|---|
| APTOS out-of-fold (selection) | 3247 | 0.404 | 0.900 | 0.926 | 0.891 | 40.8% | 0.974 |
| APTOS test | 366 | 0.374 | 0.920 | 0.934 | 0.894 | 38.5% | **0.983** |
| IDRiD | 455 | 0.668 | 0.816 | 1.000 | 1.000 | 54.5% | **0.984** |
| Messidor-2 | 1744 | 0.262 | 0.260 | 0.991 | 0.908 | 7.5% | **0.819** |

### Discrimination transfers to IDRiD; calibration does not

This is the cleanest statement of what the IDRiD test found. Messidor-2 then
showed discrimination does not always transfer either (section above).

**ROC AUC is 0.983 on APTOS test and 0.984 on IDRiD.** The model ranks patients
on an unseen population exactly as well as on its own. Nothing about its ability
to separate referable from non-referable degraded.

What moved is where the cut sits. At the fixed threshold IDRiD sensitivity falls
to 0.816 while specificity reaches 1.000 - the score distribution shifted down
relative to the boundary, so a threshold calibrated on APTOS under-refers on
IDRiD.

Expected calibration error says the same thing:

| set | ECE |
|---|---|
| APTOS out-of-fold | 0.0190 |
| APTOS test | 0.0317 |
| **IDRiD** | **0.1174** |
| **Messidor-2** | **0.1516** |

And the reliability table gives it a direction: every IDRiD bin is
*under*-confident. The model predicts 0.27 probability of referable in a bin
where 80% are referable. It is not confused about who is sick; it is
systematically too cautious about saying so.

On IDRiD that is a one-parameter problem: refitting the cut on local data
would recover the sensitivity, and the AUC says the information is there.
Messidor-2 is under-confident in the same direction (+0.149, ECE 0.152), but
there it sits on top of a real loss of discrimination, so no cut recovers it.

### Recalibrating at a new site

With two external sets, one can play the site that supplies a few labels and
the other the site the result is judged on. Refit the Platt calibrator and the
sensitivity-0.90 operating point on one, apply both to the other:

| fitted on | applied to | ECE, APTOS calibrator | ECE, refitted | cut | sensitivity | specificity |
|---|---|---|---|---|---|---|
| IDRiD | Messidor-2 | 0.152 | 0.097 | 1.336 -> 1.082 | 0.260 -> 0.341 | 0.991 -> 0.988 |
| Messidor-2 | IDRiD | 0.117 | 0.095 | 1.336 -> 0.202 | 0.816 -> 0.997 | 1.000 -> **0.132** |

Calibration error falls in both directions, so a probability learned at one new
site is better than one learned on APTOS. The operating point does not travel:
the Messidor-2 cut, pushed down to reach 90% sensitivity there, turns IDRiD
specificity from 1.000 into 0.132. **A deployment would have to set its
threshold on its own labelled data, site by site.**

### Decision curve

Net benefit against treating everyone and treating no one, across thresholds
from 0.05 to 0.70: the model beats **both trivial policies at every threshold
tested, on APTOS and IDRiD**, and at 79% of them on Messidor-2. Miscalibrated
and still useful - those are separate questions, and separating them is the
point of running this.

Generated report: `reports/calibration.md`.

---

## Grad-CAM: an instrument too noisy to answer the question

Grad-CAM was meant to test from the inside what IDRiD tested from the outside:
whether the model attends to the retina or to the frame. It is measured rather
than only drawn - the share of heatmap mass inside the retina divided by the
share of area the retina occupies, so 1.00 means no preference.

The first reading was striking and wrong. Fold 1 gave **0.72x**, apparently
concentrating on the black frame; against an ImageNet baseline of 1.41 with
p < 0.0001, it looked like training had pulled attention off the retina.

Both numbers were single draws, and the p-value pooled per-image values across
draws, counting each image several times. Repeated properly - five head
initialisations for each untrained baseline, all five trained folds, and the
draws treated as the independent units:

| weights | draws | concentration | range across draws |
|---|---|---|---|
| random | 5 | 0.895 | 0.71-1.06 |
| ImageNet, never shown a fundus photograph | 5 | 1.088 | 0.73-1.39 |
| trained (all five folds) | 5 | **0.940** | 0.72-1.11 |

Mann-Whitney, trained against untrained draws: **p = 0.768**. Fold 1 was simply
the lowest of the five.

This is a result about the instrument, not about the model. Five models trained
identically on overlapping data disagree with each other by as much as they
differ from untrained noise, so Grad-CAM concentration on this setup cannot say
where the model looks. It neither supports nor contradicts the IDRiD finding,
which never depended on it.

It is the third instance in this project of the same lesson - after CLAHE on
three seeds and squash on five folds - that one run is an anecdote. This time
the anecdote was the analyst's own, and it was caught by the control rather
than by a reviewer.

A consequence for the serving layer: `?explain=true` deliberately does not
return a heatmap. One that looks informative without being so is worse than
none. Generated report: `reports/attention.md`.

---

## The label ceiling

The dataset contains the same image more than once. Where it does, the labels
should agree.

| measure | value |
|---|---|
| Verified duplicate groups | 131 |
| Same-image pairs | 148 |
| Pairs with conflicting labels | **43 (29.1%)** |
| Agreement rate | 70.9% |
| Estimated single-label accuracy | **~84%** |

Disagreement sizes: 33 pairs differ by one grade, 9 by two, 1 by three.

The five-fold baseline ensemble's test accuracy is 0.8033 — near that ceiling. Part of the remaining
error belongs to the labels, not the model. This is a **lower bound**: it only
measures noise visible in duplicated images.

---

## Data quality

| check | result |
|---|---|
| Images scanned | 3662 |
| Unreadable | 0 |
| Labels without an image / images without a label | 0 / 0 |
| Unusably dark or bright | 0 |
| Brightness outliers (MAD, k=3.5) | 0 |
| Verified duplicate groups | 131 |
| Duplicates spanning splits | 48 |

**Duplicate detection needed two stages.** dHash alone flagged 312 candidate
groups; every fundus image is a bright disc on black, so different retinas
collide. Pixel-level verification eliminated **181 false positives (58%)**,
leaving 131 real groups.

48 of those cross the splits — 6% of test images have a copy in training.
Measured impact: removing them moves test QWK from 0.8960 to **0.8983**, so the
score was not inflated. Runs still exclude them (`--exclude-leaked` drops 49
training images), because measuring that a flaw is harmless is not the same as
fixing it.

The splits themselves are statistically consistent: class proportions
chi-square p=0.53, image properties Mann-Whitney p>0.42. There is no
distribution shift — the valid/test disagreements come from small samples, not
from different populations.

---

## Preprocessing decisions, measured

**Auto-crop threshold.** `tol=7` was validated rather than assumed. Cropping at
tol=2 removes 3.49% of the area, tol=7 removes 11.25%, tol=15 removes 11.48%.
The near-identical results at 7 and 15 show the crop boundary sits on a stable
plateau: the retina edge is well above both thresholds. tol=2 under-crops,
leaving sensor noise inside.

**Squaring the image.** Geometry measured on 80 samples: the retinal disc is
clipped by the sensor at the top (75% of images) and bottom (52%), never at the
sides. A cropped image is therefore naturally wide (median aspect 1.27) and
already 86.5% retina.

| method | black area | retina kept | effective retina px (512px output) |
|---|---|---|---|
| pad | 28.5% | 100% | 188k |
| **squash** | **13.4%** | **100%** | **227k** |
| centre crop | 6.2% | 89.4% | 246k |

`squash` was chosen: no tissue lost, 21% more effective retina pixels. The
centre crop leaves less black but discards 10.6% of the retina, and peripheral
lesions matter.

**This change was never validated by training.** The geometric argument is
sound but, as CLAHE showed, a sound argument is not a result. It is documented
as unmeasured.

---

## Dataset

3662 images, all readable, all three-channel RGB.

| grade | train | valid | test |
|---|---|---|---|
| 0 No DR | 1434 | 172 | 199 |
| 1 Mild | 300 | 40 | 30 |
| 2 Moderate | 808 | 104 | 87 |
| 3 Severe | 154 | 22 | 17 |
| 4 Proliferative DR | 234 | 28 | 33 |

Imbalance ratio (most / least frequent): train 9.31, valid 7.82, test 11.71.

17 distinct resolutions between 474x358 and 4288x2848; the most common is
1050x1050 (974 images, 26.6%). Brightness spans 15.0-129.6 with a median of
69.0 — fundus photographs are inherently dark, which is why a general-purpose
"too bright" threshold of 240 never fires here. Auto-crop removes 10.4% of the
area on average. Preprocessing turns 8 GiB of PNGs into 184 MB of 512px JPEGs.

---

## What was not finished

Seven gaps listed here previously are now closed: cross-validation completed,
`squash` was validated by training and came back null, the shortcut was tested
rather than only reported, and external validation was run on IDRiD (all on
24 September 2026); seed 42 was rerun and the ported trainer shown identical to
the original (25 September); confound-aware fold splitting was run and came
back null, and a second external set was added (26 September). The second set
closed one gap and opened another. What remains:

- **CLAHE parameters were never properly tuned.** Only clip=2.0 on the LAB
  lightness channel has been trained; a sweep with training-free proxies could
  not discriminate between settings. So the finding is "CLAHE at these settings
  does nothing", not "CLAHE cannot help". Given that two independent designs
  now put the effect at zero, further tuning looks a poor use of GPU time, but
  it has not been ruled out.

- **Calibration is measured, and correcting it has to be done per site.** The
  regression score is turned into a referral probability by Platt scaling fitted
  on out-of-fold predictions: ECE 0.019 out of fold, 0.032 on APTOS test. Both
  external sets are under-confident (ECE 0.117 on IDRiD, 0.152 on Messidor-2).
  Refitting on one external set lowers the other's ECE, but the operating point
  does not transfer between them (*Recalibrating at a new site*), so a
  deployment would need labelled data from each site it serves.

- **The deployed model under-grades Moderate disease against an adjudicated
  standard.** Messidor-2 found it (83% of Moderate eyes graded below 2), and
  fine-tuning on adjudicated labels was shown to fix it (45%, with a same-image
  control that does not improve). But the fine-tuned model over-grades APTOS, so
  the deployed one is unchanged. Closing this properly needs one labelling
  standard across all training data - APTOS regraded by adjudication - which
  this project does not have.

- **Per-class test figures remain thin.** 17 Severe images in the test split;
  one case moves that class's recall by 0.06. Cross-validation improved the
  *training* pool's use, not the test split's size.

---

## Reproducing

```bash
export APTOS_GCP_PROJECT=your-project      # optional; BigQuery is not required
python scripts/prepare_bq_csv.py
python scripts/scan_images.py --no-bq
python scripts/image_report.py
python scripts/quality_report.py
python scripts/confound_analysis.py
python scripts/preprocess_images.py --size 512
python scripts/make_figures.py
python -m aptos.training.single --mode reg --exclude-leaked
```

Every report under `reports/` is generated, not hand-written, so the numbers
above can be regenerated from the raw Kaggle download.
