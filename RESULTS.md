# Results

Every number here was produced by the scripts in this repository. Runs were
originally logged to BigQuery; that project is no longer available, so the
tables are reproduced in full below.

Setup for all runs: EfficientNet-B0 (ImageNet weights), 384px input, batch 16,
15 epochs, AdamW at 3e-4, cosine schedule, mixed precision, early stopping on
validation QWK (patience 5), best checkpoint kept. Regression mode with four
thresholds tuned on validation.

---

## Headline

| | |
|---|---|
| Best test QWK | **0.9112** (CLAHE, seed 42) |
| Baseline test QWK, 3 seeds | 0.8986 ± 0.0033 |
| CLAHE test QWK, 3 seeds | 0.8954 ± 0.0146 |
| Metadata-only floor | **QWK 0.652** — without looking at the retina |
| Estimated label ceiling | **~84%** single-label accuracy |

The score is not the interesting part of this project. Public APTOS solutions
reach 0.93+. What follows — the shortcut floor, the label ceiling, and a
preprocessing technique that did not survive replication — is.

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

## The metadata shortcut

A RandomForest trained **only on file properties** — resolution, aspect ratio,
brightness, contrast, file size — never seeing a single pixel of retina.
Five-fold cross-validation.

| measure | metadata only | always predict 0 | the real model |
|---|---|---|---|
| QWK | **0.652** | 0.000 | ~0.90 |
| Accuracy | 0.708 | 0.493 | ~0.82 |

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

The model's test accuracy is 0.82 — near that ceiling. Part of the remaining
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

- **Cross-validation was built but never completed.** `train_cv.py` works and
  passes a smoke test, but three attempts to run the full sweep died: twice to
  environment problems (a concurrent GPU job exhausting the Windows commit
  limit; editing the running script, which on Windows kills the dataloader
  workers because they re-import it by path) and once when the project ended.
  With a 366-image validation set, single-split measurements are fragile — the
  CLAHE difference swinging between +0.015 and −0.020 across seeds is the
  evidence. Cross-validation over the full 3247-image pool is the fix.

- **`squash` was never validated by training**, as noted above.

- **CLAHE parameters were never properly tuned**, only swept with
  training-free proxies that could not discriminate.

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
python scripts/train.py --mode reg --exclude-leaked --no-bq
```

Every report under `reports/` is generated, not hand-written, so the numbers
above can be regenerated from the raw Kaggle download.
