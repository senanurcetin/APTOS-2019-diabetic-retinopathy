# APTOS-2019 Diabetic Retinopathy Grading

[![tests](https://github.com/senanurcetin/APTOS-2019-diabetic-retinopathy/actions/workflows/tests.yml/badge.svg)](https://github.com/senanurcetin/APTOS-2019-diabetic-retinopathy/actions/workflows/tests.yml) [![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

Predicting diabetic retinopathy severity (ICDRSS grades 0-4) from 3662 retinal
fundus photographs — with an emphasis on checking whether the model earns its
score for the right reason.

Data: Kaggle [`mariaherrerot/aptos2019`](https://www.kaggle.com/datasets/mariaherrerot/aptos2019),
the APTOS 2019 Blindness Detection set pre-split into train/valid/test.

**Best test QWK: 0.9098**, a five-fold ensemble. That is not the interesting
part — public solutions reach 0.93+. The four findings below are.

---

## Four findings

### A model can score well without looking at the retina

A RandomForest trained **only on file metadata** — resolution, aspect ratio,
brightness, contrast, file size — reaches **QWK 0.652** under five-fold
cross-validation. It never sees a pixel of retina.

![Shortcut](reports/figures/07_resolution_confound.png)

The cause is concentrated in one resolution: **92.5% of the 1050x1050 images
are `No DR`**, against 33.6% at every other size. APTOS data was collected
across several sites with different cameras, and the device signature
correlates with disease prevalence. Resizing to 512px does not remove it —
aspect ratio, sharpness and edge geometry still carry it.

This does not invalidate the results, but any reported score should be read
against that floor. `scripts/confound_analysis.py` measures it — and the fourth
finding below tests whether the model actually depends on it.

### The label ceiling is around 84%

The dataset contains the same image more than once. Of 148 same-image pairs,
**43 (29.1%) carry conflicting labels**. That puts inter-rater agreement at
70.9% and a single label's accuracy at roughly **84%**.

The ensemble's test accuracy is 0.8033 — near the ceiling. Part of the remaining
error belongs to the labels, not the model.

### Two preprocessing ideas, both null

**CLAHE.** The first run suggested it improved test QWK by +0.0152. Across three
seeds the difference was **+0.0152, −0.0201, −0.0049** — mean −0.0033, paired
t-test p = 0.78. Five-fold cross-validation, an independent design on a larger
evaluation pool, agrees and tightens it: **−0.0005, p = 0.865**.

**`squash` squaring.** The geometric case was measured and correct — padding
leaves 28.5% of a 512px frame black against squash's 13.4%, so squash carries
21% more retina pixels. Trained and cross-validated, it bought **+0.0038 QWK,
p = 0.468**, winning 2 of 5 folds.

Neither beats the control. The second one is the sharper lesson: the argument
for squash was quantitative and right about the pixels, and the pixels turned
out not to be what limits this model.

The largest effect found anywhere in the project is not a preprocessing choice
at all — it is ensembling the five folds, worth **+0.019 QWK**, about five times
the biggest gap between any two variants.

### The model reads the retina, not the camera

Given the shortcut above, the obvious question is whether the score survives
without it. Within APTOS it cannot be removed — excluding the confounded
resolution still leaves sixteen others whose No DR share runs from 0% to 100%.

**IDRiD** removes it: all 455 of its images are 4288x2848, so geometry carries
no label information, and the prior on that geometry is *inverted* — in APTOS
all 52 images at that resolution are diseased, in IDRiD 129 of 455 are healthy.

A prediction was committed before the run
([`docs/external-validation-prediction.md`](docs/external-validation-prediction.md)):
a shortcut-dependent model would over-call disease, and specificity on the
healthy eyes would collapse.

It was falsified. No fine-tuning, APTOS thresholds not refitted:

| metric | APTOS test | IDRiD |
|---|---|---|
| QWK | 0.9091 | 0.8045 |
| referable sensitivity | 0.956 | 0.885 |
| **referable specificity** | **0.917** | **0.987** |

1.6% of healthy eyes were called referable. Specificity is *higher* on a
population the model has never seen. What does degrade is calibration at the
top of the scale — 8 grade-4 predictions against 64 true cases — while missed
referrals stay at 2 of 148. It compresses the scale rather than failing to see
disease.

Full numbers, tables and statistics: **[RESULTS.md](RESULTS.md)**.

---

## Pipeline

Shared functions live in `src/aptos/preprocessing.py`. `scripts/preprocessing.py`
remains as a thin re-export, because the Colab notebook clones this repository at
run time and imports it by path.

```
Read -> Quality check -> Auto-crop -> CLAHE -> Square -> Resize -> Normalise
```

![Pipeline](reports/figures/05_pipeline_stages.png)

| function | what it does |
|---|---|
| `auto_crop()` | Removes the black frame outside the retina (11.25% of the area at `tol=7`) |
| `apply_clahe()` | CLAHE on the LAB L channel only, so colour balance survives |
| `to_square()` | `squash` or `pad`; squash cuts black area from 28.5% to 13.4% — measured, but see below |
| `image_quality()` | Brightness and contrast measures, unusable-image detection |
| `brightness_outliers()` | MAD-based outlier detection |
| `dhash()` | Perceptual hash, used as a duplicate candidate generator |
| `preprocess()` | The full six-stage pipeline |

Decisions were measured rather than assumed:

- **`tol=7` for auto-crop** — cropping at tol=7 and tol=15 give nearly identical
  results (11.25% vs 11.48% removed), so the boundary sits on a stable plateau.
- **`squash` vs `pad` — measured, then trained, then dropped as a claim.** The
  retinal disc is clipped top and bottom but never at the sides, so a cropped
  image is naturally wide and already 86.5% retina. Padding dilutes that with
  black bars; squashing keeps all tissue and yields 21% more effective retina
  pixels. All of that is true and none of it helped: cross-validated against
  pad, squash was worth +0.0038 QWK at p = 0.468. **Every published number in
  this repository was produced with `pad`**, and the caches on disk record that
  in their own `_manifest.json`.
- **CLAHE after auto-crop** — on an uncropped image the wide black border skews
  the histogram.
- **Normalisation in the training transforms, not the pipeline** — one place
  only, so it cannot happen twice.

Each processed directory carries a `_manifest.json` recording the settings that
produced it. Training checks it rather than merely printing it: a cache that
does not match what the config asked for stops the run.

## Model and training

EfficientNet-B0 with ImageNet weights, 384px input.

**Augmentation.** Random resized crop (0.85-1.0), horizontal *and vertical*
flip, ±20° rotation, mild brightness and contrast jitter. Vertical flip is
included because a fundus photograph has no meaningful up/down orientation;
colour jitter is kept mild so it does not fight CLAHE.

![Augmentation](reports/figures/08_augmentation.png)

**Loss.** `--mode reg` uses a single output with MSE and four thresholds tuned
on validation; `--mode cls` uses five-way softmax with class-weighted
cross-entropy.

**Training loop.** AdamW, cosine schedule, mixed precision, early stopping on
validation QWK, best checkpoint kept rather than last. `aptos.training.cv` adds
stratified K-fold cross-validation with the test set held out entirely, and is
resumable: each fold persists its predictions and metrics before the next
starts, so an interrupted sweep costs one fold rather than all of them.

## Choosing the metric

The headline metric is **quadratic weighted kappa**, not accuracy. 49% of the
dataset is "No DR"; a model that always predicts 0 scores 49% accuracy and 0 on
QWK. The problem is also **ordinal** — grades form a severity scale, so calling
a Severe case Mild is worse than calling it Proliferative.

QWK should be read together with macro F1. This model scores **QWK 0.91 but
macro F1 0.55**: errors land on neighbouring grades, which QWK penalises
lightly, so the weakness on minority classes shows up only in macro F1.

## Data

![Class distribution](reports/figures/01_class_distribution.png)

3662 images, all readable, all three-channel RGB. Heavily imbalanced — 1434 No
DR against 154 Severe in training, a ratio of 9.3. Resolution varies widely: 17
distinct sizes between 474x358 and 4288x2848.

Reports: [`image_properties.md`](reports/image_properties.md),
[`data_quality.md`](reports/data_quality.md),
[`confounds_and_noise.md`](reports/confounds_and_noise.md). All generated by
scripts, not written by hand.

## Setup

```bash
pip install -e .            # preprocessing and analysis, no torch
pip install -e ".[train]"   # add the training stack (quote it: some shells glob [ ])
```

For GPU support install PyTorch from the CUDA index **only** — adding
`--extra-index-url` makes pip prefer the CPU build on PyPI:

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Runs are tracked locally with MLflow, in `mlflow.db` beside the code:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

BigQuery is not required for anything. The project originally logged every run
to a GCP project, and when that access was withdrawn the entire experimental
record went with it — which is why tracking is local and travels with the
repository. `load_to_bigquery.py` and `analyze_runs.py` are the only files that
still expect it, and nothing depends on them.

## Running

One entry point, with the stage order enforced rather than documented:

```bash
kaggle datasets download mariaherrerot/aptos2019 -p data/images --unzip

python -m aptos.pipeline list        # the stage graph
python -m aptos.pipeline check       # what could run right now
python -m aptos.pipeline run all     # everything, in dependency order
```

Every stage declares what it reads, so a stage whose inputs are missing stops
with a message naming the stage you skipped. That matters here: `quality`
verifies duplicate candidates against thumbnails from the processed cache, and
running it before `preprocess` used to yield zero duplicates and an empty leak
list *silently*. The published run order in earlier versions of this file had
exactly that mistake.

Training and cross-validation:

```bash
python -m aptos.training.cv --config configs/cv.yaml --variant baseline
bash scripts/run_cv.sh                      # baseline, squash, clahe in turn
```

The sweep is resumable — completed folds are skipped, so re-running after an
interruption picks up where it stopped.

External validation, with no fine-tuning and no threshold refitting:

```bash
kaggle datasets download mariaherrerot/idrid-dataset -p data/external --unzip
python -m aptos.evaluation.external --sweep models/cv/baseline-<id>
python -m aptos.evaluation.confound --sweeps models/cv/baseline-<id>
```

## Tests

```bash
pytest                      # 79 tests
pytest -m pure              # the 62 that need neither torch nor the dataset
python tests/test_preprocessing.py   # 31 of those, without pytest at all
```

The preprocessing tests build synthetic fundus images, so they run on a machine
that has never downloaded the dataset — which is why CI checks them on Python
3.10, 3.11 and 3.12, and why the Colab notebook can run them too. A second CI
job installs CPU torch and runs everything.

## Things worth knowing

- **Duplicate detection needs two stages.** dHash alone flagged 312 candidate
  groups; 181 were false positives, because every fundus image is a bright disc
  on black. Candidates are verified at pixel level, leaving 131 real groups, 48
  of which cross the splits. `--exclude-leaked` drops the 49 affected training
  images. Measured impact on the score was small (test QWK 0.8960 → 0.8983),
  but measuring that a flaw is harmless is not the same as fixing it.
- **The test split holds 366 images**, only 17 of them Severe. Select on
  validation, look at test once, and treat per-class test figures as
  indicative.
- **Brightness thresholds must be data-aware.** Brightness spans 15.0-129.6
  here, so the general-purpose "too bright" cutoff of 250 never fires. The
  quality report uses MAD-based outlier detection alongside fixed thresholds.
- **The Kaggle archive is inconsistent**: the validation split lives under
  `val_images/`, not `valid_images/`.
- **`test.csv` ends with ~500 blank lines.** Loading it raw with schema
  autodetection produces a broken table.
- **Long GPU runs are fragile on Windows.** Do not start a second GPU job
  alongside one — the commit limit is exhausted and dataloader workers die with
  `error code 1455`. Do not edit `train_cv.py` while it runs either; the
  workers re-import it by path. The same applies to anything under `src/aptos/`.

## Serving

```bash
pip install -e ".[train,serve]"
uvicorn serving.app:app --reload
```

`POST /predict` takes a fundus photograph and returns a grade, the referral
decision, the raw ordinal score and the spread across the five folds.
`GET /model-card` returns what the model is known to get wrong. The page states
that this is not a medical device, and lists the confound, the label ceiling and
the calibration drift, because a demo that omits them would contradict the
analysis it exists to demonstrate.

The referable flag is the headline rather than the grade: external validation
showed the binary decision transfers (ROC AUC 0.984) while the five-way grade
compresses under distribution shift.

## Known gaps

Documented rather than hidden. Four earlier entries here are now closed —
cross-validation completed, `squash` was trained and came back null, the
shortcut was tested rather than only reported, and IDRiD was run. What remains:

- **Calibration is measured but deliberately not corrected.** On IDRiD the ROC
  AUC is 0.984 against APTOS test's 0.983 — discrimination transfers intact —
  but ECE rises from 0.019 to 0.117 and the model becomes systematically
  *under*-confident, so a threshold fitted on APTOS under-refers elsewhere
  (sensitivity 0.816 against the 0.90 it was set for). It is a one-parameter
  problem and fixing it needs labelled data from the target population, which is
  what a deployment would have to obtain and this project does not have. See
  [`reports/calibration.md`](reports/calibration.md).

- **CLAHE parameters were never tuned.** Only clip=2.0 on the LAB lightness
  channel has been trained. So the finding is "CLAHE at these settings does
  nothing", not "CLAHE cannot help" — though two independent designs now put
  the effect at zero.
- **Confound-aware fold splitting is implemented but unrun.**
  `configs/cv_resolution.yaml` stratifies folds jointly on (diagnosis,
  resolution). After the IDRiD result its value dropped: it makes folds
  comparable to each other without removing the shortcut, and IDRiD answers the
  underlying question outright. Left undone deliberately.
- **The external result rests on one mirror of one dataset.** 455 images from a
  Kaggle copy of IDRiD rather than the full official distribution, and 129
  healthy eyes is a small denominator for the specificity the conclusion leans
  on — it moves by 0.008 per image.
- **The serving surface runs locally but is not deployed.** `serving/app.py`
  exposes `/predict`, `/health` and `/model-card` with a minimal page, and its
  grades match the offline evaluation on held-out images. Nothing is hosted, and
  Grad-CAM is stubbed rather than implemented.
- **Serving preprocesses from source while training read cached JPEGs.** The
  cache went through a quality-95 JPEG round-trip that an upload does not, so
  raw scores differ slightly — up to 0.12 on six held-out images, with every
  predicted grade agreeing. Small, real, and not worth hiding.
- **`aptos_2019.ipynb` is broken.** It calls `APTOSDataset(use_crop=...)`, a
  parameter that does not exist, uses `pd` and `plt` without importing either,
  and has no training loop. It is excluded from linting and scheduled for
  replacement.
- **Several legacy scripts are unported.** `analyze_runs.py` builds a BigQuery
  client at module scope and cannot be imported without credentials;
  `load_to_bigquery.py`'s default table names can never match what `train.py`
  queries; `run_seeds.sh` omits `--exclude-leaked` while `run_cv.sh` includes
  it. The package path does not depend on any of them.

## Layout

```
src/aptos/
  config.py            every constant, one place; YAML + dataclasses
  preprocessing.py     the shared image pipeline
  pipeline.py          stage graph, preconditions, CLI entry point
  tracking.py          MLflow, with BigQuery as an optional extra sink
  data/                labels, datasets, cache manifests
  modeling/            QWK, threshold search, clinical metrics (no torch)
  training/            model, epoch loop, resumable cross-validation
  evaluation/          confound stratification, external validation
configs/               base + one file per variant, and the CV variants
scripts/               legacy scripts, ported progressively
  preprocessing.py     thin re-export, kept for the Colab notebook
  run_cv.sh            the sweep runner
tests/                 79 tests; `-m pure` needs neither torch nor the dataset
docs/                  the pre-registered external-validation prediction
reports/               generated markdown, CSVs and 16 figures
```

## Licence

Code: MIT — see [LICENSE](LICENSE).

The dataset is not included in this repository and is not covered by that
licence; see [NOTICE](NOTICE).

## Credits

Built as a final-year data science project. The pipeline, analysis, reports,
figures, tests and cross-validation in this repository were developed by
**Senanur Çetin**; see the commit history for the breakdown.

Initial project scaffolding and the first Colab notebook came from **Ceren
Kocabaş** and **Ruveyda Karakoyun**.
