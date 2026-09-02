# APTOS-2019 Diabetic Retinopathy Grading

Predicting diabetic retinopathy severity (ICDRSS grades 0-4) from 3662 retinal
fundus photographs — with an emphasis on checking whether the model earns its
score for the right reason.

Data: Kaggle [`mariaherrerot/aptos2019`](https://www.kaggle.com/datasets/mariaherrerot/aptos2019),
the APTOS 2019 Blindness Detection set pre-split into train/valid/test.

**Best test QWK: 0.9112.** That is not the interesting part — public solutions
reach 0.93+. The three findings below are.

---

## Three findings

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
against that floor. `scripts/confound_analysis.py` measures it.

### The label ceiling is around 84%

The dataset contains the same image more than once. Of 148 same-image pairs,
**43 (29.1%) carry conflicting labels**. That puts inter-rater agreement at
70.9% and a single label's accuracy at roughly **84%**.

The model's test accuracy is 0.82 — near the ceiling. Part of the remaining
error belongs to the labels, not the model.

### CLAHE did not survive replication

The first run suggested CLAHE improved test QWK by +0.0152. Across three seeds
the difference was **+0.0152, −0.0201, −0.0049** — mean −0.0033, paired t-test
p = 0.78. No benefit on test could be demonstrated, and CLAHE made results
*less* stable (test QWK spread 4.4x the baseline's).

It is implemented and kept as an option, documented as unproven.

Full numbers, tables and statistics: **[RESULTS.md](RESULTS.md)**.

---

## Pipeline

Shared functions live in `scripts/preprocessing.py`; every other script and the
Colab notebook imports from it rather than keeping a copy.

```
Read -> Quality check -> Auto-crop -> CLAHE -> Square -> Resize -> Normalise
```

![Pipeline](reports/figures/05_pipeline_stages.png)

| function | what it does |
|---|---|
| `auto_crop()` | Removes the black frame outside the retina (10.4% of the area on average) |
| `apply_clahe()` | CLAHE on the LAB L channel only, so colour balance survives |
| `to_square()` | `squash` or `pad`; squash cuts black area from 28.5% to 13.4% |
| `image_quality()` | Brightness and contrast measures, unusable-image detection |
| `brightness_outliers()` | MAD-based outlier detection |
| `dhash()` | Perceptual hash, used as a duplicate candidate generator |
| `preprocess()` | The full six-stage pipeline |

Decisions were measured rather than assumed:

- **`tol=7` for auto-crop** — cropping at tol=7 and tol=15 give nearly identical
  results (11.25% vs 11.48% removed), so the boundary sits on a stable plateau.
- **`squash` over `pad`** — the retinal disc is clipped top and bottom but never
  at the sides, so a cropped image is naturally wide and already 86.5% retina.
  Padding dilutes that with black bars; squashing keeps all tissue and yields
  21% more effective retina pixels.
- **CLAHE after auto-crop** — on an uncropped image the wide black border skews
  the histogram.
- **Normalisation in the training transforms, not the pipeline** — one place
  only, so it cannot happen twice.

Each processed directory carries a `_manifest.json` recording the settings that
produced it, and the training scripts print it, so every run states which
preprocessing it used.

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
validation QWK, best checkpoint kept rather than last. `train_cv.py` adds
stratified K-fold cross-validation with the test set held out entirely.

## Choosing the metric

The headline metric is **quadratic weighted kappa**, not accuracy. 49% of the
dataset is "No DR"; a model that always predicts 0 scores 49% accuracy and 0 on
QWK. The problem is also **ordinal** — grades form a severity scale, so calling
a Severe case Mild is worse than calling it Proliferative.

QWK should be read together with macro F1. This model scores **QWK 0.90 but
macro F1 0.57**: errors land on neighbouring grades, which QWK penalises
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
pip install -r requirements.txt
```

For GPU support install PyTorch from the CUDA index **only** — adding
`--extra-index-url` makes pip prefer the CPU build on PyPI:

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

BigQuery is optional. The pipeline reads and writes local CSVs when it is
unavailable; only `load_to_bigquery.py` and `analyze_runs.py` require it.
Cloud identifiers come from the environment:

```bash
export APTOS_GCP_PROJECT=your-project     # default: datascientis
export APTOS_BQ_DATASET=your-dataset      # default: APTOS_2019
export APTOS_GCS_BUCKET=your-bucket       # default: aptos2019-retina-images
```

## Running

```bash
# 1. Raw data
kaggle datasets download mariaherrerot/aptos2019 -p data/images --unzip

# 2. Labels
python scripts/prepare_bq_csv.py

# 3. Scan image properties
python scripts/scan_images.py --no-bq

# 4. Reports
python scripts/image_report.py
python scripts/quality_report.py
python scripts/confound_analysis.py

# 5. Prepare images
python scripts/preprocess_images.py --size 512              # with CLAHE
python scripts/preprocess_images.py --size 512 --no-clahe   # control set

# 6. Figures
python scripts/make_figures.py

# 7. Train
python scripts/train.py --mode reg --epochs 30 --patience 5 --exclude-leaked
python scripts/train_cv.py --folds 5 --exclude-leaked       # cross-validated
```

## Tests

```bash
python tests/test_preprocessing.py    # pytest not required
```

31 tests over the shared module, using synthetic images — they run on a machine
that has never downloaded the dataset.

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
  here, so a general-purpose "too bright" cutoff of 240 never fires. The
  quality report uses MAD-based outlier detection alongside fixed thresholds.
- **The Kaggle archive is inconsistent**: the validation split lives under
  `val_images/`, not `valid_images/`.
- **`test.csv` ends with ~500 blank lines.** Loading it raw with schema
  autodetection produces a broken table.
- **Long GPU runs are fragile on Windows.** Do not start a second GPU job
  alongside one — the commit limit is exhausted and dataloader workers die with
  `error code 1455`. Do not edit `train_cv.py` while it runs either; the
  workers re-import it by path.

## Known gaps

Documented rather than hidden:

- **Cross-validation was built but never completed.** `train_cv.py` works and
  passes a smoke test, but no full sweep finished before the project ended.
- **`squash` was never validated by training.** The geometric case is sound,
  but CLAHE is a reminder that a sound argument is not a result.
- **CLAHE parameters were never properly tuned.** A training-free proxy sweep
  could not discriminate between settings.

## Layout

```
scripts/
  preprocessing.py       shared preprocessing functions
  prepare_bq_csv.py      clean and enrich the label CSVs
  load_to_bigquery.py    load labels into BigQuery
  scan_images.py         scan image properties
  image_report.py        image properties summary
  quality_report.py      data quality + duplicate analysis
  confound_analysis.py   metadata shortcut and label noise
  preprocess_images.py   prepare images for training
  make_figures.py        report figures
  train.py               training and evaluation (single split)
  train_cv.py            K-fold cross-validation
  analyze_runs.py        run comparison and error analysis
  run_cv.sh              cross-validation for every variant
  run_seeds.sh           multi-seed comparison

tests/test_preprocessing.py   31 tests over the shared module
reports/                      three generated reports + 16 figures
RESULTS.md                    every measurement, in full
```

## Credits

Built as a final-year data science project. The pipeline, analysis, reports,
figures, tests and cross-validation in this repository were developed by
**Senanur Çetin**; see the commit history for the breakdown.

Initial project scaffolding and the first Colab notebook came from **Ceren
Kocabaş** and **Ruveyda Karakoyun**.
