# Open research workflow — NEXUS / APTOS 2019

**Scope: academic and open-source research, not productization.** This repo
remains public under its existing MIT code license. No paid features, customer
accounts, operational screening workflow, commercialization plan or clinical
deployment claims are part of the project. An existing demonstration page is
only a research illustration.

## Reproduce an evidence summary offline

The project deliberately stores experiment outputs in versioned reports.
No GPU, dataset, Kaggle token, cloud project or MLflow database is required:

```sh
pip install -e ".[dev]"
python -m aptos.reporting.research_evidence --reports-dir reports
python -m aptos.reporting.research_evidence --reports-dir reports --format json
pytest -m pure tests/test_research_evidence.py tests/test_label_integrity.py tests/test_bigquery_research.py
```

The program validates every five-fold CV mean/SD against the underlying fold
rows, verifies every external confusion matrix, checks all true/predicted grade
counts and the frozen model thresholds, and attaches SHA-256 hashes of committed
inputs. It does **not** rerun imaging inference or claim new accuracy.

For a portable data manifest after downloading the Kaggle dataset and preparing
labels:

```sh
kaggle datasets download mariaherrerot/aptos2019 -p data/images --unzip
python -m aptos.pipeline run prepare
python -m aptos.data.integrity --csv data/bq/aptos_labels.csv --expected-rows 3662 \
  --out reports/local_label_manifest.json
```

The label manifest checks unique *identifiers*. The separate perceptual-duplicate
pipeline checks repeated *images with different IDs*. Do not confuse them.
Generated local manifests can contain source-specific identifiers, so inspect
before committing. The dataset and raw fundus images remain outside Git.

## Kaggle analysis protocol

- Download `mariaherrerot/aptos2019` explicitly and record the Kaggle dataset
  version/date in the experiment notebook. Kaggle mirrors may change.
- Keep the existing APTOS test holdout untouched for hyperparameter selection.
- Use the same fold partitions/seeds for paired model comparisons, and persist
  per-image out-of-fold predictions with image IDs.
- Report QWK with macro-F1 and each grade's recall, plus referral sensitivity,
  specificity and false negatives; show support and uncertainty.
- For IDRiD and Messidor-2, show the **unchanged** original thresholds first.
  External-site recalibration experiments require fresh development/holdout
  partitions; never report tuning-set performance as unbiased evaluation.
- No cross-site accuracy improvement is claimed until it has been measured.

## Optional BigQuery descriptive analysis

The existing `scripts/load_to_bigquery.py` exports prepared labels into
`PROJECT.APTOS_2019.aptos_labels`. No BigQuery connection is needed to produce
the SQL. The following are **SELECT-only**:

```sh
python -m aptos.reporting.bigquery_research --project your-project-id \
  --out-dir reports/local_sql
# Optional, requires GCP credentials and an existing exported table.
python -m aptos.reporting.bigquery_research --project your-project-id --dry-run
```

Four queries cover label integrity, grade frequency, referral prevalence and
identical IDs spanning splits. They cannot detect perceptually duplicated
images with *different* IDs and do not read actual retinal pixels. BigQuery is
not needed for training, inference or this offline audit. Query generation
never modifies BigQuery tables. Avoid embedding GCP secrets in notebooks.

## Research acceptance criteria

1. CI pure tests pass across Python 3.10–3.12, full CPU tests pass, and Ruff passes.
2. Reproduced descriptive reports exactly match committed source artifacts.
3. Any new imaging-model experiment reports its datasets, seed/fold, training
   config, checkpoint provenance and held-out metrics, including failures.
4. Clearly separate measured observations, mechanistic hypotheses and clinical
   claims; none of the existing results establishes clinical safety.
5. No commercial feature, external patient upload requirement or active service
   deployment is necessary to consider the research repository complete.

## Interpretation

The 29.1% discordance among exact-image pairs does **not** identify a formal
inter-rater ceiling for the full data distribution; it is a selected-pairs
estimate, with dependence and label provenance limitations. The IDRiD
experiment rejects an exclusive reliance on the *measured simple acquisition
proxy* but does not prove independence from every possible camera confound.
Messidor-2 documents a major external failure; the observed differences cannot
be uniquely causally assigned to grading standards from aggregate reports.
