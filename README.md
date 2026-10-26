<p><img src="serving/static/nexus-logo.svg" alt="NEXUS" width="270"></p>

# APTOS: retinal grading under distribution shift

**NEXUS Open Research · Noncommercial data science · English-first**

[![tests](https://github.com/senanurcetin/APTOS-2019-diabetic-retinopathy/actions/workflows/tests.yml/badge.svg)](https://github.com/senanurcetin/APTOS-2019-diabetic-retinopathy/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A reproducible investigation of whether an EfficientNet-B0 retinopathy grading
model earns its internal score for transferable reasons. The project examines
acquisition shortcuts, conflicting labels, preprocessing, external failure and
sensitivity to grading targets. It includes completed training experiments,
independent prediction audits and a static NEXUS evidence explorer.

**[Research explorer](https://senanurcetin.github.io/APTOS-2019-diabetic-retinopathy/)** ·
[Results](RESULTS.md) · [Scientific assessment](docs/SCIENTIFIC_REVIEW.md) ·
[Kaggle notebook](https://www.kaggle.com/code/senanuretin/aptos-three-predictions-i-wrote-before-the-data) ·
[Documentation](docs/README.md)

## Evidence at a glance

These rows describe different experiments; they are not one model leaderboard.

| Experiment | Observed result | Interpretation |
|---|---|---|
| Historical APTOS benchmark | Best ensemble test QWK **0.9098** | Reused 366-eye holdout; not a fresh confirmatory estimate |
| Metadata baseline | Historical QWK **0.652**; Kaggle rerun **0.641** with different folds | Low-level file/image summaries predict labels; not a mathematical performance floor |
| Historical external baseline | Referable AUC **0.984** on IDRiD, **0.819** on Messidor-2 | Ranking and referral thresholds do not transfer uniformly |
| Historical fine-tuning | Messidor AUC **0.925** | Six test/development examination overlaps; exploratory |
| Examination-disjoint v2 | Adjudicated AUC **0.9435**, control **0.8534** | Training targets differ; adjudicated validation targets are common |
| v2 transfer cost | APTOS accuracy **80.6% → 64.2%** | Messidor improvement comes with substantial retention loss |

The v2 test comprises **872 eyes / 437 supplied examinations**. Its primary
adjudicated-minus-control AUC difference is **+0.0901**, with a 95% paired
examination-bootstrap interval of **[+0.0614, +0.1204]**. The unchanged baseline
AUC is **0.8405**. This run uses `baseline-c8232af68337`, distinct from the
historical external baseline `baseline-6e66147526`.

See the [audited v2 results](reports/research/messidor2_v2_results.md) for fixed
and validation-fitted thresholds, rare-grade recall, retention and provenance.
No fine-tuned model was promoted to the inference demonstration.

## Run the evidence explorer

Python 3.10 or newer is required. These commands need no dataset, weights,
GPU, cloud account or credentials:

```bash
python -m pip install -e ".[dev]"
python -m aptos.reporting.portfolio --check
python -m http.server 8080 --directory serving/static
# Open http://localhost:8080/research.html
```

For dataset preparation, training, audits and the optional inference
illustration, use the [reproduction guide](docs/REPRODUCIBILITY.md).

## Repository structure

```text
src/aptos/       Installable package: data, training, evaluation and reporting
configs/        Experiment configuration
scripts/        Packaging, acquisition and legacy CLI adapters
notebooks/      Evidence notebooks, GPU experiment template and Kaggle source
serving/        Static NEXUS explorer and separate inference illustration
reports/        Committed aggregate evidence, tables, figures and audit records
docs/           Documentation index, methods, runbooks and dated protocols
tests/          Synthetic-data, integrity, training and serving contracts
.github/        CI and static Pages deployment
aptos_2019.ipynb Original root-level notebook, retained for existing links
```

Downloaded data, checkpoints, MLflow stores and local execution outputs belong
in ignored `data/`, `models/`, `mlruns/` and `outputs/`. They are not part of the
public source distribution. Runtime code follows the existing `src` layout;
`serving/static` is retained because FastAPI and Pages share those assets.

## Scientific boundaries

- The APTOS holdout and external cohorts have already been inspected. Further
  comparisons on them are exploratory.
- Examination separation uses a third-party pairing transcription. Original
  ADCIS spreadsheet parity and longitudinal patient identity are unverified.
- The v2 adjudicated arm recalls only **20.8%** of the 24 grade-4 test eyes;
  validation contains only two grade-4 eyes.
- Duplicate-label disagreement does not establish a population accuracy ceiling.
  Nonsignificant preprocessing or attention tests do not prove equivalence.
- No prospective clinical utility, demographic fairness or multi-seed robustness
  is established. This repository is not a diagnostic service.

The [scientific review](docs/SCIENTIFIC_REVIEW.md) separates completed findings,
methodological limitations and the remaining data requirements. Verification
records are linked there rather than repeated throughout the documentation.

## Contributing and licensing

See [CONTRIBUTING.md](CONTRIBUTING.md) for checks and evidence requirements, and
[SECURITY.md](SECURITY.md) for private disclosure. Code is [MIT-licensed](LICENSE).
Datasets and model artifacts have separate terms; see [NOTICE](NOTICE) and the
[model card](serving/MODEL_CARD.md).

Developed by **Senanur Çetin**. Initial project scaffolding and the first Colab
notebook were contributed by **Ceren Kocabaş** and **Ruveyda Karakoyun**.
Citation metadata is provided in [CITATION.cff](CITATION.cff).
