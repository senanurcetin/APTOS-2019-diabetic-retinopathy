# Scientific assessment — APTOS / NEXUS

Assessment date: **24 October 2026**. This is a repository-based methodological
audit by the development assistant, not independent peer review or a clinical
certification. Historical outputs, current source, Kaggle execution artifacts and
relevant primary methodological references were inspected. No new model was trained
for this assessment. The project remains noncommercial research.

## Overall judgement

**Strong applied data science portfolio; exploratory model-development research
with meaningful external failure analysis. Clinical readiness is not established.**
The strongest contribution is the investigation of shortcuts, label sensitivity
and transportability, rather than the maximum internal QWK. The external failure
and controlled follow-up are worth reporting. Broad causal claims, guaranteed
patient independence and prospective usefulness exceed the available evidence.

## 1. Completed GPU result audit

The authorized Kaggle job `senanuretin/aptos-research-reg-seed42` reported COMPLETE.
Its output inventory required **eight provider pages / 3,894 files**. Reading just
the CLI's first output page would have missed the training artifacts.

The audit downloaded the five JSON metric files, five NPZ prediction files,
configuration, provenance and ensemble summary. All five `.pt` checkpoint names
were also present in the inventory. Checkpoint tensors were not downloaded or
executed. A checkpoint filename is evidence of presence, not weight integrity.

`baseline-c8232af68337` has five aligned prediction arrays, each for **366 unique
image IDs**. Labels match exactly across folds. Fold metrics and ensemble metrics
were recomputed from these saved scores, with the stored validation-fitted cuts.
Fold means/SDs, selected epochs, epoch histories and mean ensemble thresholds were
checked against the records. The audit rejects missing folds, changed IDs,
nonfinite scores, wrong cohort sizes and inconsistent summary values.

| Recomputed seed-42 scalar MSE ensemble | Result |
|---|---:|
| Test QWK | 0.900763 |
| Test accuracy | 0.808743 |
| Test macro F1 | 0.551072 |
| Referable sensitivity | 132 / 137 = 96.35% |
| Referable specificity | 208 / 229 = 90.83% |
| Missed referable images | 5 |
| Mild-class recall | 13.33% |
| Proliferative-class recall | 27.27% |

The fold mean test QWK is **0.887517**, with across-model SD **0.011166**.
The ensemble exceeds that mean by **0.013247**. This is a descriptive comparison
on one shared holdout, not five independent estimates of generalization.

The new QWK is below the historical external-evaluation baseline's **0.909109**
and the historical squash ensemble's **0.9098**. It does not establish a meaningful
degradation or improvement: this is one run, not a prespecified paired multi-seed
comparison. Historical reports and the deployed ONNX model remain separate. This
new sweep has no audited IDRiD/Messidor-2 inference and no completed ordinal arm.

**Artifacts:** [machine-readable recomputation audit](../reports/research/kaggle_reg42_audit.json).
It contains aggregate results and SHA-256 fingerprints, without image IDs or
individual prediction arrays. The execution log records source commit
`fc4aa0a9c54c3719e2fd2d33b4686d666828f62c`; sweep provenance records
`run_kind=full` and a label fingerprint. Neither is independent verification of
the source pixels. No inference-parity or checkpoint-safety claim is made here.

Reproduce with authorized Kaggle credentials configured locally:

```bash
python -m pip install kaggle
python scripts/download_kaggle_audit.py \
  --kernel senanuretin/aptos-research-reg-seed42 --out outputs/gpu_audit
python -m aptos.reporting.sweep_audit \
  --sweep outputs/gpu_audit/aptos-research/models/cv/baseline-c8232af68337 \
  --expected-n 366 --output outputs/recomputed_sweep_audit.json
```

The downloader excludes fundus images and pickle checkpoints. Use the existing
project dependencies for the audit. Downloads and original predictions remain
under ignored `outputs/`; no licensed image data is redistributed.

## 2. Correction: proxy groups are not authenticated patients

The historical fine-tuning experiment derived groups from exam-date prefixes
and consecutive `IM` filename chains. The committed split contains **1,744 eyes,
379 proxy groups**, with **698 tune-train / 174 tune-valid / 872 test** images.
No proxy group spans partitions. The largest group contains **118 images**.

These facts verify separation of the implemented groups. They do **not** verify
that every patient's eyes stay together: no authentic patient mapping was supplied.
Prior wording promising that separation has been corrected in the narrative,
results and code documentation. The historical protocol is retained with an
explicit methodological correction; its original timestamp is not rewritten.

The controlled fine-tuning result still supports a role for the provided labels:
recorded Messidor-2 test-half AUC is 0.830 unchanged, 0.925 with adjudicated labels,
and 0.834 with model-generated control labels. However, unverified patient
separation and one experimental setting limit the strength of causal and
transportability claims. It does not isolate every camera or optimization effect.

**Unresolved by aggregate data:** patient-cluster bootstrap and a guarantee of no
patient leakage. The existing uncertainty code accepts genuine patient IDs, but
the filename proxies must never be passed off as those IDs. Fabricating a mapping
or relabeling an eye-level interval would create a false result.

The [official Messidor-2 provider](https://www.adcis.net/en/third-party/messidor2/)
states that the original distribution has 1,748 images in 874 examinations and
includes an image-pairing spreadsheet. The project's 1,744-image gradable mirror
must be reconciled to that original pairing table before examination-level
inference. Examination pairing alone would still not prove longitudinal patient
identity. Official access currently requires a personal-information form and
license acceptance; neither was submitted on the researcher's behalf in this
audit. Existing authorized pairing data can be used locally when available.

## 3. Internal evaluation and model selection

Strengths include leakage checks, explicit test identifiers, saved predictions,
validation-fitted thresholds and comparative preprocessing experiments. The new
audit confirms numerical consistency of the recorded seed-42 predictions.

Limitations remain:

- The same 366-image test split has been inspected across multiple variants and
  follow-up questions. It is no longer a fresh confirmatory test for new decisions.
- Each fold chooses epochs and ordinal cuts using its validation data. Its
  optimized validation QWK is a tuning statistic, not an unbiased outer estimate.
- Five fitted models evaluated on the same test images have dependent errors.
  Their SD is not a confidence interval for performance on a new population.
- Paired tests over five fold scores provide weak evidence of equivalence; a
  nonsignificant result does not prove that preprocessing has no effect.

The general selection-bias concern is supported by
[Cawley and Talbot, JMLR 2010](https://jmlr.csail.mit.edu/beta/papers/v11/cawley10a.html).
For the next confirmatory comparison, freeze architecture, seeds, operating rule
and primary outcome before accessing a new holdout. If tuning is necessary, use
an inner development loop and evaluate the selected procedure on untouched outer
data. Additional test inspections must be recorded as exploratory.

## 4. External validity, calibration and uncertainty

The original model was tested with unchanged grade thresholds on two external
mirrors. IDRiD sensitivity was **88.49%** (35 missed / 304 referable); Messidor-2
was **32.82%** (307 missed / 457). This **55.66 percentage-point** difference is a
descriptive contrast between non-exchangeable cohorts, not a causal estimate.

The separately selected APTOS operating point gives **81.58%** and **26.04%**.
The explorer correctly keeps these two decisions separate. ROC AUC describes
ranking; it does not establish that an operational sensitivity transfers.
Site recalibration is model updating and must be tested on a separate site holdout.

Eye-based Wilson intervals are explicitly conditional on eye independence.
They do not account for bilateral clustering, dataset acquisition or label
uncertainty. The audit's new-run intervals have the same limitation. Genuine
patient linkage and per-eye external predictions are needed for defensible
patient-cluster inference. No significance test comparing the two cohorts is
claimed. Fairness across demographic groups is unassessed because adequate
demographic metadata and subgroup supports are unavailable.

## 5. Shortcut, label-noise and explanation claims

The completed Kaggle narrative recomputed metadata-only QWK **0.641**, versus
historical **0.652** with a different fold split. This is evidence that low-level
file/image summaries predict labels. Brightness and contrast still derive from
pixels; “no retinal pixel information” would be too strong. These baselines are
not a mathematical floor under the image model's QWK. Duplicate grouping across
metadata CV folds was not established as patient-independent.

The approximate 84% number derived from duplicate-label agreement is not a formal
population accuracy ceiling or measured inter-rater reliability. Selected
duplicate pairs and uncertain label provenance do not identify either quantity.

Withholding untrustworthy heatmaps is a strength. However, the reported Grad-CAM
concentration control and p=0.768 do not prove equivalence or establish that every
Grad-CAM map is invalid. The small, heterogeneous weight draws and overlapping
training data limit inference. The defensible conclusion is that this experiment
does not support a lesion-localization claim. This interpretation is consistent
with the need for model/data randomization controls discussed by
[Adebayo et al., NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html).

## 6. Reporting quality and acceptance boundaries

This review uses selected concepts from
[TRIPOD+AI, BMJ 2024](https://www.bmj.com/content/385/bmj-2023-078378): transparent
data provenance, separation of development and evaluation, performance reporting
and explicit limitations. It is not a completed 27-item compliance assessment.
Software test coverage verifies behavior contracts; it does not establish model
validity, generalization or clinical utility.

| Scope | Assessment |
|---|---|
| Professional portfolio | Strong: modular code, tests, reproducible evidence, honest failures and coherent presentation |
| Scientific contribution | Useful exploratory case study of shortcut and label/domain shift |
| New GPU result | Recorded-prediction audit completed; checkpoint inference not rerun |
| Patient-level inference | Not established; authentic linkage is missing |
| Ordinal-vs-MSE superiority | Not tested by the completed scalar-only run |
| Clinical use | Unsupported; no prospective utility or safety evaluation |
| Visual acceptance | Browser policy stopped native inspection; no desktop/mobile visual sign-off claimed |

The next scientifically necessary work is data-dependent: authentic patient
linkage, independent confirmatory data and matching frozen external predictions.
These cannot be replaced by additional styling, synthetic identities or another
score on the reused APTOS test split. Under the zero-budget constraint, retain the
current model and static explorer; do not launch more GPU experiments without a
specific falsifiable question and available free quota.

## Verification of this revision

- [Full CI run](https://github.com/senanurcetin/APTOS-2019-diabetic-retinopathy/actions/runs/40083142586):
  **393 tests passed**, coverage **72.37%**; Ruff passed. Python 3.10, 3.11
  and 3.12 each passed 274 pure tests (5 skipped, 66 deselected).
- Six new audit tests cover complete input, missing folds, mismatched identifiers,
  nonfinite scores, a changed ensemble summary and wrong cohort size.
- [Static publication](https://github.com/senanurcetin/APTOS-2019-diabetic-retinopathy/actions/runs/40083142575)
  succeeded. Live HTML, CSS, JavaScript and official logo matched local sources
  after normalizing line endings. The cohort rendering check passed without
  browser automation.
- Kaggle narrative version 8 was uploaded and its complete cell source read back
  exactly. Its executable cells are unchanged from completed version 6; the code
  SHA-256 is `e9d2b3d6d8ecb955433d3b62813f57382f2f2ff7f6951ae15f5a506c5fe53dee`.
  This source-equivalence check does not itself establish completion of the new
  execution. Check its live status before citing version 8 as executed.
