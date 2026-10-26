# Scientific assessment — APTOS / NEXUS

Assessment dates: **25–26 October 2026**. This is a repository-based methodological
audit by the development assistant, not independent peer review or a clinical
certification. Historical outputs, current source, Kaggle execution artifacts and
relevant primary methodological references were inspected. The initial review did
not train a model; the separately documented v2 paired experiment subsequently
completed on 26 October. The project remains noncommercial research.

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

A newly retrieved author-released pairing table attributed to ADCIS contains
**874 examinations / 1,748 unique eyes**. All **1,744** historical split IDs match;
Google's gradable labels match the historical diagnoses exactly. There are
**870 complete pairs and four examinations with one gradable eye**.

**The historical split is not examination-disjoint:** seven examinations cross
partitions: five train/test, one validation/test and one train/validation.
Six of 872 test eyes have a development-set partner. The size or direction of
resulting optimism cannot be estimated from the aggregate results.

The recorded test-half AUCs (0.830 unchanged, 0.925 adjudicated, 0.834 control)
remain historical exploratory observations. They cannot support an independent
examination-level confirmation or definitive causal attribution. The original
frozen APTOS model's external failure analysis is a separate experiment and is
not caused by this later fine-tuning split error.

The pair table is an **author-released transcription**, pinned to RFM commit
`b8c9322863b5e2d79df964e524d74ad53d1d1b7a`. Its README attributes it to the official
ADCIS spreadsheet; original-file parity has not been independently verified.
It identifies examinations, not longitudinal patients. Raw mapping/images remain
local. [Acquisition, sources and reproduction](MESSIDOR2_ACQUISITION.md) document
this distinction; [aggregate split audit](../reports/research/messidor2_examination_audit.json)
and [download fingerprints](../reports/research/messidor2_sources.json) preserve evidence.

**Still unresolved:** matching per-eye predictions for both historical fine-tuned
arms were not found in the authenticated Kaggle inventory or pinned model release.
Images and labels alone cannot reconstruct those outputs. Clustered confidence
intervals cannot fix training/test overlap. A new prespecified examination-grouped
split and retraining would be a new experiment; deleting six test eyes after
seeing the result would not restore a fresh confirmatory test.

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
| Original scalar GPU audit | Saved predictions checked; that initial audit did not rerun inference |
| Examination v2 follow-up | Ten new fine-tuned models; per-eye metrics and intervals independently recomputed |
| Patient-level inference | Longitudinal identity unverified; supplied exam pairs reveal split overlap |
| Ordinal-vs-MSE superiority | Not tested by the completed scalar-only run |
| Clinical use | Unsupported; no prospective utility or safety evaluation |
| Screen-based visual acceptance | Excluded by user instruction; CLI/API/source checks passed, no pixel-rendering claim |

The next scientifically necessary work is data-dependent: independently verified
linkage, fresh confirmatory cohorts and multi-seed robustness.
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
  Version 8 subsequently reported **COMPLETE**. Its downloaded execution log
  confirms 3,662 images, metadata QWK **0.641** and HTML conversion.
  [Final execution record](../reports/research/kaggle_publication_v8.json) includes
  the code/log hashes and source readback result. No imaging model was retrained.
- Final checks used CLI/API, source files, Node VM and tests only. No PC-screen
  access or Playwright was used after the user's exclusion instruction.

## 25 October acquisition revision: verification

- [Linux CI](https://github.com/senanurcetin/APTOS-2019-diabetic-retinopathy/actions/runs/40216299248)
  passed **401 tests**. Python 3.10/3.11/3.12 each passed 282 pure tests;
  Ruff passed. The local Windows run passed 281 pure tests but one existing
  symlink staging test lacked OS privileges; that same test passed on Linux.
- Eight new pairing tests cover overlap, single included eyes, disjoint exams,
  duplicate members/IDs, missing IDs/partitions and invalid filenames.
- The generated fine-tuning report now preserves its split limitation; the
  formatter regression check and all eight pairing tests passed locally.
- [Pages deployment](https://github.com/senanurcetin/APTOS-2019-diabetic-retinopathy/actions/runs/40216092250)
  succeeded; live HTML/CSS/JS/logo matched source. Node VM cohort rendering passed.
- Kaggle narrative v9 was submitted and its full cell source read back exactly.
  Executable cells are identical to completed v8. Version 9 subsequently reported
  **COMPLETE**; its downloaded log confirms 3,662 images, metadata QWK **0.641**
  and HTML conversion without recorded exceptions. The
  [v9 verification record](../reports/research/kaggle_publication_v9.json)
  contains source/code/log fingerprints. No imaging model was retrained.

## Examination-disjoint follow-up preparation

The [v2 follow-up split](MESSIDOR2_FOLLOWUP.md) is now prepared with 698 training,
174 validation and 872 test eyes, representing 350/87/437 examinations. Under
the supplied pairing transcription, **no examination spans partitions**.
The old split was preserved byte-for-byte. The allocation, seed, software
versions, source/code hashes and class counts are recorded in the
[aggregate plan](../reports/research/messidor2_examination_v2_plan.json).

At allocation closeout this was a methodological repair; the subsequent model
experiment is documented below. In particular,
339 new test eyes were in historical training and 87 in historical validation;
old fine-tuned checkpoints must not be reused. Grade-4 support is only two in
validation and 24 in test. Examination bootstrap is now an explicit software
option, distinct from patient bootstrap. At that preparation stage there were no
v2 model outputs; these were subsequently produced and audited below. The historical
Kaggle narrative and all historical performance values retain their prior scope.

## Verification completed — 26 October 2026

[CI for source commit 02a0608](https://github.com/senanurcetin/APTOS-2019-diabetic-retinopathy/actions/runs/40218008457)
passed **418 tests**, coverage **73.69%**, and Ruff. Python 3.10/3.11/3.12 each
passed 299 pure tests. Real-cohort checks reconfirmed all labels and IDs,
zero follow-up cross-partition examinations, unchanged historical CSV bytes,
and matching split/source-code hashes. The
[verification record](../reports/research/messidor2_examination_v2_verification.json)
pins the CI source, downloaded log and aggregate plan. This verifies software
and allocation; no new model was trained and no new performance was measured.

## Completed v2 paired model experiment — 26 October 2026

The [executable protocol](MESSIDOR2_V2_PROTOCOL.md) and new training run are now
complete. [Per-eye audited results](../reports/research/messidor2_v2_results.md)
use the newer scalar seed-42 baseline, not the historical deployed ensemble.
On 872 test eyes / 437 supplied examinations, referable AUC is **0.8405** unchanged,
**0.9435** adjudicated and **0.8534** control. The primary adjudicated-minus-control
difference is **+0.0901**, 95% paired examination interval **[+0.0614, +0.1204]**.
Only training targets differ; adjudicated validation targets are common to both
arms. The 10 new histories, threshold decisions, per-eye outputs and paired
intervals passed [independent recomputation](../reports/research/messidor2_v2_audit.json).

At common original cuts, sensitivity rises from **36.8% to 72.8%**, specificity
changes from **98.1% to 95.5%**, and missed referable eyes fall from **144 to 62**.
This is a conditional improvement with remaining errors, not clinical readiness.
The adjudicated arm's grade-4 recall is **20.8%** on 24 test eyes. On the reused
APTOS holdout, accuracy falls from **80.6% to 64.2%** and QWK from **0.8998 to 0.8579**.
No model was promoted. Baseline re-inference has maximum raw difference 0.021574
and one grade mismatch versus the saved base scores; the paired comparison uses
the re-inferred baseline consistently. Original ADCIS parity, longitudinal
patient identity, new independent cohorts and multi-seed robustness remain open.
The frozen allocation plan records its earlier preparation state; current
completion is recorded separately in the [launch manifest](../reports/research/messidor2_v2_launch.json).
