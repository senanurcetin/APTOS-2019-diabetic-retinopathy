# Messidor-2 v2: executable follow-up protocol

Fixed on **26 October 2026, before starting the v2 GPU run**. This is an
exploratory follow-up on a previously inspected cohort, not prospective
registration or fresh external confirmation.

## Question and baseline

Does training on adjudicated rather than model-generated labels improve
referable discrimination on examination-disjoint Messidor-2 eyes, given identical
starting models, images, validation targets and optimization settings?

The baseline is **`baseline-c8232af68337`**, the completed scalar MSE seed-42 run
in `senanuretin/aptos-research-reg-seed42`. Its metadata and saved APTOS predictions
must match [the existing audit](../reports/research/kaggle_reg42_audit.json)
byte-for-byte before any tensor is loaded. Require all five checkpoints; load
only tensor/state metadata with `weights_only=True`. Record each weight hash.
This baseline differs from the historical deployed ensemble and the experiment
that reported AUC 0.925. Do not combine their numerical results.

## Data and frozen design

- Preserve the [frozen examination allocation](MESSIDOR2_FOLLOWUP.md): 698 training,
  174 validation, 872 test eyes; zero cross-partition supplied examinations.
- Verify Google grade bytes, the complete image-tree fingerprint, pair-table
  bytes and the exact frozen split CSV hash. Recreate using scikit-learn 1.8.0
  and pandas 2.2.3. Stop on any mismatch; never search for a favorable seed.
- Resolve `.jpg`/`.JPG` filename casing by unique, exact image stem only. Keep
  Google's adjudicated grades unchanged; reject ambiguous or missing files.
- Both arms start from the same five frozen APTOS-only models. No historical
  fine-tuned weight is eligible: 426 new test eyes were in old development.
- Use the existing 512-pixel preprocessing/JPEG cache and 384-pixel training
  transforms. Record configuration, preprocessing signature, source commit and
  GPU/PyTorch versions. Source images/caches remain under temporary storage.
- AdamW, learning rate `1e-4`, weight decay `1e-4`, cosine schedule, **8 epochs**,
  seed **42**, MSE. Select the best validation QWK epoch for each model.
- **Only training targets vary.** The adjudicated arm uses panel grades; the
  control uses the unchanged ensemble's grades at its original frozen cuts.
  Validation targets are adjudicated for **both** arms. This differs from the
  historical control's pseudo-labeled validation and is intentional: model
  selection and validation threshold fitting use a common reference standard.
- Use a new output directory; refuse existing directories. No implicit resume,
  cached checkpoint reuse, additional seed, alternate hyperparameter or model
  promotion is permitted by this protocol.

## Endpoints and interpretation

The **primary contrast** is adjudicated-minus-control referable ROC AUC on the
same 872 held-out eyes. Resample the **437 supplied test examinations** with all
their eyes together, using the same draws for all arms: 2,000 draws, seed 42,
percentile 95% interval. Report the effect even if negative or its interval
includes zero. An interval excluding zero supports this fitted comparison, not
a universal causal or clinical claim.

Secondary comparisons are adjudicated-minus-baseline and control-minus-baseline.
They are descriptive, with no multiplicity-adjusted superiority claim. Report
QWK, macro F1, accuracy, class support/recall, referral confusion and paired
examination intervals at the **common original APTOS cuts**. Separately show
each arm's validation-fitted cuts; do not mix threshold adaptation with ranking.

Also evaluate retention on the exact 366 cached APTOS test eyes attached to the
audited base run. Report the unchanged baseline's re-inference difference from
its saved scores, then all arms' metrics at original and Messidor-fitted cuts.
This is a reused holdout diagnostic, not a fresh estimate or selection criterion.
No new IDRiD, longitudinal-patient or prospective clinical evaluation is implied.

## Required outputs and cost boundary

Save a launch/completion manifest, five checkpoint/history files per arm,
per-eye Messidor and APTOS raw scores, all thresholds, paired comparison CSVs,
aggregate results and hashes. Download per-eye outputs into ignored `outputs/`;
publish only aggregate findings and code. Retain the historical reports unchanged.

Use one **private, free Kaggle T4 run** with attached source-kernel outputs and
the already identified image/grade datasets. Stop if CUDA or quota is unavailable;
do not switch to paid infrastructure. No training service, subscription, image
redistribution or automatic model deployment is introduced.

The runner is `aptos.training.examination_followup`; the pinned private bundle is
created by `scripts/package_examination_followup.py`. Packaging requires a clean
committed source. The existing public Kaggle narrative is updated only after
the actual run and prediction audit have completed.
