# Paired external-validation uncertainty (research-only)

The project's historical external reports are **aggregate counts**. They support
the published sensitivity/specificity estimates and descriptive Wilson
intervals, but they **cannot** reconstruct paired per-eye predictions, patient
linkage, or patient-level uncertainty. No such data is fabricated.

This module compares two fixed models scored on the **same eyes** with frozen
operating thresholds. It is used by the completed examination-disjoint v2
follow-up; see [audited results](../reports/research/messidor2_v2_results.md). Run it locally using
the original data holder's permitted data; do not commit identifiable records.

## Input CSV contract

Required columns: id_code, diagnosis, baseline_grade, candidate_grade.
diagnosis and the two model predictions are ordinal grades 0–4. One row is
one eye. Each image ID must appear exactly once. For patient-level cluster
resampling also supply a genuine patient_id column; **never derive patient
identity from image filename patterns**.

Explicitly select the sampling unit:

    python -m aptos.evaluation.paired_uncertainty \
      --predictions data/local/paired_predictions.csv --unit eye \
      --draws 2000 --seed 42 --output reports/local_paired_uncertainty.json

    python -m aptos.evaluation.paired_uncertainty \
      --predictions data/local/paired_predictions.csv --unit patient \
      --draws 2000 --seed 42

Patient mode resamples whole patient clusters, preserving both eyes and all
within-patient correlation if the linkage keys are accurate. Eye mode assumes
independence of each eye and may yield undercoverage if patients contribute
multiple eyes. Neither is an independent prospective clinical validation.

## Methodology and safeguards

- Paired sampling: both models are evaluated on identical resampled eye rows.
  Never use this method to directly contrast independent IDRiD and Messidor-2
  cohorts; those data are *not paired*. Site transfer needs its own design.
- Fixed threshold: referable equals predicted grade >= 2. Evaluation must use
  thresholds frozen **before** scoring its held-out samples. Do not choose
  a threshold using these samples and then report this interval as unbiased.
- Statistics: percentile 95% intervals for the *candidate minus baseline*
  change in sensitivity, specificity and referral rate. Bootstrap resamples
  the declared unit with replacement and uses a local seeded NumPy RNG.
- Rare class draws: replicates missing a class are counted and discarded;
  the function rejects studies with more than 20% unusable draws. Selection
  changes the bootstrap distribution; inspect the rejected count and seek
  larger cohorts when positive/negative patients are scarce.
- No p-values, QWK uncertainty or clinical safety certifications are inferred
  from this module. It cannot prove interchangeability or generalizability.
- Output includes only aggregate effects, confidence limits and an input
  SHA-256 fingerprint. Patient names and eye identifiers are not returned.

When actual paired predictions are unavailable, continue to use the
content-addressed committed evidence manifest rather than reconstructing
imaginary patient-level measurements:

    python -m aptos.reporting.research_evidence

The research code remains public and non-commercial; data and real patient
identifiers must stay outside this repository.
