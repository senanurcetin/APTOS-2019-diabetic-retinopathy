# Messidor-2: examination-disjoint follow-up results

Noncommercial exploratory research. New paired fine-tuning run; distinct from the historical AUC 0.925 experiment.

Baseline: `baseline-c8232af68337`. Test: **872 eyes / 437 supplied examinations**. [Protocol](../../docs/MESSIDOR2_V2_PROTOCOL.md) · [frozen allocation](../../docs/MESSIDOR2_FOLLOWUP.md).

## Primary contrast

Adjudicated minus control referable AUC: **+0.0901**, 95% examination-bootstrap interval **[+0.0614, +0.1204]**.

The adjudicated arm has higher referable AUC than the control in this fitted comparison. The cohort was inspected previously; this is not fresh external confirmation.

## Same original APTOS cuts for all arms

AUC uses continuous scores. Classification metrics use the same original APTOS thresholds, without external refitting.

| Arm | Referable AUC | QWK | Macro F1 | Sensitivity | Specificity | False negatives |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.8405 | 0.5129 | 0.2947 | 36.8% | 98.1% | 144 |
| adjudicated | 0.9435 | 0.7920 | 0.4809 | 72.8% | 95.5% | 62 |
| control | 0.8534 | 0.6111 | 0.4167 | 46.1% | 96.4% | 123 |

Original cuts: `0.708, 1.124, 2.42, 3.284`.

## Separately fitted Messidor validation cuts

Both arms use adjudicated validation targets; only training targets differ. The control's training labels are the unchanged model's own grades.

| Arm | QWK | Sensitivity | Specificity | False negatives | Validation cuts |
|---|---:|---:|---:|---:|---|
| baseline | 0.5129 | 36.8% | 98.1% | 144 | 0.7080, 1.1240, 2.4200, 3.2840 |
| adjudicated | 0.8108 | 64.5% | 98.0% | 81 | 0.6520, 1.3560, 2.0600, 3.4760 |
| control | 0.6037 | 40.8% | 96.7% | 135 | 0.6120, 1.2440, 2.4200, 3.4760 |

## Severity support and recall at original cuts

| True grade | Eyes | Baseline recall | Adjudicated recall | Control recall |
|---|---:|---:|---:|---:|
| 0 | 509 | 91.4% | 88.6% | 87.6% |
| 1 | 135 | 8.9% | 30.4% | 17.8% |
| 2 | 162 | 17.3% | 46.9% | 27.2% |
| 3 | 42 | 14.3% | 52.4% | 40.5% |
| 4 | 24 | 8.3% | 20.8% | 16.7% |

## APTOS retention: reused 366-eye holdout

These diagnostics were not used for epoch selection or a new model promotion. All rows below use the original APTOS cuts.

| Arm | Referable AUC | QWK | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| baseline | 0.9809 | 0.8998 | 0.8060 | 0.5496 |
| adjudicated | 0.9677 | 0.8579 | 0.6421 | 0.4128 |
| control | 0.9791 | 0.8955 | 0.7377 | 0.4981 |

Baseline re-inference differs from its saved scores by at most 0.021574 raw-score units, with 1 grade mismatch(es). This comparison consistently uses the re-inferred baseline; numerical parity is not bit-exact.

## Provenance and limitations

GPU source commit: `17be6cff382c2bcdc438c68e73773fa1fe812767`. Ten new fold histories were checked; results and paired intervals were independently recomputed from saved scores. Weight tensors were not loaded by the independent audit.

Bootstrap: 2,000 paired examination draws, seed 42, percentile 95% intervals. The primary contrast was fixed before the run; secondary contrasts are descriptive. Validation has only two grade-4 eyes; test has 24. Examination pairing is a third-party ADCIS transcription, not verified longitudinal patient identity.

- Exploratory follow-up on a previously inspected cohort; not fresh external confirmation.
- Third-party examination mapping; original ADCIS parity and patient identity unverified.
- New baseline sweep differs from the historical 0.925 AUC experiment.
- Grade-4 support: two validation eyes and 24 test eyes; rare-grade metrics are unstable.
- Secondary contrasts are descriptive; bootstrap conditions on these fitted models.
- APTOS retention reuses the inspected 366-eye holdout; no fresh IDRiD or prospective assessment.
- No automatic model promotion or clinical-use claim.
