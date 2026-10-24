# Reviewer brief — APTOS 2019 Retinopathy Grading

**Role:** Applied computer vision, experimental design and responsible medical-image ML. **Scope:** research demonstration, not a clinical diagnostic device.

**[Explore the NEXUS research dashboard](https://senanurcetin.github.io/APTOS-2019-diabetic-retinopathy/)** · [Executed Kaggle narrative](https://www.kaggle.com/code/senanuretin/aptos-three-predictions-i-wrote-before-the-data) · [Delivery evidence](PORTFOLIO.md)

## Question and evidence

Can a retinal-image classifier generalize beyond dataset shortcuts, label noise and a high internal metric?

| Review question | Evidence |
|---|---|
| What is the measured result? | **Test QWK 0.9098** for the squash ensemble; the externally evaluated baseline scored **0.9091**. These are distinct experiments; see [result details](../RESULTS.md) |
| Are dataset shortcuts present? | Image metadata alone achieved **QWK 0.652**; [confounding figure](../reports/figures/07_resolution_confound.png) |
| Were preprocessing gains replicated? | CLAHE and geometric squash improvements did not achieve convincing consistent gains; [pre-registered experiments](WRITEUP.md) |
| Is there out-of-domain evidence? | IDRiD **QWK 0.8045**, Messidor-2 **QWK 0.4928**. At the original grade-2 boundary, referral sensitivity fell from **88.5% to 32.8%**. [Fixed-threshold evidence](PORTFOLIO.md#interpretation-boundaries) |
| Can a reviewer examine the model? | [Serving model card](../serving/MODEL_CARD.md), [full README](../README.md), [source tests](../tests/) |

## Scientific review

[Methodological assessment](SCIENTIFIC_REVIEW.md) records a completed five-fold
saved-prediction audit (new-run QWK 0.900763), the correction from patient groups
to filename proxies, and remaining limits on causal and patient-level inference.
The new run does not replace the historical model or its external evidence.

## Engineering choices

- Cross-validation, threshold search and explicit label-noise analysis.
- Testing whether model behavior depends on retinal structure rather than camera metadata.
- Separating a headline ordinal score from minority-class performance and domain shift.
- Auditing aggregate report consistency, source hashes and two distinct operating
  points, with descriptive uncertainty and explicit patient-clustering limitations.
- Publishing an independent static explorer without image uploads, model weights,
  paid infrastructure or a dependency on the inference service.

## Restrictions

The data and its original terms are separate from the [MIT code license](../LICENSE) (see [NOTICE](../NOTICE)). No prediction should be used for patient care. A QWK around 0.91 does **not** imply equivalent sensitivity, specificity or reliability across clinical subgroups.

**Further reading:** [Three predictions written before testing](WRITEUP.md).
