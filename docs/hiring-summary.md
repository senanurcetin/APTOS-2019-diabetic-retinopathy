# Reviewer brief — APTOS 2019 Retinopathy Grading

**Role:** Applied computer vision, experimental design and responsible medical-image ML. **Scope:** research demonstration, not a clinical diagnostic device.

## Question and evidence

Can a retinal-image classifier generalize beyond dataset shortcuts, label noise and a high internal metric?

| Review question | Evidence |
|---|---|
| What is the measured result? | **Test QWK 0.9098** for the five-fold ensemble; see [result details](../RESULTS.md) |
| Are dataset shortcuts present? | Image metadata alone achieved **QWK 0.652**; [confounding figure](../reports/figures/07_resolution_confound.png) |
| Were preprocessing gains replicated? | CLAHE and geometric squash improvements did not achieve convincing consistent gains; [pre-registered experiments](WRITEUP.md) |
| Is there out-of-domain evidence? | External IDRiD result **QWK 0.8045**, below the internal test QWK; [experimental results](../RESULTS.md) |
| Can a reviewer examine the model? | [Serving model card](../serving/MODEL_CARD.md), [full README](../README.md), [source tests](../tests/) |

## Engineering choices

- Cross-validation, threshold search and explicit label-noise analysis.
- Testing whether model behavior depends on retinal structure rather than camera metadata.
- Separating a headline ordinal score from minority-class performance and domain shift.

## Restrictions

The data and its original terms are separate from the [MIT code license](../LICENSE) (see [NOTICE](../NOTICE)). No prediction should be used for patient care. A QWK around 0.91 does **not** imply equivalent sensitivity, specificity or reliability across clinical subgroups.

**Further reading:** [Three predictions written before testing](WRITEUP.md).
