# Research notebooks

The research notebooks are reproducible evidence **readers**, not clinical
screening applications or substitutes for retraining.

- [research_evidence.ipynb](research_evidence.ipynb) — compare recorded five-fold
  QWK results, external referral performance and source SHA-256 fingerprints.
  The notebook does not require GPU, cloud credentials or the Kaggle images.
  To run it, install the repository as a Python package and start Jupyter from
  the repo root.
- [Operating-point evidence audit](../docs/operating-point-audit.md) — a
  dependency-light CLI comparing two different recorded referral decisions
  without inventing per-eye predictions.

Raw image datasets and labels remain under their own licenses and are not
included in this public repository. For Kaggle workflows, see
[open research protocol](../docs/open-research-workflow.md).

## NEXUS Identity 2.0 — research visuals

Both `research_evidence.ipynb` and `ordinal_experiment_matrix.ipynb` now
include the **unaltered official transparent NEXUS SVG wordmark** as a standard
notebook attachment, so the approved identity remains visible without a network
request or a Drive sign-in. The source vector is
`serving/static/nexus-logo.svg` (same bytes as the Google Drive approved asset).
No generated or unofficial substitute logo is used.

The evidence notebook applies the identity's midnight/cyan/blue/lavender
palette to **chart presentation only**. Underlying metrics, cohorts,
error bars, bootstrap intervals and clinical warnings are unchanged.
The GPU notebook uses exactly the same wordmark and an explicit
one-mode/one-seed experimental contract; it still defaults to dry-run.
Kaggle kernel packaging copies the attachment unmodified.

Graphic styling does not constitute training, a valid model result, a
deployment, a medical device, or productization. The associated official
source and color specifications are recorded in
[the identity design audit](../docs/nexus-clinical-research.md).
