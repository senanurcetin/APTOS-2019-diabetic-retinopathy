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
