# Research notebooks

Run notebooks from the repository root after installing the package. Dataset
licenses and experiment provenance apply independently of notebook styling.

| Notebook | Role | Required inputs |
|---|---|---|
| [research_evidence.ipynb](research_evidence.ipynb) | Inspect committed historical aggregates, external results and fingerprints | Repository package and reports; no GPU or images |
| [ordinal_experiment_matrix.ipynb](ordinal_experiment_matrix.ipynb) | Configure a separate scalar/ordinal GPU experiment; defaults to dry-run | Authorized Kaggle inputs and free GPU quota for an actual run |
| [Kaggle publication source](kaggle_publication) | Public research narrative and executable metadata classifier | Attached APTOS mirror for execution |
| [Original project notebook](../aptos_2019.ipynb) | Root-level historical notebook retained for existing Colab/repository links | Follow its cells; it is separate from the current Kaggle publication |

The evidence notebook is a reader, not a substitute for retraining. The GPU
notebook is an experiment template: synthetic tests and a dry-run do not
establish a completed ordinal result. Use the
[ordinal benchmark](../docs/ordinal-benchmark.md) and
[GPU runbook](../docs/kaggle-gpu-launch.md) for its protocol.

## Published Kaggle narrative

`kaggle_publication/` preserves the existing public notebook ID. Version 10
completed and all 20 source cells matched the repository readback; see the
[verification record](../reports/research/kaggle_publication_v10.json).
Its CPU execution recomputes the metadata classifier. Historical imaging tables
come from the v1.0.0 release. The completed v2 imaging experiment is reported in
Markdown and links to its separate audited artifacts; it is not retrained by the
public CPU notebook. A future upload starts another execution and must be
verified before being described as complete.

## NEXUS identity

The evidence and GPU notebooks embed the unmodified official SVG wordmark as a
standard attachment for offline portability. The published notebook retains its
NEXUS attachment. See [identity provenance](../docs/PORTFOLIO.md#official-nexus-identity)
for source assets and colors. Chart styling does not change numeric evidence or
establish clinical validity.

For installation and commands, use the
[reproduction guide](../docs/REPRODUCIBILITY.md). Raw images, per-eye execution
artifacts and model checkpoints remain outside Git.
