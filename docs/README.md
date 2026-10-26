# Documentation

The repository is a noncommercial investigation of retinal grading under
acquisition and label shift. Start with the results; use the implementation
runbooks when reproducing an experiment.

## Reading paths

| Purpose | Document |
|---|---|
| Project overview and layout | [Repository README](../README.md) |
| Complete numerical record | [Results](../RESULTS.md) |
| Current completed follow-up | [Audited Messidor-2 v2 results](../reports/research/messidor2_v2_results.md) |
| Methodological assessment and limitations | [Scientific review](SCIENTIFIC_REVIEW.md) |
| Narrative explanation | [Research write-up](WRITEUP.md) |
| Installation, data, training and audits | [Reproduction guide](REPRODUCIBILITY.md) |
| NEXUS explorer, identity and static hosting | [Explorer guide](PORTFOLIO.md) |
| Notebook roles and published source | [Notebook index](../notebooks/README.md) |

## Methods and operational runbooks

| Document | Scope |
|---|---|
| [Open research workflow](open-research-workflow.md) | Offline evidence checks, label integrity, optional read-only SQL |
| [Operating-point audit](operating-point-audit.md) | Original grade decisions versus the separately selected referral rule |
| [Paired uncertainty](paired-uncertainty.md) | Aligned per-eye predictions and resampling contracts |
| [Ordinal benchmark](ordinal-benchmark.md) | Optional cumulative-link comparison; no completed superiority claim |
| [Kaggle GPU launch](kaggle-gpu-launch.md) | Private experiment packaging, input mounts and free-quota boundaries |
| [Messidor acquisition](MESSIDOR2_ACQUISITION.md) | Sources, filename resolution and historical examination overlap |
| [Messidor follow-up allocation](MESSIDOR2_FOLLOWUP.md) | Examination-disjoint split and uncertainty interface |

## Dated hypotheses and frozen design

These are research records, not current installation instructions. Their dates,
initial predictions, appended outcomes and design constraints are retained.

- [IDRiD prediction and outcome](external-validation-prediction.md).
- [Messidor-2 prediction and outcome](second-external-validation-prediction.md).
- [Historical fine-tuning prediction and outcome](finetune-prediction.md).
- [Frozen examination-disjoint v2 protocol](MESSIDOR2_V2_PROTOCOL.md).

## Evidence locations

`reports/` contains historical generated reports and figures. `reports/research/`
contains newer aggregate audits, allocation snapshots, execution manifests and
v2 results. A preparation snapshot may say "not started" because it records the
state at freezing; current completion is in the
[final verification record](../reports/research/messidor2_v2_completion.json).
Do not rewrite a frozen plan to make it resemble a completion report.

Raw images, per-eye execution downloads and model checkpoints are kept outside
Git. Artifact hashes document identity and consistency; they do not independently
prove data provenance or clinical validity.
