# NEXUS clinical evidence and design layer

This enhancement separates **model performance evidence** from the appearance of
the live demo. The branding follows the *approved NEXUS Identity 2.0* assets,
not a derivative logo.

## Branding provenance

- Official vector wordmark and X mark: Google Drive `Logo/NEXUS_Identity_2_0`.
- NEXUS palette: midnight `#0B0F19`, deep blue `#061524`,
  cyan `#00E0FF`, blue `#4EA8FF`, lavender `#8F9BFF`,
  violet `#6E55C6`, and slate `#B9C5CE`.
- Figma brand source: https://www.figma.com/design/oA9wyvurYfypG4onVoa4pS
- Front-end files: `serving/static/nexus-theme.css`,
  `serving/static/nexus-logo.svg`, and `serving/static/nexus-x.svg`.

The research hero's retinal illustration retains anatomy-inspired warm colors.
Cyan/blue/lavender are used for the Nexus interface and the model comparison.

## Scientific reporting (no dataset/GPU/credentials required)

```bash
pip install -e ".[dev]"
python -m aptos.reporting.clinical_audit
python -m aptos.reporting.clinical_audit --format json
pytest -m pure tests/test_clinical_audit.py
```

The audit reads only the committed JSON evidence for IDRiD and Messidor-2.
It checks confusion-matrix totals and published sensitivity/specificity before
computing Wilson 95% *descriptive* intervals. It does not reconstruct CIs from
inaccessible image-level predictions. Any bilateral-patient clustering is
ignored by Wilson and **must** be addressed before inferential use.

## Source integrations

- **Kaggle** remains the source of downloaded, licensed APTOS / IDRiD /
  Messidor image datasets and the reproducible notebook. The new audit never
  requires Kaggle credentials or attempts a dataset download.
- **BigQuery** remains an optional label export at
  `scripts/load_to_bigquery.py`; it is not a model training dependency.
  This change does not alter, truncate, or query the connected BigQuery datasets.
- **Inference** uses the existing ONNX/PyTorch ensemble and scientific thresholds.
  This change adds an application-level size and media-type check for uploads;
  it does not retrain or modify model weights.

## Clinical boundaries

Messidor-2 referable sensitivity is **150 / 457 = 32.8%**. The UI and API must
not imply reliable clinical referral performance from the IDRiD result alone.
The app accepts JPG/PNG images up to 10 MiB and keeps no persistent copy of
images in its own code path. External hosting infrastructure may still retain
request metadata. Do not upload identifiable patient data.

Run the existing `pytest` suite and validate the live application before merging
this change. A completed PR is *not* a deployment.
