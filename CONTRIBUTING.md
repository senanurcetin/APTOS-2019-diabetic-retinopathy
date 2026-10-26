# Contributing to APTOS-2019 Diabetic Retinopathy Grading

Contributions are welcome for reproducibility, documentation, tests and well-scoped technical improvements. Open an issue describing the change before proposing a major feature or architecture rewrite.

## Local checks

```bash
python -m pip install -e ".[dev]"
pytest -m pure -q
ruff check .
python -m aptos.reporting.portfolio --check
```

The complete training/evaluation suite requires additional scientific dependencies and public retinal image datasets. Dataset access is subject to its source terms and should not be committed to the repository.

## Pull request requirements

- Explain the problem, the exact code or data paths changed and how the result was validated.
- Add tests or reproducible evidence for changes affecting model behavior, metrics or user-facing features.
- Document dataset provenance, license constraints, split methodology and evaluation limitations.
- Include no secrets, personally identifiable data, or bulky downloaded training datasets.
- Keep metrics honest: label CV, holdout and demo/synthetic results separately.

## Responsible disclosure

Please use the private process in [SECURITY.md](SECURITY.md) for vulnerabilities rather than a public issue.

## Documentation and research records

Use the [documentation index](docs/README.md) and [reproduction guide](docs/REPRODUCIBILITY.md).
Preserve dated hypotheses, frozen protocols and historical output artifacts.
Corrections should identify the affected experiment rather than silently
rewriting its design. Regenerate the explorer when its input reports change.
Keep per-eye downloads and model weights outside Git. This project remains
noncommercial research; software checks do not establish clinical validity.
