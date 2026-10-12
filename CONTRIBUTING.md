# Contributing to APTOS-2019 Diabetic Retinopathy Grading

Contributions are welcome for reproducibility, documentation, tests and well-scoped technical improvements. Open an issue describing the change before proposing a major feature or architecture rewrite.

## Local checks

```bash
python -m pip install -e .
python -m pip install pytest ruff
pytest -m pure -q
ruff check .
```

The complete training/evaluation suite requires additional scientific dependencies and public retinal image datasets. Dataset access is subject to its source terms and should not be committed to the repository.

## Pull request requirements

- Explain the problem, the exact code or data paths changed and how the result was validated.
- Add tests or reproducible evidence for changes affecting model behavior, metrics or user-facing features.
- Document dataset provenance, license constraints, split methodology and evaluation limitations.
- Include no secrets, personally identifiable data, proprietary plant data, or bulky downloaded training datasets.
- Keep metrics honest: label CV, holdout and demo/synthetic results separately.

## Responsible disclosure

Please use the private process in [SECURITY.md](SECURITY.md) for vulnerabilities rather than a public issue.
