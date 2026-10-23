# Kaggle GPU experiment launch — no CI and no credential commits

This is a public, non-productized research repository. The APTOS experiment
requires genuine GPU runs, not placeholder performance numbers.

## Security and access

Kaggle's current CLI supports the access-token environment variable
`KAGGLE_API_TOKEN` or a protected `~/.kaggle/access_token` file. Do **not**
put the token in this repository, Kaggle notebook cells, kernel metadata,
issues, GitHub Actions secrets or shell command arguments. Tokens pasted into
a chat should be rotated from Kaggle account settings. To keep a new token
out of shell history, configure it securely on a computer you control.

On 23 October 2026 the authorized local Kaggle CLI was reachable. The existing
`aptos-research-reg-seed42` kernel reported COMPLETE. This status alone does
not establish five complete trained folds; inspect metrics and artifacts before
publishing new results. The public narrative uses CPU and is independent of GPU
training. See [portfolio delivery](PORTFOLIO.md).

## One model and seed per private GPU kernel

On a computer with the authorized Kaggle CLI and this repository checked out:

```sh
python -m pip install -U kaggle
python scripts/package_kaggle_experiment.py --owner senanuretin --mode reg --seed 42
python scripts/package_kaggle_experiment.py --owner senanuretin --mode ordinal --seed 42
```

Those commands create local bundles under `outputs/kaggle_research/`
**without contacting Kaggle** and with GPU execution disabled. After the
credential is securely configured, upload and run *one model at a time*:

```sh
python scripts/package_kaggle_experiment.py --owner senanuretin --mode reg --seed 42 --run --push
kaggle kernels status senanuretin/aptos-research-reg-seed42
# When the first job is complete:
python scripts/package_kaggle_experiment.py --owner senanuretin --mode ordinal --seed 42 --run --push
kaggle kernels status senanuretin/aptos-research-ordinal-seed42
```

The owner `senanuretin` is taken from the Kaggle notebook referenced in
README; override `--owner` if the connected account differs.
Use `kaggle kernels output <owner>/aptos-research-reg-seed42` to download
actual experiment artifacts after completion. Following the first verified
comparison, repeat with seeds 43 and 44 as quota permits. A failed or partial
CV run is **not** an experiment result.

## What the notebook actually does

Each generated bundle is private, requests a Kaggle GPU and internet access,
attaches the public mirror `mariaherrerot/aptos2019`, clones the public
GitHub repository, records its commit, installs training requirements, then
runs:

1. Strict APTOS label and PNG image preflight (3,662 unique eyes, exactly one
   PNG per ID). Original images are mounted read-only rather than duplicated.
2. Historical pipeline `quality` dependencies: prepare labels, scan images,
   preprocess 512px control images and check duplicate/leakage candidates.
3. Full five-fold CV on the selected model mode and seed, `--no-track` to
   avoid MLflow/BigQuery or cloud credentials.
4. Verify that the five fold checkpoint, metrics and prediction files exist.
   An incomplete training session causes an explicit error, not a new metric.

**The notebook does not run IDRiD or Messidor-2 automatically.** Those are
separate external experiments that require authorized cohorts, complete
matching model checkpoints, authentic per-eye identity, and frozen APTOS
thresholds. See `docs/ordinal-benchmark.md`.

### Limits

Kaggle dataset versions, actual job status, GPU quotas, run times and
measured outcomes can only be confirmed after real Kaggle execution. The
preflight checks label/image IDs and original input layout but does not hash
all source retinal pixel bytes. Preserve dataset versions for the actual run.
This notebook is not a clinical diagnostic tool.

## Verified CLI upload semantics

The official Kaggle CLI `kernels push` **starts a kernel run**; the local
CLI 1.8.4 does not accept a `--no-run` flag. The bundle builder therefore
rejects `--push` unless `--run` is also supplied. Generating a bundle
without `--push` remains a completely offline dry run.

The user's connected SentinelX Windows host now has an isolated Kaggle CLI
installation under `C:\\ProgramData\\SentinelX\\workspace\\aptos-kaggle-cli`.
This only verifies that the command-line program launches. It does **not**
configure authentication or establish Kaggle GPU availability. API secrets
must be configured locally, never committed or echoed into logs.

## Kaggle input mount path mismatch (October 2026)

The first Kaggle MSE/seed42 attempt terminated after approximately 27 seconds
with `FileNotFoundError` because the old notebook assumed
`/kaggle/input/aptos2019` exists. A T4 GPU allocation alone does not
establish that the dataset was attached.

The corrected notebook now searches under `/kaggle/input` for one
dataset directory containing all three **specific** historical
`train_1.csv`, `valid.csv` and `test.csv` label files, then runs the
existing strict preflight over this resolved mount path. No fallback uses
unrelated images or fabricates missing data. If nothing is found, the
notebook reports the visible mount directory names and instructs the operator
to use Kaggle's **Add Input → mariaherrerot/aptos2019**. If multiple matching
mounts exist, it aborts to prevent ambiguous research provenance.

This correction is not evidence of successful model training; a new kernel
version must run and report five complete folds before any QWK metric is
claimed. A Kaggle API credential previously shared in chat may be revoked or
expired; configure a fresh private token on the authorized computer and
do not paste it in chat.
