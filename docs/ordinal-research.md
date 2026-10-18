# Ordinal research extension (no productization)

This optional **cumulative-link** training experiment is separate from the
historical five-fold MSE regression benchmark. A timm model emits a single
latent severity score. Four learnable cutpoints are strictly ordered via
softplus-positive gaps, giving four noncrossing estimates of P(grade > k).
The ordinal target is `[grade>0, grade>1, grade>2, grade>3]` and uses
`BCEWithLogitsLoss`. Expected severity is the sum of four sigmoid probabilities.

**No new model performance is claimed.** The code and synthetic unit tests do
not train on the APTOS, IDRiD, or Messidor-2 images.

## Reproducible experiment design

Prepare Kaggle APTOS images using the existing pipeline. Do not move held-out
test cases into model selection. Then run paired MSE and ordinal experiments:

```sh
# CPU/GPU smoke test (no checkpoint or MLflow run; local diagnostic files may be written):
python -m aptos.training.cv --config configs/cv.yaml --mode ordinal --limit 25 --folds 5 --epochs 1 --no-track
# Five-fold, identical splits/seed:
python -m aptos.training.cv --config configs/cv.yaml --mode reg --seed 42
python -m aptos.training.cv --config configs/cv.yaml --mode ordinal --seed 42
```

Repeat with at least three seeds. Compare per-grade recall and macro F1,
quadratic-weighted kappa, referral sensitivity, specificity, calibration and
statistical uncertainty, not only aggregate QWK. Keep all seeds/configs,
checkpoint hashes and aligned OOF predictions.

The default model remains `reg`, and existing checkpoints/results are
unchanged. The research-only external evaluation adapter now understands
cumulative ordinal logits and converts them to expected grades before applying
**unchanged APTOS validation thresholds**. It checks the saved sweep's original
training mode, model, input shape and preprocessing settings before reading
even cached scores.

```sh
python -m aptos.evaluation.external --sweep models/cv/baseline-ORDINAL_RUN_ID \
  --dataset idrid --mode ordinal
python -m aptos.evaluation.external --sweep models/cv/baseline-ORDINAL_RUN_ID \
  --dataset messidor2 --mode ordinal
```

Replace the placeholder with the actual new sweep folder. Reports for ordinal
runs are written below `reports/research/`, never over the committed clinical
baseline report. The legacy ONNX serving path remains **scalar-regression only**;
do not load an ordinal checkpoint into the demo. No external dataset has been
scored by the new model yet.

The data remain under their respective licenses and outside this public
repository; no clinical decisions, claims of safety, or product rollout.
