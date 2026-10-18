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
# CPU/GPU smoke test, never saved to historical results:
python -m aptos.training.cv --config configs/cv.yaml --mode ordinal --limit 10 --no-track
# Five-fold, identical splits/seed:
python -m aptos.training.cv --config configs/cv.yaml --mode reg --seed 42
python -m aptos.training.cv --config configs/cv.yaml --mode ordinal --seed 42
```

Repeat with at least three seeds. Compare per-grade recall and macro F1,
quadratic-weighted kappa, referral sensitivity, specificity, calibration and
statistical uncertainty, not only aggregate QWK. Keep all seeds/configs,
checkpoint hashes and aligned OOF predictions.

The default model remains `reg`, and existing checkpoints/results are
unchanged. These experiments do NOT yet have an ordinal-aware external
IDRiD/Messidor-2 inference adapter. Do not load four-logit checkpoints
through existing single-output external evaluation or ONNX serving. Implement
and test that adapter before any external-performance claims.

The data remain under their respective licenses and outside this public
repository; no clinical decisions, claims of safety, or product rollout.
