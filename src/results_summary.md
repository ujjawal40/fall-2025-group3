# Latent Manifold Estimation — Results Summary

**Target:** PPSQFT

## Train Metrics

| Metric | Value |
|---|---|
| < 5% within | 25.54% |
| < 10% within | 49.10% |
| < 15% within | 67.54% |
| Median abs rel | 0.1022 |

## Test Metrics

| Metric | Value |
|---|---|
| < 5% within | 18.10% |
| < 10% within | 35.45% |
| < 15% within | 50.62% |
| Median abs rel | 0.1479 |

## Figures

- `em_loss.png` — EM iteration loss (inner-train energy)
- `mstep_losses.png` — M-step losses per EM (inner-train)
- `pred_vs_true_test.png` — Predicted vs True (TEST)
- `abs_rel_hist_test.png` — Absolute Relative Error (TEST)
