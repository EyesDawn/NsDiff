# Traffic Stage 3 Diagnosis

## Current Phenomena

- On Traffic, Stage 3 training loss keeps decreasing, but `val_crps` improves only very early and then usually degrades.
- Early stopping is still triggered quickly because the best `val_crps` appears within the first few epochs.
- The same behavior is much weaker on ETTm1.

## What We Confirmed

- The frozen Stage 2 uncertainty estimator is stable during Stage 3.
- `sigma_*` metrics stay almost exactly unchanged across Stage 3 epochs and across different Stage 3 reruns.
- So the main problem is not that `sigma` is drifting during Stage 3.

Representative fixed sigma metrics on Traffic Stage 3:

- `sigma_gauss_nll ~= 0.459052`
- `sigma_gauss_crps ~= 0.211263`
- `sigma_cov_90 ~= 0.953113`
- `sigma_width_90 ~= 1.587653`
- `sigma_pit_ks ~= 0.161241`

## Results of the Recent Changes

| Run | Main change | Best val CRPS | Best test CRPS | Best test PICP | Best test QICE | Result |
|---|---|---:|---:|---:|---:|---|
| `run-20260330_081911-xa9iumrm` | Baseline after log-variance refactor, `lr=2e-4` | `0.203467` | `0.230052` | `0.791673` | `0.018689` | Unstable, clear CRPS degradation after epoch 1 |
| `run-20260330_111531-oflfbv30` | Scheduler uses `val_crps`, `val_num_samples=100`, `lr=1e-4`, `lr_patience=3` | `0.200473` | `0.227612` | `0.837381` | `0.016358` | Clear improvement, but best epoch still epoch 1 |
| `run-20260330_151723-w2z68sts` | Keep `lr=1e-4`, change `lr_patience=2` | `0.200473` | `0.227612` | `0.837381` | `0.016358` | Almost no meaningful change |
| `run-20260331_032720-8gnk5r4b` | `lr=5e-5`, `lr_patience=3` | `0.200249` (epoch 3) | `0.228228` | `0.796041` | `0.017301` | Slightly better val CRPS, but worse probabilistic test quality than the best `1e-4` run |

## Interpretation

- The first useful change was not `lr_patience`; it was:
  - validating with more samples (`val_num_samples=100`)
  - using `val_crps` instead of `val_loss` for the scheduler
  - lowering Stage 3 LR from `2e-4` to `1e-4`
- That change improved both validation stability and test probabilistic quality.
- Changing `lr_patience` from `3` to `2` did not solve anything.
- Lowering Stage 3 LR further to `5e-5` slightly improved the best validation CRPS, but did not improve test CRPS, and test coverage got worse again.

## Core Problem

The current core problem is still **Stage 3 objective mismatch**, not sigma.

- Stage 3 optimizes `velocity_loss`.
- Model selection is based on `val_crps`.
- On high-dimensional Traffic (`862` variables, only `6` val batches), improving `velocity_loss` does not reliably improve probabilistic forecast quality.

Evidence:

- `sigma_*` metrics are fixed, so Stage 2 uncertainty is not being corrupted.
- Best `val_crps` still appears very early.
- Small LR and scheduler changes only move the curve slightly; they do not change the basic behavior.
- A slightly better validation CRPS at `5e-5` did not translate into a better test CRPS.

## Bottom Line

- There was real improvement from the first control changes (`val_crps` scheduler + more validation samples + `lr=1e-4`).
- There is still room for improvement on Traffic.
- But the remaining bottleneck is no longer simple LR tuning.
- The main issue is that Stage 3 is trained with a surrogate objective that is only weakly aligned with the final probabilistic metrics on Traffic, and the validation signal is still noisy.
