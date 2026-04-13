# ETTm1 Decoupling Case Study Diagnosis

## Symptom

When running `scripts/Analysis/decoupling_case_study.sh` on ETTm1, the figure
[`ETTm1_decoupling_case_study_macro.pdf`](/workspace/NsDiff/results/analysis/ETTm1/exp1_decoupling_case_study/ETTm1_decoupling_case_study_macro.pdf)
looks incorrect:

- the predicted `mu` appears almost as a flat line;
- the `sigma` band is nearly invisible.

## Conclusion

The issue is not primarily caused by the export scale of
`ETTm1_decoupling_case_study_data.npz`. Instead, it is caused by two factors
stacking together:

1. The selected ETTm1 windows already have nearly constant exported `mu_hat`
   and very small `sigma_hat`.
2. The plotting code reuses the raw-space y-axis range for the macro panel,
   which visually compresses the `mu ± sigma` band even further.

## Evidence

The selected samples recorded in
[`ETTm1_decoupling_case_study.json`](/workspace/NsDiff/results/analysis/ETTm1/ETTm1_decoupling_case_study.json)
are:

- `feature_dim = 2`
- selected indices: `9600, 9599, 768, 767, 766, 765`

For these windows, the exported values in
[`ETTm1_decoupling_case_study_data.npz`](/workspace/NsDiff/results/analysis/ETTm1/ETTm1_decoupling_case_study_data.npz)
show that `mu_hat` and `sigma_hat` are indeed almost constant for several
selected samples:

- Window `9600`
  - `y.std = 8.315`
  - `mu.std = 0.007`
  - `sigma = 0.0349`, essentially constant over the whole horizon
- Window `9599`
  - `mu.std = 0.0077`
  - `sigma = 0.0349`, constant
- Window `768`
  - `mu.std = 0.0071`
  - `sigma = 0.0349`, constant
- Window `767`
  - `mu.std = 0.0077`
  - `sigma = 0.0349`, constant
- Only windows `766` and `765` show slightly more variation, but even there
  the band remains small:
  - `sigma` roughly in `0.15 - 0.23`

So the observed “flat `mu`” is not a PDF rendering issue. On these selected
windows, the model output itself is already nearly flat.

## Plotting Issue

There is also a visualization design problem in
[`decoupling_case_study.py`](/workspace/NsDiff/src/analysis/decoupling_case_study.py#L427)
and
[`decoupling_case_study.py`](/workspace/NsDiff/src/analysis/decoupling_case_study.py#L466).

The current implementation computes the macro-panel y-axis limits from the raw
trajectories and then applies the same range to the macro panel:

- `y_limits_raw` is computed from `raw Y + mu`
- `ax_macro.set_ylim(...)` reuses that raw range

For the selected ETTm1 windows:

- pooled raw `Y` span is about `32.19`
- pooled `mu` span is only about `3.46`
- pooled `sigma` span is only about `0.199`
- `mean(sigma) / raw_y_span ≈ 0.00284`

This means the `sigma` band occupies only a tiny fraction of the raw-space
vertical range, so it becomes almost invisible even if the export is correct.

## Why This Happens With Current Selection Logic

The current selector prefers windows that are:

- high-drift;
- highly similar in residual shape.

On ETTm1, this can easily select very similar or adjacent windows, such as:

- `9600` and `9599`
- `768, 767, 766, 765`

Because these windows are highly similar already, their predictive macro
components are also very similar, which further increases the chance that the
middle panel looks nearly flat.

## What This Means

The main issue is:

- not that the export script necessarily corrupts `mu` or `sigma`;
- not mainly that the raw target scale is wrong;
- but that the combination of sample selection and shared raw y-axis makes the
  macro panel unreadable on ETTm1.

## Recommended Fixes

To make Figure 1 more informative for reviewers, the following changes are
recommended:

1. Give the macro panel its own y-axis range, computed from `mu ± sigma`,
   instead of reusing the raw panel range.
2. Add temporal de-duplication or a minimum index-gap constraint to sample
   selection, so that near-adjacent windows are not selected together.

These two changes should make the ETTm1 macro plot substantially more readable
without changing the core PDN analysis logic.
