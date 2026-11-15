# Regression assessment

For finite one-dimensional arrays y (nonnegative reference ppm) and p (prediction ppm),
define e=p−y. Empty, shape-mismatched, nonfinite inputs and duplicate sample IDs raise
ValueError. A separate prediction_sample_id can enforce ordering. Multi-model tables
must agree on IDs, references, groups, folds, feature version and evaluation scope.
Order differences raise unless align_by_id=True is explicitly chosen.

| Metric key | Formula | Unit / applicability |
| --- | --- | --- |
| mse | mean(e²) | ppm² |
| mae | mean(abs(e)) | ppm |
| rmse | sqrt(mean(e²)) | ppm |
| mean_bias_error | mean(e) | ppm; positive = overestimate |
| rmsle | sqrt(mean((log1p(p)−log1p(y))²)) | dimensionless; nonnegative p and y, no clipping |
| r2 | 1−sum(e²)/sum((y−mean(y))²) | dimensionless; undefined for constant y or n<2; negative values retained |
| adjusted_r2 | 1−(1−r2)(n−1)/(n−p_count−1) | only full-rank unpenalized Linear training diagnostic, intercept, integer p_count>=1, n>p_count+1 |

Adjusted R² requires explicit adjusted_context(p,fit_intercept=True,full_rank=True).
The caller must establish rank; it is not inferred by pretending the number of nonlinear
inputs is effective model complexity. It is N/A for ordinary OOF/test comparisons.

Metric objects carry value=None plus status/reason when undefined or not applicable.
They never substitute zero. Evaluation rows include sample count, model, feature version,
scope and Bias convention. Missing inputs are not silently removed per model.

The scalar API accepts single_test_set, per_fold, pooled_oof and training_diagnostic.
The table API separately exports pooled_oof, per_fold, fold_mean and fold_sd. Fold means
weight valid folds equally; the table records partial status if some folds are undefined.
Pooled OOF weights each retained cycle equally. These are different statistics.

Classification and clustering metrics are not applicable to this regression task.
CSV is the canonical output and preserves full precision (17 significant digits).
