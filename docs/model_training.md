# Model training

The registry contains LinearRegression, Ridge, SVR, RandomForestRegressor and
XGBRegressor. Model defaults and search spaces inherit the audited 08 script:
Linear defaults; Ridge alpha 1/10; SVR C 1/10; RF 60 trees, leaf size 2, depth 2/None;
XGB 60 trees, depth 2, learning rate .05, squarederror, lambda 1/10. RF/XGB seed 42
and one thread by default. Unspecified parameters use the pinned framework defaults.
Full effective estimator parameters are exported, not just best_params_.

`models.yaml` configures params and search; every search uses a fresh sklearn Pipeline:
SimpleImputer(median,keep_empty_features=True) -> StandardScaler -> regressor.
GridSearchCV clones and fits this Pipeline inside each inner training split.
Scoring defaults to neg_mean_absolute_error; neg_mean_squared_error and
neg_root_mean_squared_error are explicit alternatives. Higher search score is better.
Ties retain configured model order, matching the prior iteration-based behavior.

`nested_validate` produces OOF predictions and search logs. `refit_final` is separate,
uses full training data grouped CV for deployment selection, and returns no test metrics.
`train_from_config` orchestrates both into a new, exclusively created run directory.
Run failure is recorded and raised; it never becomes a successful partial comparison.

No train/test random split, deep networks or feature selection is introduced by default.
Any future learned feature selector must be inside the Pipeline. Raw sample grouping
is supplied by a reviewed manifest; filename-based concentration inference is removed.
