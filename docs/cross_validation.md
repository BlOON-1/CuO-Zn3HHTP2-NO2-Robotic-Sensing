# Grouped validation and interpretation

Default outer LeaveOneGroupOut holds out a complete source group. Optional outer
GroupKFold requires an explicit outer_folds setting. Inner GroupKFold uses only outer
training groups. The splitter rejects unsupported strategies, insufficient groups and
group overlap. Every sample receives exactly one outer prediction per candidate.

All cycles of an original source ID must share a group. Raw byte hashes and exact
time/resistance hashes cannot cross groups. These checks do not establish physical
independence or detect every near-duplicate; device/day/batch dependencies need study review.

Candidate families share one materialized outer split list and test sample set.
Inner search chooses parameters per family; Nested_selection chooses the highest inner
score's family in each outer fold. Each test sample records its selected family.
Outer model-family performance comparisons do not provide an unbiased assessment of
choosing the best family after seeing outer errors.

Output validation_splits.csv lists each sample's outer role and group;
inner_validation_splits.csv records inner train/validation/outer-test groups.
hyperparameter_search.json preserves complete cv_results_ and final-refit selection logs.

pooled_oof metrics use all outer predictions together. Per-fold metrics and unweighted
fold mean/SD are separate scopes and need not equal pooled metrics. SD uses ddof=1.
Undefined folds are counted explicitly as n_valid_folds vs n_folds, with partial/undefined
status. Technical-cycle weighting differs from equal-file weighting.

No independent external test was created or claimed here. Single-test evaluation is an
available metric scope for externally supplied records, not proof of independence.
No bootstrap intervals or statistical superiority claims are added in P0.
