# Feature definitions and migration

Canonical raw data: seconds and ohms; response `S(t)=Ra/R(t)` is dimensionless.
Only reviewed_v1 approved boundaries are usable for model input. Baseline is the median
of finite positive QC-valid raw resistance in `[max(baseline_start,onset−45),onset)`;
at least 20 points; population SD/median <=0.10. Exposure requires at least 30 valid
points. Ceiling in baseline/exposure rejects the cycle. No interpolation in feature values.

| Definition | Legacy @1.0.0 | Paper @1.0.0 |
| --- | --- | --- |
| Peak | Max valid S in complete exposure, not max(S−1) | Same |
| Slope | OLS S against real t−onset in [0,30] s; >=6 valid points | Same, plus first/last endpoint distance and maximum gap <=5 s |
| Response at 60 s | Median valid S in [58,62] s; exposure duration >=60 s | Not in feature set |
| AUC | Not in feature set | Trapezoidal integral of S−1 at valid times in [0,120] s; duration >=120 s, >=2 points, endpoint distances and largest gap <=5 s |

Peak and timed response are dimensionless; slope is s⁻¹; AUC is seconds.
The integration uses available timestamps, without adding fictional endpoint values;
actual sampled coverage can differ by at most the stated endpoint tolerance.
Unsupported windows return NaN with a feature_not_available warning. All-missing feature
rows and entirely missing training columns are rejected. Baseline/QC failures reject the cycle.

Names and order:
- legacy_peak_r60_slope@1.0.0: max_response, resp_60s, initial_slope.
- paper_peak_slope_auc@1.0.0: max_response, initial_slope, auc_0_120.

The definition fingerprint hashes the immutable full FeatureSpec. A second fingerprint
binds the full QC configuration; both are stored and checked during inference.
Training and prediction use the same extract_features.
New features/parameters require a new definition version, not reused old performance.

Historical migration:
- Peak/r60/slope formulas follow scripts/08_exploratory_models.py.
- Historical raw-point validity requires legacy08_mask_v1 and an explicit archived anomaly mask.
  The whole channel is invalid when any resistance reaches its configured ceiling.
  This mask mode alone does not reproduce the historical candidate-count/sustained-drop
  eligibility filter; approved manifests replace that unverified automatic eligibility.
- conservative_v1 uses time-neighborhood median/MAD outlier flags and no raw replacement.
  It is a new QC algorithm, not a claim to reproduce upstream index-based historical flags.
- Candidate detector legacy08_1s_v1 follows the index detector only for exactly 1 s spacing;
  other sampling is refused. time_grid_v1 explicitly interpolates for detection only.
- Old pipeline short-spike interpolation and unverified manual-boundary propagation are
  not silently carried into formal features. Approved external boundaries remain authoritative.

Full exposure Peak and reviewed boundaries require offline information. This software
does not claim a causal, streaming or real-time predictor.
