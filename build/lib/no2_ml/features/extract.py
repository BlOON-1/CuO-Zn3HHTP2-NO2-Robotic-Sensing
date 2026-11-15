from dataclasses import dataclass
import numpy as np
from .versions import feature_spec
from .response import baseline_resistance, response_curve, peak_response, initial_slope, timed_response, auc_response
from no2_ml.preprocessing.quality import quality_control, QCConfig


@dataclass
class FeatureResult:
    values: dict
    feature_version: str
    feature_fingerprint: str
    baseline_ohm: float
    response: np.ndarray
    quality: object
    issues: tuple


def extract_features(record, channel, boundary, version="legacy_peak_r60_slope", *, qc_config=QCConfig(), candidate_anomaly=None):
    """Same pure signal-to-feature function for training and prediction; offline."""
    spec = feature_spec(version)
    boundary.validate(record.time_s)
    if channel not in record.resistance_ohm:
        raise ValueError(f"Required channel {channel} missing")
    t, r = record.time_s, record.resistance_ohm[channel]
    qc = quality_control(t,r,qc_config,candidate_anomaly=candidate_anomaly)
    start = max(boundary.baseline_start_s,boundary.exposure_start_s-spec.baseline_max_s)
    ra = baseline_resistance(t,r,qc.valid,start,boundary.exposure_start_s,
                             min_points=spec.baseline_min_points,max_cv=spec.baseline_max_cv)
    exposure = (t>=boundary.exposure_start_s)&(t<=boundary.exposure_end_s)
    if (exposure&qc.valid).sum()<spec.exposure_min_points:
        raise ValueError("Insufficient valid exposure samples")
    if qc.ceiling[((t>=start)&(t<=boundary.exposure_end_s))].any():
        raise ValueError("Instrument ceiling in baseline/exposure; features rejected")
    s = response_curve(r,ra,qc.valid)
    rel = t[exposure]-boundary.exposure_start_s
    se = s[exposure]
    paper = spec.name == "paper_peak_slope_auc"
    values = {"max_response":peak_response(se[np.isfinite(se)]),
              "initial_slope":initial_slope(rel,se,end_s=spec.slope_end_s,min_points=spec.slope_min_points,
                                             max_gap_s=spec.max_feature_gap_s if paper else None)}
    duration = boundary.exposure_end_s-boundary.exposure_start_s
    if paper:
        values["auc_0_120"] = auc_response(rel,se,end_s=spec.auc_end_s,max_gap_s=spec.max_feature_gap_s,exposure_duration_s=duration)
    else:
        values["resp_60s"] = timed_response(rel,se,at_s=spec.response_time_s,tolerance_s=spec.response_tolerance_s,exposure_duration_s=duration)
    ordered = {name:values[name] for name in spec.columns}
    issues = tuple(record.issues)+tuple(f"feature_not_available:{name}" for name,value in ordered.items() if not np.isfinite(value))
    return FeatureResult(ordered,spec.version,spec.fingerprint,ra,s,qc,issues)
