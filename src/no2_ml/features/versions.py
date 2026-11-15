"""Immutable definitions. Algorithm changes require a new version ID."""
from dataclasses import dataclass, asdict
import hashlib
import json


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    version: str
    columns: tuple
    baseline_max_s: float = 45.
    baseline_min_points: int = 20
    exposure_min_points: int = 30
    baseline_max_cv: float = .10
    slope_end_s: float = 30.
    slope_min_points: int = 6
    response_time_s: float = 60.
    response_tolerance_s: float = 2.
    auc_end_s: float = 120.
    max_feature_gap_s: float = 5.

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def feature_spec(name):
    specs = {
        "legacy_peak_r60_slope": FeatureSpec("legacy_peak_r60_slope", "legacy_peak_r60_slope@1.0.0",
                                           ("max_response", "resp_60s", "initial_slope")),
        "paper_peak_slope_auc": FeatureSpec("paper_peak_slope_auc", "paper_peak_slope_auc@1.0.0",
                                          ("max_response", "initial_slope", "auc_0_120")),
    }
    for spec in specs.values():
        if name in {spec.name, spec.version}:
            return spec
    raise ValueError(f"Unknown feature version: {name}")
