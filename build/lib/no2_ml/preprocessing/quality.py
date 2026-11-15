from dataclasses import dataclass, asdict
import hashlib
import json
import numpy as np
from scipy.ndimage import median_filter


@dataclass(frozen=True)
class QCConfig:
    version: str = "conservative_v1"
    ceiling_ohm: float = 510e6
    hampel_window_s: float = 15.
    hampel_sigma: float = 8.
    minimum_spike_fraction: float = .02

    def __post_init__(self):
        if self.version not in {"conservative_v1", "legacy08_mask_v1"}:
            raise ValueError("Unknown QC version")
        if not all(np.isfinite(v) and v > 0 for v in
                   [self.ceiling_ohm, self.hampel_window_s, self.hampel_sigma, self.minimum_spike_fraction]):
            raise ValueError("QC thresholds must be finite and positive")

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()


@dataclass
class QCResult:
    valid: np.ndarray
    invalid: np.ndarray
    ceiling: np.ndarray
    anomaly: np.ndarray
    detection_signal: np.ndarray
    version: str


def quality_control(time_s, resistance_ohm, config=QCConfig(), *, candidate_anomaly=None):
    """Detection interpolation never replaces raw signal or fills feature values.

    legacy08_mask_v1 requires an explicit archived anomaly mask; its creation
    depended on upstream historical QC and cannot be silently reconstructed.
    """
    t, r = np.asarray(time_s, float), np.asarray(resistance_ohm, float)
    if t.ndim != 1 or r.shape != t.shape or len(t)<2 or not np.isfinite(t).all() or not (np.diff(t)>0).all():
        raise ValueError("QC requires matching arrays and strictly increasing time")
    finite = np.isfinite(r) & (r > 0)
    if not finite.any():
        raise ValueError("No valid resistance for QC")
    fill = np.interp(t, t[finite], r[finite])
    ceiling = finite & (r >= config.ceiling_ohm)
    if config.version == "legacy08_mask_v1":
        if candidate_anomaly is None:
            raise ValueError("legacy08_mask_v1 requires archived candidate_anomaly mask")
        anomaly = np.asarray(candidate_anomaly)
        if anomaly.dtype != bool or anomaly.shape != r.shape:
            raise ValueError("Archived anomaly mask must be boolean and match resistance")
    else:
        # Time-based local median/MAD supports unequal sample intervals.
        anomaly = np.zeros(len(t), bool)
        for i, value in enumerate(fill):
            lo, hi = np.searchsorted(t, [t[i]-config.hampel_window_s/2, t[i]+config.hampel_window_s/2], side="left")
            window = fill[lo:max(hi, lo+1)]
            med = np.median(window)
            mad = 1.4826*np.median(abs(window-med))
            anomaly[i] = abs(value-med) > max(config.hampel_sigma*mad, config.minimum_spike_fraction*abs(med))
    valid = finite & ~ceiling & ~anomaly
    if config.version == "legacy08_mask_v1" and ceiling.any():
        valid[:] = False  # Historical whole-file/channel ceiling exclusion.
    return QCResult(valid, ~finite, ceiling, anomaly.copy(), fill, config.version)
