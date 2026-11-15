"""Canonical units: seconds, ohms, ppm. Invalid resistance remains visible."""
from dataclasses import dataclass, field
from typing import Mapping, Tuple
import numpy as np


@dataclass(frozen=True)
class SignalRecord:
    time_s: np.ndarray
    resistance_ohm: Mapping[int, np.ndarray]
    source_id: str
    source_path: str = ""
    issues: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        t = np.array(self.time_s, dtype=float, copy=True)
        if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or not (np.diff(t) > 0).all():
            raise ValueError("Time must contain >=2 finite, strictly increasing seconds")
        if not self.source_id or not self.resistance_ohm:
            raise ValueError("source_id and resistance channels are required")
        channels = {}
        for channel, values in self.resistance_ohm.items():
            r = np.array(values, dtype=float, copy=True)
            if not isinstance(channel, int) or channel < 1 or r.shape != t.shape:
                raise ValueError("Channel IDs must be positive integers; lengths must match time")
            if not (np.isfinite(r) & (r > 0)).any():
                raise ValueError(f"Channel {channel} has no positive finite resistance")
            r.setflags(write=False)
            channels[channel] = r
        t.setflags(write=False)
        object.__setattr__(self, "time_s", t)
        object.__setattr__(self, "resistance_ohm", channels)


@dataclass(frozen=True)
class CycleBoundary:
    baseline_start_s: float
    exposure_start_s: float
    exposure_end_s: float
    recovery_end_s: float
    approved: bool = False
    boundary_version: str = "reviewed_v1"

    def validate(self, time_s, require_approved=True):
        values = np.array([self.baseline_start_s, self.exposure_start_s,
                           self.exposure_end_s, self.recovery_end_s], dtype=float)
        if not np.isfinite(values).all() or not (np.diff(values) > 0).all():
            raise ValueError("Boundary times must be finite and strictly ordered")
        if values[0] < time_s[0] or values[-1] > time_s[-1]:
            raise ValueError("Cycle boundary outside measured time range")
        if require_approved and (self.approved is not True or self.boundary_version != "reviewed_v1"):
            raise ValueError("Features require explicitly approved reviewed_v1 boundaries")
