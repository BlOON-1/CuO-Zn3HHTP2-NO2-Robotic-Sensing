"""Detection proposals are never approved automatically."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks, peak_widths
from no2_ml.io.schema import CycleBoundary


def detect_cycles(time_s, detection_signal, *, version="legacy08_1s_v1",
                  expected_cycles=None, sigma_s=8., separation_s=180., baseline_s=45.):
    t, r = np.asarray(time_s, float), np.asarray(detection_signal, float)
    if t.ndim != 1 or r.shape != t.shape or len(t)<3 or not np.isfinite(np.r_[t,r]).all() or not (np.diff(t)>0).all():
        raise ValueError("Invalid detection arrays")
    if version not in {"legacy08_1s_v1", "time_grid_v1"}:
        raise ValueError("Unknown cycle detection version")
    if min(sigma_s, separation_s, baseline_s) <= 0:
        raise ValueError("Detection time windows must be positive")
    if version == "legacy08_1s_v1" and not np.allclose(np.diff(t), 1., rtol=0, atol=1e-8):
        raise ValueError("Historical index detector requires 1 s sampling; use time_grid_v1 explicitly")
    if version == "time_grid_v1":
        grid = np.arange(t[0], t[-1]+1e-9, 1.)
        signal = np.interp(grid, t, r)
    else:
        grid, signal = t, r
    sm = gaussian_filter1d(median_filter(signal, 5), sigma_s)
    dynamic = np.quantile(sm,.95)-np.quantile(sm,.05)
    peaks, _ = find_peaks(-sm, prominence=max(.12*dynamic,np.median(abs(np.diff(sm)))*25), distance=separation_s)
    _, _, left, right = peak_widths(-sm, peaks, rel_height=.85)
    derivative = np.gradient(sm, grid)
    candidates = []
    for k, pk in enumerate(peaks):
        lo, hi = int(left[k]), int(right[k])
        search = np.arange(max(1,lo-100),pk)
        if len(search)<40:
            continue
        idx = search[np.argmin(derivative[search])]
        threshold = .15*abs(derivative[idx])
        onset = idx
        while onset>max(1,lo-100) and derivative[onset-1]<-threshold:
            onset -= 1
        if grid[pk]-grid[onset]<40 or grid[hi]-grid[pk]<30 or onset<20:
            continue
        end = min(len(grid)-1, hi+45)
        candidates.append(CycleBoundary(max(grid[0],grid[onset]-baseline_s),grid[onset],grid[pk],grid[end],False,version))
    if expected_cycles is not None and len(candidates) != expected_cycles:
        raise ValueError("Detected count differs from explicitly configured expectation; review required")
    return candidates
