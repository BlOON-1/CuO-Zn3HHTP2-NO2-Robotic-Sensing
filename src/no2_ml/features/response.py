import numpy as np


def baseline_resistance(time_s, resistance_ohm, valid, start_s, end_s, *, min_points=20, max_cv=.10):
    t, r, valid = np.asarray(time_s), np.asarray(resistance_ohm), np.asarray(valid)
    mask = (t>=start_s)&(t<end_s)&valid&np.isfinite(r)&(r>0)
    if mask.sum()<min_points:
        raise ValueError("Insufficient valid baseline samples")
    ra = float(np.median(r[mask]))
    if np.std(r[mask], ddof=0)/ra > max_cv:
        raise ValueError("Unstable baseline: CV exceeds configured threshold")
    return ra


def response_curve(resistance_ohm, ra, valid):
    r, mask = np.asarray(resistance_ohm,float), np.asarray(valid,bool)
    if r.ndim != 1 or mask.shape != r.shape or not np.isfinite(ra) or ra<=0:
        raise ValueError("Invalid resistance/valid mask/baseline")
    out = np.full(r.shape,np.nan)
    good = mask&np.isfinite(r)&(r>0)
    out[good] = ra/r[good]
    return out


def peak_response(response):
    s = np.asarray(response,float)
    return float(np.max(s)) if s.size and np.isfinite(s).all() else np.nan


def initial_slope(time_relative_s, response, *, end_s=30., min_points=6, max_gap_s=None):
    t, s = np.asarray(time_relative_s,float), np.asarray(response,float)
    mask = (t>=0)&(t<=end_s)&np.isfinite(s)
    x, y = t[mask], s[mask]
    if len(x)<min_points or np.ptp(x)<=0:
        return np.nan
    if max_gap_s is not None and (x[0]>max_gap_s or end_s-x[-1]>max_gap_s or np.diff(x).max()>max_gap_s):
        return np.nan
    return float(np.polyfit(x,y,1)[0])


def timed_response(time_relative_s, response, *, at_s=60., tolerance_s=2., exposure_duration_s):
    t, s = np.asarray(time_relative_s,float), np.asarray(response,float)
    ids = (abs(t-at_s)<=tolerance_s)&np.isfinite(s)
    return float(np.median(s[ids])) if exposure_duration_s>=at_s and ids.any() else np.nan


def auc_response(time_relative_s, response, *, end_s=120., max_gap_s=5., exposure_duration_s):
    """Integrate S-1 at available timestamps. No endpoint invention/interpolation."""
    t, s = np.asarray(time_relative_s,float), np.asarray(response,float)
    ids = (t>=0)&(t<=end_s)&np.isfinite(s)
    x, y = t[ids], s[ids]
    if exposure_duration_s<end_s or len(x)<2 or x[0]>max_gap_s or end_s-x[-1]>max_gap_s or np.diff(x).max()>max_gap_s:
        return np.nan
    return float(np.trapz(y-1,x))
