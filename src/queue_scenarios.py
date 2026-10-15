"""Capacity-constrained fluid scenarios; no minute-level observations inferred."""
import math
import numpy as np


def shaped_arrivals(hourly, shape, sigma, center=30, delay=5):
    counts=np.asarray(hourly,dtype=float)
    if counts.ndim!=1 or not len(counts) or not np.isfinite(counts).all() or (counts<0).any():
        raise ValueError('Expected nonnegative hourly forecasts')
    if shape not in {'uniform','broad','concentrated'}: raise ValueError('Unknown shape')
    if not isinstance(delay,int) or delay<0 or not 0<=center<60: raise ValueError('Invalid timing')
    profile=np.ones((len(counts),60))/60
    # Use the FORECAST peak hour, never the observed peak hour. Other hours stay uniform.
    if shape!='uniform':
        if sigma is None or not math.isfinite(sigma) or sigma<=0: raise ValueError('Invalid sigma')
        w=np.exp(-.5*((np.arange(60)+.5-center)/sigma)**2)
        if w.sum()==0: raise ValueError('Profile numerically degenerate')
        profile[int(np.argmax(counts))]=w/w.sum()
    arrivals=(counts[:,None]*profile).ravel()
    if not np.allclose(arrivals.reshape(-1,60).sum(axis=1),counts,atol=1e-9):
        raise AssertionError('Hourly conservation failed')
    return np.r_[np.zeros(delay),arrivals]


def remaining_capacity(maximum, onboard_before, alighting):
    """Spaces after alighting and before boarding; no invented occupancy counts."""
    if any(not math.isfinite(v) or v<0 for v in (maximum,onboard_before,alighting)):
        raise ValueError('Negative/nonfinite capacity inputs')
    if alighting>onboard_before: raise ValueError('Cannot alight more than aboard')
    return max(0.,maximum-(onboard_before-alighting))
