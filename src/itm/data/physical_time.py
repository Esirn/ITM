"""Explicit seconds-axis interpolation without endpoint clamping or time warps."""
import numpy as np


def resample_positions_at_times(values, source_times, target_times):
    values=np.asarray(values,dtype=np.float64)
    source=np.asarray(source_times,dtype=np.float64)
    target=np.asarray(target_times,dtype=np.float64)
    if values.ndim < 2 or source.shape != (len(values),) or len(source)<2 or target.ndim!=1 or not len(target):
        raise ValueError("matching source samples and nonempty target timestamps required")
    if not np.isfinite(values).all() or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("finite inputs required; invalid stencils must be handled separately")
    if (np.diff(source)<=0).any() or (np.diff(target)<=0).any():
        raise ValueError("strictly increasing timestamps required")
    if target[0]<source[0] or target[-1]>source[-1]:
        raise ValueError("target time is outside source support; no clamping or stretching")
    flat=values.reshape(len(values),-1)
    sampled=np.stack([np.interp(target,source,flat[:,j]) for j in range(flat.shape[1])],-1)
    return sampled.reshape((len(target),*values.shape[1:]))
