"""Metrics computed after fitting generated motion to a sensor-bearing SMPL body."""

from __future__ import annotations

import numpy as np


def rotation_geodesic_error(predicted, target, mask=None):
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.shape[-2:] != (3, 3):
        raise ValueError("rotation arrays must have equal (..., 3, 3) shapes")
    relative = np.swapaxes(predicted, -1, -2) @ target
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    errors = np.arccos(cosine)
    return _masked_mean(errors, mask)


def vector_l2_error(predicted, target, mask=None):
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.shape[-1] != 3:
        raise ValueError("vector arrays must have equal (..., 3) shapes")
    return _masked_mean(np.linalg.norm(predicted - target, axis=-1), mask)


def _masked_mean(values, mask):
    if mask is None:
        return float(np.mean(values))
    active = np.asarray(mask, dtype=bool)
    while active.ndim < values.ndim:
        active = active[None]
    active = np.broadcast_to(active, values.shape)
    if not np.any(active):
        raise ValueError("metric mask contains no active values")
    return float(values[active].mean())
