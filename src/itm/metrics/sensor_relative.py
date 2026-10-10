"""Supplementary joint proxies; root translation removed, world heading retained.

This does not change canonical_b0 metrics or reconstruct sensor vertex signals.
"""
import numpy as np


SENSOR_JOINTS = {'head': (15,), 'wrists': (20, 21)}


def evaluate_sensor_relative(prediction, target, config, fps=20.0):
    p, g = (np.asarray(x, dtype=np.float64) for x in (prediction, target))
    if p.shape != g.shape or p.ndim != 3 or p.shape[1:] != (22, 3) or len(p) < 4:
        raise ValueError('Expected equal complete (T>=4,22,3) sequences')
    if not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError('Nonfinite sequence')
    if config not in SENSOR_JOINTS or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Unknown sensor configuration or invalid frame rate')
    ids = list(SENSOR_JOINTS[config])
    local_p, local_g = (x[:, ids] - x[:, :1] for x in (p, g))
    e = local_p - local_g
    rms = lambda x: float(np.sqrt(np.mean(np.sum(x * x, axis=-1))))
    bias = e.mean(axis=0, keepdims=True)
    root_e = (p - g)[:, :1]
    global_e = (p - g)[:, ids]
    result = {
        'observed_relative_position_mean_m': float(np.linalg.norm(e, axis=-1).mean()),
        'observed_relative_position_RMSE_m': rms(e),
        'observed_relative_velocity_RMSE_m_s': rms(np.diff(e, axis=0) * fps),
        'observed_relative_static_error_RMS_m': rms(bias),
        'observed_relative_dynamic_error_RMS_m': rms(e - bias),
        'observed_global_position_MSE_m2': rms(global_e)**2,
        'root_position_MSE_m2': rms(root_e)**2,
        'root_local_error_cross_m2': float(2 * np.mean(np.sum(root_e * e, axis=-1))),
    }
    for x, prefix in ((local_p, ''), (local_g, 'GT_')):
        result[prefix + 'observed_relative_spread_RMS_m'] = rms(x - x.mean(axis=0, keepdims=True))
        result[prefix + 'observed_relative_speed_RMS_m_s'] = rms(np.diff(x, axis=0) * fps)
        result[prefix + 'observed_relative_jerk_mean_m_s3'] = float(
            np.linalg.norm(np.diff(x, n=3, axis=0) * fps**3, axis=-1).mean())
    return result
