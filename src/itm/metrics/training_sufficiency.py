"""Fixed-state diagnostics in normalized HumanML263 units, NumPy only.

These measures describe saved-state x0 predictions, not decoded full motions.
There is deliberately no convergence, semantic-success, or checkpoint selector.
"""
import numpy as np

from itm.metrics.feature_balance import BLOCKS

VIEWS = dict(uniform=list(range(263)), root_ric67=list(range(67)),
             remaining196=list(range(67, 263)), **BLOCKS)
PRIMARY = ('root', 'ric', 'rotation6d', 'local_velocity', 'contact')


def _features(prediction, target, valid=None):
    p, g = (np.asarray(x, dtype=np.float64) for x in (prediction, target))
    if p.shape != g.shape or p.ndim != 2 or p.shape[1] != 263 or not len(p):
        raise ValueError('Equal nonempty (T,263) features required')
    mask = np.ones(len(p), dtype=bool) if valid is None else np.asarray(valid)
    if mask.dtype != np.bool_ or mask.shape != (len(p),) or not mask.any():
        raise ValueError('Nonempty boolean (T,) valid mask required')
    if not np.isfinite(p[mask]).all() or not np.isfinite(g[mask]).all():
        raise ValueError('Finite valid features required')
    return p, g, mask


def _bound(*values):
    return 64 * np.finfo(np.float64).eps * max(1., *(abs(v) for v in values))


def temporal_decomposition(prediction, target, valid=None):
    """Separate mean-pose error from temporal error without fitting/alignment.

    The static baseline is target's per-coordinate temporal mean, not a new
    model prediction. First differences include only originally adjacent valid
    frames; omitted frames must not create an artificial time interval.
    """
    p, g, mask = _features(prediction, target, valid)
    a, b = p[mask], g[mask]
    pm, gm = a.mean(0), b.mean(0)
    pc, gc = a - pm, b - gm
    adjacent = mask[:-1] & mask[1:]
    result = {}
    for name, ids in VIEWS.items():
        mse = float(np.mean((a[:, ids] - b[:, ids]) ** 2))
        mean_error = float(np.mean((pm[ids] - gm[ids]) ** 2))
        temporal_error = float(np.mean((pc[:, ids] - gc[:, ids]) ** 2))
        pe, ge = float(np.mean(pc[:, ids] ** 2)), float(np.mean(gc[:, ids] ** 2))
        covariance = float(np.mean(pc[:, ids] * gc[:, ids]))
        error = max(abs(mse - mean_error - temporal_error),
                    abs(temporal_error - pe - ge + 2 * covariance))
        bound = _bound(mse, mean_error, temporal_error, pe, ge, 2 * covariance)
        if error > bound:
            raise ArithmeticError('Temporal decomposition identity failed')
        result[name] = dict(dimensions=len(ids), mse=mse, mean_error=mean_error,
                            temporal_error=temporal_error, prediction_temporal_energy=pe,
                            target_temporal_energy=ge, temporal_covariance=covariance,
                            target_mean_baseline_mse=ge,
                            temporal_energy_ratio=pe/ge if ge > 0 else None,
                            identity_error=error, identity_bound=bound,
                            adjacent_valid_pairs=int(adjacent.sum()),
                            prediction_difference_rms=float(np.sqrt(np.mean(np.diff(p[:, ids], axis=0)[adjacent] ** 2))) if adjacent.any() else None,
                            target_difference_rms=float(np.sqrt(np.mean(np.diff(g[:, ids], axis=0)[adjacent] ** 2))) if adjacent.any() else None)
    result['balanced'] = {k: .5 * result['root_ric67'][k] + .5 * result['remaining196'][k]
                          for k in ('mse', 'mean_error', 'temporal_error', 'prediction_temporal_energy',
                                    'target_temporal_energy', 'temporal_covariance', 'target_mean_baseline_mse')}
    _partition(result, 'mse')
    return result


def _partition(result, field):
    weighted = sum(result[k][field] * len(VIEWS[k]) / 263 for k in PRIMARY)
    if abs(weighted - result['uniform'][field]) > _bound(weighted, result['uniform'][field]):
        raise ArithmeticError('Primary feature partition reconstruction failed')


def same_state_delta(baseline, prediction, target, valid=None):
    """Signed MSE change = cross term + residual energy; no causal state claim."""
    p, g, mask = _features(prediction, target, valid)
    b, _, other = _features(baseline, target, valid)
    if not np.array_equal(mask, other):
        raise ValueError('Different state masks')
    error, delta = b[mask] - g[mask], p[mask] - b[mask]
    result = {}
    for name, ids in VIEWS.items():
        old = float(np.mean(error[:, ids] ** 2))
        new = float(np.mean((p[mask][:, ids] - g[mask][:, ids]) ** 2))
        cross = float(2 * np.mean(error[:, ids] * delta[:, ids]))
        energy = float(np.mean(delta[:, ids] ** 2))
        change = new - old
        bound = _bound(old, new, cross, energy)
        if abs(change - cross - energy) > bound:
            raise ArithmeticError('Residual decomposition identity failed')
        result[name] = dict(dimensions=len(ids), baseline_mse=old, prediction_mse=new,
                            delta_mse=change, cross=cross, residual_energy=energy,
                            prediction_difference_rms=float(np.sqrt(energy)),
                            prediction_difference_max_abs=float(np.max(np.abs(delta[:, ids]))),
                            count=int(len(ids)*mask.sum()),
                            identity_error=abs(change-cross-energy), identity_bound=bound)
    result['balanced'] = {k: .5 * result['root_ric67'][k] + .5 * result['remaining196'][k]
                          for k in ('baseline_mse', 'prediction_mse', 'delta_mse', 'cross', 'residual_energy')}
    _partition(result, 'delta_mse')
    return result
