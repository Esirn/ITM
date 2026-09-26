"""Frozen canonical-B0-v1 metrics; no spatial/time fitting or success threshold."""
import numpy as np


def extent(x):
    return {'min_xyz': x.min(axis=(0, 1)).tolist(), 'max_xyz': x.max(axis=(0, 1)).tolist(),
            'range_xyz': np.ptp(x, axis=(0, 1)).tolist(),
            'pelvis_path_length_m': float(np.linalg.norm(np.diff(x[:, 0], axis=0), axis=-1).sum())}


def evaluate_motion(prediction, target, config, contacts=None):
    p, g = (np.asarray(x, dtype=np.float64) for x in (prediction, target))
    if p.shape != g.shape or p.ndim != 3 or p.shape[1:] != (22, 3) or len(p) < 4:
        raise ValueError('Expected equal full sequences (T>=4,22,3)')
    if not (np.isfinite(p).all() and np.isfinite(g).all()):
        raise ValueError('Nonfinite motion')
    if config not in ('head', 'wrists'):
        raise ValueError('Unknown sensor configuration')
    error = p - g
    joint_ids = [15] if config == 'head' else [20, 21]
    result = {
        'global_MPJPE_m': float(np.linalg.norm(error, axis=-1).mean()),
        'root_relative_MPJPE_m': float(np.linalg.norm(error-error[:, :1], axis=-1).mean()),
        'root_path_RMSE_m': float(np.sqrt(np.square(error[:, 0]).sum(-1).mean())),
        'head_or_wrists_position_joint_proxy_m': float(np.linalg.norm(error[:, joint_ids], axis=-1).mean()),
        'velocity_joint_proxy_RMSE_m_s': float(np.sqrt(np.square(np.diff(error, axis=0)*20).sum(-1).mean())),
        'jerk_absolute_m_s3': float(np.linalg.norm(np.diff(p, n=3, axis=0)*20**3, axis=-1).mean()),
        'GT_jerk_absolute_m_s3': float(np.linalg.norm(np.diff(g, n=3, axis=0)*20**3, axis=-1).mean()),
        'motion_extent_m': extent(p), 'GT_motion_extent_m': extent(g),
        'foot_contact_speed_proxy_m_s': None, 'GT_foot_contact_speed_proxy_m_s': None,
        'foot_contact_stencils': 0, 'foot_status': 'not_applicable_nonflat_support',
        'frames': len(p), 'velocity_stencils': len(p)-1, 'jerk_stencils': len(p)-3,
    }
    if contacts is not None:
        c = np.asarray(contacts)
        if c.shape != (len(p), 4) or not np.isin(c, [0, 1]).all():
            raise ValueError('Expected delivered binary GT contacts (T,4)')
        mask = c[:-1].astype(bool)  # final GT contact has no generated endpoint
        result['foot_contact_stencils'] = int(mask.sum())
        result['foot_status'] = 'GT_contact_proxy' if mask.any() else 'no_GT_contact_stencils'
        if mask.any():
            for x, name in [(p, 'foot_contact_speed_proxy_m_s'), (g, 'GT_foot_contact_speed_proxy_m_s')]:
                speeds = np.linalg.norm(np.diff(x[:, [7, 10, 8, 11]], axis=0)*20, axis=-1)
                result[name] = float(speeds[mask].mean())
    return result
