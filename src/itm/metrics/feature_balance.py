"""Saved-array F evaluation arithmetic, with the 2026-10-03 frozen rules.

No model, device discovery, fitting, randomness, or semantic classifier.
"""
from collections import defaultdict

import numpy as np

from itm.metrics.canonical_b0 import evaluate_motion

GUARDRAILS = (
    'global_MPJPE_m', 'root_relative_MPJPE_m', 'root_path_RMSE_m',
    'head_or_wrists_position_joint_proxy_m', 'velocity_joint_proxy_RMSE_m_s',
)
SCALARS = GUARDRAILS + (
    'jerk_absolute_m_s3', 'GT_jerk_absolute_m_s3',
    'foot_contact_speed_proxy_m_s', 'GT_foot_contact_speed_proxy_m_s',
)
BLOCKS = dict(root=list(range(4)), ric=list(range(4, 67)),
              rotation6d=list(range(67, 193)), local_velocity=list(range(193, 259)),
              contact=list(range(259, 263)))
for _name, _joints in dict(ric_legs=[1, 2, 4, 5, 7, 8, 10, 11],
                          ric_trunk_head=[3, 6, 9, 12, 15],
                          ric_arms=[13, 14, 16, 17, 18, 19, 20, 21]).items():
    BLOCKS[_name] = [4 + 3 * (j - 1) + k for j in _joints for k in range(3)]


def feature_views(prediction, target):
    """Inputs are (T,263), already normalized using the bound Mean/Std."""
    p, g = (np.asarray(x, dtype=np.float64) for x in (prediction, target))
    if p.shape != g.shape or p.ndim != 2 or p.shape[1] != 263 or not len(p):
        raise ValueError('Expected equal nonempty (T,263) features')
    if not (np.isfinite(p).all() and np.isfinite(g).all()):
        raise ValueError('Nonfinite features')
    e = (p - g) ** 2
    d, a = float(e[:, :67].mean()), float(e[:, 67:].mean())
    return dict(uniform=float(e.mean()), root_ric67=d, remaining196=a,
                balanced=.5 * d + .5 * a,
                **{k: float(e[:, ids].mean()) for k, ids in BLOCKS.items()})


def motion_metrics(p, g, config, split, motion_id, gt_features):
    contacts = gt_features[:, 259:263] if split == 'val' and motion_id in ('000433', '000382') else None
    result = evaluate_motion(p, g, config, contacts)
    if split == 'train':
        result['foot_status'] = 'not_applicable_train_floor_admission_not_frozen'
    return result


def curves(motion):
    p = np.asarray(motion, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (22, 3) or not np.isfinite(p).all():
        raise ValueError('Finite (T,22,3) motion required')
    result = dict(frames=list(range(len(p))), seconds=(np.arange(len(p)) / 20).tolist(),
                  pelvis_xyz_m=p[:, 0].tolist(), pelvis_height_m=p[:, 0, 1].tolist(),
                  pelvis_path_m=np.r_[0, np.cumsum(np.linalg.norm(np.diff(p[:, 0], axis=0), axis=-1))].tolist())
    for name, joint in dict(left_ankle=7, right_ankle=8, left_toe=10, right_toe=11).items():
        result[name + '_relative_xyz_m'] = (p[:, joint] - p[:, 0]).tolist()
        result[name + '_absolute_y_m'] = p[:, joint, 1].tolist()
    for side, hip, knee, ankle in [('left', 1, 4, 7), ('right', 2, 5, 8)]:
        a, b = p[:, hip] - p[:, knee], p[:, ankle] - p[:, knee]
        denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        valid = denominator > 0
        angles = np.zeros(len(p))
        angles[valid] = np.degrees(np.arccos(np.clip((a[valid] * b[valid]).sum(-1) / denominator[valid], -1, 1)))
        result[side + '_knee_degrees'] = [float(x) if ok else None for x, ok in zip(angles, valid)]
        result[side + '_knee_NA_count'] = int((~valid).sum())
    return result


def subtract(a, b):
    """Recursive signed differences, including per-axis extents; NA propagates."""
    if a is None or b is None:
        return None
    if isinstance(a, dict):
        return {k: subtract(a[k], b[k]) for k in a}
    if isinstance(a, list):
        return [subtract(x, y) for x, y in zip(a, b)]
    return a - b


def contrast(a, b, predictions):
    result = dict(case_id=a['case_id'], reference_case_id=b['case_id'])
    if a['status'] != 'valid' or b['status'] != 'valid':
        return dict(result, status='not_comparable', reasons=[a.get('reason'), b.get('reason')])
    delta = predictions[a['case_id']].astype(float) - predictions[b['case_id']]
    return dict(result, status='valid', metrics={k: subtract(a['metrics'][k], b['metrics'][k]) for k in SCALARS},
                extent_delta=subtract(a['metrics']['motion_extent_m'], b['metrics']['motion_extent_m']),
                feature_delta=subtract(a['feature_views'], b['feature_views']),
                mean_joint_distance_m=float(np.linalg.norm(delta, axis=-1).mean()),
                max_joint_distance_m=float(np.linalg.norm(delta, axis=-1).max()),
                max_abs_m=float(np.abs(delta).max()))


def gate(new, reference):
    if new['status'] != 'valid' or reference['status'] != 'valid':
        return dict(status='inconclusive', metrics={})
    result = {}
    for k in GUARDRAILS:
        n, r = new['metrics'].get(k), reference['metrics'].get(k)
        if n is None or r is None or not np.isfinite([n, r]).all():
            return dict(status='inconclusive', metrics=result)
        epsilon = 1e-6 + 1e-6 * abs(r)
        result[k] = dict(new=n, reference=r, delta=n-r, epsilon=epsilon,
                         nonworse=bool(n <= r+epsilon), strictly_improved=bool(n < r-epsilon))
    passed = all(v['nonworse'] for v in result.values()) and result['root_relative_MPJPE_m']['strictly_improved']
    return dict(status='passed' if passed else 'failed', metrics=result)


def primary_decision(f, u, text, static, f_behavior, u_behavior):
    checks = {k: gate(f, row) for k, row in [('U_paired', u), ('F_text', text), ('F_static', static)]}
    behavior_known = f_behavior in ('complete', 'incomplete') and u_behavior in ('complete', 'incomplete')
    behavior_pass = f_behavior == 'complete' and u_behavior == 'incomplete'
    # An observed failure defeats support even if another component is unknown.
    failed = any(x['status'] == 'failed' for x in checks.values()) or (behavior_known and not behavior_pass)
    if f_behavior == 'incomplete' or u_behavior == 'complete':
        failed = True
    decision = 'not_supported' if failed else 'inconclusive'
    if behavior_pass and all(x['status'] == 'passed' for x in checks.values()):
        decision = 'supported_local_only'
    return dict(decision=decision, F_behavior=f_behavior, U_behavior=u_behavior, numerical=checks)


def aggregate(rows, group_keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)
    result = []
    for group, items in sorted(groups.items()):
        def flatten(value, prefix=''):
            out = {}
            for k, v in value.items():
                key = prefix + k
                if isinstance(v, dict): out.update(flatten(v, key + '.'))
                elif isinstance(v, list): out.update({key + '.' + str(i): x for i, x in enumerate(v)})
                elif v is None or isinstance(v, (int, float)): out[key] = v
            return out
        values = [flatten({k: r[k] for k in ('metrics', 'feature_views', 'feature_delta', 'extent_delta') if k in r})
                  for r in items if r['status'] == 'valid']
        names = sorted({'metrics.' + k for k in SCALARS}.union(*(v.keys() for v in values)))
        metrics = {}
        for k in names:
            xs = [v[k] for v in values if v.get(k) is not None]
            metrics[k] = dict(valid_n=len(xs), planned_n=len(items), mean=float(np.mean(xs)) if xs else None,
                              min=min(xs) if xs else None, max=max(xs) if xs else None)
        result.append(dict(zip(group_keys, group), planned_n=len(items), valid_n=len(values),
                           missing=[dict(case_id=r.get('case_id'), reason=r.get('reason', r.get('reasons')))
                                    for r in items if r['status'] != 'valid'], metrics=metrics))
    return result
