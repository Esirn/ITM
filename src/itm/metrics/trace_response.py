"""CPU algebra of saved factorized responses, in normalized feature units.

Temporal variance is not motion correctness. Each decomposition belongs to one
saved state; changing the state or diffusion time is not an intervention.
"""
from itertools import combinations

import numpy as np

from itm.metrics.training_sufficiency import VIEWS, _bound, _features


def response_decomposition(branches, valid=None):
    f = {k: np.asarray(branches[k], dtype=np.float64)
         for k in ('f00', 'f10', 'f01', 'f11')}
    _, _, mask = _features(f['f00'], f['f11'], valid)
    for value in f.values():
        _features(value, f['f00'], mask)
    components = dict(base=f['f00'], text=f['f10']-f['f00'],
                      imu=f['f01']-f['f00'],
                      interaction=f['f11']-f['f10']-f['f01']+f['f00'])
    c = {k: v[mask]-v[mask].mean(0) for k, v in components.items()}
    p = {k: v[mask]-v[mask].mean(0) for k, v in f.items()}
    out = {}
    for name, ids in VIEWS.items():
        energy = {k: float(np.mean(v[:, ids]**2)) for k, v in c.items()}
        cross = {a+'__'+b: float(2*np.mean(c[a][:, ids]*c[b][:, ids]))
                 for a, b in combinations(c, 2)}
        branch_energy = {k: float(np.mean(v[:, ids]**2)) for k, v in p.items()}
        reconstructed = sum(energy.values())+sum(cross.values())
        err = abs(reconstructed-branch_energy['f11'])
        bound = _bound(*energy.values(), *cross.values(), branch_energy['f11'])
        if err > bound:
            raise ArithmeticError('Four-component temporal identity failed')
        changes = {}
        for label, a, b in [('text_at_no_imu','f00','f10'),
                            ('imu_at_no_text','f00','f01'),
                            ('imu_at_text','f10','f11'),
                            ('text_at_imu','f01','f11'),
                            ('joint_at_base','f00','f11')]:
            delta = p[b][:, ids]-p[a][:, ids]
            de = float(np.mean(delta**2))
            cov = float(2*np.mean(p[a][:, ids]*delta))
            change = branch_energy[b]-branch_energy[a]
            error = abs(change-cov-de)
            limit = _bound(branch_energy[a], branch_energy[b], cov, de)
            if error > limit:
                raise ArithmeticError('Conditional temporal identity failed')
            raw = f[b][mask][:, ids]-f[a][mask][:, ids]
            changes[label] = dict(delta_temporal_energy=change,
                twice_base_delta_covariance=cov, delta_centered_energy=de,
                delta_raw_rms=float(np.sqrt(np.mean(raw**2))),
                delta_mean_energy=float(np.mean(raw.mean(0)**2)),
                identity_error=error, identity_bound=limit)
        out[name] = dict(component_energy=energy, twice_component_covariance=cross,
            branch_energy=branch_energy, reconstruction=reconstructed,
            identity_error=err, identity_bound=bound, conditional_changes=changes)
    return out


def paired_rows(rows, specs):
    """Keep both arm slots and missing/invalid denominators for every pair."""
    lookup = {(r['arm'],r['trace_id']):r for r in rows}
    pairs = []
    for s in specs:
        f = lookup.get(('F',s['trace_id']))
        u = lookup.get(('U',s['historical_trace_id']))
        status = 'valid' if all(x and x['status']=='valid' for x in (f,u)) else 'incomplete'
        pairs.append(dict(F_trace=s['trace_id'],U_trace=s['historical_trace_id'],
            timestep=s['timestep'],case_id=s['case_id'],status=status,
            F_status=f['status'] if f else 'missing',U_status=u['status'] if u else 'missing'))
    return pairs
