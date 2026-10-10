#!/usr/bin/env python
"""CPU-only supplementary evaluation of complete, previously generated motions."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from itm.metrics.sensor_relative import SENSOR_JOINTS, evaluate_sensor_relative


def read(path):
    return json.loads(Path(path).read_text())


def descriptor(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify(d):
    if descriptor(d['path']) != d:
        raise ValueError('Artifact changed: ' + d['path'])
    return Path(d['path'])


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def alternative_proof(p, g, sensor, r):
    """Joint-wise reduction and explicit differences, including signed cross term."""
    p, g = p.astype(np.float64), g.astype(np.float64)
    sums = defaultdict(list)
    for j in SENSOR_JOINTS[sensor]:
        a, b = p[:, j] - p[:, 0], g[:, j] - g[:, 0]
        e = (p[:, j] - g[:, j]) - (p[:, 0] - g[:, 0])
        norm = lambda x: np.sqrt(np.einsum('ti,ti->t', x, x))
        bias = e.mean(0)
        sums['position_mean'].append(norm(e).mean())
        sums['position_square'].append((norm(e)**2).mean())
        sums['velocity_square'].append((norm((e[1:] - e[:-1])*20)**2).mean())
        sums['static_square'].append(np.dot(bias, bias))
        sums['dynamic_square'].append((norm(e - bias)**2).mean())
        sums['global_square'].append((norm(p[:, j] - g[:, j])**2).mean())
        sums['cross'].append(2*np.einsum('ti,ti->t', p[:, 0]-g[:, 0], e).mean())
        for x, prefix in ((a, ''), (b, 'GT_')):
            sums[prefix+'spread_square'].append((norm(x-x.mean(0))**2).mean())
            sums[prefix+'speed_square'].append((norm((x[1:]-x[:-1])*20)**2).mean())
            d3 = x[3:]-3*x[2:-1]+3*x[1:-2]-x[:-3]
            sums[prefix+'jerk_mean'].append(norm(d3*8000).mean())
    mean = lambda k: float(np.mean(sums[k]))
    alternative = {
        'observed_relative_position_mean_m': mean('position_mean'),
        'observed_relative_position_RMSE_m': np.sqrt(mean('position_square')),
        'observed_relative_velocity_RMSE_m_s': np.sqrt(mean('velocity_square')),
        'observed_relative_static_error_RMS_m': np.sqrt(mean('static_square')),
        'observed_relative_dynamic_error_RMS_m': np.sqrt(mean('dynamic_square')),
        'observed_global_position_MSE_m2': mean('global_square'),
        'root_position_MSE_m2': float(np.einsum('ti,ti->t', p[:,0]-g[:,0], p[:,0]-g[:,0]).mean()),
        'root_local_error_cross_m2': mean('cross'),
    }
    for prefix in ('', 'GT_'):
        alternative[prefix+'observed_relative_spread_RMS_m'] = np.sqrt(mean(prefix+'spread_square'))
        alternative[prefix+'observed_relative_speed_RMS_m_s'] = np.sqrt(mean(prefix+'speed_square'))
        alternative[prefix+'observed_relative_jerk_mean_m_s3'] = mean(prefix+'jerk_mean')
    for key in r:
        np.testing.assert_allclose(r[key], alternative[key], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(r['observed_relative_position_RMSE_m']**2,
        r['observed_relative_static_error_RMS_m']**2+r['observed_relative_dynamic_error_RMS_m']**2,
        atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(r['observed_global_position_MSE_m2'],
        r['root_position_MSE_m2']+r['observed_relative_position_RMSE_m']**2+r['root_local_error_cross_m2'],
        atol=1e-12, rtol=1e-12)
    return len(r) + 2


def evaluate(plan_path, output):
    if output.exists(): raise FileExistsError(output)
    plan = read(plan_path)
    for d in plan['bound_artifacts']: verify(d)
    dataset = read(verify(plan['dataset']))
    records = {r['motion_id']: r for r in dataset['records']}
    stages = {s['name']: read(verify(s['rows'])) for s in plan['stages']}
    if {k:len(v) for k,v in stages.items()} != plan['expected_stage_rows']:
        raise ValueError('Stage denominator changed')
    output.mkdir(parents=True)
    cache, rows, checks = {}, [], 0
    for stage, old_rows in stages.items():
        for old in old_rows:
            if old['status'] != 'valid': raise ValueError('Incomplete saved output')
            key = (old['source']['path'], old['sensor_config'])
            r = records[old['motion_id']]
            if old['split'] != r['split']: raise ValueError('Split identity mismatch')
            if key not in cache:
                with np.load(verify(old['source']), allow_pickle=False) as z:
                    p, g, mask = z['motion'][0], z['gt'][0], z['frame_mask'][0]
                if p.shape != (r['frames'],22,3) or mask.dtype != bool or mask.shape != (r['frames'],) or not mask.all():
                    raise ValueError('Incomplete sequence or mask')
                with np.load(verify(r['input']), allow_pickle=False) as z:
                    np.testing.assert_array_equal(g, z['gt'])
                m = evaluate_sensor_relative(p, g, old['sensor_config'])
                checks += alternative_proof(p, g, old['sensor_config'], m)
                cache[key] = dict(metrics=m, motion_id=old['motion_id'], split=old['split'])
            cached = cache[key]
            if cached['motion_id'] != old['motion_id'] or cached['split'] != old['split']:
                raise ValueError('Reused output identity mismatch')
            row = {k:old[k] for k in ('arm','motion_id','split','sensor_config','population','frames','source')}
            row.update(stage=stage, status='valid', metrics=cached['metrics'], inherited_metrics=old['metrics'])
            row['amplitude_ratios'] = {}
            for stem in ('spread_RMS_m','speed_RMS_m_s'):
                gt = cached['metrics']['GT_observed_relative_'+stem]
                row['amplitude_ratios'][stem] = cached['metrics']['observed_relative_'+stem]/gt if gt > 1e-12 else None
            rows.append(row)
    if len(cache) != plan['expected_unique_outputs']: raise ValueError('Output denominator changed')
    grouped = defaultdict(list)
    for r in rows: grouped[r['stage'],r['population'],r['arm'],r['sensor_config']].append(r)
    aggregates = []
    for (stage,pop,arm,sensor), rr in grouped.items():
        ratios = {}
        for k in rr[0]['amplitude_ratios']:
            vals = [r['amplitude_ratios'][k] for r in rr if r['amplitude_ratios'][k] is not None]
            ratios[k] = dict(mean=float(np.mean(vals)) if vals else None, valid_n=len(vals), omitted_zero_GT_n=len(rr)-len(vals))
        aggregates.append(dict(stage=stage,population=pop,arm=arm,sensor_config=sensor,
            planned_n=len(rr),valid_n=len(rr),
            metrics={k:float(np.mean([r['metrics'][k] for r in rr])) for k in rr[0]['metrics']},
            amplitude_ratios=ratios))
    lookup = {(r['stage'],r['population'],r['arm'],r['motion_id'],r['sensor_config']):r for r in rows}
    contrasts = []
    for r in rows:
        if r['arm'] != 'expanded_U' or r['population'] != 'matched_val8': continue
        for arm in ('MDM_text_g2.5','simple_text_imu','simple_imu_only'):
            ref = lookup[r['stage'],r['population'],arm,r['motion_id'],r['sensor_config']]
            if r['frames'] != ref['frames']: raise ValueError('Mismatched comparison length')
            contrasts.append(dict(stage=r['stage'],motion_id=r['motion_id'],sensor_config=r['sensor_config'],reference=arm,
                delta={k:r['metrics'][k]-ref['metrics'][k] for k in r['metrics']}))
    comparative = []
    cc = defaultdict(list)
    for r in contrasts: cc[r['stage'],r['sensor_config'],r['reference']].append(r)
    for (stage,sensor,ref), rr in cc.items():
        comparative.append(dict(stage=stage,sensor_config=sensor,reference=ref,n=len(rr),
            lower_error_sources={k:sum(r['delta'][k] < 0 for r in rr) for k in
                ('observed_relative_position_mean_m','observed_relative_velocity_RMSE_m_s')},
            mean_delta={k:float(np.mean([r['delta'][k] for r in rr])) for k in rr[0]['delta']}))
    summary = dict(status='completed_supplementary_development_evaluation',
        rows=len(rows),unique_saved_outputs=len(cache),scalar_checks=checks,missing=0,
        new_model_calls=0,GPU_queries=0,formal_test=False,strict_IMU_consistency_certified=False,
        behavior_success_assigned=False,old_metrics_changed=False,
        scope='Observed joint proxies in pelvis-relative positions, original world axes/heading, complete 20Hz clips.',
        limits=['Post-hoc supplementary analysis; not a replacement of historical frozen gates.',
                'No rotation/scale/time fitting, smoothing, frame trimming, or generated-signal calibration.',
                'Joint 20/21 are not wrist sensor mounting vertices; orientation/acceleration closure not certified.',
                'One seed/K1; matched main comparison uses 8 val sources, all32 simple baselines remain separate.',
                'Velocity/jerk from joint differences are kinematic proxies, not physical IMU sensor signals.',
                'Root/global errors retained; squared decomposition includes a signed cross term, not additive MPJPE shares.'])
    for name,value in [('rows',rows),('aggregates',aggregates),('contrasts',contrasts),('comparison_summary',comparative),('summary',summary)]:
        write(output/(name+'.json'),value)
    write(output/'provenance.json',dict(plan=descriptor(plan_path),bound_artifacts=plan['bound_artifacts']))
    write(output/'artifact_index.json',[descriptor(p) for p in sorted(output.iterdir())])
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    evaluate(args.plan.resolve(),args.output.resolve())
