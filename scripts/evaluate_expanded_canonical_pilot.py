#!/usr/bin/env python
"""Evaluate complete saved motions on exactly matched development populations."""
from collections import defaultdict
import argparse
from pathlib import Path

import numpy as np
import torch

from itm.metrics.canonical_b0 import evaluate_motion
from itm.models.mdm_correction import read, artifact, verify
from itm.data.humanml_torch import recover_from_ric_torch
from run_expanded_canonical_training import write

KEYS = ['global_MPJPE_m', 'root_relative_MPJPE_m', 'root_path_RMSE_m',
        'head_or_wrists_position_joint_proxy_m', 'velocity_joint_proxy_RMSE_m_s', 'jerk_absolute_m_s3']


def scalar_proof(p, g, sensor, metrics):
    e = p.astype(np.float64) - g.astype(np.float64)
    norm = lambda a: np.sqrt(np.einsum('...i,...i->...', a, a))
    observed = [15] if sensor == 'head' else [20, 21]
    alternative = [norm(e).mean(), norm(e - e[:, :1]).mean(),
                   np.sqrt(np.einsum('ti,ti->t', e[:, 0], e[:, 0]).mean()),
                   norm(e[:, observed]).mean(), np.sqrt(np.mean(norm(np.diff(e, axis=0) * 20)**2)),
                   norm(np.diff(p.astype(np.float64), n=3, axis=0) * 8000).mean()]
    for k, actual in zip(KEYS, alternative):
        np.testing.assert_allclose(metrics[k], actual, atol=1e-12, rtol=1e-12)
    return len(alternative)


def evaluate(training_plan, training_run, sampling_plan, sampling_run, output):
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    tp, sp = read(training_plan), read(sampling_plan)
    ts, ss = read(training_run / 'queue_status.json'), read(sampling_run / 'queue_status.json')
    if ts['status'] != 'completed' or ss['status'] != 'completed': raise ValueError('Complete queues required')
    data = read(verify(tp['dataset'])); records = {r['motion_id']: r for r in data['records']}
    mean, std = [np.load(verify(data[k])).astype(np.float32) for k in ('motion_mean', 'motion_std')]
    rows, sources, proofs = [], {}, 0
    cached_rows = {}
    reused_metric_outputs = 0
    if sp.get('cached_evaluation'):
        for r in read(verify(sp['cached_evaluation'])):
            cached_rows[r['source']['path']] = r
    for key in ('mdm_calls', 'encoder_calls', 'caption_calls', 'samples', 'decodes'):
        if ss['worker'][key] != sp['counts'][key]: raise ValueError('Incomplete sampling budget')
    for key in ('updates', 'exposures', 'mdm_calls', 'encoder_calls', 'caption_calls', 'baseline_updates', 'baseline_train_examples', 'baseline_validation_calls'):
        if ts['worker'][key] != tp['counts'][key]: raise ValueError('Incomplete training budget')
    for e in tp['expected_artifacts'] + sp['expected_artifacts'] + tp['input_artifacts']: verify(e)

    def metric(path, arm, mid, split, sensor, population):
        nonlocal reused_metric_outputs
        path = Path(path); sources[str(path)] = artifact(path)
        if str(path) in cached_rows:
            cached = cached_rows[str(path)]; verify(cached['source'])
            if cached['motion_id'] != mid or cached['sensor_config'] != sensor:
                raise ValueError('Reused metric identity mismatch')
            reused_metric_outputs += 1
            return dict(cached, arm=arm, population=population), 0
        with np.load(path, allow_pickle=False) as z:
            p, g, mask = z['motion'][0], z['gt'][0], z['frame_mask'][0]
            features = z['normalized_humanml263'][0]
        r = records[mid]
        if p.shape != (r['frames'], 22, 3) or mask.shape != (r['frames'],) or not mask.all():
            raise ValueError('Whole sequence/mask mismatch')
        with np.load(verify(r['input']), allow_pickle=False) as z:
            np.testing.assert_array_equal(g, z['gt'])
        vectors = (features.reshape(263, -1).T * std + mean).astype(np.float32)
        decoded = recover_from_ric_torch(torch.from_numpy(vectors)).numpy()
        np.testing.assert_allclose(p, decoded, atol=2e-6, rtol=1e-6)
        result = evaluate_motion(p, g, sensor)
        result['foot_status'] = 'not_evaluated_support_not_reviewed'
        f = features.reshape(263, -1).T.astype(np.float64)
        centered = f[:, :67] - f[:, :67].mean(0)
        result['normalized_root_ric_temporal_energy'] = float(np.mean(centered**2))
        row = dict(arm=arm, motion_id=mid, split=split, sensor_config=sensor, population=population,
                   frames=r['frames'], status='valid', source=artifact(path), metrics=result)
        return row, scalar_proof(p, g, sensor, result)

    noises = {}
    for r in sp['rows'] + sp.get('reused_baselines', []):
        path = verify(r['reuse_final']) if r.get('reuse_final') else sampling_run / 'artifacts' / (r['case_id'] + '.npz')
        metadata = read(path.with_suffix('.json')); verify(metadata['raw']); verify(metadata['final'])
        arm = 'expanded_U' if r['condition'] == 'paired' else 'MDM_text_g2.5'
        row, n = metric(path, arm, r['motion_id'], r['split'], r['sensor_config'], 'matched_val8' if r['split'] == 'val' else 'training_anchors')
        rows.append(row); proofs += n
        key = r['noise_group']; signature = metadata['random_plans']
        if key in noises and noises[key] != signature: raise ValueError('Paired/text noise mismatch')
        noises[key] = signature
    matched = {(r['motion_id'], r['sensor_config']) for r in sp['rows'] if r['split'] == 'val'}
    for arm in ('simple_text_imu', 'simple_imu_only'):
        for r in data['records']:
            if r['split'] != 'val': continue
            for sensor in ('head', 'wrists'):
                path = training_run / 'artifacts/validation' / arm / (r['motion_id'] + '-' + sensor + '.npz')
                row, n = metric(path, arm, r['motion_id'], 'val', sensor, 'all_val32')
                rows.append(row); proofs += n
                if (r['motion_id'], sensor) in matched:
                    rows.append(dict(row, population='matched_val8'))
    tables = defaultdict(list)
    for r in rows:
        tables[r['population'], r['arm'], r['sensor_config']].append(r)
    aggregate = [dict(population=pop, arm=arm, sensor_config=sensor, planned_n=len(values), valid_n=len(values),
                      metrics={k:float(np.mean([r['metrics'][k] for r in values])) for k in KEYS})
                 for (pop, arm, sensor), values in tables.items()]
    lookup = {(r['population'], r['arm'], r['motion_id'], r['sensor_config']):r for r in rows}
    contrasts = []
    for mid, sensor in sorted(matched):
        target = lookup['matched_val8', 'expanded_U', mid, sensor]
        for reference in ('MDM_text_g2.5', 'simple_text_imu', 'simple_imu_only'):
            ref = lookup['matched_val8', reference, mid, sensor]
            contrasts.append(dict(motion_id=mid, sensor_config=sensor, reference=reference,
                                  delta={k:target['metrics'][k] - ref['metrics'][k] for k in KEYS}))
    summary = dict(status='complete_development_comparison', generated_samples=sp['counts']['samples'],
        reused_text_samples=len(sp.get('reused_baselines', [])),
        reused_metric_outputs=reused_metric_outputs,
        baseline_validation_outputs=128, matched_validation_groups=8, configurations=2,
        scalar_metric_checks=proofs, missing=0,
        training_elapsed_seconds=ts['elapsed_seconds'], sampling_elapsed_seconds=ss['elapsed_seconds'],
        strict_IMU_consistency_certified=False, behavior_success_assigned=False,
        formal_test=False, standard_text_motion_metrics_computed=False,
        limits=['K=1 and one training seed.', 'Matched main-method comparison uses 8 validation sources, not all32.',
                'New captions use original annotations without manual certification.',
                'No FID/R-precision or significance claim.', 'Training anchors remain separate from validation.'])
    write(output / 'rows.json', rows); write(output / 'aggregates.json', aggregate)
    write(output / 'contrasts.json', contrasts); write(output / 'summary.json', summary)
    write(output / 'sources.json', list(sources.values()))
    write(output / 'artifact_index.json', [artifact(p) for p in sorted(output.iterdir())])
    print(summary)
    for r in aggregate:
        if r['population'] == 'matched_val8': print(r)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('training-plan', 'training-run', 'sampling-plan', 'sampling-run', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); evaluate(a.training_plan.resolve(), a.training_run.resolve(), a.sampling_plan.resolve(), a.sampling_run.resolve(), a.output.resolve())
