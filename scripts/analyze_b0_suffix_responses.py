#!/usr/bin/env python
"""NumPy-only full saved suffix response analysis, never a sampler or model."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from evaluate_b0_native_training_sufficiency import read,verify,artifact,arrays,descriptor,matrix,write
from itm.metrics.training_sufficiency import temporal_decomposition
from itm.metrics.trace_response import response_decomposition

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/itm_rework/protocol/b0_suffix_response_CPU_20261008_v1/plan.json'
PLAN_SHA='e6bfa66b25e656f574144149fc8f69d3e452927aa7a6999a2c33674ae58434ed'
F200_SHA='d0e4f39f8c7ee5b980243baac48bc1bc02dc8f7c15c7a1f47f0adabb0b7e0ad3'
NAMES=('x_t','f00','f10','f01','f11','actual_combined_prediction')


def compact(path,data):
    with Path(path).open('x') as f:json.dump(data,f,ensure_ascii=False,allow_nan=False,separators=(',',':'));f.write('\n')


def validate_raw(job,meta,z):
    if meta['status']!='completed' or meta['trajectory_id']!=job['trajectory_id'] or meta['timestep']!=job['timestep']:
        raise ValueError('Suffix row/time/status identity')
    if meta['checkpoint']['sha256']!=F200_SHA or meta['raw_artifact']!=job['raw']:raise ValueError('Suffix checkpoint/raw identity')
    if set(z)!=set(NAMES)|{'frame_mask'} or meta['arrays']!={k:descriptor(v) for k,v in z.items()}:raise ValueError('Saved raw descriptors')
    for k in NAMES:
        if z[k].dtype!=np.float32 or z[k].shape!=(1,263,1,job['frames']) or not np.isfinite(z[k]).all():raise ValueError('Native finite full-frame prediction')
    mask=z['frame_mask']
    if mask.dtype!=np.bool_ or mask.shape!=(1,job['frames']) or not mask.all():raise ValueError('Original full-frame mask')
    if job['condition']=='switch_imu_off':
        for a,b in [('f00','f01'),('f10','f11')]:
            if not np.all(abs(z[a].astype(float)-z[b])<=1e-6+1e-6*abs(z[b])):raise ValueError('Inherited strict-absence software equation')


def gate(plan):
    if artifact(PLAN)['sha256']!=PLAN_SHA or plan['planned_rows']!=3600 or len(plan['jobs'])!=3600:raise ValueError('Exact complete CPU plan')
    for a in plan['sources']:verify(a)
    proof=read(verify(plan['software_gate']))
    if proof['status']!='passed' or proof['software_equations']!=7244 or proof['posterior_transition_hash_links']!=3600:raise ValueError('Finished suffix software proof')
    run=Path(plan['run_directory']);status=read(run/'runtime/status.json')
    if status['status']!='completed' or status['implementation']!=plan['implementation'] or any(r['status']!='completed' for r in status['checks']):raise ValueError('Complete exact suffix queue required')
    table={r['id']:r for r in status['steps']}
    for job in plan['jobs']:
        slot=table[job['row_id']]
        if slot['status']!='completed' or slot['diagnostic_only'] or slot['raw_artifact']!=job['raw']:raise ValueError('Frozen raw slot identity')
    return status


def prepare(output):
    if output.exists():raise FileExistsError('Fresh CPU implementation required')
    plan=read(PLAN);gate(plan);sources={a['path']:a for a in plan['sources']}
    for path in [PLAN,Path(__file__).resolve(),ROOT/'tests/test_suffix_response_adapter.py',
        ROOT/'scripts/evaluate_b0_native_training_sufficiency.py',ROOT/'src/itm/metrics/training_sufficiency.py',ROOT/'src/itm/metrics/trace_response.py']:
        a=artifact(path);sources[a['path']]=a
    for job in plan['jobs']:
        z=arrays(verify(job['raw']));validate_raw(job,read(verify(job['metadata'])),z)
    output.mkdir(parents=True)
    write(output/'implementation.json',dict(version='itm-suffix-response-CPU-v1',plan=artifact(PLAN),expected_artifacts=[sources[k] for k in sorted(sources)],
        planned_rows=3600,validated_raw_jobs=3600,new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0))
    print(dict(implementation=artifact(output/'implementation.json'),dependencies=len(sources)))


def evaluate(manifest_path,output):
    if output.exists():raise FileExistsError('Fresh CPU evaluation required')
    manifest=read(manifest_path)
    for a in manifest['expected_artifacts']:verify(a)
    plan=read(verify(manifest['plan']));gate(plan)
    oldplan=read(ROOT/'docs/itm_rework/protocol/b0_saved_trace_response_20261008_v1/plan.json')
    mean,std=[np.load(verify(oldplan['normalization'][k]),allow_pickle=False) for k in ('motion_mean','motion_std')]
    slots={c['case_id']:c for c in read(ROOT/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/sampling_slots.json')}
    targets={cid:(np.load(Path(slots[cid]['gt_path']).with_name('motion.npy'),allow_pickle=False)-mean)/std for cid in {j['panel_case_id'] for j in plan['jobs']}}
    rows=[];contrasts=[];paired={}
    for job in plan['jobs']:
        row=dict(job,status='missing',source_n=1)
        try:
            z=arrays(verify(job['raw']));validate_raw(job,read(verify(job['metadata'])),z)
            branches={k:matrix(z[k]) for k in NAMES};g=targets[job['panel_case_id']]
            row.update(status='valid',temporal={k:temporal_decomposition(v,g) for k,v in branches.items()},
                response=response_decomposition(branches))
            key=(job['sensor_config'],job['timestep']);other=paired.pop(key,None)
            if other is None:paired[key]=(row,branches['x_t'].astype(float))
            else:
                a,xt=other;off,keep=(row,a) if row['condition']=='switch_imu_off' else (a,row)
                d=branches['x_t'].astype(float)-xt
                contrasts.append(dict(sensor_config=job['sensor_config'],timestep=job['timestep'],status='valid',source_n=1,
                    comparison='IMU_off_minus_keep',state_difference_rms={n:float(np.sqrt(np.mean(d[:,ids]**2))) for n,ids in [('uniform',range(263)),('root_ric67',range(67)),('remaining196',range(67,263))]},
                    actual_prediction={view:{k:off['temporal']['actual_combined_prediction'][view][k]-keep['temporal']['actual_combined_prediction'][view][k]
                        for k in ('mse','mean_error','temporal_error','prediction_temporal_energy','target_temporal_energy','temporal_covariance')}
                        for view in off['temporal']['actual_combined_prediction']},
                    interpretation='Matched initial history/noise and t, but current states differ. Within-state response algebra is not a cross-path counterfactual.'))
        except (ValueError,KeyError,OSError,ArithmeticError) as e:row.update(status='invalid',reason=str(e))
        rows.append(row)
        if len(rows)%900==0:print('CPU rows',len(rows),flush=True)
    present={(c['sensor_config'],c['timestep']) for c in contrasts}
    for config in ('head','wrists'):
        for t in plan['timesteps']:
            if (config,t) not in present:contrasts.append(dict(sensor_config=config,timestep=t,status='inconclusive',source_n=1))
    contrasts.sort(key=lambda r:(r['sensor_config'],-r['timestep']))
    summary=dict(status='completed' if all(r['status']=='valid' for r in rows+contrasts) else 'inconclusive',planned_rows=3600,
        row_counts=dict(Counter(r['status'] for r in rows)),planned_contrasts=1800,contrast_counts=dict(Counter(r['status'] for r in contrasts)),
        source_count=1,panel_configs=2,new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,
        interpretation=plan['interpretation'],original_behavior='All4 suffix outputs incomplete/not_supported; no new semantic label or model certification.')
    output.mkdir(parents=True)
    for name,data in [('rows',rows),('path_contrasts',contrasts),('summary',summary),('sources',[artifact(manifest_path),*manifest['expected_artifacts']])]:compact(output/(name+'.json'),data)
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.glob('*.json'))]);print(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-dir',type=Path);p.add_argument('--implementation',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.prepare_dir:prepare(a.prepare_dir.resolve())
    else:evaluate(a.implementation.resolve(),a.output.resolve())
