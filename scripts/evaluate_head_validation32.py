#!/usr/bin/env python
"""Head-only full-population comparison; reuse bound previously evaluated outputs."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from itm.models.mdm_correction import read,verify,artifact
from itm.data.humanml_torch import recover_from_ric_torch
from itm.metrics.canonical_b0 import evaluate_motion
from itm.metrics.sensor_relative import evaluate_sensor_relative
from evaluate_expanded_canonical_pilot import KEYS,scalar_proof
from evaluate_sensor_relative_saved_v2 import alternative_proof
from run_expanded_canonical_training import write


def evaluate(plan_path,run,output):
    if output.exists():raise FileExistsError(output)
    plan,status=read(plan_path),read(run/'queue_status.json')
    if status['status']!='completed':raise ValueError('Complete48sampling queue required')
    for k in ['samples','mdm_calls','encoder_calls','caption_calls','decodes']:
        if status['worker'][k]!=plan['counts'][k]:raise ValueError('Incomplete '+k)
    for d in plan['expected_artifacts']+plan['input_artifacts']:verify(d)
    data=read(verify(plan['dataset']));records={r['motion_id']:r for r in data['records']}
    expected={r['motion_id'] for r in data['records'] if r['split']=='val'}
    if len(expected)!=32:raise ValueError('Fixed32val required')
    mean,std=[np.load(verify(data[k])).astype(np.float32) for k in ['motion_mean','motion_std']]
    rows=[];sources={};noises={};checks=0
    cached=read(verify(plan['supplementary_references']))
    for r in cached:
        if r['stage']!='U100_text2.5' or r['sensor_config']!='head' or r['split']!='val':continue
        is_simple=r['arm'] in ['simple_text_imu','simple_imu_only']
        if r['population']!=('all_val32' if is_simple else 'matched_val8'):continue
        p=verify(r['source']);sources[str(p)]=artifact(p)
        metrics=dict(r['inherited_metrics'],**r['metrics'])
        rows.append(dict(arm='original_U100' if r['arm']=='expanded_U' else r['arm'],motion_id=r['motion_id'],
            split='val',sensor_config='head',population='head_all_val32',frames=r['frames'],status='valid',source=r['source'],metrics=metrics))
        if not is_simple:
            metadata=read(p.with_suffix('.json'))
            group=metadata['row']['noise_group']
            if group in noises and noises[group]!=metadata['random_plans']:raise ValueError('Cached paired noise mismatch')
            noises[group]=metadata['random_plans']
    if len(rows)!=80:raise ValueError('Fixed80 cached outputs required')
    for r in plan['rows']:
        p=run/'artifacts'/(r['case_id']+'.npz');metadata=read(p.with_suffix('.json'))
        verify(metadata['final']);verify(metadata['raw'])
        if metadata['checkpoint']!=plan['checkpoint']:raise ValueError('Checkpoint changed')
        group=r['noise_group']
        if group in noises and noises[group]!=metadata['random_plans']:raise ValueError('Paired noise mismatch')
        noises[group]=metadata['random_plans']
        with np.load(p,allow_pickle=False) as z:
            prediction,gt,mask,features=z['motion'][0],z['gt'][0],z['frame_mask'][0],z['normalized_humanml263'][0]
        record=records[r['motion_id']]
        if r['split']!='val' or r['sensor_config']!='head' or prediction.shape!=(record['frames'],22,3) or not mask.all() or mask.shape!=(record['frames'],) or mask.dtype!=bool:
            raise ValueError('Completehead val sequence required')
        with np.load(verify(record['input']),allow_pickle=False) as z:np.testing.assert_array_equal(gt,z['gt'])
        decoded=recover_from_ric_torch(torch.from_numpy(features.reshape(263,-1).T*std+mean)).numpy()
        np.testing.assert_allclose(prediction,decoded,atol=2e-6,rtol=1e-6)
        gm=evaluate_motion(prediction,gt,'head');gm['foot_status']='not_evaluated_support_not_reviewed'
        lm=evaluate_sensor_relative(prediction,gt,'head')
        checks+=scalar_proof(prediction,gt,'head',gm)+alternative_proof(prediction,gt,'head',lm)
        rows.append(dict(arm='original_U100' if r['condition']=='paired' else 'MDM_text_g2.5',
            motion_id=r['motion_id'],split='val',sensor_config='head',population='head_all_val32',
            frames=record['frames'],status='valid',source=artifact(p),metrics=dict(gm,**lm)))
        sources[str(p)]=artifact(p)
    keys=KEYS+list(lm);groups=defaultdict(list)
    for r in rows:groups[r['arm']].append(r)
    if len(rows)!=128 or set(groups)!={'original_U100','MDM_text_g2.5','simple_text_imu','simple_imu_only'} or len(noises)!=32:
        raise ValueError('Incomplete comparison population')
    aggregate=[]
    for arm,rr in groups.items():
        if len(rr)!=32 or {r['motion_id'] for r in rr}!=expected:raise ValueError('Missing source for '+arm)
        ratios={}
        for key in ['observed_relative_spread_RMS_m','observed_relative_speed_RMS_m_s']:
            values=[r['metrics'][key]/r['metrics']['GT_'+key] for r in rr if r['metrics']['GT_'+key]>1e-12]
            ratios[key]=dict(mean=float(np.mean(values)),valid_n=len(values),zero_GT_omitted=32-len(values))
        aggregate.append(dict(arm=arm,planned_n=32,valid_n=32,metrics={k:float(np.mean([r['metrics'][k] for r in rr])) for k in keys},amplitude_ratios=ratios))
    lookup={(r['arm'],r['motion_id']):r for r in rows};contrasts=[];comparisons=[]
    for ref in ['MDM_text_g2.5','simple_text_imu','simple_imu_only']:
        dd=[]
        for mid in sorted(expected):
            u,b=lookup['original_U100',mid],lookup[ref,mid]
            delta={k:u['metrics'][k]-b['metrics'][k] for k in keys};dd.append(delta)
            contrasts.append(dict(motion_id=mid,reference=ref,delta=delta))
        comparisons.append(dict(reference=ref,n=32,lower_error_sources={k:sum(d[k]<0 for d in dd) for k in keys},
            mean_delta={k:float(np.mean([d[k] for d in dd])) for k in keys}))
    summary=dict(status='complete_head_all32_development_comparison',new_samples=48,new_decode_checks=48,
        new_scalar_checks=checks,reused_outputs=80,total_compared_outputs=128,methods=4,sources_per_method=32,
        same_noise_groups=32,missing=0,formal_test=False,strict_IMU_consistency_certified=False,
        behavior_success_assigned=False,semantic_metrics_computed=False,elapsed_seconds=status['elapsed_seconds'],
        checkpoint=plan['checkpoint'],limits=['Single seed/K1; development val, not formal test.',
            'Ground-truth text and joint proxy metrics do not certify semantic success or physical sensor closure.',
            'Head comparison only; known wrists costs retained elsewhere.'])
    output.mkdir(parents=True)
    for name,value in [('rows',rows),('aggregates',aggregate),('contrasts',contrasts),('comparison_summary',comparisons),('summary',summary)]:
        write(output/(name+'.json'),value)
    write(output/'sources.json',[artifact(plan_path),artifact(run/'queue_status.json'),artifact(Path(__file__).resolve()),plan['supplementary_references']]+list(sources.values()))
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.iterdir())])
    print(summary)
    for r in aggregate:print(r)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['plan','run','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();evaluate(a.plan.resolve(),a.run.resolve(),a.output.resolve())
