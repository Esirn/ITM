#!/usr/bin/env python
"""Complete CPU comparison of head candidate with bound existing references."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from itm.models.mdm_correction import read, artifact, verify
from itm.data.humanml_torch import recover_from_ric_torch
from itm.metrics.canonical_b0 import evaluate_motion
from itm.metrics.sensor_relative import evaluate_sensor_relative
from evaluate_expanded_canonical_pilot import KEYS, scalar_proof
from evaluate_sensor_relative_saved_v2 import alternative_proof
from run_expanded_canonical_training import write

ROOT=Path(__file__).resolve().parents[1]


def evaluate(tp_path,train_run,sp_path,sample_run,reference_path,output):
    if output.exists():raise FileExistsError(output)
    tp,sp=read(tp_path),read(sp_path)
    ts,ss=read(train_run/'queue_status.json'),read(sample_run/'queue_status.json')
    if ts['status']!='completed' or ss['status']!='completed':raise ValueError('Complete queues required')
    for status,plan in [(ts,tp),(ss,sp)]:
        for k,n in plan['counts'].items():
            if k in status['worker'] and status['worker'][k]!=n:raise ValueError('Incomplete '+k)
        for a in plan['expected_artifacts']+plan['input_artifacts']:verify(a)
    data=read(verify(tp['dataset']));records={r['motion_id']:r for r in data['records']}
    mean,std=[np.load(verify(data[k])).astype(np.float32) for k in ['motion_mean','motion_std']]
    old_rows=[r for r in read(reference_path) if r['stage']=='U100_text2.5' and r['sensor_config']=='head']
    if len(old_rows)!=99:raise ValueError('Fixed99 reference views required')
    rows=[];sources={};metric_checks=0
    for old in old_rows:
        path=verify(old['source']);sources[str(path)]=artifact(path)
        row={k:old[k] for k in ['motion_id','split','sensor_config','population','frames','source','status']}
        row.update(arm='original_U100' if old['arm']=='expanded_U' else old['arm'],
                   metrics=dict(old['inherited_metrics'],**old['metrics']))
        rows.append(row)
    original={r['motion_id']:r for r in rows if r['arm']=='original_U100'}
    for row in sp['rows']:
        if row['sensor_config']!='head':raise ValueError('Head-only sampling required')
        path=sample_run/'artifacts'/(row['case_id']+'.npz')
        metadata=read(path.with_suffix('.json'));verify(metadata['final']);verify(metadata['raw'])
        old_meta=read(Path(original[row['motion_id']]['source']['path']).with_suffix('.json'))
        if metadata['random_plans']!=old_meta['random_plans']:raise ValueError('Original paired noise changed')
        if metadata['checkpoint']!=sp['checkpoint']:raise ValueError('Sampling checkpoint changed')
        with np.load(path,allow_pickle=False) as z:
            p,g,mask,features=z['motion'][0],z['gt'][0],z['frame_mask'][0],z['normalized_humanml263'][0]
        record=records[row['motion_id']]
        if p.shape!=(record['frames'],22,3) or mask.shape!=(record['frames'],) or mask.dtype!=bool or not mask.all():
            raise ValueError('Whole sequence required')
        with np.load(verify(record['input']),allow_pickle=False) as z:np.testing.assert_array_equal(g,z['gt'])
        decoded=recover_from_ric_torch(torch.from_numpy(features.reshape(263,-1).T*std+mean)).numpy()
        np.testing.assert_allclose(p,decoded,atol=2e-6,rtol=1e-6)
        global_m=evaluate_motion(p,g,'head');global_m['foot_status']='not_evaluated_support_not_reviewed'
        local_m=evaluate_sensor_relative(p,g,'head')
        metric_checks+=scalar_proof(p,g,'head',global_m)+alternative_proof(p,g,'head',local_m)
        rows.append(dict(arm='head_supervised_U',motion_id=row['motion_id'],split=row['split'],sensor_config='head',
            population='matched_val8' if row['split']=='val' else 'training_anchors',frames=record['frames'],
            source=artifact(path),status='valid',metrics=dict(global_m,**local_m)))
        sources[str(path)]=artifact(path)
    keys=KEYS+list(local_m)
    groups=defaultdict(list)
    for r in rows:groups[r['population'],r['arm']].append(r)
    aggregates=[dict(population=pop,arm=arm,planned_n=len(rr),valid_n=len(rr),
        metrics={k:float(np.mean([r['metrics'][k] for r in rr])) for k in keys})
        for (pop,arm),rr in groups.items()]
    lookup={(r['population'],r['arm'],r['motion_id']):r for r in rows}
    contrasts=[]
    for r in rows:
        if r['arm']!='head_supervised_U':continue
        refs=['original_U100']+(['MDM_text_g2.5','simple_text_imu','simple_imu_only'] if r['split']=='val' else [])
        for ref in refs:
            b=lookup[r['population'],ref,r['motion_id']]
            contrasts.append(dict(population=r['population'],motion_id=r['motion_id'],reference=ref,
                delta={k:r['metrics'][k]-b['metrics'][k] for k in keys}))
    comparative=[]
    for ref in ['original_U100','MDM_text_g2.5','simple_text_imu','simple_imu_only']:
        rr=[r for r in contrasts if r['population']=='matched_val8' and r['reference']==ref]
        if len(rr)!=8:raise ValueError('Same8val references required')
        comparative.append(dict(reference=ref,n=8,lower_error_sources={k:sum(r['delta'][k]<0 for r in rr) for k in keys},
            mean_delta={k:float(np.mean([r['delta'][k] for r in rr])) for k in keys}))
    # Compare every actual update receipt to its fixed original exposure/noise address.
    reference_train=ROOT/'outputs/itm_rework/method/expanded_canonical_training_authorized_20261009_v2/artifacts/updates'
    receipts=[];aux=0
    for i in range(1,1601):
        receipt=read(train_run/'artifacts/updates'/f'{i:04d}.json')
        prior=read(reference_train/f'{i:04d}.json')
        group=tp['exposures'][(i-1)*16:i*16]
        if len(receipt['exposures'])!=16:raise ValueError('Incomplete update')
        for actual,ref,expected in zip(receipt['exposures'],prior['exposures'],group):
            if any(actual[k]!=ref[k] for k in ['exposure','frames','noise_sha256']):raise ValueError('Exposure/noise changed')
            active=expected['sensor_config']=='head' and expected['imu_present'];aux+=int(active)
            if actual['auxiliary_active']!=active:raise ValueError('Auxiliary activation changed')
            parts=actual['loss_parts']
            np.testing.assert_allclose(actual['loss'],parts['base']+parts['auxiliary'],atol=1e-6,rtol=1e-6)
            if not active and any(parts[k]!=0 for k in ['auxiliary','position','velocity']):
                raise ValueError('Auxiliary active in forbidden exposure')
        receipts.append(artifact(train_run/'artifacts/updates'/f'{i:04d}.json'))
    if aux!=11528:raise ValueError('Incomplete auxiliary population')
    checkpoint=torch.load(verify(sp['checkpoint']),map_location='cpu',weights_only=True)
    for field in ['imu_encoder','adapters']:
        if not all(torch.isfinite(v).all() for v in checkpoint[field].values()):raise ValueError('Nonfinite checkpoint')
    summary=dict(status='complete_head_development_comparison',new_full_samples=11,new_decode_checks=11,
        scalar_metric_checks=metric_checks,reused_reference_outputs=83,matched_val_sources=8,training_anchor_sources=3,
        matched_noise_checks=11,updates_verified=1600,exposures_verified=25600,auxiliary_exposures_verified=aux,
        missing=0,formal_test=False,strict_IMU_consistency_certified=False,behavior_success_assigned=False,
        semantic_metrics_computed=False,training_elapsed_seconds=ts['elapsed_seconds'],sampling_elapsed_seconds=ss['elapsed_seconds'],
        limits=['Single seed/K1;8 development validation sources, not formal test.',
                'Head primary; no new wrists generation or efficacy claim.',
                'Local position/velocity/jerk are joint proxies; no sensor orientation/vertex acceleration closure.',
                'Text semantics/FID not certified; captions not all manually reviewed.'])
    output.mkdir(parents=True)
    for name,value in [('rows',rows),('aggregates',aggregates),('contrasts',contrasts),('comparison_summary',comparative),('summary',summary)]:
        write(output/(name+'.json'),value)
    provenance=[artifact(p) for p in [tp_path,sp_path,reference_path,train_run/'queue_status.json',sample_run/'queue_status.json',Path(__file__).resolve()]]
    write(output/'sources.json',provenance+list(sources.values()))
    write(output/'receipt_index.json',receipts)
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.iterdir())])
    print(summary)
    for r in aggregates:
        if r['population']=='matched_val8':print(r)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['training-plan','training-run','sampling-plan','sampling-run','references','output']:
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();evaluate(a.training_plan.resolve(),a.training_run.resolve(),a.sampling_plan.resolve(),
        a.sampling_run.resolve(),a.references.resolve(),a.output.resolve())
