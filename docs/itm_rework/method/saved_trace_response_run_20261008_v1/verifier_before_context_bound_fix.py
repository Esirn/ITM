#!/usr/bin/env python
"""Independent CPU covariance calculation and complete source/hash replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def matrix(v):return v[0,:,0,:].T.astype(np.float64)


def main(plan_path,result_path,output):
    if output.exists():raise ValueError('Refusing to overwrite verification')
    plan=read(plan_path);rows=read(result_path/'traces.json');sample_rows=read(result_path/'samples.json')
    for e in plan['sources']+read(result_path/'artifact_index.json'):
        if sha(e['path'])!=e['sha256']:raise ValueError('Source/artifact changed: '+e['path'])
    views=dict(uniform=list(range(263)),root_ric67=list(range(67)),remaining196=list(range(67,263)),
        root=list(range(4)),ric=list(range(4,67)),rotation6d=list(range(67,193)),
        local_velocity=list(range(193,259)),contact=list(range(259,263)))
    for n,j in dict(ric_legs=[1,2,4,5,7,8,10,11],ric_trunk_head=[3,6,9,12,15],ric_arms=[13,14,16,17,18,19,20,21]).items():
        views[n]=[4+3*(x-1)+k for x in j for k in range(3)]
    mean,std=[np.load(plan['normalization'][k]['path'],allow_pickle=False) for k in ('motion_mean','motion_std')]
    root=Path(__file__).resolve().parents[1]
    cases={c['case_id']:c for c in read(root/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/sampling_slots.json')}
    targets={k:((np.load(Path(c['gt_path']).with_name('motion.npy'),allow_pickle=False)-mean)/std).astype(np.float64) for k,c in cases.items()}
    cells=0;max_error=0.
    def check(actual,expected):
        nonlocal cells,max_error
        cells+=1;error=abs(actual-expected);max_error=max(max_error,error)
        if error>64*np.finfo(float).eps*max(1.,abs(actual),abs(expected)):
            raise ValueError('Independent scalar mismatch: '+str((actual,expected)))
    def covariance(a,b):return float(np.einsum('ij,ij->',a,b)/(a.size))
    def verify_temporal(p,g,metrics):
        pc=p-p.mean(0);gc=g-g.mean(0)
        for name,ids in views.items():
            a=p[:,ids];b=g[:,ids];ac=pc[:,ids];bc=gc[:,ids];v=metrics[name]
            check(v['mse'],covariance(a-b,a-b))
            check(v['mean_error'],float(np.dot((a-b).mean(0),(a-b).mean(0))/len(ids)))
            check(v['temporal_error'],covariance(ac-bc,ac-bc))
            check(v['prediction_temporal_energy'],covariance(ac,ac))
            check(v['target_temporal_energy'],covariance(bc,bc))
            check(v['temporal_covariance'],covariance(ac,bc))
    for row in rows:
        if row['status']!='valid':raise ValueError('Incomplete real trace denominator')
        with np.load(row['path'],allow_pickle=False) as z:f={k:matrix(z[k]) for k in ('f00','f10','f01','f11','x_t','actual_combined_prediction')}
        for k,p in f.items():verify_temporal(p,targets[row['pair_case_id']],row['metrics']['temporal'][k])
        # A Gram matrix computes the component variances and all cross terms at once.
        components=np.stack([f['f00'],f['f10']-f['f00'],f['f01']-f['f00'],f['f11']-f['f10']-f['f01']+f['f00']])
        centered=components-components.mean(axis=1,keepdims=True)
        names=['base','text','imu','interaction']
        for name,ids in views.items():
            c=centered[:,:,ids].reshape(4,-1);gram=c@c.T/c.shape[1]
            v=row['metrics']['response'][name]
            for i,n in enumerate(names):check(v['component_energy'][n],float(gram[i,i]))
            for i in range(4):
                for j in range(i+1,4):check(v['twice_component_covariance'][names[i]+'__'+names[j]],float(2*gram[i,j]))
            check(v['reconstruction'],float(gram.sum()))
            for n,a,b in [('text_at_no_imu','f00','f10'),('imu_at_no_text','f00','f01'),('imu_at_text','f10','f11'),('text_at_imu','f01','f11'),('joint_at_base','f00','f11')]:
                base=f[a][:,ids]-f[a][:,ids].mean(0);delta=f[b][:,ids]-f[a][:,ids];dc=delta-delta.mean(0)
                c=v['conditional_changes'][n]
                check(c['delta_centered_energy'],covariance(dc,dc))
                check(c['twice_base_delta_covariance'],2*covariance(base,dc))
                check(c['delta_temporal_energy'],covariance(base+dc,base+dc)-covariance(base,base))
                check(c['delta_raw_rms'],np.sqrt(covariance(delta,delta)))
                check(c['delta_mean_energy'],float(delta.mean(0)@delta.mean(0)/len(ids)))
    features={}
    for row in sample_rows:
        if row['status']!='valid':raise ValueError('Incomplete real sample denominator')
        with np.load(row['path'],allow_pickle=False) as z:p=matrix(z['normalized_humanml263'])
        verify_temporal(p,targets[row['pair_case_id']],row['metrics'])
        features[(row['arm'],row['pair_case_id'])]=p
    equalities=[]
    for cid,c in cases.items():
        if c['condition'] in ('text_only','physical_static','backbone_bypass'):
            a,b=[features[(arm,cid)] for arm in ('F','U')]
            if not np.array_equal(a,b):raise ValueError('Absent/static/bypass cross-arm final mismatch')
            equalities.append(dict(case_id=cid,bitwise=True))
    result=dict(status='passed',bound='64eps float64 formula checks; original scientific tolerances unchanged',
        source_hashes=len(plan['sources']),result_artifacts=len(read(result_path/'artifact_index.json')),
        trace_points=len(rows),final_samples=len(sample_rows),independent_scalar_checks=cells,
        max_scalar_absolute_error=max_error,absent_static_bypass_cross_arm_final_equalities=equalities,
        new_GPU_queries=0,new_model_calls=0,independent_review=False,
        plan_sha256=sha(plan_path),verifier_sha256=sha(__file__))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print({k:v for k,v in result.items() if k!='absent_static_bypass_cross_arm_final_equalities'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    main(a.plan.resolve(),a.results.resolve(),a.output.resolve())
