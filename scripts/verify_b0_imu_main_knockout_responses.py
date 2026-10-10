#!/usr/bin/env python
"""Independent Gram/dot calculations over all saved900-step response rows."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from evaluate_b0_native_training_sufficiency import read,verify,artifact,arrays,matrix,write
from itm.metrics.training_sufficiency import VIEWS


def run(plan_path,results,output):
    if output.exists():raise FileExistsError('Fresh complete response verification required')
    plan=dict(jobs=read(results/'jobs.json'),sources=read(results/'sources.json'))
    for a in plan['sources']+read(results/'sources.json')+read(results/'artifact_index.json'):verify(a)
    rows=read(results/'rows.json')
    if len(rows)!=1800 or any(r['status']!='valid' for r in rows):raise ValueError('Full saved response denominators')
    root=Path(__file__).resolve().parents[1];oldplan=read(root/'docs/itm_rework/protocol/b0_saved_trace_response_20261008_v1/plan.json')
    mean,std=[np.load(verify(oldplan['normalization'][k]),allow_pickle=False) for k in ('motion_mean','motion_std')]
    cases={s['case_id']:s for s in read(root/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/sampling_slots.json')}
    targets={cid:((np.load(Path(cases[cid]['gt_path']).with_name('motion.npy'),allow_pickle=False)-mean)/std).astype(float) for cid in {j['panel_case_id'] for j in plan['jobs']}}
    cells=0;maxerror=0.
    def check(a,b,*scale):
        nonlocal cells,maxerror
        error=abs(a-b);maxerror=max(maxerror,error);cells+=1
        if error>64*np.finfo(float).eps*max(1.,abs(a),abs(b),*(abs(v) for v in scale)):raise ArithmeticError('Second-formula saved response mismatch: '+str((a,b)))
    def cov(a,b):return float(np.einsum('ij,ij->',a,b)/a.size)
    def temporal(p,g,metrics):
        pc=p-p.mean(0);gc=g-g.mean(0)
        for view,ids in VIEWS.items():
            a,b,ac,bc=[v[:,ids] for v in (p,g,pc,gc)];m=metrics[view];mu=(a-b).mean(0)
            check(m['mse'],cov(a-b,a-b));check(m['mean_error'],mu@mu/len(ids));check(m['temporal_error'],cov(ac-bc,ac-bc))
            pe,ge=cov(ac,ac),cov(bc,bc);check(m['prediction_temporal_energy'],pe);check(m['target_temporal_energy'],ge)
            check(m['temporal_covariance'],cov(ac,bc));check(m['target_mean_baseline_mse'],ge)
            if ge:check(m['temporal_energy_ratio'],pe/ge)
            elif m['temporal_energy_ratio'] is not None:raise ValueError('Undefined GT ratio')
            for key,v in [('prediction_difference_rms',a),('target_difference_rms',b)]:
                d=np.diff(v,axis=0);check(m[key],np.sqrt(cov(d,d)))
            if m['dimensions']!=len(ids) or m['adjacent_valid_pairs']!=len(p)-1:raise ValueError('Frame/view denominator changed')
        for k,v in metrics['balanced'].items():check(v,.5*metrics['root_ric67'][k]+.5*metrics['remaining196'][k])
    for job,row in zip(plan['jobs'],rows):
        if job['row_id']!=row['row_id'] or any(job[k]!=row[k] for k in ('condition','sensor_config','timestep','panel_case_id')):raise ValueError('Frozen row role changed')
        z=arrays(verify(job['raw']));f={k:matrix(z[k]).astype(float) for k in row['temporal']};g=targets[job['panel_case_id']]
        for k,p in f.items():temporal(p,g,row['temporal'][k])
        components=dict(base=f['f00'],text=f['f10']-f['f00'],imu=f['f01']-f['f00'],interaction=f['f11']-f['f10']-f['f01']+f['f00'])
        centered={k:v-v.mean(0) for k,v in components.items()};branch={k:f[k]-f[k].mean(0) for k in ('f00','f10','f01','f11')};names=list(components)
        for view,ids in VIEWS.items():
            c=np.stack([centered[k][:,ids].ravel() for k in names]);gram=c@c.T/c.shape[1];m=row['response'][view]
            for i,k in enumerate(names):check(m['component_energy'][k],gram[i,i])
            for i,j in combinations(range(4),2):check(m['twice_component_covariance'][names[i]+'__'+names[j]],2*gram[i,j])
            for k,v in branch.items():check(m['branch_energy'][k],cov(v[:,ids],v[:,ids]))
            check(m['reconstruction'],gram.sum(),*gram.diagonal(),*(2*gram[i,j] for i,j in combinations(range(4),2)))
            for label,a,b in [('text_at_no_imu','f00','f10'),('imu_at_no_text','f00','f01'),('imu_at_text','f10','f11'),('text_at_imu','f01','f11'),('joint_at_base','f00','f11')]:
                base,new=branch[a][:,ids],branch[b][:,ids];d=new-base;raw=f[b][:,ids]-f[a][:,ids];cm=m['conditional_changes'][label]
                de,bc=cov(d,d),2*cov(base,d);old,newenergy=cov(base,base),cov(new,new)
                check(cm['delta_centered_energy'],de);check(cm['twice_base_delta_covariance'],bc)
                check(cm['delta_temporal_energy'],newenergy-old,newenergy,old,de,bc)
                check(cm['delta_raw_rms'],np.sqrt(cov(raw,raw)));mu=raw.mean(0);check(cm['delta_mean_energy'],mu@mu/len(ids))
        aa,bb,cc,dd=[z[k] for k in ('f00','f10','f01','f11')]
        unit=matrix(aa+np.float32(1.)*(bb-aa)+np.float32(1.)*(cc-aa)+np.float32(1.)*(dd-bb-cc+aa)).astype(float)
        temporal(unit,g,row['derived_native111_temporal'])
        error=unit-g;delta=f['actual_combined_prediction']-unit
        for view,ids in VIEWS.items():
            m=row['same_state_knockout_minus_native111'][view];a,b=error[:,ids],delta[:,ids];old,new=cov(a,a),cov(a+b,a+b);cross,energy=2*cov(a,b),cov(b,b)
            check(m['baseline_mse'],old);check(m['prediction_mse'],new);check(m['delta_mse'],new-old,old,new,cross,energy);check(m['cross'],cross);check(m['residual_energy'],energy);check(m['prediction_difference_rms'],np.sqrt(energy));check(m['prediction_difference_max_abs'],float(np.max(abs(b))))
            if m['count']!=b.size or m['dimensions']!=len(ids):raise ValueError('Same-state denominator')
        for k,v in row['same_state_knockout_minus_native111']['balanced'].items():check(v,.5*row['same_state_knockout_minus_native111']['root_ric67'][k]+.5*row['same_state_knockout_minus_native111']['remaining196'][k])
        if row['timestep']==0:print('Verified saved response',row['row_id'],flush=True)
    proof=dict(status='passed',rows=1800,derived_native111_rows=1800,independent_scalar_checks=cells,max_scalar_absolute_error=maxerror,
        new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,independent_review=False,
        bound='Inherited float64 64eps times context scale; scientific software tolerances unchanged',jobs=artifact(results/'jobs.json'),verifier=artifact(Path(__file__).resolve()))
    output.parent.mkdir(parents=True,exist_ok=True);write(output,proof);print(proof)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('results','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(None,a.results.resolve(),a.output.resolve())
