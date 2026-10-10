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
    plan=read(plan_path)
    for a in plan['sources']+read(results/'sources.json')+read(results/'artifact_index.json'):verify(a)
    rows=read(results/'rows.json');contrasts=read(results/'path_contrasts.json')
    if len(rows)!=3600 or len(contrasts)!=1800 or any(r['status']!='valid' for r in rows+contrasts):raise ValueError('Full saved response denominators')
    root=Path(__file__).resolve().parents[1];oldplan=read(root/'docs/itm_rework/protocol/b0_saved_trace_response_20261008_v1/plan.json')
    mean,std=[np.load(verify(oldplan['normalization'][k]),allow_pickle=False) for k in ('motion_mean','motion_std')]
    cases={s['case_id']:s for s in read(root/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/sampling_slots.json')}
    targets={cid:((np.load(Path(cases[cid]['gt_path']).with_name('motion.npy'),allow_pickle=False)-mean)/std).astype(float) for cid in {j['panel_case_id'] for j in plan['jobs']}}
    cells=0;maxerror=0.;pairs={};table={(c['sensor_config'],c['timestep']):c for c in contrasts}
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
        key=(row['sensor_config'],row['timestep']);prior=pairs.pop(key,None)
        if prior is None:pairs[key]=(row,f['x_t'])
        else:
            other,xt=prior;off,keep=(row,other) if row['condition']=='switch_imu_off' else (other,row);delta=f['x_t']-xt;contrast=table[key]
            for view,ids in [('uniform',range(263)),('root_ric67',range(67)),('remaining196',range(67,263))]:check(contrast['state_difference_rms'][view],np.sqrt(cov(delta[:,ids],delta[:,ids])))
            for view,fields in contrast['actual_prediction'].items():
                for k,value in fields.items():
                    a=off['temporal']['actual_combined_prediction'][view][k];b=keep['temporal']['actual_combined_prediction'][view][k];check(value,a-b,a,b)
        if (len(pairs)==0 or int(row['timestep'])==0):print('Verified',row['row_id'],flush=True)
    if pairs:raise ValueError('Missing paired path state')
    proof=dict(status='passed',rows=3600,path_contrasts=1800,independent_scalar_checks=cells,max_scalar_absolute_error=maxerror,
        new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,independent_review=False,
        bound='Inherited float64 64eps times context scale; scientific software tolerances unchanged',plan=artifact(plan_path),verifier=artifact(Path(__file__).resolve()))
    output.parent.mkdir(parents=True,exist_ok=True);write(output,proof);print(proof)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('plan','results','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.plan.resolve(),a.results.resolve(),a.output.resolve())
