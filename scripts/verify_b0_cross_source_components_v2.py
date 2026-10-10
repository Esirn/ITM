#!/usr/bin/env python
"""Independent raw dot formulas for every frozen cross-source row and F/U contrast."""
import argparse
from pathlib import Path
import numpy as np
from evaluate_b0_native_training_sufficiency import read,verify,artifact,arrays,matrix,flatten,write
from itm.metrics.training_sufficiency import VIEWS

ROOT=Path(__file__).resolve().parents[1]

def run(plan_path,results,output):
    if output.exists():raise FileExistsError('Fresh complete verification required')
    manifest=read(Path(output).parent/'implementation.json')
    for a in manifest['expected_artifacts']:verify(a)
    if artifact(results/'artifact_index.json')!=manifest['inherited_results_index']:raise ValueError('Exact reused CPU results identity')
    plan=read(plan_path)
    for a in plan['sources']+read(results/'sources.json')+read(results/'artifact_index.json'):verify(a)
    rows=read(results/'rows.json');pairs=read(results/'pairs.json')
    if len(rows)!=480 or len(pairs)!=240 or any(r['status']!='valid' for r in rows+pairs):raise ValueError('Complete row/contrast denominator')
    slots={r['case_id']:r for r in read(ROOT/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/sampling_slots.json')}
    mean,std=[np.load(verify(plan['normalization'][k]),allow_pickle=False) for k in ('motion_mean','motion_std')]
    targets={cid:((np.load(Path(slots[cid]['gt_path']).with_name('motion.npy'),allow_pickle=False)-mean)/std).astype(float) for cid in {j['pair_case_id'] for j in plan['jobs']}}
    checks=0;maximum=0.;nulls=0
    def check(a,b,*scale):
        nonlocal checks,maximum,nulls
        checks+=1
        if a is None or b is None:
            if a is not None or b is not None:raise ValueError('Undefined scalar not preserved')
            nulls+=1;return
        e=abs(a-b);maximum=max(maximum,e)
        if e>64*np.finfo(float).eps*max(1.,abs(a),abs(b),*(abs(s) for s in scale)):raise ArithmeticError('Independent component/error mismatch: '+str((a,b)))
    def dot(a,b):return float(np.einsum('ij,ij->',a,b)/a.size)
    def stats(p,g,ids):
        a,b=p[:,ids],g[:,ids];pc=a-a.mean(0);gc=b-b.mean(0);e=a-b;mu=e.mean(0)
        return dict(mse=dot(e,e),mean_error=float(mu@mu/len(ids)),temporal_error=dot(pc-gc,pc-gc),prediction_temporal_energy=dot(pc,pc),target_temporal_energy=dot(gc,gc),temporal_covariance=dot(pc,gc),target_mean_baseline_mse=dot(gc,gc))
    def temporal(p,g,m):
        expected={}
        for view,ids in VIEWS.items():
            s=stats(p,g,ids);expected[view]=s
            for k,v in s.items():check(m[view][k],v,*s.values())
            ge=s['target_temporal_energy'];check(m[view]['temporal_energy_ratio'],s['prediction_temporal_energy']/ge if ge else None)
            for name,v in [('prediction_difference_rms',p[:,ids]),('target_difference_rms',g[:,ids])]:d=np.diff(v,axis=0);check(m[view][name],np.sqrt(dot(d,d)))
            if m[view]['dimensions']!=len(ids) or m[view]['adjacent_valid_pairs']!=len(p)-1:raise ValueError('Temporal frame/view denominator')
        for k,v in m['balanced'].items():check(v,.5*expected['root_ric67'][k]+.5*expected['remaining196'][k])
        return expected
    table={}
    for job,row in zip(plan['jobs'],rows):
        if any(job[k]!=row[k] for k in job) or row['row_id']!=job['arm']+':'+job['trace_id'] or row['source_n']!=1:raise ValueError('Exact frozen row identity')
        key=(row['pair_case_id'],row['timestep'],row['arm'])
        if key in table:raise ValueError('Duplicate row')
        table[key]=row;z=arrays(verify(job['raw']));f={k:matrix(z[k]).astype(float) for k in ('f00','f10','f01','f11')};g=targets[job['pair_case_id']]
        a=f['f10'];u=f['f01']-f['f00'];j=f['f11']-f['f10']-f['f01']+f['f00'];expected={}
        for label,p in [('main_only',a+u),('joint_only',a+j)]:
            h=row['hypotheses'][label];temporal(p,g,h['temporal']);e=a-g;d=p-a;de={}
            for view,ids in VIEWS.items():
                b,c=e[:,ids],d[:,ids];old,new=dot(b,b),dot(b+c,b+c);cross,energy=2*dot(b,c),dot(c,c)
                m=dict(baseline_mse=old,prediction_mse=new,delta_mse=new-old,cross=cross,residual_energy=energy,prediction_difference_rms=np.sqrt(energy),prediction_difference_max_abs=float(np.max(abs(c))));de[view]=m
                for k,v in m.items():check(h['error_delta'][view][k],v,old,new,cross,energy)
                if h['error_delta'][view]['dimensions']!=len(ids) or h['error_delta'][view]['count']!=len(a)*len(ids):raise ValueError('Error/view denominator')
            for k,v in h['error_delta']['balanced'].items():check(v,.5*de['root_ric67'][k]+.5*de['remaining196'][k])
        for view,ids in VIEWS.items():
            b,p,m,n=[stats(x,g,ids) for x in (a,f['f11'],a+u,a+j)]
            uu,jj=u[:,ids],j[:,ids];uc,jc=uu-uu.mean(0),jj-jj.mean(0)
            expected[view]=dict(baseline_energy=b['prediction_temporal_energy'],target_energy=b['target_temporal_energy'],main_delta_energy=m['prediction_temporal_energy']-b['prediction_temporal_energy'],joint_delta_energy=n['prediction_temporal_energy']-b['prediction_temporal_energy'],combined_delta_energy=p['prediction_temporal_energy']-b['prediction_temporal_energy'],twice_centered_main_joint_covariance=2*dot(uc,jc),baseline_mse=b['mse'],main_delta_mse=m['mse']-b['mse'],joint_delta_mse=n['mse']-b['mse'],combined_delta_mse=p['mse']-b['mse'],twice_raw_main_joint_inner_product=2*dot(uu,jj),main_delta_temporal_error=m['temporal_error']-b['temporal_error'],joint_delta_temporal_error=n['temporal_error']-b['temporal_error'],main_delta_mean_error=m['mean_error']-b['mean_error'],joint_delta_mean_error=n['mean_error']-b['mean_error'])
        expected['balanced']={k:.5*expected['root_ric67'][k]+.5*expected['remaining196'][k] for k in expected['root_ric67']}
        for view,fields in expected.items():
            scale=list(fields.values())
            for k,v in fields.items():check(row['component_terms'][view][k],v,*scale)
            for k in ('main_delta_energy','joint_delta_energy','combined_delta_energy','twice_centered_main_joint_covariance'):
                ge=fields['target_energy'];ratio_scale=[abs(v)/ge for v in scale] if ge else []
                check(row['component_terms'][view][k+'_over_GT'],fields[k]/ge if ge else None,*ratio_scale)
        if len(table)%120==0:print('Verified raw rows',len(table),flush=True)
    if len(table)!=480:raise ValueError('Raw row count')
    seen=set()
    for pair in pairs:
        key=pair['pair_case_id'],pair['timestep']
        if key in seen:raise ValueError('Duplicate F/U contrast')
        seen.add(key);f,u=[table[(*key,arm)] for arm in ('F','U')]
        if pair['F_row']!=f['row_id'] or pair['U_row']!=u['row_id'] or pair['source_n']!=1:raise ValueError('Pair sign/role identity')
        for label in ('component_terms','hypotheses'):
            left,right,value=[flatten(r[label]) for r in (f,u,pair)]
            if set(left)!=set(right) or set(left)!=set(value):raise ValueError('Contrast metric denominator')
            for k,v in value.items():
                a,b=left[k],right[k];check(v,None if a is None or b is None else a-b,*([a,b] if a is not None and b is not None else []))
    if seen!={(r['pair_case_id'],r['timestep']) for r in rows}:raise ValueError('Complete F/U pairing')
    # Check equal-source aggregation without dropping undefined ratios or invalid denominators.
    for group in read(results/'aggregates.json'):
        rs=[r for r in rows if all(r[k]==group[k] for k in ('arm','data_split','sensor_config','condition','timestep'))]
        if group['planned_sources']!=len(rs) or group['valid_sources']!=len(rs) or sorted(group['sources'])!=sorted(r['motion_id'] for r in rs):raise ValueError('Equal-source population')
        for view,fields in group['metrics'].items():
            for k,m in fields.items():
                values=[r['component_terms'][view][k] for r in rs if r['component_terms'][view][k] is not None]
                if m['planned_n']!=len(rs) or m['valid_n']!=len(values):raise ValueError('Undefined/source denominator')
                for name,v in [('mean',sum(values)/len(values) if values else None),('min',min(values) if values else None),('max',max(values) if values else None)]:check(m[name],v,*values)
    proof=dict(status='passed',rows=480,F_U_pairs=240,independent_scalar_checks=checks,undefined_scalar_checks=nulls,max_absolute_error=maximum,new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,independent_review=False,bound='Immutable float64 64eps times context scale; no scientific tolerance change',implementation=artifact(Path(output).parent/'implementation.json'),plan=artifact(plan_path),results_index=artifact(results/'artifact_index.json'),verifier=artifact(Path(__file__).resolve()))
    write(output,proof);print(proof,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('plan','results','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.plan.resolve(),a.results.resolve(),a.output.resolve())
