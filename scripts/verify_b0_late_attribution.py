#!/usr/bin/env python
"""Independent late-state Gram formulas and raw old-trajectory temporal checks."""
import argparse
from pathlib import Path

import numpy as np

from evaluate_b0_native_training_sufficiency import read,artifact,verify,arrays,matrix,write,flatten
from itm.metrics.training_sufficiency import VIEWS


def run(plan_path,results,output):
    if output.exists():raise FileExistsError('Fresh CPU verification required')
    plan=read(plan_path)
    for a in plan['sources']+read(results/'artifact_index.json')+read(results/'sources.json'):verify(a)
    sources={};targets={};counts=0;maxerror=0.;rows=read(results/'rows.json')
    if len(rows)!=plan['planned_rows'] or len(rows)!=96:raise ValueError('Changed row denominator')
    def compare(a,b,*scale):
        nonlocal counts,maxerror
        error=abs(a-b);maxerror=max(maxerror,error);counts+=1
        if error>64*np.finfo(float).eps*max(1.,abs(a),abs(b),*(abs(x) for x in scale)):raise ValueError('Independent component formula mismatch')
    for job,row in zip(plan['jobs'],rows):
        if row['row_id']!=job['row_id'] or row['status']!='valid':raise ValueError('Missing/misidentified decomposition row')
        values=[]
        for ref in [job['prediction'],job['intermediate'],job['reference']]:
            key=ref['job_id']
            if key not in sources:sources[key]=arrays(verify(ref['raw']))
            values.append(matrix(sources[key][ref['branch']]).astype(float))
        p,i,r=values;first,second=p-i,i-r;d,s=(first,second) if job['direct_is_first'] else (second,first)
        cid=job['target']['path']
        if cid not in targets:targets[cid]=matrix(arrays(verify(job['target']))[job['target_array']]).astype(float)
        g=targets[cid];rc=r-r.mean(0);pc=p-p.mean(0);dc=d-d.mean(0);sc=s-s.mean(0)
        for name,ids in VIEWS.items():
            n=len(ids)*len(p);dd=d[:,ids].ravel();ss=s[:,ids].ravel();error=(r-g)[:,ids].ravel()
            gram=np.stack([dd,ss])@np.stack([dd,ss]).T/n;m=row['metrics'][name]
            old=float(error@error/n);updated=((p-g)[:,ids].ravel());new=float(updated@updated/n)
            compare(m['reference_mse'],old);compare(m['prediction_mse'],new)
            compare(m['direct_energy'],gram[0,0]);compare(m['state_energy'],gram[1,1]);compare(m['twice_direct_state_covariance'],2*gram[0,1])
            compare(m['direct_error_cross'],2*error@dd/n);compare(m['state_error_cross'],2*error@ss/n)
            compare(m['delta_mse'],new-old,old,new,2*error@dd/n,2*error@ss/n,gram.sum())
            total=(p-r)[:,ids].ravel();compare(m['total_difference_energy'],total@total/n)
            compare(m['direct_rms'],np.sqrt(gram[0,0]));compare(m['state_rms'],np.sqrt(gram[1,1]))
            dd,ss=dc[:,ids].ravel(),sc[:,ids].ravel();base=rc[:,ids].ravel();pred=pc[:,ids].ravel()
            compare(m['direct_centered_energy'],dd@dd/n);compare(m['state_centered_energy'],ss@ss/n)
            compare(m['twice_centered_covariance'],2*dd@ss/n)
            compare(m['reference_temporal_energy'],base@base/n);compare(m['prediction_temporal_energy'],pred@pred/n)
            compare(m['delta_temporal_energy'],pred@pred/n-base@base/n,pred@pred/n,base@base/n)
            compare(m['twice_reference_direct_temporal_covariance'],2*base@dd/n)
            compare(m['twice_reference_state_temporal_covariance'],2*base@ss/n)
    keys=['data_split','sensor_config','timestep','comparison','ordering'];grouped={}
    for row in rows:grouped.setdefault(tuple(row[k] for k in keys),[]).append(row)
    for item in read(results/'aggregates.json'):
        group=tuple(item[k] for k in keys);records=sorted(grouped[group],key=lambda r:r['motion_id'])
        for name,value in item['metrics'].items():
            v=[flatten(r['metrics'])[name] for r in records]
            compare(value['mean'],float(np.mean(v)));compare(value['min'],min(v));compare(value['max'],max(v))
            if value['valid_n']!=len(v) or value['planned_n']!=len(v):raise ValueError('Grouped source denominator changed')
    # Recheck the balanced summary and both alternate telescoping paths.
    for row in rows:
        for key,value in row['metrics']['balanced'].items():
            compare(value,.5*row['metrics']['root_ric67'][key]+.5*row['metrics']['remaining196'][key])
    pairs={}
    for row in rows:
        pairs.setdefault(tuple(row[k] for k in ('data_split','motion_id','sensor_config','timestep')),[]).append(row)
    for pair in pairs.values():
        if len(pair)!=2 or len({r['direct_is_first'] for r in pair})!=2:raise ValueError('Both orderings required')
        for name in VIEWS:
            for key in ('reference_mse','prediction_mse','delta_mse','total_difference_energy',
                        'reference_temporal_energy','prediction_temporal_energy','delta_temporal_energy'):
                compare(pair[0]['metrics'][name][key],pair[1]['metrics'][name][key])
    root=Path(__file__).resolve().parents[1]
    oldplan=read(root/'docs/itm_rework/protocol/b0_saved_trace_response_20261008_v1/plan.json')
    for a in oldplan['sources']:verify(a)
    mean,std=[np.load(verify(oldplan['normalization'][k]),allow_pickle=False) for k in ('motion_mean','motion_std')]
    cases={c['case_id']:c for c in read(root/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/sampling_slots.json')}
    oldtargets={cid:((np.load(Path(c['gt_path']).with_name('motion.npy'),allow_pickle=False)-mean)/std).astype(float) for cid,c in cases.items()}
    def temporal(p,g,metrics):
        pc=p-p.mean(0);gc=g-g.mean(0)
        for name,ids in VIEWS.items():
            a,b,ac,bc=[v[:,ids].ravel() for v in (p,g,pc,gc)];n=len(a);m=metrics[name]
            compare(m['mse'],(a-b)@(a-b)/n)
            mu=(p-g)[:,ids].mean(0);compare(m['mean_error'],mu@mu/len(ids))
            compare(m['temporal_error'],(ac-bc)@(ac-bc)/n)
            pe,ge=ac@ac/n,bc@bc/n
            compare(m['prediction_temporal_energy'],pe);compare(m['target_temporal_energy'],ge)
            compare(m['temporal_covariance'],ac@bc/n);compare(m['target_mean_baseline_mse'],ge)
            if ge:compare(m['temporal_energy_ratio'],pe/ge)
            elif m['temporal_energy_ratio'] is not None:raise ValueError('Zero-energy target ratio must stay undefined')
            for key,v in [('prediction_difference_rms',p),('target_difference_rms',g)]:
                delta=np.diff(v[:,ids],axis=0).ravel();compare(m[key],np.sqrt(delta@delta/len(delta)))
            if m['dimensions']!=len(ids) or m['adjacent_valid_pairs']!=len(p)-1:raise ValueError('Temporal denominator changed')
        for key,value in metrics['balanced'].items():compare(value,.5*metrics['root_ric67'][key]+.5*metrics['remaining196'][key])
    olddir=root/'outputs/itm_rework/protocol/saved_trace_response_evaluation_20261008_v1'
    trace_source={(r['arm'],r['trace_id']):r for r in read(olddir/'traces.json')}
    final_source={r['case_id']:r for r in read(olddir/'samples.json')}
    refs=read(results/'reference_traces.json');finals=read(results/'reference_finals.json')
    if len(refs)!=480 or len(finals)!=120:raise ValueError('Lost reference/negative denominator')
    for row in refs:
        original=trace_source[(row['arm'],row['row_id'].split('-',1)[1])]
        if row['temporal']!=original['metrics']['temporal']['actual_combined_prediction']:raise ValueError('Changed descriptive reference')
        for key in ('arm','data_split','motion_id','sensor_config','condition','timestep'):
            if row[key]!=original[key]:raise ValueError('Changed reference identity')
        temporal(matrix(arrays(original['path'])['actual_combined_prediction']).astype(float),oldtargets[original['pair_case_id']],row['temporal'])
    for row in finals:
        original=final_source[row['row_id']]
        if row['temporal']!=original['metrics']:raise ValueError('Changed original final')
        for key in ('arm','data_split','motion_id','sensor_config','condition'):
            if row[key]!=original[key]:raise ValueError('Changed final identity')
        temporal(matrix(arrays(original['path'])['normalized_humanml263']).astype(float),oldtargets[original['pair_case_id']],row['temporal'])
    refkeys=('arm','data_split','sensor_config','condition','timestep');refgroups={}
    for row in refs:refgroups.setdefault(tuple(row[k] for k in refkeys),[]).append(row)
    for item in read(results/'reference_aggregates.json'):
        population=sorted(refgroups[tuple(item[k] for k in refkeys)],key=lambda r:r['motion_id'])
        for name,value in item['metrics'].items():
            raw=[flatten(r['temporal'])[name] for r in population];values=[v for v in raw if v is not None]
            if values:
                compare(value['mean'],float(np.mean(values)));compare(value['min'],min(values));compare(value['max'],max(values))
            elif any(value[k] is not None for k in ('mean','min','max')):raise ValueError('Undefined ratio became numeric')
            if value['valid_n']!=len(values) or value['planned_n']!=len(population):raise ValueError('Reference source denominator changed')
    result=dict(status='passed',rows=len(rows),scalar_checks=counts,max_scalar_absolute_error=maxerror,
        input_jobs_verified=len(sources),trace_references_verified=len(refs),final_references_verified=len(finals),original_source_hashes=len(oldplan['sources']),new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,
        independent_review=False,interpretation='Main-session second formula; ordering-sensitive algebra, not unique causal mechanism.',
        plan=artifact(plan_path),verifier=artifact(Path(__file__).resolve()))
    output.parent.mkdir(parents=True,exist_ok=True);write(output,result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('plan','results','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.plan.resolve(),a.results.resolve(),a.output.resolve())
