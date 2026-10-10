#!/usr/bin/env python
"""Independent Gram/dot-product checks of both saved-prediction orderings."""
import argparse
from pathlib import Path

import numpy as np

from evaluate_b0_native_training_sufficiency import read,artifact,verify,arrays,matrix,write,flatten
from itm.metrics.training_sufficiency import VIEWS


def run(plan_path,results,output):
    if output.exists():raise FileExistsError('Fresh CPU verification required')
    plan=read(plan_path);run_dir=Path(plan['run_directory'])
    for a in plan['sources']+read(results/'artifact_index.json'):verify(a)
    status=read(run_dir/'runtime/status.json')
    if status['status']!='completed' or status['implementation']!=plan['implementation']:raise ValueError('Complete planned queue required')
    receipt=read(verify(status['receipt']));sources={};counts=0;maxerror=0.
    for a in receipt['artifacts']:verify(a)
    probes={r['id']:r for r in status['probes']};targets={};rows=read(results/'rows.json')
    def compare(a,b,*scale):
        nonlocal counts,maxerror
        error=abs(a-b);maxerror=max(maxerror,error);counts+=1
        if error>64*np.finfo(float).eps*max(1.,abs(a),abs(b),*(abs(x) for x in scale)):raise ValueError('Independent component formula mismatch')
    for job,row in zip(plan['jobs'],rows):
        if row['row_id']!=job['row_id'] or row['status']!='valid':raise ValueError('Missing/misidentified decomposition row')
        values=[]
        for ref in [job['prediction'],job['intermediate'],job['reference']]:
            key=ref['probe_id']
            if key not in sources:sources[key]=arrays(verify(probes[key]['raw_artifact']))
            values.append(matrix(sources[key][ref['branch']]).astype(float))
        p,i,r=values;first,second=p-i,i-r;d,s=(first,second) if job['direct_is_first'] else (second,first)
        cid=job['panel_case_id']
        if cid not in targets:targets[cid]=matrix(arrays(run_dir/'prefix/inputs'/(cid+'.npz'))['target']).astype(float)
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
    result=dict(status='passed',rows=len(rows),scalar_checks=counts,max_scalar_absolute_error=maxerror,
        input_probes_verified=len(sources),new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,
        independent_review=False,interpretation='Main-session second formula; ordering-sensitive algebra, not unique causal mechanism.',
        plan=artifact(plan_path),verifier=artifact(Path(__file__).resolve()))
    output.parent.mkdir(parents=True,exist_ok=True);write(output,result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('plan','results','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.plan.resolve(),a.results.resolve(),a.output.resolve())
