#!/usr/bin/env python
"""Independent CPU equations, complete state chain and old-noise provenance."""
import argparse
from pathlib import Path

import numpy as np

from evaluate_b0_native_training_sufficiency import read,verify,artifact,arrays,descriptor,matrix,write,flatten
from itm.models.mdm_suffix_history_factorial import read_receipt
from itm.metrics.training_sufficiency import VIEWS


def run(run_dir,results,output):
    if output.exists():raise FileExistsError('Fresh suffix verification required')
    status=read(run_dir/'runtime/status.json')
    if status['status']!='completed':raise ValueError('Complete queue required for primary verification')
    receipt=read_receipt(verify(status['receipt']));verify(status['receipt_journal'])
    manifest=read(verify(status['implementation']))
    for a in manifest['expected_artifacts']+receipt['artifacts']+read(results/'artifact_index.json'):verify(a)
    files={Path(a['path']).stem:verify(a) for a in read(verify(manifest['package']))['artifacts']}
    paths=read(files['trajectories']);anchors=read(files['historical_anchors']);noise={r['noise_id']:r['draws'] for r in read(files['noise'])}
    table={r['id']:r for r in receipt['steps']};eq=0;links=0;draws=0;scalars=0;maxerror=0.
    def equal(a,b,bitwise=False):
        nonlocal eq
        eq+=1
        if np.asarray(a).shape!=np.asarray(b).shape:raise ValueError('Equation shape')
        if bitwise:
            if not np.array_equal(a,b):raise ValueError('Bitwise mask/shared start equation')
        elif not np.all(abs(np.asarray(a,dtype=float)-b)<=1e-6+1e-6*abs(b)):raise ValueError('Inherited numerical equation')
    starts={}
    for path in paths:
        tid=path['trajectory_id'];cid=path['panel_case_id'];start=arrays(run_dir/'suffix/starts'/(tid+'.npz'))['x_899']
        equal(start,arrays(verify(path['start']))['x_899'],True)
        if cid in starts:equal(start,starts[cid],True)
        else:starts[cid]=start
        metadata=read(run_dir/'suffix/trajectories'/(tid+'.json'))
        if metadata['noise_draws']!=noise[path['noise_id']]:raise ValueError('Old-noise suffix descriptors')
        draws+=len(metadata['noise_draws'])
        for step in path['diffusion_steps']:
            key=tid+'-'+str(step);z=arrays(verify(table[key]['raw_artifact']));meta=read(run_dir/'suffix/steps'/(key+'.json'))
            if meta['arrays']!={k:descriptor(v) for k,v in z.items()}:raise ValueError('Raw step descriptors')
            a,b,c,d=[z[n].astype(float) for n in ('f00','f10','f01','f11')];actual=z['actual_combined_prediction'].astype(float)
            scale=abs(a)+(abs(b)+abs(a))+(abs(c)+abs(a))+(abs(d)+abs(b)+abs(c)+abs(a));u=2.**-24
            bound=16*u/(1-16*u)*np.maximum(1.,scale)
            if not all(np.all(abs(actual-ref)<=bound) for ref in (a+(b-a)+(c-a)+(d-b-c+a),d)):raise ValueError('Inherited float32 CFG')
            eq+=1
            if not path['imu_present']:equal(z['f00'],z['f01']);equal(z['f10'],z['f11'])
            nxt=arrays(run_dir/'suffix/raw_finals'/(tid+'.npz'))['normalized_humanml263'] if step==0 else arrays(run_dir/'suffix/steps'/(tid+'-'+str(step-1)+'.npz'))['x_t']
            if descriptor(nxt)!=meta['next_x']:raise ValueError('Posterior transition chain')
            if step:
                if meta['noise']!=noise[path['noise_id']][899-step]:raise ValueError('Per-step old-noise mapping')
            elif meta['noise'] is not None:raise ValueError('Unexpected t0 noise')
            if step==899 and not np.array_equal(start,z['x_t']):raise ValueError('Start transition changed')
            links+=1
    # The four load-to-source checks above are additional to two frozen shared-pair checks.
    source_load_checks=4
    for a in anchors:
        if a.get('kind')=='final':
            current=arrays(run_dir/'suffix/finals'/(a['trajectory_id']+'.npz'));old=arrays(verify(a['raw']))
            for name in a['arrays']:equal(current[name],old['motion' if name=='joints' else name],name=='frame_mask')
        else:
            current=arrays(run_dir/'suffix/steps'/(a['trajectory_id']+'-'+str(a['timestep'])+'.npz'));old=arrays(verify(a['raw']))
            for name in a['arrays']:equal(current[name],old[name])
    if eq-source_load_checks!=7244 or links!=3600 or draws!=3596:raise ValueError('Independent software denominator')
    root=Path(__file__).resolve().parents[1]
    cases={s['case_id']:s for s in read(root/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/sampling_slots.json')}
    plan=read(root/'docs/itm_rework/protocol/b0_saved_trace_response_20261008_v1/plan.json')
    mean,std=[np.load(verify(plan['normalization'][k]),allow_pickle=False) for k in ('motion_mean','motion_std')]
    def check(a,b,*scale):
        nonlocal scalars,maxerror
        error=abs(a-b);maxerror=max(maxerror,error);scalars+=1
        if error>64*np.finfo(float).eps*max(1.,abs(a),abs(b),*(abs(v) for v in scale)):raise ArithmeticError('Independent temporal formula')
    def temporal(p,g,metrics):
        p=p.astype(float);g=g.astype(float);pc=p-p.mean(0);gc=g-g.mean(0)
        for name,ids in VIEWS.items():
            a,b,ac,bc=[v[:,ids].ravel() for v in (p,g,pc,gc)];m=metrics[name];n=len(a)
            check(m['mse'],(a-b)@(a-b)/n);mu=(p-g)[:,ids].mean(0);check(m['mean_error'],mu@mu/len(ids))
            check(m['temporal_error'],(ac-bc)@(ac-bc)/n);pe,ge=ac@ac/n,bc@bc/n
            check(m['prediction_temporal_energy'],pe);check(m['target_temporal_energy'],ge);check(m['temporal_covariance'],ac@bc/n)
            if ge:check(m['temporal_energy_ratio'],pe/ge)
            elif m['temporal_energy_ratio'] is not None:raise ValueError('Undefined ratio')
            for key,v in [('prediction_difference_rms',p),('target_difference_rms',g)]:
                d=np.diff(v[:,ids],axis=0).ravel();check(m[key],np.sqrt(d@d/len(d)))
        for key,value in metrics['balanced'].items():check(value,.5*metrics['root_ric67'][key]+.5*metrics['remaining196'][key])
    targets={cid:((np.load(Path(cases[cid]['gt_path']).with_name('motion.npy'),allow_pickle=False)-mean)/std).astype(float) for cid in starts}
    by_tid={p['trajectory_id']:p for p in paths};finals=read(results/'finals.json');traces=read(results/'traces.json')
    if len(finals)!=4 or len(traces)!=12 or any(r['status']!='valid' for r in finals+traces):raise ValueError('Complete evaluation denominator')
    for row in finals:
        z=arrays(verify(row['raw']));g=targets[row['panel_case_id']];temporal(matrix(z['normalized_humanml263']),g,row['temporal'])
        for name,ids in VIEWS.items():
            d=(matrix(z['normalized_humanml263']).astype(float)-g)[:,ids].ravel();check(row['feature_views'][name],d@d/len(d))
    for row in traces:
        g=targets[by_tid[row['trajectory_id']]['panel_case_id']];z=arrays(verify(row['raw']))
        for name,metrics in row['temporal'].items():temporal(matrix(z[name]),g,metrics)
    for c in read(results/'contrasts.json'):
        off=next(r for r in finals if r['sensor_config']==c['sensor_config'] and r['condition']=='switch_imu_on')
        keep=next(r for r in finals if r['sensor_config']==c['sensor_config'] and r['condition']=='keep_text_only')
        for name,value in flatten(c['temporal']).items():
            left,right=flatten(off['temporal'])[name],flatten(keep['temporal'])[name]
            if left is None or right is None:
                if value is not None:raise ValueError('Undefined contrast')
            else:check(value,left-right,left,right)
    result=dict(status='passed',software_equations=eq-source_load_checks,additional_source_start_checks=4,
        posterior_transition_hash_links=links,old_noise_descriptors=draws,source_hashes=len(manifest['expected_artifacts']),worker_artifacts=len(receipt['artifacts']),
        independent_metric_scalars=scalars,max_scalar_absolute_error=maxerror,verifier=artifact(Path(__file__).resolve()),
        new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,independent_review=False)
    output.parent.mkdir(parents=True,exist_ok=True);write(output,result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('run','results','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.run.resolve(),a.results.resolve(),a.output.resolve())
