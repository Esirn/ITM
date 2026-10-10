#!/usr/bin/env python
"""CPU hash/trajectory-chain replay and independent formula checks; no noise/model."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from evaluate_b0_native_training_sufficiency import read,artifact,verify,arrays,descriptor,matrix,write
from itm.metrics.training_sufficiency import VIEWS
from itm.models.mdm_prefix_mechanism import read_receipt


def run(run_dir,results,output,diagnostic=False):
    if output.exists():raise FileExistsError('Fresh verification required')
    status=read(run_dir/'runtime/status.json');receipt=read_receipt(verify(status['receipt']))
    if status.get('receipt_journal'):verify(status['receipt_journal'])
    complete=status['status']=='completed' and receipt['status']=='completed'
    if not complete and not diagnostic:raise ValueError('Complete real queue required unless explicitly diagnostic')
    manifest=read(verify(status['implementation']))
    for e in manifest['expected_artifacts']+receipt['artifacts']+read(results/'artifact_index.json'):verify(e)
    package=read(verify(manifest['package']));files={Path(a['path']).stem:verify(a) for a in package['artifacts']}
    trajectories=read(files['trajectories']);states=read(files['states']);probes=read(files['probes']);anchors=read(files['historical_anchors'])
    noise={n['noise_id']:n['draws'] for n in read(files['noise'])}
    table={k:{r['id']:r for r in receipt[k]} for k in ('steps','states','probes','checks')}
    eq=0;scalars=0;maxerr=0.;noise_descriptors=0;chains=0
    def equality(a,b,bitwise=False):
        nonlocal eq
        a,b=(np.asarray(v,dtype=np.float64) for v in (a,b));eq+=1
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Equation shape/finite')
        if bitwise:
            if not np.array_equal(a,b):raise ValueError('Bitwise shared initial noise mismatch')
        elif not np.all(abs(a-b)<=1e-6+1e-6*abs(b)):raise ValueError('Inherited software equation failed')
    def compare(a,b,*scale):
        nonlocal scalars,maxerr
        scalars+=1;error=abs(a-b);maxerr=max(maxerr,error)
        if error>64*np.finfo(float).eps*max(1.,abs(a),abs(b),*(abs(v) for v in scale)):raise ArithmeticError('Independent metric formula mismatch')
    initial={}
    for t in trajectories:
        tid=t['trajectory_id'];cid=t['panel_case_id'];x=arrays(run_dir/'prefix/initial'/(tid+'.npz'))['x_t']
        initial.setdefault(cid,{})[t['path_arm']]=x
        meta=read(run_dir/'prefix/trajectories'/(tid+'.json'))
        if meta['noise_draws']!=noise[t['noise_id']]:raise ValueError('Old noise descriptor changed')
        noise_descriptors+=len(meta['noise_draws'])
        for step in t['diffusion_steps']:
            key=tid+'-'+str(step);z=arrays(verify(table['steps'][key]['raw_artifact']))
            metadata=read(run_dir/'prefix/steps'/(key+'.json'))
            if metadata['arrays']!={k:descriptor(v) for k,v in z.items()}:raise ValueError('Step raw descriptors changed')
            a,b,c,d=[z[k].astype(float) for k in ('f00','f10','f01','f11')];actual=z['actual_combined_prediction'].astype(float)
            scale=abs(a)+(abs(b)+abs(a))+(abs(c)+abs(a))+(abs(d)+abs(b)+abs(c)+abs(a))
            u=2.**-24;bound=16*u/(1-16*u)*np.maximum(1.,scale)
            if not all(np.all(abs(actual-ref)<=bound) for ref in (a+(b-a)+(c-a)+(d-b-c+a),d)):
                raise ValueError('Frozen float32 CFG bound')
            eq+=1
            if t['path_arm']=='T':equality(z['f00'],z['f01']);equality(z['f10'],z['f11'])
            nxt=arrays(run_dir/'prefix/terminal'/(tid+'.npz'))['x_899'] if step==900 else arrays(run_dir/'prefix/steps'/(tid+'-'+str(step-1)+'.npz'))['x_t']
            if descriptor(nxt)!=metadata['next_x']:raise ValueError('Posterior transition chain changed')
            chains+=1
    for cid,paths in initial.items():
        if set(paths)!=set(('F','U','T')):raise ValueError('Incomplete initial states')
        equality(np.stack([paths['F'],paths['U']]),np.stack([paths['T'],paths['T']]),bitwise=True)
    for a in anchors:
        current=arrays(run_dir/'prefix/steps'/(a['trajectory_id']+'-900.npz'));reference=arrays(verify(a['npz']))
        for b in a['arrays']:equality(current[b],reference[b])
    tmap={t['trajectory_id']:t for t in trajectories};smap={s['state_id']:s for s in states}
    pmap={(p['state_id'],p['checkpoint']):p['probe_id'] for p in probes}
    targets={}
    for t in trajectories:
        cid=t['panel_case_id']
        if cid not in targets:targets[cid]=matrix(arrays(run_dir/'prefix/inputs'/(cid+'.npz'))['target']).astype(float)
    pred={p['probe_id']:arrays(verify(table['probes'][p['probe_id']]['raw_artifact'])) for p in probes if table['probes'][p['probe_id']].get('raw_artifact')}
    saved={s['state_id']:arrays(verify(table['states'][s['state_id']]['artifact'])) for s in states}
    for p in probes:
        s=smap[p['state_id']];t=tmap[s['trajectory_id']]
        if p['probe_id'] not in pred:continue
        if t['path_arm']=='T' or p['checkpoint']==t['checkpoint']:
            for b in (('f00','f10') if t['path_arm']=='T' else ('f00','f10','f01','f11')):
                if table['checks']['OWN-'+p['probe_id']+'-'+b]['status']=='completed':equality(pred[p['probe_id']][b],saved[s['state_id']][b])
    for s in states:
        for b in ('f00','f10'):
            if table['checks']['CROSS-'+s['state_id']+'-'+b]['status']=='completed':equality(pred[pmap[(s['state_id'],'U200')]][b],pred[pmap[(s['state_id'],'F200')]][b])
    expected_eq=sum(r['status']=='completed' for r in receipt['checks'])
    if eq!=expected_eq or chains!=3600 or noise_descriptors!=3636:raise ValueError('Independent equation/population denominator')
    def cov(a,b):return float(np.einsum('ij,ij->',a,b)/a.size)
    def temporal(p,g,metrics):
        p=p.astype(float);g=g.astype(float);pc=p-p.mean(0);gc=g-g.mean(0)
        for name,ids in VIEWS.items():
            a=p[:,ids];b=g[:,ids];ac=pc[:,ids];bc=gc[:,ids];m=metrics[name]
            compare(m['mse'],cov(a-b,a-b));compare(m['mean_error'],float((a-b).mean(0)@(a-b).mean(0)/len(ids)))
            compare(m['temporal_error'],cov(ac-bc,ac-bc));compare(m['prediction_temporal_energy'],cov(ac,ac))
            compare(m['target_temporal_energy'],cov(bc,bc));compare(m['temporal_covariance'],cov(ac,bc))
    def response(z,metrics):
        f={b:matrix(z[b]).astype(float) for b in ('f00','f10','f01','f11')}
        components=dict(base=f['f00'],text=f['f10']-f['f00'],imu=f['f01']-f['f00'],interaction=f['f11']-f['f10']-f['f01']+f['f00'])
        c={k:v-v.mean(0) for k,v in components.items()}
        centered={k:v-v.mean(0) for k,v in f.items()}
        for name,ids in VIEWS.items():
            m=metrics[name]
            for k,v in c.items():compare(m['component_energy'][k],cov(v[:,ids],v[:,ids]))
            for a,b in combinations(c,2):compare(m['twice_component_covariance'][a+'__'+b],2*cov(c[a][:,ids],c[b][:,ids]))
            for label,a,b in [('text_at_no_imu','f00','f10'),('imu_at_no_text','f00','f01'),('imu_at_text','f10','f11'),('text_at_imu','f01','f11'),('joint_at_base','f00','f11')]:
                old=centered[a][:,ids];new=centered[b][:,ids];delta=new-old;m2=m['conditional_changes'][label]
                compare(m2['delta_centered_energy'],cov(delta,delta));compare(m2['twice_base_delta_covariance'],2*cov(old,delta))
                compare(m2['delta_temporal_energy'],cov(new,new)-cov(old,old),cov(new,new),cov(old,old),cov(delta,delta),2*cov(old,delta))
    for r in read(results/'prefix_traces.json'):
        if r['status']!='valid' and not (diagnostic and r['status']=='diagnostic_only'):raise ValueError('Incomplete primary traces')
        z=saved[r['state_id']];g=targets[r['panel_case_id']]
        for k,m in r['metrics']['temporal'].items():temporal(matrix(z[k]),g,m)
        response(z,r['metrics']['response'])
    for r in read(results/'same_state_branches.json'):
        if r['status']=='missing' and diagnostic:continue
        if r['status']!='valid' and not (diagnostic and r['status']=='diagnostic_only'):raise ValueError('Incomplete primary branches')
        temporal(matrix(pred[r['probe_id']][r['branch']]),targets[r['panel_case_id']],r['temporal'])
    for r in read(results/'probes.json'):
        if r['probe_id'] in pred:response(pred[r['probe_id']],r['response'])
    for r in read(results/'weight_contrasts.json'):
        if r['status']=='not_comparable' and diagnostic:continue
        s=r['state_id'];b=r['branch'];g=targets[r['panel_case_id']]
        old=matrix(pred[pmap[(s,'U200')]][b]).astype(float);new=matrix(pred[pmap[(s,'F200')]][b]).astype(float)
        for name,ids in VIEWS.items():
            e=old[:,ids]-g[:,ids];delta=new[:,ids]-old[:,ids];m=r['delta'][name]
            baseline=cov(e,e);updated=cov(e+delta,e+delta)
            compare(m['baseline_mse'],baseline);compare(m['prediction_mse'],updated)
            compare(m['cross'],2*cov(e,delta));compare(m['residual_energy'],cov(delta,delta))
            compare(m['delta_mse'],updated-baseline,updated,baseline,2*cov(e,delta),cov(delta,delta))
    for r in read(results/'terminals.json'):
        z=arrays(run_dir/'prefix/terminal'/(r['row_id']+'.npz'))
        temporal(matrix(z['x_899']),targets[r['panel_case_id']],r['temporal'])
    trace_map={r['state_id']:r for r in read(results/'prefix_traces.json')}
    state_map={(tmap[s['trajectory_id']]['panel_case_id'],tmap[s['trajectory_id']]['path_arm'],s['timestep']):s['state_id'] for s in states}
    for r in read(results/'path_contrasts.json'):
        a,b=r['comparison'].split('-');cid=r['panel_case_id'];t=r['timestep']
        left,right=[state_map[(cid,p,t)] for p in (a,b)]
        for name,ids in [('uniform',list(range(263))),('root_ric67',list(range(67))),('remaining196',list(range(67,263)))]:
            delta=matrix(saved[left]['x_t']).astype(float)[:,ids]-matrix(saved[right]['x_t']).astype(float)[:,ids]
            compare(r['metrics'][name]['x_t_difference_rms'],np.sqrt(cov(delta,delta)))
            le=trace_map[left]['metrics']['temporal']['f11'][name]['prediction_temporal_energy'];re=trace_map[right]['metrics']['temporal']['f11'][name]['prediction_temporal_energy']
            compare(r['metrics'][name]['energy_delta'],le-re,le,re)
    result=dict(status='passed',scope='diagnostic_only; incomplete queue, no primary certification' if not complete else 'complete queue self-check',source_hashes=len(manifest['expected_artifacts']),worker_artifact_hashes=len(receipt['artifacts']),
        software_equations=eq,posterior_transition_hash_links=chains,old_noise_descriptors=noise_descriptors,
        independent_metric_scalars=scalars,max_scalar_absolute_error=maxerr,new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,
        independent_review=False,verifier=artifact(Path(__file__).resolve()),implementation=status['implementation'])
    output.parent.mkdir(parents=True,exist_ok=True);write(output,result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('run','results','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--diagnostic',action='store_true')
    a=p.parse_args();run(a.run.resolve(),a.results.resolve(),a.output.resolve(),a.diagnostic)
