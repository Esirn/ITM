#!/usr/bin/env python
"""CPU evaluation of every prescribed prefix state and fixed-state weight probe."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from evaluate_b0_native_training_sufficiency import read,artifact,verify,arrays,matrix,write,aggregate
from itm.metrics.trace_response import response_decomposition
from itm.metrics.training_sufficiency import temporal_decomposition,same_state_delta

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'docs/itm_rework/protocol/b0_prefix_mechanism_20261008_v1/package.json'
PACKAGE_SHA='8b32b863da7a530cc71350384663678adfd072f8db53bbb01add176b3040853c'
BRANCHES=('f00','f10','f01','f11')


def evaluate(run,output,package=PACKAGE,*,fixture=False):
    run=Path(run).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh prefix evaluation required')
    if not fixture and artifact(package)['sha256']!=PACKAGE_SHA:raise ValueError('Exact prefix package required')
    sources={}
    def bind(p):a=artifact(p);sources[a['path']]=a;return a
    bind(package);files={Path(a['path']).stem:verify(a) for a in read(package)['artifacts']}
    for p in files.values():bind(p)
    trajectories=read(files['trajectories']);states=read(files['states']);probes=read(files['probes'])
    tmap={r['trajectory_id']:r for r in trajectories};smap={r['state_id']:r for r in states}
    pmap={r['probe_id']:r for r in probes}
    status=read(run/'runtime/status.json');bind(run/'runtime/status.json')
    if not fixture:bind(verify(status['implementation']))
    receipt=read(run/'runtime/prefix.receipt.json') if (run/'runtime/prefix.receipt.json').exists() else None
    if receipt:
        bind(run/'runtime/prefix.receipt.json')
        for e in receipt['artifacts']:bind(verify(e))
    if status.get('receipt_journal'):bind(verify(status['receipt_journal']))
    complete=status['status']=='completed' and receipt is not None and receipt['status']=='completed'
    if complete:
        if (receipt['denoising_calls']!=3600 or receipt['branch_examples']!=15840 or receipt['updates_attempted'] or receipt['updates_completed']):raise ValueError('Wrong prefix work')
        for kind,cap,calls in [('mdm',15840,3960),('encoder',48,48),('caption',12,12),('noise',3636,3636)]:
            c=receipt['counts'][kind]
            if any(c[f]!=cap for f in ('reserved','actual','completed')) or any(c[f]!=calls for f in ('reserved_calls','actual_calls','completed_calls')):raise ValueError('Wrong prefix counts')
        for k in ('trajectories','steps','states','probes','branches','checks'):
            if any(r['status']!='completed' for r in receipt[k]):raise ValueError('Incomplete prefix ledger')
    source_rows={k:{r['id']:r for r in status[k]} for k in ('states','probes','trajectories')}
    def load(table,key,path):
        row=source_rows[table][key]
        if not path.exists():return None,'missing'
        entry=row.get('raw_artifact',row.get('artifact'))
        if entry is None:raise ValueError('Unbound prefix raw array: '+key)
        verify(entry);bind(path)
        z=arrays(path)
        state=smap[key] if table=='states' else smap[pmap[key]['state_id']]
        frames=tmap[state['trajectory_id']]['frames']
        if set(z)!=set((*BRANCHES,'x_t','actual_combined_prediction','frame_mask')):raise ValueError('Native raw population')
        if any(v.dtype!=np.float32 or v.shape!=(1,263,1,frames) or not np.isfinite(v).all() for k,v in z.items() if k!='frame_mask'):
            raise ValueError('Native finite float32 raw shape')
        if z['frame_mask'].dtype!=np.bool_ or z['frame_mask'].shape!=(1,frames) or not z['frame_mask'].all():raise ValueError('Original complete frames')
        return z,'valid' if complete and row['status']=='completed' and not row.get('diagnostic_only') else 'diagnostic_only'
    targets={}
    def target(cid):
        if cid not in targets:
            path=run/'prefix/inputs'/(cid+'.npz');bind(path);targets[cid]=matrix(arrays(path)['target'])
        return targets[cid]
    def identity(t):return {k:t[k] for k in ('path_arm','data_split','motion_id','sensor_config','panel_case_id')}
    traces=[];saved_values={}
    for s in states:
        t=tmap[s['trajectory_id']];z,valid=load('states',s['state_id'],run/'prefix/states'/(s['state_id']+'.npz'))
        r=dict(row_id=s['state_id'],state_id=s['state_id'],trajectory_id=s['trajectory_id'],timestep=s['timestep'],**identity(t),status=valid)
        if z:
            g=target(t['panel_case_id']);saved_values[s['state_id']]=z
            r['metrics']=dict(temporal={k:temporal_decomposition(matrix(z[k]),g) for k in (*BRANCHES,'x_t','actual_combined_prediction')},
                response=response_decomposition({k:matrix(z[k]) for k in BRANCHES}))
        traces.append(r)
    pred_rows=[];branch_rows=[];pred_values={};branch_lookup={}
    for p in probes:
        s=smap[p['state_id']];t=tmap[s['trajectory_id']];z,valid=load('probes',p['probe_id'],run/'prefix/probes'/(p['probe_id']+'.npz'))
        r=dict(row_id=p['probe_id'],probe_id=p['probe_id'],state_id=p['state_id'],checkpoint=p['checkpoint'],timestep=s['timestep'],**identity(t),status=valid)
        if z:
            pred_values[p['probe_id']]=z;g=target(t['panel_case_id'])
            r['response']=response_decomposition({k:matrix(z[k]) for k in BRANCHES})
        pred_rows.append(r)
        for b in BRANCHES:
            br=dict(r,row_id=p['probe_id']+'-'+b,branch=b);br.pop('response',None)
            if z:br['temporal']=temporal_decomposition(matrix(z[b]),g)
            branch_rows.append(br);branch_lookup[(p['state_id'],p['checkpoint'],b)]=br
    weight=[];probe_map={(p['state_id'],p['checkpoint']):p for p in probes}
    for s in states:
        t=tmap[s['trajectory_id']]
        for b in BRANCHES:
            f=probe_map[(s['state_id'],'F200')];u=probe_map[(s['state_id'],'U200')]
            a=branch_lookup[(s['state_id'],'F200',b)];c=branch_lookup[(s['state_id'],'U200',b)]
            row=dict(row_id=s['state_id']+'-'+b+'-F200_minus_U200',state_id=s['state_id'],timestep=s['timestep'],branch=b,**identity(t),status='not_comparable')
            if f['probe_id'] in pred_values and u['probe_id'] in pred_values:
                row['status']='valid' if a['status']==c['status']=='valid' else 'diagnostic_only'
                row['delta']=same_state_delta(matrix(pred_values[u['probe_id']][b]),matrix(pred_values[f['probe_id']][b]),target(t['panel_case_id']))
            weight.append(row)
    paths=[];trace_lookup={(r['panel_case_id'],r['path_arm'],r['timestep']):r for r in traces}
    for cid in sorted({t['panel_case_id'] for t in trajectories}):
        for timestep in [999,975,950,925,900]:
            for a,b in [('F','U'),('F','T'),('U','T')]:
                left=trace_lookup[(cid,a,timestep)];right=trace_lookup[(cid,b,timestep)]
                row=dict(row_id=cid+'-'+str(timestep)+'-'+a+'-'+b,comparison=a+'-'+b,timestep=timestep,
                    **{k:left[k] for k in ('data_split','motion_id','sensor_config','panel_case_id')},status='not_comparable',
                    interpretation='Matched-noise total prefix path contrast; different states/weights, not an isolated same-state weight effect')
                if left['state_id'] in saved_values and right['state_id'] in saved_values:
                    la=saved_values[left['state_id']];rb=saved_values[right['state_id']]
                    row['status']='valid' if left['status']==right['status']=='valid' else 'diagnostic_only'
                    row['metrics']={k:dict(energy_delta=left['metrics']['temporal']['f11'][k]['prediction_temporal_energy']-right['metrics']['temporal']['f11'][k]['prediction_temporal_energy'],
                        x_t_difference_rms=float(np.sqrt(np.mean((matrix(la['x_t'])[:,ids]-matrix(rb['x_t'])[:,ids])**2))))
                        for k,ids in [('uniform',list(range(263))),('root_ric67',list(range(67))),('remaining196',list(range(67,263)))]}
                paths.append(row)
    terminals=[]
    for t in trajectories:
        path=run/'prefix/terminal'/(t['trajectory_id']+'.npz')
        row=dict(row_id=t['trajectory_id'],**identity(t),timestep=899,status='missing')
        if path.exists():
            bind(path);z=arrays(path);row.update(status='valid' if complete else 'diagnostic_only',temporal=temporal_decomposition(matrix(z['x_899']),target(t['panel_case_id'])))
        terminals.append(row)
    summary=dict(package_sha256=artifact(package)['sha256'],queue_status=status['status'],failure=status.get('failure'),
        traces=dict(Counter(r['status'] for r in traces)),probes=dict(Counter(r['status'] for r in pred_rows)),
        branches=dict(Counter(r['status'] for r in branch_rows)),weight_contrasts=dict(Counter(r['status'] for r in weight)),
        path_contrasts=dict(Counter(r['status'] for r in paths)),terminals=dict(Counter(r['status'] for r in terminals)),
        new_evaluation_GPU_queries=0,new_evaluation_model_calls=0,new_full_motion_decodes=0,
        interpretation='Conditional fixed-seed/panel short-prefix diagnostics only. No full-motion behavior repair, unique mechanism, training sufficiency or generalization certification.')
    output.mkdir(parents=True)
    for name,value in [('prefix_traces',traces),('probes',pred_rows),('same_state_branches',branch_rows),('weight_contrasts',weight),('path_contrasts',paths),('terminals',terminals),('summary',summary)]:write(output/(name+'.json'),value)
    for name,value in [('trace_aggregates',aggregate(traces,['data_split','sensor_config','path_arm','timestep'],'metrics')),
        ('branch_aggregates',aggregate(branch_rows,['data_split','sensor_config','path_arm','timestep','checkpoint','branch'],'temporal')),
        ('weight_aggregates',aggregate(weight,['data_split','sensor_config','path_arm','timestep','branch'],'delta'))]:write(output/(name+'.json'),value)
    write(output/'sources.json',list(sources.values()));write(output/'artifact_index.json',[artifact(p) for p in sorted(output.glob('*.json'))]);print(summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();evaluate(a.run,a.output)
