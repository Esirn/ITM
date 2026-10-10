#!/usr/bin/env python
"""All planned saved-state predictions/contrasts; NumPy only, no model calls."""
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from itm.metrics.training_sufficiency import temporal_decomposition,same_state_delta

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'docs/itm_rework/protocol/b0_training_sufficiency_20261008_v4/package.json'
PACKAGE_SHA='95283f4c9744c6624ca3d4e6f0d2a199427f85e50edb2c975039512208b0416a'
BRANCHES=('T','P','S','R')


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def artifact(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def verify(a):
    p=ROOT/a['path']
    if sha(p)!=a['sha256']:raise ValueError('Evaluation artifact identity changed: '+str(p))
    return p
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def descriptor(a):return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
def matrix(a):return a.reshape(263,-1).T
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def flatten(value,prefix=''):
    result={}
    for k,v in value.items():
        name=prefix+k
        if isinstance(v,dict):result.update(flatten(v,name+'.'))
        elif v is None or isinstance(v,(int,float)):result[name]=v
    return result


def aggregate(rows,keys,metric_key):
    """Both Q seeds required per source, then equal source means; retain missing."""
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    result=[]
    for group,items in sorted(groups.items()):
        per_source=defaultdict(list)
        for r in items:per_source[r['motion_id']].append(r)
        sources=[]
        for motion_id,records in sorted(per_source.items()):
            eligible=[r for r in records if r['status']=='valid']
            complete=len(eligible)==len(records)
            metrics=[flatten(r[metric_key]) for r in eligible]
            names=sorted({k for m in metrics for k in m})
            averaged={k:float(np.mean([m[k] for m in metrics])) if complete and all(m.get(k) is not None for m in metrics) else None for k in names}
            sources.append(dict(motion_id=motion_id,planned_replicates=len(records),valid_replicates=len(eligible),
                                status='valid' if complete else 'incomplete',metrics=averaged,
                                missing=[dict(row_id=r['row_id'],status=r['status'],reason=r.get('reason')) for r in records if r['status']!='valid']))
        names=sorted({k for r in sources for k in r['metrics']})
        metrics={}
        for k in names:
            values=[r['metrics'][k] for r in sources if r['status']=='valid' and r['metrics'].get(k) is not None]
            metrics[k]=dict(planned_n=len(sources),valid_n=len(values),mean=float(np.mean(values)) if values else None,
                            min=min(values) if values else None,max=max(values) if values else None)
        result.append(dict(zip(keys,group),planned_sources=len(sources),valid_sources=sum(r['status']=='valid' for r in sources),
                           metrics=metrics,sources=sources))
    return result


def evaluate(run,output,package=PACKAGE):
    run=Path(run).resolve();output=Path(output).resolve();package=Path(package).resolve()
    if output.exists():raise FileExistsError('Evaluation must use a fresh directory')
    if sha(package)!=PACKAGE_SHA:raise ValueError('Exact frozen diagnostic package required')
    sources={}
    def bind(p):
        a=artifact(p);sources[a['path']]=a;return a
    bind(package)
    files={Path(a['path']).stem:verify(a) for a in read(package)['artifacts']}
    for p in files.values():bind(p)
    states=read(files['states']);jobs=read(files['jobs']);evaluation=read(files['evaluation'])
    status=read(run/'runtime/status.json');bind(run/'runtime/status.json')
    if status['implementation']['sha256']!=status['execution_record']['implementation_sha256']:
        raise ValueError('Queue implementation identity changed')
    bind(verify(status['implementation']))
    if status.get('receipt_journal'):bind(verify(status['receipt_journal']))
    receipt=None
    if (run/'runtime/diagnostic.receipt.json').exists():
        receipt=read(run/'runtime/diagnostic.receipt.json');bind(run/'runtime/diagnostic.receipt.json')
        for entry in receipt['artifacts']:bind(verify(entry))
    queue_complete=status['status']=='completed' and receipt is not None and receipt['status']=='completed'
    if queue_complete:
        for kind,cap,calls in (('mdm',5184,1152),('encoder',432,432),('caption',72,24)):
            c=receipt['counts'][kind]
            if (any(c[k]!=cap for k in ('reserved','actual','completed'))
                    or any(c[k]!=calls for k in ('reserved_calls','actual_calls','completed_calls'))):
                raise ValueError('Completed queue has wrong diagnostic counts')
        if any(receipt[k] for k in ('updates_attempted','updates_completed','denoising_calls')):
            raise ValueError('Completed queue performed forbidden work')
        for key in ('states','jobs','branches','checks'):
            if any(r['status']!='completed' for r in receipt[key]):raise ValueError('Completed queue has incomplete ledger')
    state_map={r['state_id']:r for r in states};job_status={r['job_id']:r for r in status['jobs']}
    predictions={};targets={};rows=[]
    for job in jobs:
        state=state_map[job['state_id']];jr=job_status[job['job_id']]
        raw=run/'diagnostic/predictions'/(job['job_id']+'.npz')
        values=None;reason=None
        if raw.exists():
            if not jr.get('raw_artifact'):raise ValueError('Unbound raw prediction')
            verify(jr['raw_artifact']);bind(raw);values=arrays(raw)
            if set(values)!=set(BRANCHES) or any(v.dtype!=np.float32 or v.shape!=(263,1,state['frames']) or not np.isfinite(v).all() for v in values.values()):
                raise ValueError('Raw prediction branch/dtype/shape/finite mismatch')
            target_path=verify(state['target_artifact']);bind(target_path)
            with np.load(target_path,allow_pickle=False) as z:target=z['target'].copy()
            if descriptor(target)!=state['arrays']['target']:raise ValueError('Frozen target changed')
            targets[state['state_id']]=matrix(target)
        if not queue_complete:reason='queue_failed_or_incomplete: '+str(status.get('failure'))
        if values is None:reason=jr['status']
        for branch in BRANCHES:
            rid=job['job_id']+'-'+branch
            row={k:state[k] for k in ('state_id','state_family','data_split','motion_id','sensor_config','timestep','noise_seed')}
            row.update(row_id=rid,job_id=job['job_id'],snapshot_id=job['snapshot_id'],branch=branch,
                       status='missing' if values is None else ('valid' if queue_complete and jr['status']=='completed' and not jr.get('diagnostic_only') else 'diagnostic_only'),
                       reason=reason)
            if values is not None:
                p=matrix(values[branch]);predictions[rid]=p
                row['temporal']=temporal_decomposition(p,targets[state['state_id']])
            rows.append(row)
    row_map={r['row_id']:r for r in rows}
    def contrast(left,right,label):
        a,b=row_map[left],row_map[right]
        out={k:a[k] for k in ('state_id','state_family','data_split','motion_id','sensor_config','timestep','noise_seed')}
        out.update(row_id=left+'--'+right,comparison=label,prediction_id=left,reference_id=right,
                   snapshot_id=a['snapshot_id'],reference_snapshot_id=b['snapshot_id'],branch=a['branch'],reference_branch=b['branch'],
                   status='not_comparable',reason=[a['reason'],b['reason']])
        if left in predictions and right in predictions:
            out['status']='valid' if a['status']==b['status']=='valid' else 'diagnostic_only'
            out['delta']=same_state_delta(predictions[right],predictions[left],targets[a['state_id']])
        return out
    within=[]
    for job in jobs:
        for a,b in evaluation['branch_contrasts']:
            within.append(contrast(job['job_id']+'-'+a,job['job_id']+'-'+b,a+'-'+b))
    progress=[]
    for state in states:
        for a,b in evaluation['checkpoint_contrasts']:
            for branch in BRANCHES:
                progress.append(contrast(a+'-'+state['state_id']+'-'+branch,b+'-'+state['state_id']+'-'+branch,a+'-'+b))
    summary=dict(version='itm-sufficiency-evaluation-v1',package_sha256=PACKAGE_SHA,run_directory=str(run),
                 queue_status=status['status'],planned=dict(states=192,jobs=1152,branches=4608,checks=2880,within_contrasts=4608,checkpoint_contrasts=4608),
                 branch_statuses=dict(Counter(r['status'] for r in rows)),within_statuses=dict(Counter(r['status'] for r in within)),
                 checkpoint_statuses=dict(Counter(r['status'] for r in progress)),counts=status['counts'],
                 elapsed_seconds=status['finished_monotonic']-status['started_monotonic'],
                 peak_allocated_bytes=status.get('peak_allocated_bytes'),
                 decision='completed_local_diagnostic_no_convergence_or_motion_repair_certification' if queue_complete else 'inconclusive_incomplete_or_failed_software',
                 limitations=evaluation['limitations'],new_training=0,new_samples=0,new_model_forwards=0,GPU_queries=0)
    key=('snapshot_id','branch','state_family','data_split','sensor_config','timestep')
    aggregates=dict(branches=aggregate(rows,key,'temporal'),
                    within=aggregate(within,key+('comparison',),'delta'),
                    progress=aggregate(progress,key+('reference_snapshot_id','comparison'),'delta'))
    if not (len(rows)==len(within)==len(progress)==4608):raise ValueError('Incomplete planned evaluation population')
    output.mkdir(parents=True)
    for name,value in dict(branches=rows,within_contrasts=within,checkpoint_contrasts=progress,aggregates=aggregates,summary=summary,sources=list(sources.values())).items():
        write(output/(name+'.json'),value)
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.iterdir())])
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(evaluate(a.run_dir,a.output),ensure_ascii=False))
