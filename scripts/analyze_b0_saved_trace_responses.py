#!/usr/bin/env python
"""Freeze then evaluate existing F/U trace arrays on CPU; never loads a model."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from audit_b0_feature_balance_traces import arithmetic, array_descriptor
from evaluate_b0_feature_balance import (
    FROZEN, IMPLEMENTATION_SHA, PACKAGE_SHA, ROOT, read, require, sha, write,
)
from itm.metrics.trace_response import paired_rows, response_decomposition
from itm.metrics.training_sufficiency import temporal_decomposition

RUNS = dict(F=ROOT/'outputs/itm_rework/method/feature_balance_authorized_20261008_v1',
            U=ROOT/'outputs/itm_rework/method/sampling_alignment_authorized_20260927_v1')
CHECKPOINTS = dict(F='d0e4f39f8c7ee5b980243baac48bc1bc02dc8f7c15c7a1f47f0adabb0b7e0ad3',
                   U='91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642')
FIELDS = ('data_split','motion_id','sensor_config','condition','frames')
TIMES = [900,500,250,50]


def matrix(a):
    require(a.ndim==4 and a.shape[:3]==(1,263,1),'Feature array shape')
    return a[0,:,0,:].T


def freeze(output):
    require(not output.exists(),'Refusing to overwrite analysis plan')
    require(sha(FROZEN/'package.json')==PACKAGE_SHA,'Original F package identity')
    cases=read(FROZEN/'sampling_slots.json');specs=read(FROZEN/'trace_slots.json')
    require(len(cases)==60 and len(specs)==240,'Original complete population')
    previous=read(ROOT/'outputs/itm_rework/protocol/feature_balance_evaluation_20261008_v2/sources.json')
    previous+=read(ROOT/'outputs/itm_rework/protocol/feature_balance_trace_audit_20261008_v1/traces.json')['sources']
    anchors={str((ROOT/r['path']).resolve()):r['sha256'] for r in previous}
    paths={Path(__file__),ROOT/'src/itm/metrics/trace_response.py',
           ROOT/'tests/test_trace_response.py',ROOT/'src/itm/metrics/training_sufficiency.py',
           ROOT/'src/itm/metrics/feature_balance.py',
           ROOT/'scripts/audit_b0_feature_balance_traces.py',
           ROOT/'scripts/evaluate_b0_feature_balance.py',FROZEN/'package.json',
           FROZEN/'sampling_slots.json',FROZEN/'trace_slots.json',FROZEN/'design.json'}
    normalization=read(FROZEN/'design.json')['common']['training_input_provenance']['payload']['artifacts']
    norm={k:normalization[k] for k in ('motion_mean','motion_std')}
    for e in norm.values():
        require(sha(ROOT/e['path'])==e['sha256'],'Normalization provenance')
        paths.add((ROOT/e['path']).resolve())
    traces=[];samples=[]
    case_map={c['case_id']:c for c in cases}
    for arm,run in RUNS.items():
        paths.add(run/'runtime/status.json')
        status=read(run/'runtime/status.json');require(status['status']=='completed','Complete historical run')
        paths.add(Path(status['implementation']['path']))
        require(sha(status['implementation']['path'])==status['implementation']['sha256'],'Manifest anchor')
        for s in specs:
            c=case_map[s['case_id']];tid=s['trace_id'] if arm=='F' else s['historical_trace_id']
            path=run/'traces'/(tid+'.npz');paths.update([path,path.with_suffix('.json')])
            traces.append(dict(arm=arm,trace_id=tid,case_id=c['case_id'] if arm=='F' else c['historical_case_id'],
                pair_case_id=c['case_id'],timestep=s['timestep'],path=str(path),**{k:c[k] for k in FIELDS}))
        for c in cases:
            cid=c['case_id'] if arm=='F' else c['historical_case_id']
            path=run/'samples'/(cid+'.npz');paths.add(path)
            paths.add(Path(c['gt_path']).with_name('motion.npy'))
            samples.append(dict(arm=arm,case_id=cid,pair_case_id=c['case_id'],path=str(path),**{k:c[k] for k in FIELDS}))
    sources=[]
    for path in sorted(paths):
        actual=sha(path);prior=anchors.get(str(path))
        require(prior is None or prior==actual,'Previously bound array/source changed: '+str(path))
        sources.append(dict(path=str(path),sha256=actual,prior_anchor=prior is not None))
    output.mkdir(parents=True)
    plan=dict(version='b0-saved-trace-response-v1',date='2026-10-08',
        authorization='Main experimental session; CPU saved-array analysis',
        population=dict(traces=480,paired_trace_points=240,samples=120,
            condition_counts_per_arm=dict(Counter(c['condition'] for c in cases)),timesteps=TIMES),
        metrics=dict(views='Existing 11 HumanML263 views; no new behavioral threshold',
            components=['f00','f10-f00','f01-f00','f11-f10-f01+f00'],
            temporal_energy='Mean squared temporally centered normalized features',
            identity='E(f11) = four component energies + six twice-covariances',
            conditional='E(b)-E(a) = 2 Cov(center(a),center(b-a)) + E(center(b-a))',
            rounding='Inherited float64 64eps algebra bounds; inherited CFG/absent-branch checks unchanged'),
        aggregation='Equal source means, min/max/planned/valid n; retain split/config/condition/t/arm separately. '
            'No pooling diffusion times or treating frames as independent sources.',
        interpretation='Different F/U x_t and different times are descriptive only. GT is fixed-time '
            'reference, not a conditional expectation. Variance is not behavior correctness. '
            'All negative cases and incomplete denominators retained; not blind or independent review.',
        prohibited=['model load/forward','GPU query','training','noise generation','sampling','decode','checkpoint selection'],
        normalization=norm,traces=traces,samples=samples,sources=sources)
    write(output/'plan.json',plan)
    print(dict(plan=str(output/'plan.json'),sha256=sha(output/'plan.json'),sources=len(sources),**plan['population']))


def flatten(value,prefix=''):
    result={}
    for k,v in value.items():
        path=prefix+k
        if isinstance(v,dict):result.update(flatten(v,path+'/'))
        elif isinstance(v,(int,float)) and not isinstance(v,bool):result[path]=float(v)
    return result


def aggregates(rows, group_fields):
    groups=defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in group_fields)].append(row)
    out=[]
    for key,population in sorted(groups.items()):
        values=defaultdict(list)
        for r in sorted(population,key=lambda x:x['motion_id']):
            if r['status']=='valid':
                for k,v in flatten(r['metrics']).items():values[k].append(v)
        out.append(dict(group=dict(zip(group_fields,key)),planned=len(population),
            valid=sum(r['status']=='valid' for r in population),sources=[r['motion_id'] for r in population],
            metrics={k:dict(n=len(v),mean=float(np.mean(v)),min=min(v),max=max(v)) for k,v in sorted(values.items())}))
    return out


def evaluate(plan_path,output):
    require(not output.exists(),'Refusing to overwrite analysis evidence')
    plan=read(plan_path);bindings={r['path']:r['sha256'] for r in plan['sources']}
    def bind(path):
        path=Path(path).resolve()
        require(str(path) in bindings and sha(path)==bindings[str(path)],'Frozen source changed: '+str(path))
        return path
    for path in [Path(__file__),ROOT/'src/itm/metrics/trace_response.py',
                 ROOT/'src/itm/metrics/training_sufficiency.py',ROOT/'src/itm/metrics/feature_balance.py',
                 ROOT/'scripts/audit_b0_feature_balance_traces.py',ROOT/'scripts/evaluate_b0_feature_balance.py']:
        bind(path)
    cases={r['case_id']:r for r in read(bind(FROZEN/'sampling_slots.json'))}
    specs=read(bind(FROZEN/'trace_slots.json'))
    mean,std=[np.load(bind(plan['normalization'][k]['path']),allow_pickle=False) for k in ('motion_mean','motion_std')]
    require(np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all(),'Normalization finite/positive')
    targets={}
    for c in cases.values():
        if c['case_id'] not in targets:
            target=np.load(bind(Path(c['gt_path']).with_name('motion.npy')),allow_pickle=False)
            targets[c['case_id']]=(target-mean)/std
    fstatus=read(bind(RUNS['F']/'runtime/status.json'))
    manifest=read(bind(fstatus['implementation']['path']))
    require(fstatus['implementation']['sha256']==IMPLEMENTATION_SHA,'F manifest identity')
    inputs={r['case_id']:r['arrays'] for r in manifest['inputs']}
    rows=[]
    for spec in plan['traces']:
        row=dict(spec,status='missing');path=Path(spec['path'])
        if path.exists():
            try:
                meta=read(bind(path.with_suffix('.json')));bind(path)
                c=cases[spec['pair_case_id']]
                for k in ('trace_id','case_id','timestep','condition'):
                    require(meta[k]==spec[k],'Trace identity: '+k)
                require(meta['status']=='completed' and meta['scaled_t']==[spec['timestep']],'Completed native timestep')
                require(meta['checkpoint_sha256']==CHECKPOINTS[spec['arm']],'Checkpoint identity')
                require(meta['guidance']==c['guidance'] and meta['random_plan_sha256']==c['historical_random_plan_sha256'],'Guidance/noise identity')
                expected=inputs[c['case_id']] if spec['arm']=='F' else c['historical_input_hashes']
                require(meta['input_hashes']==expected,'Delivered input hashes')
                if spec['arm']=='F':require(meta['package_sha256']==PACKAGE_SHA and meta['implementation_sha256']==IMPLEMENTATION_SHA,'F protocol identity')
                with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
                require({k:array_descriptor(v) for k,v in arrays.items()}==meta['arrays'],'Trace descriptors')
                require(array_descriptor(arrays['actual_combined_prediction'])['sha256']==meta['actual_return_sha256'],'Returned array identity')
                for k in ('f00','f10','f01','f11','x_t','actual_combined_prediction'):
                    require(arrays[k].shape==(1,263,1,c['frames']) and arrays[k].dtype==np.float32,'Float32 full-length trace')
                mask=arrays['frame_mask'];require(mask.shape==(1,c['frames']) and mask.dtype==np.bool_ and mask.all(),'Complete original frames')
                check=arithmetic(arrays,c);target=targets[c['case_id']]
                temporal={k:temporal_decomposition(matrix(arrays[k]),target,mask[0])
                    for k in ('x_t','f00','f10','f01','f11','actual_combined_prediction')}
                response=response_decomposition({k:matrix(arrays[k]) for k in ('f00','f10','f01','f11')},mask[0])
                row.update(status='valid',arithmetic=check,metrics=dict(temporal=temporal,response=response))
            except (ValueError,KeyError,OSError,AssertionError,ArithmeticError) as e:
                row.update(status='invalid',reason=str(e))
        rows.append(row)
    samples=[]
    for spec in plan['samples']:
        row=dict(spec,status='missing');path=Path(spec['path'])
        if path.exists():
            try:
                bind(path)
                with np.load(path,allow_pickle=False) as z:
                    p=matrix(z['normalized_humanml263']);mask=z['frame_mask'][0]
                require(p.shape==(spec['frames'],263),'Final feature shape')
                row.update(status='valid',metrics=temporal_decomposition(p,targets[spec['pair_case_id']],mask))
            except (ValueError,KeyError,OSError,ArithmeticError) as e:row.update(status='invalid',reason=str(e))
        samples.append(row)
    pairs=paired_rows(rows,specs)
    summary=dict(plan_sha256=sha(plan_path),trace_counts=dict(Counter(r['status'] for r in rows)),
        paired_counts=dict(Counter(r['status'] for r in pairs)),sample_counts=dict(Counter(r['status'] for r in samples)),
        new_GPU_queries=0,new_model_calls=0,new_noise=0,new_decodes=0,
        interpretation=plan['interpretation'])
    output.mkdir(parents=True)
    for name,value in [('traces',rows),('samples',samples),('pairs',pairs),('summary',summary),
        ('aggregates',aggregates(rows,['arm','data_split','sensor_config','condition','timestep'])),
        ('sample_aggregates',aggregates(samples,['arm','data_split','sensor_config','condition']))]:write(output/(name+'.json'),value)
    write(output/'sources.json',[dict(path=str(plan_path.resolve()),sha256=sha(plan_path))]+plan['sources'])
    write(output/'artifact_index.json',[dict(path=str(p.resolve()),sha256=sha(p)) for p in sorted(output.glob('*.json'))])
    print(summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.freeze:freeze(args.freeze.resolve())
    else:
        require(args.plan is not None and args.output is not None,'Plan/output required')
        evaluate(args.plan.resolve(),args.output.resolve())
