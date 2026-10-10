#!/usr/bin/env python
"""CPU adapter for frozen96 late-state rows and complete saved trace references."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from evaluate_b0_native_training_sufficiency import read,artifact,verify,arrays,descriptor,matrix,write,aggregate
from itm.metrics.prefix_attribution import two_component_delta

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/itm_rework/protocol/b0_late_attribution_CPU_20261008_v1/plan.json'
NATIVE=ROOT/'outputs/itm_rework/method/training_sufficiency_native_authorized_20261008_v1'
NATIVE_PACKAGE=ROOT/'docs/itm_rework/protocol/b0_training_sufficiency_20261008_v4/package.json'
NATIVE_SHA='95283f4c9744c6624ca3d4e6f0d2a199427f85e50edb2c975039512208b0416a'
REFERENCES=ROOT/'outputs/itm_rework/protocol/saved_trace_response_evaluation_20261008_v1'


def source_gate(plan):
    for a in plan['sources']:verify(a)
    status=read(NATIVE/'runtime/status.json')
    if status['status']!='completed' or any(c['status']!='completed' for c in status['checks']):raise ValueError('Original native software gate incomplete')
    implementation=read(verify(status['implementation']))
    if implementation['package']['sha256']!=NATIVE_SHA or artifact(NATIVE_PACKAGE)['sha256']!=NATIVE_SHA:raise ValueError('Native protocol identity changed')
    receipt=read(verify(status['receipt']))
    if receipt['status']!='completed':raise ValueError('Original native receipt incomplete')
    verify(status['receipt_journal'])
    for k,examples,calls in [('mdm',5184,1152),('encoder',432,432),('caption',72,24)]:
        c=receipt['counts'][k]
        if any(c[f]!=examples for f in ['reserved','actual','completed']) or any(c[f]!=calls for f in ['reserved_calls','actual_calls','completed_calls']):raise ValueError('Native count mismatch')
    if any(receipt[k] for k in ['updates_attempted','updates_completed','denoising_calls']):raise ValueError('Unexpected native work')
    return status,implementation


def validate_roles(job,metadata):
    first=job['direct_is_first']
    expected=[('F200','F_paired_trace'),('U200','F_paired_trace') if first else ('F200','U_paired_trace'),('U200','U_paired_trace')]
    if job['ordering']!=('direct_at_F_state' if first else 'direct_at_reference_state'):raise ValueError('Ordering role mismatch')
    for meta,(weight,family) in zip(metadata,expected):
        if meta['status']!='completed' or meta['diagnostic_only']:raise ValueError('Uncertified raw slot')
        if meta['job']['snapshot_id']!=weight or meta['state']['state_family']!=family:raise ValueError('Checkpoint/state family mismatch')
        for k in ('data_split','motion_id','sensor_config','timestep'):
            if meta['state'][k]!=job[k]:raise ValueError('Case/time identity mismatch: '+k)
        if meta['job']['branch_order']!=['T','P','S','R'] or meta['job']['caption_batch']!=1 or meta['job']['input_batch_shape'][0]!=4:
            raise ValueError('Native saved-trajectory path changed')
    if len({m['state']['frames'] for m in metadata})!=1:raise ValueError('Different full-frame lengths')


def load_job(job,status,checkpoints,cache,bindings):
    table={j['job_id']:j for j in status['jobs']};metadata=[];values=[]
    for key in ['prediction','intermediate','reference']:
        ref=job[key];jid=ref['job_id'];entry=table[jid]
        if ref['branch']!='P' or entry['status']!='completed' or entry.get('diagnostic_only'):raise ValueError('Non-paired or uncertified prediction')
        path=verify(ref['raw'])
        if ref['raw']!=entry['raw_artifact']:raise ValueError('Receipt raw identity mismatch')
        meta_path=path.parent.parent/'jobs'/(jid+'.json')
        if str(meta_path) not in bindings or artifact(meta_path)['sha256']!=bindings[str(meta_path)]:raise ValueError('Unbound native metadata')
        if jid not in cache:
            z=arrays(path);meta=read(meta_path)
            if meta['raw_artifact']!=ref['raw'] or meta['job']['job_id']!=jid or meta['checkpoint']!=checkpoints[meta['job']['snapshot_id']]:raise ValueError('Checkpoint/job/raw identity')
            if set(z)!=set(['T','P','S','R']) or {k:descriptor(v) for k,v in z.items()}!=meta['arrays']:raise ValueError('Raw descriptors')
            frames=meta['state']['frames']
            if any(v.dtype!=np.float32 or v.shape!=(263,1,frames) or not np.isfinite(v).all() for v in z.values()):raise ValueError('Native float32 finite shape')
            state=arrays(verify(meta['state_artifact']))
            if {k:descriptor(v) for k,v in state.items()}!=meta['state']['arrays']:raise ValueError('Saved state descriptors')
            if not state['frame_mask'].all():raise ValueError('Original full frames required')
            cache[jid]=(z,meta,state)
        z,meta,state=cache[jid];values.append(matrix(z['P']));metadata.append(meta)
    validate_roles(job,metadata)
    target_path=verify(job['target']);target=arrays(target_path)[job['target_array']]
    if target.dtype!=np.float32 or descriptor(target)!=metadata[0]['state']['arrays']['target']:raise ValueError('Target dtype/identity')
    for meta in metadata:
        if meta['state']['arrays']['target']!=descriptor(target):raise ValueError('Different targets across state families')
    return values,matrix(target)


def prepare(directory,plan_path=PLAN):
    if directory.exists():raise FileExistsError('Fresh CPU implementation directory required')
    plan=read(plan_path);status,implementation=source_gate(plan)
    snapshots={s['snapshot_id']:s['artifact'] for s in implementation['snapshots']}
    bindings={s['path']:s['sha256'] for s in plan['sources']};cache={}
    for job in plan['jobs']:load_job(job,status,snapshots,cache,bindings)
    index=read(REFERENCES/'artifact_index.json')
    for a in index:verify(a)
    sources={s['path']:s for s in plan['sources']}
    for p in [plan_path,NATIVE_PACKAGE,NATIVE/'runtime/status.json',Path(status['receipt']['path']),Path(status['receipt_journal']['path']),
              Path(status['implementation']['path']),REFERENCES/'artifact_index.json',REFERENCES/'samples.json',
              Path(__file__).resolve(),ROOT/'scripts/evaluate_b0_native_training_sufficiency.py',
              ROOT/'src/itm/metrics/prefix_attribution.py',ROOT/'src/itm/metrics/training_sufficiency.py',
              ROOT/'src/itm/metrics/feature_balance.py',ROOT/'tests/test_late_attribution_adapter.py']:
        a=artifact(p);sources[a['path']]=a
    for _,meta,_ in cache.values():
        for a in [meta['state_artifact'],meta['checkpoint']]:verify(a);sources[a['path']]=a
    directory.mkdir(parents=True)
    manifest=dict(version='itm-late-attribution-CPU-implementation-v1',plan=artifact(plan_path),native_implementation=status['implementation'],
        expected_artifacts=[sources[k] for k in sorted(sources)],validated_jobs=len(cache),planned_rows=96,
        synthetic_helper_tests_inherited=True,new_GPU_queries=0,new_model_calls=0,new_noise_draws=0)
    write(directory/'implementation.json',manifest);print(dict(implementation=artifact(directory/'implementation.json'),validated_jobs=len(cache),dependencies=len(sources)))


def evaluate(manifest_path,output):
    if output.exists():raise FileExistsError('Fresh late attribution evidence required')
    manifest=read(manifest_path)
    for a in manifest['expected_artifacts']:verify(a)
    plan=read(verify(manifest['plan']));status,implementation=source_gate(plan)
    snapshots={s['snapshot_id']:s['artifact'] for s in implementation['snapshots']}
    bindings={s['path']:s['sha256'] for s in plan['sources']};cache={};rows=[]
    for job in plan['jobs']:
        row=dict(job,status='missing',comparison='F_minus_U')
        try:
            values,g=load_job(job,status,snapshots,cache,bindings)
            row.update(status='valid',metrics=two_component_delta(*values,g,direct_is_first=job['direct_is_first']))
        except (ValueError,KeyError,OSError,ArithmeticError) as e:row.update(status='invalid',reason=str(e))
        rows.append(row)
    traces=read(REFERENCES/'traces.json');samples=read(REFERENCES/'samples.json')
    reference=[]
    for r in traces:
        if r['status']!='valid':raise ValueError('Incomplete original full trace reference')
        reference.append(dict(row_id=r['arm']+'-'+r['trace_id'],**{k:r[k] for k in ['arm','data_split','motion_id','sensor_config','condition','timestep']},
            status='valid',reference_kind='actual_saved_own_trajectory',temporal=r['metrics']['temporal']['actual_combined_prediction'],
            interpretation='Across conditions/arms the states differ; not a same-state conditional prediction or new intervention.'))
    finals=[]
    for r in samples:
        if r['status']!='valid':raise ValueError('Incomplete final generated reference')
        finals.append(dict(row_id=r['case_id'],**{k:r[k] for k in ['arm','data_split','motion_id','sensor_config','condition']},
            status='valid',reference_kind='old_final_features',temporal=r['metrics']))
    summary=dict(status='completed' if all(r['status']=='valid' for r in rows) else 'inconclusive',planned=96,counts=dict(Counter(r['status'] for r in rows)),
        trace_references=len(reference),final_references=len(finals),new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,
        interpretation='Saved-prediction algebra/old text-only trajectory reference. No isolated causal state intervention, behavior repair or training sufficiency claim.')
    output.mkdir(parents=True)
    for name,data in [('rows',rows),('aggregates',aggregate(rows,['data_split','sensor_config','timestep','comparison','ordering'],'metrics')),
        ('reference_traces',reference),('reference_finals',finals),('reference_aggregates',aggregate(reference,['arm','data_split','sensor_config','condition','timestep'],'temporal')),
        ('summary',summary),('sources',[artifact(manifest_path),*manifest['expected_artifacts']])]:write(output/(name+'.json'),data)
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.glob('*.json'))]);print(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-dir',type=Path);p.add_argument('--implementation',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.prepare_dir:prepare(a.prepare_dir.resolve())
    else:evaluate(a.implementation.resolve(),a.output.resolve())
