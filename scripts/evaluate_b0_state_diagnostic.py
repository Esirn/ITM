#!/usr/bin/env python
"""Owner02 NumPy-only V evaluation, complete frozen population, no neural calls."""
import argparse
import builtins
from collections import defaultdict
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import subprocess
from unittest.mock import patch
import numpy as np
from freeze_b0_state_diagnostic import ROOT, BLOCKS, PRIMARY, decomposition, sha, describe, read, write

FROZEN=ROOT/'docs/itm_rework/protocol/b0_state_diagnostic_20260928_v1'
RUN=ROOT/'outputs/itm_rework/method/state_diagnostic_authorized_20261003_v1'
METHOD=ROOT/'docs/itm_rework/method/state_diagnostic_run_20261003_v1'
OUT=ROOT/'docs/itm_rework/protocol/b0_state_diagnostic_evaluation_20261003_v1'
PACKAGE_SHA='6f050dbfbe57c508e86d83a30b3cdfb505f7700b6781cb52b43d85b6ffc0b012'
MANIFEST_SHA='71cb4e2306559295e59ee5c179ce57e971a1a60dcb9f021cf1ddc33c4c45ee46'
CHECKPOINT_SHA='91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642'
BRANCHES=('T','P','S','R')
KEYS=('data_split','sensor_config','timestep','motion_id','contrast')


def features(a):
    a=np.asarray(a)
    if a.ndim==4 and a.shape[0]==1:a=a[0]
    if a.ndim!=3 or a.shape[:2]!=(263,1) or a.shape[2]==0 or not np.isfinite(a).all():
        raise ValueError('Finite nonempty [263,1,T] required')
    return a.astype(np.float64).reshape(263,-1)


def magnitude(a):
    a=np.asarray(a,np.float64)
    if not a.size or not np.isfinite(a).all():raise ValueError('Invalid magnitude input')
    return dict(count=a.size,rms=float(np.sqrt(np.mean(a*a))),max_abs=float(np.max(np.abs(a))))


def branch_metrics(pred,target):
    p,g=features(pred),features(target)
    if p.shape!=g.shape:raise ValueError('Feature/frame shape mismatch')
    return {k:dict(dimensions=len(ids),mse=float(np.mean((p[ids]-g[ids])**2)),
        prediction=magnitude(p[ids]),target=magnitude(g[ids]),error=magnitude(p[ids]-g[ids])) for k,ids in BLOCKS.items()}


def contrast_metrics(pred,base,target):
    p,b,g=features(pred),features(base),features(target)
    if p.shape!=b.shape or b.shape!=g.shape:raise ValueError('Contrast shape mismatch')
    values=decomposition(base,pred,target)
    for k,ids in BLOCKS.items():values[k].update(prediction_difference=magnitude(p[ids]-b[ids]))
    return values


def source_means(rows,expected_replicates):
    grouped=defaultdict(list)
    for row in rows:grouped[tuple(row[k] for k in KEYS)].append(row)
    out=[]
    metrics=('baseline_mse','conditioned_mse','delta_mse','cross','residual_energy','weighted_delta')
    for key,items in sorted(grouped.items()):
        if len(items)!=expected_replicates or len({r['replicate'] for r in items})!=expected_replicates:
            raise ValueError('Missing or duplicate noise replicate')
        blocks={k:{m:float(np.mean([r['blocks'][k][m] for r in items])) for m in metrics} for k in BLOCKS}
        out.append(dict(zip(KEYS,key),n_replicates=len(items),blocks=blocks))
    return out


def aggregate(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[tuple(row[k] for k in KEYS[:-2])+ (row['contrast'],)].append(row)
    result=[]
    for key,items in sorted(grouped.items()):
        if len({r['motion_id'] for r in items})!=len(items):raise ValueError('Replicates must be averaged per source first')
        if len(items)!=3:raise ValueError('This complete-run analysis requires all3 frozen sources')
        blocks={}
        for block in BLOCKS:
            blocks[block]={}
            for metric in ('baseline_mse','conditioned_mse','delta_mse','cross','residual_energy','weighted_delta'):
                vals=[r['blocks'][block][metric] for r in items]
                blocks[block][metric]=dict(mean=float(np.mean(vals)),min=min(vals),max=max(vals))
            vals=[r['blocks'][block]['delta_mse'] for r in items]
            blocks[block]['delta_sign_counts']=dict(negative=sum(v<0 for v in vals),zero=sum(v==0 for v in vals),positive=sum(v>0 for v in vals))
        result.append(dict(zip(('data_split','sensor_config','timestep','contrast'),key),n_sources=3,planned_sources=3,
            missing=[],source_ids=[r['motion_id'] for r in items],blocks=blocks))
    return result


def paired_state_comparison(v,q):
    vm={tuple(r[k] for k in KEYS):r for r in v};qm={tuple(r[k] for k in KEYS):r for r in q}
    if len(vm)!=len(v) or len(qm)!=len(q) or vm.keys()!=qm.keys():raise ValueError('V/Q source/config/t/contrast population mismatch')
    return [dict(zip(KEYS,k),V_replicates=vm[k]['n_replicates'],Q_replicates=qm[k]['n_replicates'],
        blocks={b:dict(V=vm[k]['blocks'][b],Q=qm[k]['blocks'][b],
            V_minus_Q_delta_mse=vm[k]['blocks'][b]['delta_mse']-qm[k]['blocks'][b]['delta_mse']) for b in BLOCKS}) for k in sorted(vm)]


def completed_chain(parent,receipt,final,slot_id):
    # Intermediate sidecars are intentionally pending; accept only complete final chain.
    for state in (parent,receipt,final):
        if state['status']!='completed':raise ValueError('Incomplete run/group cannot enter primary means')
        row=next(r for r in state['branches'] if r['slot_id']==slot_id)
        if row['status']!='completed' or not row['executed'] or row['diagnostic_only']:
            raise ValueError('Incomplete branch cannot enter primary means')


def evaluate(output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    sources={}
    def bind(p,expected=None):
        p=ROOT/Path(p);key=str(p)
        if key not in sources:sources[key]=dict(path=key,sha256=sha(p))
        if expected is not None and sources[key]['sha256']!=expected:raise ValueError('Source hash mismatch: '+key)
        return p
    def get(p):return read(bind(p))
    def verify(a):return bind(a['path'],a['sha256'])
    def load(p):
        with np.load(bind(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files if k!='metadata'}
    package=get(bind(FROZEN/'package.json',PACKAGE_SHA))
    for a in package['artifacts']:verify(a)
    groups=get(FROZEN/'state_groups.json');slots=get(FROZEN/'branch_slots.json');cases=get(FROZEN/'input_cases.json')
    specs=get(FROZEN/'software_checks.json');protocol=get(FROZEN/'evaluation.json')
    assert get(FROZEN/'feature_partition.json')['blocks']==BLOCKS
    status=get(RUN/'runtime/status.json');receipt=get(RUN/'runtime/V.receipt.json');context=get(RUN/'runtime/V.context.json')
    approval=get(ROOT/'docs/itm_rework/method/B0_STATE_DIAGNOSTIC_AUTHORIZATION_20261003.json')
    assert status['status']==receipt['status']=='completed' and status['authorization']==approval
    assert approval['authorized'] and approval['approved_by']=='user' and approval['run_directory']==str(RUN)
    assert approval['implementation_sha256']==MANIFEST_SHA and approval['package_sha256']==PACKAGE_SHA
    assert approval['checkpoint']['sha256']==CHECKPOINT_SHA
    for key in ('package','checkpoint','request'):verify(approval[key])
    verify(status['receipt']);manifest=get(verify(status['implementation']));assert status['implementation']['sha256']==MANIFEST_SHA
    assert len(manifest['expected_artifacts'])==1656
    for a in manifest['expected_artifacts']:verify(a)
    assert receipt['context']==context
    assert context['implementation_sha256']==MANIFEST_SHA and context['package_sha256']==PACKAGE_SHA and context['checkpoint_sha256']==CHECKPOINT_SHA
    for a in context['expected_artifacts']:verify(a)
    elapsed=status['finished_monotonic']-status['started_monotonic']
    assert 0<elapsed<300 and status['deadline_monotonic']-status['started_monotonic']==300
    assert context['deadline_monotonic']==status['deadline_monotonic'] and receipt['last_monotonic']<=context['deadline_monotonic']
    assert 0<receipt['peak_allocated_bytes']<=receipt['allocator_cap_bytes']<=24*1024**3
    assert status['gpu_preflight']['gpu_uuid']==receipt['gpu_uuid']==context['gpu_uuid']==approval['gpu_uuid']
    assert status['gpu_preflight']['compute_processes']==0 and status['worker_result']['returncode']==0 and status['worker_result']['launched']
    assert receipt['updates_attempted']==receipt['updates_completed']==receipt['denoising_calls']==0
    for k,(examples,calls) in dict(mdm=(192,48),encoder=(72,72),caption=(12,12)).items():
        assert status['counts'][k]==receipt['counts'][k]
        for stage in ('reserved','actual','completed'):
            assert receipt['counts'][k][stage]==examples and receipt['counts'][k][stage+'_calls']==calls
    for key,frozen,idkey in (('groups',groups,'group_id'),('branches',slots,'slot_id'),('checks',specs,'check_id')):
        assert status[key]==receipt[key]
        assert [r[idkey] for r in status[key]]==[r[idkey] for r in frozen]
        assert all(r['status']=='completed' for r in status[key])
    run_index=get(METHOD/'run_artifacts.json')
    assert len(run_index)==688 and {str(p) for p in RUN.rglob('*') if p.is_file()}=={str(ROOT/a['path']) for a in run_index}
    for a in run_index:verify(a)
    assert len(receipt['artifacts'])==684 and {str(ROOT/a['path']) for a in receipt['artifacts']}=={str(p) for p in (RUN/'V').rglob('*') if p.is_file()}
    for a in receipt['artifacts']:verify(a)
    for p in (RUN/'runtime').iterdir():assert p.read_bytes()==bind(METHOD/'runtime'/p.name).read_bytes()
    for a in get(METHOD/'artifact_index.json'):verify(a)
    bind(ROOT/'outputs/itm_rework/method/state_diagnostic_authorized_20260928_v1/runtime/status.json',approval['previous_attempt']['status_sha256'])
    finite=dict(npz=0,arrays=0,elements=0)
    for p in sorted((RUN/'V').rglob('*.npz')):
        finite['npz']+=1
        for v in load(p).values():
            if v.dtype.kind in 'biufc':
                assert np.isfinite(v).all()
                finite['arrays']+=1;finite['elements']+=v.size
    assert finite==dict(npz=288,arrays=864,elements=10376160)
    caches={};input_checks=[]
    for case,prepared in zip(cases,manifest['inputs']):
        cache=get(RUN/'V/inputs'/(case['cache_id']+'.json'));assert cache['case']==case
        assert cache['arrays']==prepared['branches'] and prepared['cache_id']==case['cache_id']
        caches[case['cache_id']]=cache
        for branch,expected in zip(case['branches'],prepared['branches']):
            arrays=load(RUN/'V/inputs'/(case['cache_id']+'-'+branch['branch']+'.npz'))
            assert {k:describe(a) for k,a in arrays.items()}==expected
            for k,spec in branch['arrays'].items():assert describe(arrays[k])==spec
            input_checks.append(dict(cache_id=case['cache_id'],branch=branch['branch'],exact=True))
    branch_rows=[];vrows=[];qrows=[];anchors=[];qmap={r['group_id']:r for r in get(FROZEN/'historical_Q.json')}
    for group in groups:
        gid=group['group_id'];final=get(RUN/'V/groups'/(gid+'.json'));assert final['group']==group
        shared=load(RUN/'V/states'/(gid+'.npz'));old=load(verify(group['saved_state']))
        for key,desc in (('x_t',group['state_descriptor']),('target',group['target_descriptor']),('frame_mask',group['frame_mask'])):
            assert describe(shared[key])==desc
        assert shared['frame_mask'].all() and describe(shared['x_t'])==describe(old['x_t'])
        target=load(verify(group['target_artifact']))['target'];assert describe(target)==describe(shared['target'])
        preds={};info={k:group[k] for k in ('data_split','sensor_config','motion_id','timestep','frames')}
        for branch in BRANCHES:
            sid=gid+'-'+branch;completed_chain(status,receipt,final,sid)
            pred=load(RUN/'V/branches'/(sid+'.npz'))['predicted_x0']
            assert pred.dtype==np.float32 and pred.shape==target.shape
            meta=get(RUN/'V/branches'/(sid+'.json'));assert meta['arrays']['predicted_x0']==describe(pred)
            assert meta['group']==group and meta['slot_id']==sid and meta['branch']==branch
            assert meta['package_sha256']==PACKAGE_SHA and meta['implementation_sha256']==MANIFEST_SHA and meta['checkpoint_sha256']==CHECKPOINT_SHA
            assert Path(meta['final_group_metadata'])==RUN/'V/groups'/(gid+'.json')
            assert meta['status']=='raw_saved_pending_anchors' and meta['diagnostic_only']
            assert meta['cache_identity']==caches[group['cache_id']]['identity'] and meta['caption']==group['sample_spec']['text']
            assert meta['scaled_t']==group['scaled_t'] and meta['text_present'] and meta['imu_present']==(branch!='T')
            verify(meta['state_artifact']);verify(meta['input_artifact'])
            assert meta['input_descriptors']==caches[group['cache_id']]['arrays'][BRANCHES.index(branch)]
            preds[branch]=pred
            branch_rows.append(dict(info,group_id=gid,slot_id=sid,branch=branch,status='valid',blocks=branch_metrics(pred,target)))
        for spec in [r for r in specs if r['group_id']==gid]:
            p=preds[spec['branch']][None];reference=old[spec['reference']] if spec['reference_kind']=='saved_paired_trace' else preds['T'][None]
            delta=p.astype(float)-reference.astype(float)
            result=dict(bitwise=describe(p)==describe(reference),count=p.size,finite_count=p.size,
                max_abs=float(np.max(np.abs(delta))),rms=float(np.sqrt(np.mean(delta**2))),
                passed=bool(np.all(np.abs(delta)<=spec['atol']+spec['rtol']*np.abs(reference.astype(float)))))
            assert result['passed'];record=get(RUN/'V/checks'/(spec['check_id']+'.json'))
            assert record['atol']==record['rtol']==1e-6
            for location in (record,next(r for r in receipt['checks'] if r['check_id']==spec['check_id']),next(r for r in final['checks'] if r['check_id']==spec['check_id'])):
                for k,v in result.items():assert location[k]==v
            anchors.append(dict(check_id=spec['check_id'],**result))
        for contrast in protocol['contrasts']:
            vrows.append(dict(info,group_id=gid,family='V_actual_paired_state',replicate=group['sample_spec']['sampling_seed'],
                contrast=contrast['name'],role=contrast['role'],status='valid',blocks=contrast_metrics(preds[contrast['prediction']],preds[contrast['baseline']],target)))
        for q in qmap[gid]['existing_Q_groups']:
            qs={r['branch']:load(verify(r['artifact'])) for r in q['branches']}
            for arr in qs.values():
                assert describe(arr['target'])==describe(target)
                assert describe(arr['x_t'])==describe(qs['T']['x_t'])
            for contrast in protocol['contrasts']:
                qrows.append(dict(info,group_id=q['group_id'],V_group_id=gid,family='Q_GT_noised',replicate=q['noise_seed'],
                    contrast=contrast['name'],role=contrast['role'],status='valid',blocks=contrast_metrics(qs[contrast['prediction']]['predicted_x0'],qs[contrast['baseline']]['predicted_x0'],target)))
    assert len(branch_rows)==len(vrows)==192 and len(qrows)==384 and len(anchors)==144
    vsource=source_means(vrows,1);qsource=source_means(qrows,2);comparison=paired_state_comparison(vsource,qsource)
    assert len(vsource)==len(qsource)==len(comparison)==192
    vaggr=aggregate(vsource);qaggr=aggregate(qsource)
    # Preserve the complete-action evidence as historical context, no new semantic score.
    for name in ('B0_SAMPLING_ALIGNMENT_EVALUATION_20260928.md','b0_sampling_alignment_evaluation_20260928_v1/contrasts.json','b0_sampling_alignment_evaluation_20260928_v1/review_unmasked.json'):
        bind(ROOT/'docs/itm_rework/protocol'/name)
    for path in (Path(__file__),ROOT/'scripts/freeze_b0_state_diagnostic.py',ROOT/'tests/test_b0_state_diagnostic_evaluation.py'):
        bind(path)
    audit=dict(read_commit='260d934',status='delivery_valid_all_frozen_cases_retained',bound_files=1656,run_files=688,receipt_artifacts=684,
        elapsed_seconds=elapsed,peak_allocated_bytes=receipt['peak_allocated_bytes'],counts=receipt['counts'],finite=finite,
        runtime_copies_exact=True,old_failure_unchanged=True,software_checks=144,bitwise_checks=sum(a['bitwise'] for a in anchors),
        anchors_max_abs=max(a['max_abs'] for a in anchors),inputs_checked=48,states_checked=48,
        branch_rows=192,branch_block_cells=1728,V_contrast_rows=192,V_contrast_block_cells=1728,Q_contrast_rows=384,
        source_matched_V_Q_rows=192,failed=0,missing=0,diagnostic_only=0,
        current_model_imports=0,current_model_forwards=0,current_GPU_queries=0,current_new_samples=0,
        authorization_consumed=True,new_authorization_created=False)
    output.mkdir(parents=True)
    products={'execution_audit':audit,'anchors':anchors,'input_checks':input_checks,'branches':branch_rows,
        'V_contrasts':vrows,'Q_contrasts':qrows,'V_source_means':vsource,'Q_source_means':qsource,
        'V_aggregates':vaggr,'Q_aggregates':qaggr,'V_Q_comparison':comparison,
        'provenance':dict(sources=[sources[k] for k in sorted(sources)],protocol=protocol)}
    for name,value in products.items():write(output/(name+'.json'),value)
    # Flat all-block table preserves every source and t; no temporal pooling.
    with (output/'all_feature_source_rows.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow([*KEYS,'V_delta_mse','Q_delta_mse','V_minus_Q','V_baseline_mse','V_conditioned_mse','V_cross','V_residual_energy'])
        for r in comparison:
            a=r['blocks']['all'];writer.writerow([*(r[k] for k in KEYS),a['V']['delta_mse'],a['Q']['delta_mse'],a['V_minus_Q_delta_mse'],a['V']['baseline_mse'],a['V']['conditioned_mse'],a['V']['cross'],a['V']['residual_energy']])
    print(json.dumps(audit))


def guarded_evaluate(output):
    old=builtins.__import__
    def restrict(name,*a,**kw):
        if name.split('.')[0] in ('torch','tensorflow','jax'):raise AssertionError('Neural imports forbidden')
        return old(name,*a,**kw)
    def reject(*a,**kw):raise AssertionError('Process/GPU query forbidden')
    with ExitStack() as stack:
        stack.enter_context(patch.object(builtins,'__import__',restrict))
        for name in ('Popen','run','call','check_call','check_output'):stack.enter_context(patch.object(subprocess,name,reject))
        evaluate(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);args=p.parse_args();guarded_evaluate(args.output)
