#!/usr/bin/env python
"""Independent-formula NumPy replay of the frozen plan, not an independent review."""
import argparse
import builtins
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'docs/itm_rework/protocol/b0_training_sufficiency_20261008_v2/package.json'
SHA = '471376c4799b80975c09faee4a80e2adfa220a6119a73a325761ed2a89029593'
SNAPSHOTS = ['I','F050','F100','F150','F200','U200']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def verified(entry):
    path = ROOT/entry['path']
    if sha(path) != entry['sha256']:
        raise AssertionError('Hash changed: '+str(path))
    return path


def arrays(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files if k!='metadata'}


def descriptor(a):
    return dict(shape=list(a.shape),dtype=str(a.dtype),
                sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())


def close(new,reference):
    if new is None or reference is None:
        assert new is reference
    else:
        assert abs(new-reference) <= 64*np.finfo(float).eps*max(1.,abs(new),abs(reference)), (new,reference)


def verify(package=PACKAGE,rebuild=None):
    package = Path(package)
    assert sha(package)==SHA
    files = {Path(a['path']).stem:verified(a) for a in read(package)['artifacts']}
    sources = read(files['sources'])
    for entry in sources:
        verified(entry)
    assert len({str((ROOT/a['path']).resolve()) for a in sources})==len(sources)==2533
    checkpoints = read(files['checkpoints'])
    assert [r['snapshot_id'] for r in checkpoints]==SNAPSHOTS
    assert len({r['artifact']['sha256'] for r in checkpoints})==6
    assert len({json.dumps(r['tensor_hashes'],sort_keys=True) for r in checkpoints})==6
    assert checkpoints[0]['raw_control_config']['reference_mode']=='none'
    assert all(r['runtime_control_config']['reference_mode']=='static_delta' for r in checkpoints)
    for row in checkpoints:
        verified(row['artifact'])
    states, jobs, checks = (read(files[name]) for name in ('states','jobs','software_checks'))
    assert len(states)==192 and len(jobs)==1152 and len(checks)==2880
    assert Counter(r['state_family'] for r in states)==dict(GT_noised_Q=96,U_paired_trace=48,F_paired_trace=48)
    assert Counter(r['data_split'] for r in states)==dict(train=96,val=96)
    cases = {r['cache_id']:r for r in read(files['input_cases'])}
    common = read(files['design'])['common']
    mean,std = [np.load(verified(a)).astype(np.float32) for a in common['motion_normalization']]
    loaded = {}
    for row in states:
        raw = arrays(verified(row['saved_state']))
        target = arrays(verified(row['target_artifact']))['target']
        xt,mask = raw['x_t'],raw['frame_mask']
        if row['state_family']=='GT_noised_Q':
            xt,mask=xt[None],mask[None]
        values=dict(x_t=xt,target=target,frame_mask=mask)
        assert {k:descriptor(a) for k,a in values.items()}==row['arrays']
        assert xt.shape==(1,263,1,row['frames']) and xt.dtype==np.float32
        assert mask.shape==(1,row['frames']) and mask.dtype==np.bool_ and mask.all()
        case=cases[row['cache_id']]
        features=np.load(Path(case['sample_spec']['gt_path']).with_name('motion.npy'))
        np.testing.assert_array_equal(target,((features-mean)/std).astype(np.float32).T[:,None])
        assert row['planned_checkpoints']==SNAPSHOTS
        loaded[row['state_id']]=(row,raw,target)
    seen_text=set(); seen_control=set()
    for j in jobs:
        row=loaded[j['state_id']][0]; key=(j['snapshot_id'],j['cache_id'])
        assert j['state_sha256']==row['arrays']['x_t']['sha256']
        assert j['target_sha256']==row['arrays']['target']['sha256']
        assert j['branch_order']==['T','P','S','R']
        assert j['input_batch_shape']==[4,263,1,row['frames']]
        assert j['scaled_t']==[row['timestep']]*4
        assert j['encoder_calls']==(0 if key in seen_control else 6)
        assert j['caption_calls']==(0 if j['cache_id'] in seen_text else 1)
        seen_text.add(j['cache_id']);seen_control.add(key)
    assert len(seen_text)==12 and len(seen_control)==72
    assert Counter(j['state_id'] for j in jobs)=={r['state_id']:6 for r in states}
    assert sum(j['encoder_calls'] for j in jobs)==432
    assert sum(j['caption_calls'] for j in jobs)==12
    assert sum(j['MDM_calls'] for j in jobs)==1152
    assert sum(j['MDM_examples'] for j in jobs)==4608
    assert len({c['check_id'] for c in checks})==2880
    kinds=Counter()
    for c in checks:
        assert c['atol']==c['rtol']==1e-6
        ref=c['reference'];kinds[ref['kind']]+=1
        if ref['kind']=='new_prediction':
            assert c['snapshot_id']!='I' and c['branch']==ref['branch']=='T'
            assert ref['job_id']=='I-'+c['state_id']
        elif ref['kind']=='same_job':
            assert ref['branch']=='T'
            assert c['branch']=='S' or (c['snapshot_id']=='I' and c['branch']=='P')
        else:
            assert ref['kind']=='saved_array' and ref['snapshot_id']==c['snapshot_id']
            raw=arrays(verified(ref['artifact']))
            assert descriptor(raw[ref['array']])==ref['descriptor']
            assert c['branch']==ref['branch']
    assert kinds==dict(same_job=1344,new_prediction=960,saved_array=576)
    # Recompute the descriptive baseline without importing its implementation.
    baseline=read(files['existing_trace_temporal_baseline'])['rows']
    views=read(files['evaluation'])['feature_views']; cells=0
    for entry in baseline:
        row,raw,target=loaded[entry['state_id']]
        p=raw['f11'].reshape(263,row['frames']).T.astype(float)
        g=target.reshape(263,row['frames']).T.astype(float)
        pm,gm=p.mean(0),g.mean(0);pc,gc=p-pm,g-gm
        recomputed={}
        for name,ids in views.items():
            actual=dict(mse=float(np.mean((p[:,ids]-g[:,ids])**2)),
                        mean_error=float(np.mean((pm[ids]-gm[ids])**2)),
                        temporal_error=float(np.mean((pc[:,ids]-gc[:,ids])**2)),
                        prediction_temporal_energy=float(np.mean(pc[:,ids]**2)),
                        target_temporal_energy=float(np.mean(gc[:,ids]**2)),
                        temporal_covariance=float(np.mean(pc[:,ids]*gc[:,ids])))
            actual['target_mean_baseline_mse']=actual['target_temporal_energy']
            actual['temporal_energy_ratio']=actual['prediction_temporal_energy']/actual['target_temporal_energy'] if actual['target_temporal_energy'] else None
            actual['prediction_difference_rms']=float(np.sqrt(np.mean(np.diff(p[:,ids],axis=0)**2))) if len(p)>1 else None
            actual['target_difference_rms']=float(np.sqrt(np.mean(np.diff(g[:,ids],axis=0)**2))) if len(p)>1 else None
            for field,value in actual.items():
                close(value,entry['temporal'][name][field])
            report=entry['temporal'][name]
            assert report['identity_error']<=report['identity_bound']
            close(actual['mse'],actual['mean_error']+actual['temporal_error'])
            recomputed[name]=actual;cells+=1
        for field in entry['temporal']['balanced']:
            close(.5*recomputed['root_ric67'][field]+.5*recomputed['remaining196'][field],entry['temporal']['balanced'][field])
        cells+=1
    assert len(baseline)==96 and cells==1152
    rebuilt_files=None
    if rebuild is not None:
        original={p.name:sha(p) for p in package.parent.iterdir() if p.is_file()}
        regenerated={p.name:sha(p) for p in Path(rebuild).iterdir() if p.is_file()}
        assert original==regenerated
        rebuilt_files=len(original)
    return dict(version='itm-training-sufficiency-NumPy-verification-v1',package_sha256=SHA,
                source_files_verified=len(sources),snapshots=6,states_verified=192,
                jobs_verified=1152,checks_verified=2880,baseline_rows_recomputed=96,
                baseline_view_cells_recomputed=cells,rebuilt_files_identical=rebuilt_files,
                model_imports=0,checkpoint_deserializations=0,model_forwards=0,GPU_queries=0,
                interpretation='Main-session independent-formula self-check/replay, not independent reviewer acceptance or model-effect evidence.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,default=PACKAGE)
    p.add_argument('--rebuild',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    original=builtins.__import__
    def restricted(name,*a,**kw):
        if name.split('.')[0] in ('torch','tensorflow','jax'):
            raise AssertionError('Neural import forbidden in NumPy replay')
        return original(name,*a,**kw)
    def reject(*a,**kw):raise AssertionError('Process/GPU command forbidden in NumPy replay')
    with ExitStack() as stack:
        stack.enter_context(patch.object(builtins,'__import__',restricted))
        for name in ('Popen','run','call','check_call','check_output'):
            stack.enter_context(patch.object(subprocess,name,reject))
        result=verify(args.package,args.rebuild)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(result))
