#!/usr/bin/env python
"""Independent NumPy formulas over all saved native diagnostic outputs."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/itm_rework/method/training_sufficiency_native_authorized_20261008_v1'
EV=ROOT/'outputs/itm_rework/protocol/training_sufficiency_native_evaluation_20261008_v1'
PKG=ROOT/'docs/itm_rework/protocol/b0_training_sufficiency_20261008_v4'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def verify(a):
    p=ROOT/a['path'];assert sha(p)==a['sha256'],p
    return p
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def desc(a):return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
def matrix(a):return a.reshape(263,-1).T.astype(float)
def close(a,b,scale=1.):
    if a is None or b is None:assert a is b;return
    assert abs(a-b)<=64*np.finfo(float).eps*max(1.,abs(a),abs(b),scale),(a,b)


def main():
    status=read(RUN/'runtime/status.json');receipt=read(RUN/'runtime/diagnostic.receipt.json');assert status['status']==receipt['status']=='completed'
    impl=read(verify(status['implementation']))
    for a in impl['expected_artifacts']:verify(a)
    for a in receipt['artifacts']:verify(a)
    assert len(receipt['artifacts'])==3300
    for key,n in (('states',192),('jobs',1152),('branches',4608),('checks',2880)):
        assert len(status[key])==n and all(r['status']=='completed' for r in status[key])
    for kind,examples,calls in (('mdm',5184,1152),('encoder',432,432),('caption',72,24)):
        c=status['counts'][kind];assert all(c[k]==examples for k in ('reserved','actual','completed'));assert all(c[k]==calls for k in ('reserved_calls','actual_calls','completed_calls'))
    assert not any(receipt[k] for k in ('updates_attempted','updates_completed','denoising_calls'))
    rules=read(PKG/'evaluation.json');states=read(PKG/'states.json');jobs=read(PKG/'jobs.json');checks=read(PKG/'software_checks.json')
    state_map={r['state_id']:r for r in states};targets={};predictions={};raw={}
    for state in states:
        z=arrays(RUN/'diagnostic/states'/(state['state_id']+'.npz'))
        assert {k:desc(v) for k,v in z.items()}==state['arrays'];assert z['frame_mask'].all()
        targets[state['state_id']]=matrix(z['target'])
    for job in jobs:
        z=arrays(RUN/'diagnostic/predictions'/(job['job_id']+'.npz'));meta=read(RUN/'diagnostic/jobs'/(job['job_id']+'.json'))
        assert meta['status']=='completed' and {k:desc(v) for k,v in z.items()}==meta['arrays']
        raw[job['job_id']]=z
        for b in ('T','P','S','R'):predictions[job['job_id']+'-'+b]=matrix(z[b])
    reports={r['check_id']:r for j in jobs for r in read(RUN/'diagnostic/jobs'/(j['job_id']+'.json'))['checks']}
    for c in checks:
        a=raw[c['job_id']][c['branch']];ref=c['reference']
        if ref['kind']=='same_job':b=raw[c['job_id']][ref['branch']]
        elif ref['kind']=='new_prediction':b=raw[ref['job_id']][ref['branch']]
        else:b=arrays(verify(ref['artifact']))[ref['array']].reshape(a.shape)
        d=a.astype(float)-b.astype(float);assert np.all(np.abs(d)<=1e-6+1e-6*np.abs(b.astype(float)))
        assert reports[c['check_id']]['passed'];close(float(np.abs(d).max()),reports[c['check_id']]['max_abs']);close(float(np.sqrt(np.mean(d*d))),reports[c['check_id']]['rms'])
    rows=read(EV/'branches.json');assert len(rows)==4608 and all(r['status']=='valid' for r in rows)
    views=rules['feature_views'];cells=0
    for row in rows:
        p=predictions[row['row_id']];g=targets[row['state_id']];pm,gm=p.mean(0),g.mean(0);pc,gc=p-pm,g-gm
        for name,ids in views.items():
            r=row['temporal'][name];mse=float(np.mean((p[:,ids]-g[:,ids])**2));mean=float(np.mean((pm[ids]-gm[ids])**2));temporal=float(np.mean((pc[:,ids]-gc[:,ids])**2));pe=float(np.mean(pc[:,ids]**2));ge=float(np.mean(gc[:,ids]**2));cov=float(np.mean(pc[:,ids]*gc[:,ids]))
            for field,val in dict(mse=mse,mean_error=mean,temporal_error=temporal,prediction_temporal_energy=pe,target_temporal_energy=ge,temporal_covariance=cov,target_mean_baseline_mse=ge,temporal_energy_ratio=pe/ge if ge else None).items():close(val,r[field])
            close(mse,mean+temporal);close(temporal,pe+ge-2*cov);cells+=1
        for field,value in row['temporal']['balanced'].items():close(value,.5*row['temporal']['root_ric67'][field]+.5*row['temporal']['remaining196'][field])
    contrasts=read(EV/'within_contrasts.json')+read(EV/'checkpoint_contrasts.json');assert len(contrasts)==9216
    for row in contrasts:
        assert row['status']=='valid';p=predictions[row['prediction_id']];b=predictions[row['reference_id']];g=targets[row['state_id']];e=b-g;d=p-b
        for name,ids in views.items():
            old=float(np.mean(e[:,ids]**2));new=float(np.mean((p[:,ids]-g[:,ids])**2));cross=float(2*np.mean(e[:,ids]*d[:,ids]));energy=float(np.mean(d[:,ids]**2));r=row['delta'][name]
            for field,val in dict(baseline_mse=old,prediction_mse=new,delta_mse=new-old,cross=cross,residual_energy=energy,prediction_difference_rms=energy**.5,prediction_difference_max_abs=float(np.abs(d[:,ids]).max())).items():close(val,r[field],max(old,new))
            close(new-old,cross+energy,max(old,new));cells+=1
    # Audit auxiliary f00 against every historical Q f00; no new forward/criterion.
    qmap={g['group_id']:g for g in read(ROOT/'docs/itm_rework/protocol/b0_sampling_alignment_20260927_v1/probe_groups.json')}
    aux=0
    for job in jobs:
        if not job['auxiliary_f00']:continue
        state=state_map[job['state_id']];pid=qmap[state['original_identity']]['probe_ids'][0]
        old=arrays(ROOT/'outputs/itm_rework/method/sampling_alignment_authorized_20260927_v1/Q'/(pid+'.npz'))['predicted_x0']
        new=arrays(RUN/'diagnostic/auxiliary'/(job['job_id']+'.npz'))['f00']
        assert np.all(np.abs(new.astype(float)-old)<=1e-6+1e-6*np.abs(old));aux+=1
    for a in read(EV/'artifact_index.json'):verify(a)
    result=dict(status='passed',source_files_verified=len(impl['expected_artifacts']),run_artifacts_verified=3300,software_checks_recomputed=2880,
                temporal_rows_recomputed=4608,contrast_rows_recomputed=9216,view_cells_recomputed=cells,auxiliary_f00_checks=aux,
                model_forwards=0,GPU_queries=0,interpretation='Independent-formula main-session NumPy self-check, not independent reviewer acceptance.')
    out=ROOT/'outputs/itm_rework/method/training_sufficiency_native_implementation_20261008_v1/result_verification.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
