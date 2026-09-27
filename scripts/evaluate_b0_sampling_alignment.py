#!/usr/bin/env python
"""Owner02 saved-array evaluation; prohibits neural forward/CUDA, never launches jobs."""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import math
from pathlib import Path
import subprocess
from unittest.mock import patch
import numpy as np
import torch
from evaluate_b0_correction import (read, write, sha, ah, digest, lines, stats, averages,
    metric_delta, distance, probe_aggregate, reconstruct_noise, ROOT, DATA, KEYS)
from accept_canonical_b0_handoff import reference
from itm.data.humanml_torch import recover_from_ric_torch
from itm.metrics.canonical_b0 import evaluate_motion

FROZEN=ROOT/'docs/itm_rework/protocol/b0_sampling_alignment_20260927_v1'
RUN=ROOT/'outputs/itm_rework/method/sampling_alignment_authorized_20260927_v1'
METHOD=ROOT/'docs/itm_rework/method/sampling_alignment_run_20260927_v1'
OUT=ROOT/'docs/itm_rework/protocol/b0_sampling_alignment_evaluation_20260928_v1'
IMAGES=ROOT/'outputs/itm_rework/protocol/b0_sampling_alignment_evaluation_20260928_v1/review'
ARMS=('R_static_reference_delta','S1_R_unit_guidance')
PACKAGE_SHA='537895c45695a48cb4b5114941e4bf8cab3962c40be893fcc79b4c9eeb311d60'
CHECKPOINT_SHA='91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642'


def descriptor(x):return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=ah(x))
def reduction(x):
    s=stats(x);s['sum_squares']=s.pop('sums_sq');return s

def decompose(f00,f10,f11,target):
    a=f11.astype(float)-target;b=1.5*(f10.astype(float)-f00)
    unit=float(np.mean(a*a));legacy=float(np.mean((a+b)**2))
    cross=float(2*np.mean(a*b));energy=float(np.mean(b*b))
    error=abs(legacy-unit-cross-energy)
    bound=64*np.finfo(float).eps*max(1.,unit,legacy,abs(cross),energy)
    assert error<=bound
    return dict(unit_mse=unit,legacy_reduced_mse=legacy,legacy_minus_unit=legacy-unit,
        twice_error_text_inner_product=cross,text_energy=energy,
        text_increment_rms=float(np.sqrt(energy)),decomposition_error=error,decomposition_bound=bound)

def extent_delta(a,b):
    return {k:(a[k]-b[k] if k=='pelvis_path_length_m' else (np.array(a[k])-b[k]).tolist()) for k in a}

def frozen_betas():
    # Upstream full-step SpacedDiffusion reconstructs betas from cumulative
    # alpha even with an identity timestep map; preserve that float64 roundoff.
    alpha_bar=lambda t:math.cos((t+.008)/1.008*math.pi/2)**2
    base=np.array([min(1-alpha_bar((i+1)/1000)/alpha_bar(i/1000),.999) for i in range(1000)])
    cumulative=np.cumprod(1-base)
    return 1-cumulative/np.r_[1.,cumulative[:-1]]

def contrast(a,b,pa,pb):
    return dict(case_id=a['case_id'],comparator_case_id=b['case_id'],status='valid',
        metrics=metric_delta(a['metrics'],b['metrics']),
        extent_delta=extent_delta(a['metrics']['motion_extent_m'],b['metrics']['motion_extent_m']),**distance(pa,pb))


def audit():
    status=read(RUN/'runtime/status.json');manifest=read(status['implementation']['path'])
    assert sha(status['implementation']['path'])=='793fdd9a95a102484d2dbc7579aa398f3af0120c66422ba6ec966cc01f8cb7c4'
    assert sha(FROZEN/'package.json')==PACKAGE_SHA
    checked={}
    def check(a):
        p=ROOT/a['path'];key=str(p)
        if key not in checked:checked[key]=sha(p)
        assert checked[key]==a['sha256'],key
    for a in manifest['expected_artifacts']:check(a)
    index=read(METHOD/'artifact_index.json')
    assert {str(p) for p in RUN.rglob('*') if p.is_file()}=={str(ROOT/a['path']) for a in index}
    for a in index:check(a)
    assert status['authorization']==read(ROOT/'docs/itm_rework/method/B0_SAMPLING_ALIGNMENT_AUTHORIZATION_20260927.json')
    assert status['authorization']['implementation_sha256']==status['implementation']['sha256']
    assert status['authorization']['package_sha256']==PACKAGE_SHA
    assert status['authorization']['checkpoint']['sha256']==CHECKPOINT_SHA
    assert status['authorization']['run_directory']==str(RUN)
    assert status['status']=='completed' and all(r['status']=='completed' for key in ('probes','samples','traces','stages') for r in status[key])
    assert [len(status[k]) for k in ('probes','samples','traces')]==[480,60,240]
    receipts=[];artifacts={}
    for p in sorted((RUN/'runtime').glob('*.receipt.json')):
        r=read(p);c=read(p.with_name(p.name.replace('.receipt.','.context.')))
        assert r['context']==c and r['status']=='completed' and r['checkpoint_sha256']==CHECKPOINT_SHA
        assert c['gpu_uuid']==status['authorization']['gpu_uuid'] and r['peak_allocated_bytes']<=24*1024**3
        assert r['last_monotonic']<=c['deadline_monotonic']<=status['total_deadline_monotonic']
        assert r['updates_attempted']==r['updates_completed']==0
        assert r['completed_slots']==c['expected_slots'] and r['completed_traces']==c['expected_traces']
        if r['phase']=='Q':assert (r['mdm_examples'],r['mdm_calls'],r['encoder_examples'])==(480,96,576)
        else:assert (r['mdm_examples'],r['mdm_calls'],r['encoder_examples'],r['denoising_calls'],r['branch_examples'])==(4000,1000,2,1000,4000)
        for a in r['artifacts']:check(a);artifacts[a['path']]=a
        receipts.append(r)
    assert len(receipts)==61 and sum(r['mdm_examples'] for r in receipts)==240480
    counts=Counter()
    for p in RUN.rglob('*.npz'):
        with np.load(p,allow_pickle=False) as z:
            counts['npz']+=1
            for key in z.files:
                a=z[key]
                if np.issubdtype(a.dtype,np.number):
                    assert np.isfinite(a).all(),(p,key);counts['numeric_arrays']+=1;counts['numeric_elements']+=a.size
            if p.with_suffix('.json').exists():
                m=read(p.with_suffix('.json'))
                for k,spec in m.get('arrays',{}).items():assert descriptor(z[k])==spec
                if 'metadata' in z:assert read(p.with_suffix('.json'))==__import__('json').loads(str(z['metadata']))
    elapsed=status['finished_monotonic']-status['total_started_monotonic']
    qstart=status['stages'][0]['deadline_monotonic']-600
    sstart=status['sampling_deadline_monotonic']-2400
    assert elapsed<=3600 and sstart-qstart<=600 and status['finished_monotonic']-sstart<=2400
    result=dict(read_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        bound_files=len(manifest['expected_artifacts']),run_files=len(index),receipt_artifacts=len(artifacts),
        checked_files=len(checked),receipts=61,finite_counts=dict(counts),elapsed_seconds=elapsed,
        Q_seconds=sstart-qstart,S1_seconds=status['finished_monotonic']-sstart,
        peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in receipts),
        MDM_branch_examples=240480,encoder_examples=696,updates=0,
        complete_denominators=dict(Q=480,S1=60,trace=240),new_model_forwards=0,new_GPU_queries=0)
    write(OUT/'execution_audit.json',result)
    return manifest,status


class Inputs:
    def __init__(self):
        self.design=read(FROZEN/'design.json');norm=self.design['common']['imu_normalization']
        self.mean=np.array(norm['acceleration_mean'],np.float32);self.std=np.array(norm['acceleration_std'],np.float32)
        accepted=read(ROOT/'docs/itm_rework/method/canonical_training_input_20260925_v1.json')
        self.mm=np.load(accepted['training_args']['mean']);self.ms=np.load(accepted['training_args']['std'])
        self.refs={};self.captions={};self.targets={}
        for split in ('train','val'):
            self.captions[split]={r['motion_id']:r for r in lines(DATA/f'{split}_caption_plans.jsonl')}
            for r in lines(DATA/f'{split}_imu.jsonl'):
                mid=r['motion_id'];cp=self.captions[split][mid]
                self.refs[split,mid]=reference(r,cp['frames'])[:3]
                folder=Path(r['imu_path']).parent
                self.targets[split,mid]=(np.load(folder/'joints.npy'),np.load(folder/'motion.npy'))
        train=np.concatenate([self.refs['train',mid][0].astype(float) for mid in ('000027','000506','004728')])
        assert len(train)==398
        np.testing.assert_array_equal(train.mean(0).astype(np.float32),self.mean)
        np.testing.assert_allclose(np.sqrt(np.maximum(train.var(0),1e-8)).astype(np.float32),self.std,atol=0,rtol=1e-7)
    def check(self,z,c):
        split=c.get('data_split',c.get('split'));mid=c['motion_id'];a,o,t=self.refs[split,mid];n=len(t)
        so=np.broadcast_to(o[:1],o.shape);ref=np.concatenate([(np.zeros_like(a)-self.mean)/self.std,so.reshape(n,6,9)],-1)
        ea,eo=a,o;condition=c.get('input_transform',c['condition'])
        if condition=='physical_static':ea,eo=np.zeros_like(a),so
        if condition=='tail_reversed':i=np.r_[0,np.arange(n-1,0,-1)];ea,eo=a[i],o[i]
        u=np.concatenate([(ea-self.mean)/self.std,eo.reshape(n,6,9)],-1)
        if condition=='numeric_zero':u=np.zeros_like(u)
        np.testing.assert_array_equal(z['acceleration'][0],ea)
        for k,v in [('orientation',eo),('normalized_imu',u),('normalized_reference_imu',ref)]:np.testing.assert_allclose(z[k][0],v,atol=2e-7,rtol=0)
        np.testing.assert_allclose(z['original_orientation0'],o[0],atol=2e-7,rtol=0)
        np.testing.assert_array_equal(z['timestamps_seconds'],t)
        np.testing.assert_array_equal(z['frame_mask'],np.ones((1,n),bool))
        mask=np.zeros((1,6),bool);mask[0,[4] if c['sensor_config']=='head' else [0,1]]=True
        np.testing.assert_array_equal(z['sensor_mask'],mask)
        for kind in ('text','imu'):np.testing.assert_array_equal(z[kind+'_condition_present'],[c[kind+'_present']])
        np.testing.assert_array_equal(z['gt'][0],self.targets[split,mid][0])
        assert c['text']==self.captions[split][mid]['caption']


def samples(manifest,status,inputs):
    rows=[];pred={};plans={};decode_errors=[]
    old_frozen={r['case_id']:r for r in read(ROOT/'docs/itm_rework/protocol/b0_correction_20260926_v1/sampling_slots.json')}
    cases=read(FROZEN/'sampling_slots.json');delivered={r['case_id']:r for r in status['samples']}
    expected={r['case_id']:r for r in manifest['inputs']}
    for c in cases:
        assert delivered[c['case_id']]['spec']==c and delivered[c['case_id']]['diagnostic_complete']
        oldmeta=read(ROOT/c['historical_metadata']['path'])
        for arm,path,spec in [(ARMS[0],ROOT/c['historical_output']['path'],old_frozen[c['comparison_case_id']]),
                              (ARMS[1],RUN/'samples'/(c['case_id']+'.npz'),c)]:
            m=read(path.with_suffix('.json'));cid=spec['case_id']
            assert m['control_checkpoint_sha256']==CHECKPOINT_SHA and m['random_plans']==oldmeta['random_plans']
            assert m['guidance']==dict(text_scale=2.5 if arm==ARMS[0] else 1.,imu_scale=1.,joint_scale=1.,mode='factorized',branch_execution='batched')
            assert all(m[k]==v for k,v in spec.items() if k!='status')
            assert m['sampler']=='ddpm' and m['model_forward_calls']==1000 and m['MDM_branch_examples']==4000
            if arm==ARMS[1]:assert m['package_sha256']==PACKAGE_SHA and len(m['trace_ids'])==4
            with np.load(path,allow_pickle=False) as z:
                inputs.check(z,c);p=z['motion'][0]
                assert z['motion'].shape==(1,c['frames'],22,3) and float(z['fps'])==20
                assert z['normalized_humanml263'].shape==(1,263,1,c['frames'])
                if arm==ARMS[1]:
                    assert {k:descriptor(z[k]) for k in expected[cid]['arrays']}==expected[cid]['arrays']
                decoded=recover_from_ric_torch(torch.from_numpy(z['normalized_humanml263'][0,:,0,:].T*inputs.ms+inputs.mm)).numpy()
                np.testing.assert_allclose(decoded,p,atol=2e-6,rtol=1e-6);decode_errors.append(float(np.abs(decoded-p).max()))
            plan=m['random_plans'][0];group=c['random_plan_case_id']
            assert digest(plan)==c['historical_random_plan_sha256']
            if group in plans:assert plan==plans[group]
            else:
                assert [d['step'] for d in plan['noise_draws']]==[-1]+list(range(999,0,-1))
                assert plan['timestep_map']==list(range(1000)) and not plan['clip_denoised']
                for d in plan['noise_draws']:
                    assert d['case_id']==group and d['sample_seed']==20260926 and d['shape']==[1,263,1,c['frames']]
                    reconstruct_noise(d)
                assert digest(plan['noise_draws'])==plan['random_plan_id'];plans[group]=plan
            gt,features=inputs.targets[c['data_split'],c['motion_id']]
            contacts=features[:,259:263] if c['data_split']=='val' and c['motion_id'] in ('000433','000382') else None
            metrics=evaluate_motion(p,gt,c['sensor_config'],contacts)
            if c['data_split']=='train':metrics['foot_status']='not_applicable_train_floor_admission_not_frozen'
            rows.append(dict(case_id=cid,arm=arm,**{k:c[k] for k in ('data_split','motion_id','sensor_config','condition','frames')},
                status='valid',metrics=metrics,decode_max_abs_m=decode_errors[-1],random_plan_id=plan['random_plan_id']))
            pred[cid]=p
    index={(r['arm'],r['data_split'],r['motion_id'],r['sensor_config'],r['condition']):r for r in rows}
    within=[];between=[];identities=[]
    for key,a in index.items():
        arm,split,mid,config,condition=key;group=dict(arm=arm,data_split=split,motion_id=mid,sensor_config=config)
        def pair(b):return contrast(a,b,pred[a['case_id']],pred[b['case_id']])
        if condition=='paired':
            for base in ['text_only','physical_static']+(['tail_reversed','numeric_zero'] if split=='val' else []):
                within.append(dict(group,comparator=base,**pair(index[arm,split,mid,config,base])))
        if arm==ARMS[1]:
            between.append(dict(group,condition=condition,**pair(index[ARMS[0],split,mid,config,condition])))
            if condition in ('physical_static','backbone_bypass','imu_only'):
                b=index[ARMS[0],split,mid,config,condition] if condition=='imu_only' else index[arm,split,mid,config,'text_only']
                d=pair(b);delta=pred[a['case_id']].astype(float)-pred[b['case_id']]
                d.update(rms_m=float(np.sqrt(np.mean(delta**2))),count=delta.size,passed=d['max_abs_m']<=1e-5)
                assert d['passed'];identities.append(dict(group,condition=condition,**d))
    wi={(r['arm'],r['data_split'],r['motion_id'],r['sensor_config'],r['comparator']):r for r in within};did=[]
    for key,a in wi.items():
        if key[0]!=ARMS[1]:continue
        b=wi[(ARMS[0],*key[1:])]
        did.append(dict(data_split=key[1],motion_id=key[2],sensor_config=key[3],comparator=key[4],status='valid',
            metrics=metric_delta(a['metrics'],b['metrics']),extent_delta=extent_delta(a['extent_delta'],b['extent_delta']),
            output_distance_difference_m=a['mean_joint_distance_m']-b['mean_joint_distance_m']))
    assert (len(rows),len(within),len(between),len(did),len(identities))==(120,72,60,36,24)
    report=dict(planned_new=60,valid_new=60,planned_historical=60,valid_historical=60,rows=rows,
        denominators=[dict(arm=arm,data_split=s,planned=42 if s=='val' else 18,valid=sum(r['arm']==arm and r['data_split']==s for r in rows)) for arm in ARMS for s in ('train','val')],
        aggregates=averages(rows,('arm','data_split','sensor_config','condition')),max_decode_error_m=max(decode_errors),
        normalization_population=398,noise_groups=len(plans),noise_draws_replayed=len(plans)*1000)
    write(OUT/'samples.json',report)
    write(OUT/'contrasts.json',dict(within=within,S1_minus_oldR=between,difference_of_contrasts=did,identities=identities,
        within_aggregates=averages(within,('arm','data_split','sensor_config','comparator')),
        between_aggregates=averages(between,('data_split','sensor_config','condition')),
        difference_aggregates=averages(did,('data_split','sensor_config','comparator'))))
    with (OUT/'per_case.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case_id',*KEYS]);w.writeheader()
        for r in rows:w.writerow(dict(case_id=r['case_id'],**{k:r['metrics'][k] for k in KEYS}))
    return report


def probes(manifest,inputs):
    slots=read(FROZEN/'probe_slots.json');groups=read(FROZEN/'probe_groups.json');noises=read(FROZEN/'probe_noise.json')
    expected={r['probe_id']:r for r in manifest['Q_inputs']};arrays={};report=[];qerr=[]
    # Frozen upstream cosine schedule, evaluated directly without importing diffusion/model code.
    betas=frozen_betas()
    cum=np.cumprod(1-betas)
    for slot in slots:
        pid=slot['probe_id'];p=RUN/'Q'/(pid+'.npz');m=read(p.with_suffix('.json'))
        assert all(m[k]==v for k,v in slot.items() if k!='status')
        assert m['checkpoint_sha256']==CHECKPOINT_SHA and m['package_sha256']==PACKAGE_SHA
        assert m['noise']==noises[slot['noise_key']] and m['noise']['padding_noise'] is None
        assert m['scaled_t']==[slot['timestep']] and m['diffusion']['betas_sha256']==ah(betas)
        assert sha(m['input_artifact']['path'])==m['input_artifact']['sha256']
        with np.load(m['input_artifact']['path'],allow_pickle=False) as z:
            inputs.check(z,slot)
            assert {k:descriptor(z[k]) for k in z.files}==expected[pid]['arrays']==m['input_hashes']
        with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        assert set(a)=={'predicted_x0','target','x_t','frame_mask'} and a['frame_mask'].all()
        for k,v in a.items():assert reduction(v)==m['reductions'][k]
        target=((inputs.targets[slot['split'],slot['motion_id']][1]-inputs.mm)/inputs.ms).astype(np.float32).T[:,None,:]
        np.testing.assert_array_equal(target,a['target'])
        eps=reconstruct_noise(m['noise']['valid_noise'])[0];assert ah(eps)==m['actual_noise_sha256']
        alpha=cum[slot['timestep']];xt=np.float32(np.sqrt(alpha))*target+np.float32(np.sqrt(1-alpha))*eps
        np.testing.assert_allclose(xt,a['x_t'],atol=1e-6,rtol=1e-6);qerr.append(float(np.abs(xt-a['x_t']).max()))
        mse=float(np.mean((a['predicted_x0'].astype(float)-target)**2));assert mse==m['x0_masked_mse']
        arrays[pid]=a
        report.append(dict(**{k:slot[k] for k in ('probe_id','split','motion_id','sensor_config','condition','branch','timestep','noise_seed')},values={'x0_mse':mse}))
    derived=[];within=[];did=[];static=[];stored=read(RUN/'Q/derived.json')
    stored_d={(r['group_id'],r['condition']):r for r in stored['conditions']}
    stored_w={(r['group_id'],r['guidance'],r['comparator']):r for r in stored['within']}
    stored_i={(r['group_id'],r['comparator']):r for r in stored['difference_of_differences']}
    for g in groups:
        aa=[arrays[p] for p in g['probe_ids']];pred=[a['predicted_x0'] for a in aa];target=aa[0]['target']
        for a in aa:np.testing.assert_array_equal(a['target'],target);np.testing.assert_array_equal(a['x_t'],aa[0]['x_t'])
        np.testing.assert_allclose(pred[3],pred[1],atol=1e-6,rtol=1e-6)
        static.append(bool(np.array_equal(pred[3],pred[1])))
        base={k:g[k] for k in ('group_id','data_split','motion_id','sensor_config','timestep','noise_seed')};analyses={}
        for i,c in ((2,'paired'),(3,'physical_static'),(4,'tail_reversed')):
            d=decompose(pred[0],pred[1],pred[i],target);analyses[c]=d
            original=stored_d[g['group_id'],c]
            for k,v in d.items():
                if k in original:np.testing.assert_allclose(v,original[k],atol=2e-15,rtol=1e-12)
            derived.append(dict(base,condition=c,values=d))
        for c,i in (('physical_static',3),('tail_reversed',4)):
            values={}
            for guidance,metric in [('unit','unit_mse'),('legacy_reduced','legacy_reduced_mse')]:
                dm=analyses['paired'][metric]-analyses[c][metric];values[guidance]=dm
                raw=reduction(pred[2].astype(float)-pred[i])
                assert dm==stored_w[g['group_id'],guidance,c]['paired_minus_comparator_mse']
                assert raw==stored_w[g['group_id'],guidance,c]['prediction_difference']
                within.append(dict(base,guidance=guidance,comparator=c,values=dict(paired_minus_comparator_mse=dm,prediction_difference_rms=raw['rms'])))
            delta=values['unit']-values['legacy_reduced'];assert delta==stored_i[g['group_id'],c]['unit_minus_legacy_contrast']
            did.append(dict(base,comparator=c,values=dict(unit_minus_legacy_contrast=delta)))
    assert (len(derived),len(within),len(did))==(288,384,192)
    write(OUT/'probes.json',dict(planned=480,valid=480,rows=report,max_q_sample_error=max(qerr),groups=96,
        noise_addresses_replayed=96,static_text_bitwise_groups=sum(static),
        aggregates=probe_aggregate(report,('split','sensor_config','timestep','branch')),
        derived=derived,derived_aggregates=probe_aggregate(derived,('data_split','sensor_config','timestep','condition'))))
    write(OUT/'probe_contrasts.json',dict(within=within,difference_of_contrasts=did,
        within_aggregates=probe_aggregate(within,('data_split','sensor_config','timestep','guidance','comparator')),
        difference_aggregates=probe_aggregate(did,('data_split','sensor_config','timestep','comparator'))))


def traces(manifest):
    cases={r['case_id']:r for r in read(FROZEN/'sampling_slots.json')};inputs={r['case_id']:r['arrays'] for r in manifest['inputs']}
    rows=[];u=2.**-24;gamma=16*u/(1-16*u)
    for s in read(FROZEN/'trace_slots.json'):
        p=RUN/'traces'/(s['trace_id']+'.npz');m=read(p.with_suffix('.json'));c=cases[s['case_id']]
        assert all(m[k]==v for k,v in s.items() if k!='status') and m['status']=='completed'
        assert m['checkpoint_sha256']==CHECKPOINT_SHA and m['package_sha256']==PACKAGE_SHA and m['scaled_t']==[s['timestep']]
        assert m['input_hashes']==inputs[s['case_id']] and m['guidance']==c['guidance']
        assert m['random_plan_sha256']==c['historical_random_plan_sha256']
        with np.load(p,allow_pickle=False) as z:
            a={k:z[k] for k in z.files}
        assert ah(a['actual_combined_prediction'])==m['actual_return_sha256']
        for k in ('f00','f10','f01','f11','x_t','actual_combined_prediction'):
            assert list(a[k].shape)==s['shape'] and a[k].dtype==np.float32
        assert a['frame_mask'].all()
        f00,f10,f01,f11=[a[k].astype(float) for k in ('f00','f10','f01','f11')]
        actual=a['actual_combined_prediction'].astype(float)
        full=f00+(f10-f00)+(f01-f00)+(f11-f10-f01+f00)
        scale=np.abs(f00)+(np.abs(f10)+np.abs(f00))+(np.abs(f01)+np.abs(f00))+(np.abs(f11)+np.abs(f10)+np.abs(f01)+np.abs(f00))
        bound=gamma*np.maximum(1,scale);ratio=max(float(np.max(np.abs(actual-ref)/bound)) for ref in (full,f11));assert ratio<=1
        pairs=[]
        if not c['text_present']:pairs += [('f00','f10'),('f01','f11')]
        if not c['imu_present'] or c['condition'] in ('physical_static','backbone_bypass'):pairs += [('f00','f01'),('f10','f11')]
        for x,y in pairs:np.testing.assert_allclose(a[x],a[y],atol=1e-6,rtol=1e-6)
        delta=1.5*(f10-f00)
        rows.append(dict(trace_id=s['trace_id'],case_id=s['case_id'],**{k:c[k] for k in ('data_split','motion_id','sensor_config','condition')},timestep=s['timestep'],
            values=dict(text_increment_rms=stats(delta)['rms'],text_increment_max_abs=stats(delta)['max_abs'],
                actual_combination_rms=stats(actual)['rms'],unit_f11_rms=stats(f11)['rms'],
                legacy_local_combination_rms=stats(f11+delta)['rms'],combination_max_error_over_bound=ratio),
            absent_branch_checks=len(pairs)))
    write(OUT/'traces.json',dict(planned=240,valid=len(rows),rows=rows,
        aggregates=probe_aggregate(rows,('data_split','sensor_config','condition','timestep')),
        interpretation='Float64 legacy recombination at actual S1 states only; no legacy trajectory or counterfactual rollout; no target MSE asserted.'))


def evaluate():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'samples.json').exists(),'Preserve existing evaluation'
    with ExitStack() as stack:
        def forbidden(*a,**kw):raise AssertionError('Evaluation attempted neural forward or CUDA')
        stack.enter_context(patch.object(torch.nn.Module,'_call_impl',forbidden))
        for name in ('init','_lazy_init','is_available','device_count','get_device_properties','synchronize','current_device','set_device'):
            stack.enter_context(patch.object(torch.cuda,name,forbidden))
        manifest,status=audit();print('execution audit passed',flush=True)
        inputs=Inputs();samples(manifest,status,inputs);print('120 samples and contrasts passed',flush=True)
        probes(manifest,inputs);print('480 Q and 864 derived/contrast rows passed',flush=True)
        traces(manifest);print('240 traces passed',flush=True)
    write(OUT/'provenance.json',dict(code_sha256=sha(__file__),read_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        package_sha256=PACKAGE_SHA,checkpoint_sha256=CHECKPOINT_SHA,neural_forward_and_CUDA_prohibited=True,
        current_model_forwards=0,current_training_updates=0,current_diffusion_samples=0,current_GPU_queries=0))


def render():
    """No metric/condition labels in images; order was frozen before generation."""
    OUT.mkdir(parents=True,exist_ok=True);IMAGES.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    rows=read(FROZEN/'review_order.json')['rows'];cases={r['case_id']:r for r in rows}
    new={r['case_id']:RUN/'samples'/(r['case_id']+'.npz') for r in read(FROZEN/'sampling_slots.json')}
    old={r['comparison_case_id']:ROOT/r['historical_output']['path'] for r in read(FROZEN/'sampling_slots.json')}
    paths=dict(old,**new)
    actions={'000433':'kick, return to standing','000382':'clap, lower arms','000139':'stand/sway, then walk',
             '000506':'alternating forward lunges/squats','004728':'RIGHT leg back lift, hold, lower','000027':'punch, then turn'}
    chains=[([0,1,4,7,10],'#1479c4'),([0,2,5,8,11],'#d44a38'),([0,3,6,9,12,15],'#555555'),([9,13,16,18,20],'#1479c4'),([9,14,17,19,21],'#d44a38')]
    artifacts=[]
    for page in range(20):
        p=IMAGES/f'page_{page+1:02}.png'
        if p.exists():
            artifacts.append(dict(path=str(p),sha256=sha(p)));continue
        fig=plt.figure(figsize=(24,18));gs=GridSpec(6,13,figure=fig,hspace=.65,wspace=.65)
        for row,item in enumerate(rows[page*6:page*6+6]):
            c=cases[item['case_id']]
            with np.load(paths[c['case_id']]) as z:p=z['motion'][0]
            t=np.arange(len(p))/20;centered=p.copy();centered[:,:,0]-=p[:,0:1,0];centered[:,:,2]-=p[:,0:1,2]
            for col,f in enumerate(np.linspace(0,len(p)-1,10).round().astype(int)):
                ax=fig.add_subplot(gs[row,col]);pose=centered[f]
                for ids,color in chains:ax.plot(pose[ids,0]+.55*pose[ids,2],pose[ids,1]+.1*pose[ids,2],color=color,lw=1.5)
                ax.set(xlim=(-1.4,1.4),ylim=(-.35,2.3));ax.set_aspect('equal');ax.axis('off');ax.set_title(f'{f/20:.2f}s',fontsize=8)
                if col==0:ax.text(0,1.25,item['review_id']+' | '+actions[c['motion_id']],transform=ax.transAxes,fontsize=10,weight='bold')
            ax=fig.add_subplot(gs[row,10]);ax.plot(t,np.linalg.norm(p[:,20]-p[:,21],axis=-1),label='wrist sep');ax.plot(t,p[:,20,1]-p[:,0,1],label='L wrist-pelvis');ax.plot(t,p[:,21,1]-p[:,0,1],label='R wrist-pelvis');ax.set_title('Hands (m)',fontsize=9);ax.legend(fontsize=5)
            ax=fig.add_subplot(gs[row,11]);ax.plot(t,p[:,7,1],label='L ankle');ax.plot(t,p[:,8,1],label='R ankle');ax.plot(t,p[:,0,1],label='pelvis');ax.set_title('Height (m)',fontsize=9);ax.legend(fontsize=5)
            ax=fig.add_subplot(gs[row,12]);ax.plot(p[:,0,0],p[:,0,2]);ax.scatter(*p[0,0,[0,2]],s=15,c='green');ax.scatter(*p[-1,0,[0,2]],s=15,c='red');ax.set_title('Root X/Z (m)',fontsize=9);ax.set_aspect('equal',adjustable='datalim')
        fig.suptitle(f'Frozen-order guidance review {page+1}/20 | L blue / R red | root XZ centered for display only | full-frame traces',fontsize=12)
        p=IMAGES/f'page_{page+1:02}.png';fig.savefig(p,dpi=100,bbox_inches='tight');plt.close(fig)
        artifacts.append(dict(path=str(p),sha256=sha(p)));print(f'rendered {page+1}/20',flush=True)
    write(OUT/'review_rendering.json',dict(images=artifacts,order_sha256=sha(FROZEN/'review_order.json'),
        scope='Arm/config/condition hidden on figures. Single AI reviewer; snapshots plus all-frame traces. Historical results known and first mapping row seen during schema discovery; not a fully blinded independent human assessment. Notes recorded before bulk key decoding.'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['evaluate','render'])
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--images',type=Path,default=IMAGES)
    args=parser.parse_args();OUT=args.output;IMAGES=args.images
    {'evaluate':evaluate,'render':render}[args.mode]()
