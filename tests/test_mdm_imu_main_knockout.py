"""Synthetic full suffix, native-layout replay and bounded-stop CPU acceptance."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from itm.data.random_plan import sampling_noise
from itm.models.mdm_correction import artifact,read
from itm.models.mdm_imu_control import make_text_imu_guidance_model
from itm.models.mdm_sampling import sample_addressed_ddpm
from itm.models.mdm_sampling_alignment import numpy
from itm.models.mdm_imu_main_knockout import SuffixPackage,SuffixGuard,PACKAGE_SHA,CAPS,CALL_CAPS,ledger,read_receipt,replay_suffix,required_artifacts
from test_mdm_prefix_mechanism import script,Model as BaseModel,Encoder,Diffusion,toy_item

class Model(BaseModel):
    def forward(self,x,t,y=None):
        result=super().forward(x,t,y=y)
        control=y['imu_control']*y['imu_condition_present'][:,None,None]
        delta=self.adapter(control).transpose(1,2)[:,:,None]
        text=y['text_embed'].mean((0,2))[:,None,None,None]
        return result+.003*text*delta


def decode(features):return features[:,:66,0,:].permute(0,2,1).reshape(1,3,22,3).clone()


def fixture_package(tmp):
    real=SuffixPackage();paths=deepcopy(real.trajectories);noise={};anchors=[];weight=None;expected={}
    for trajectory in paths:trajectory['frames']=3
    def inputs(t):
        row=deepcopy(real.slots[t['input_slot_id']]);row['frames']=3
        return toy_item(row['text']),row
    for first in paths:
        cid=first['panel_case_id'];model=Model().eval();enc=Encoder().eval()
        with torch.no_grad():model.adapter.weight.fill_(1.)
        weight=dict(imu_encoder=deepcopy(enc.state_dict()),adapters=deepcopy(model.adapter.state_dict()))
        item,row=inputs(first);_,y=script('run_mdm_correction_worker').batch_inputs([item],[row],'cpu')
        observed=torch.arange(3,dtype=torch.float32)[None,:,None].repeat(1,1,4)
        y.update(text_embed=model.encode_text([row['text']]),imu_control=observed,imu_control_mode='static_delta');y.pop('imu_reference')
        guided=make_text_imu_guidance_model(model,mode='factorized',branch_execution='batched').eval();cache={}
        handle=model.register_forward_hook(lambda m,args,result:cache.update(branches=tuple(result.chunk(4))))
        class Capture(torch.nn.Module):
            njoints=263;nfeats=1
            def forward(self,x,t,y=None):
                step=int(t[0]);conditions=dict(y)
                if step>=900:conditions.update(imu_control=torch.zeros_like(observed),imu_condition_present=torch.tensor([False]))
                else:conditions.update(imu_control=observed,imu_condition_present=torch.tensor([True]),text_scale=1.,imu_scale=0.,joint_scale=1.)
                result=guided(x,t,y=conditions)
                if step==899:
                    cache['start']=numpy(x);cache['anchor']={b:numpy(v) for b,v in zip(('f00','f10','f01','f11'),cache['branches'])};cache['anchor']['x_t']=numpy(x)
                return result
        with torch.no_grad():features,plans=sample_addressed_ddpm(Capture().eval(),Diffusion(),y=y,lengths=torch.tensor([3]),case_ids=[cid],sample_seed=12,backbone_id='CPUfixture')
        expected[first['trajectory_id']]=numpy(features);handle.remove()
        sp=tmp/(first['trajectory_id']+'-start.npz');np.savez(sp,x_899=cache['start'],frame_mask=np.ones((1,3),bool));first['start']=artifact(sp)
        noise[first['noise_id']]=dict(draws=plans[0]['noise_draws'][101:])
        ap=tmp/(first['trajectory_id']+'-899-anchor.npz');np.savez(ap,**cache['anchor'])
        anchors.append(dict(trajectory_id=first['trajectory_id'],timestep=899,raw=artifact(ap),arrays=['x_t','f00','f10','f01','f11']))
    return SimpleNamespace(trajectories=paths,noise=noise,anchors=anchors,cases=real.cases,expected=expected,
        state=lambda sid:deepcopy(weight),inputs=inputs,snapshots={'F200':dict(artifact=dict(path='CPUfixture',sha256='F200'))})


def context(tmp):return dict(version='itm-main-knockout-context-v1',phase='main_knockout',package_sha256=PACKAGE_SHA,
    implementation_sha256='CPUfixture',deadline_monotonic=1200.,receipt_path=str(tmp/'receipt.json'),expected_artifacts=[])


def install_worker(pkg,monkeypatch):
    worker=script('run_mdm_imu_main_knockout_worker');model=Model().eval();enc=Encoder().eval()
    model.adapter.load_state_dict(pkg.state('F200')['adapters']);removed=[]
    monkeypatch.setattr(worker,'load_model',lambda *a:(model,Diffusion(),SimpleNamespace(adapters=model.adapter),enc,torch.nn.Identity().eval(),
        SimpleNamespace(remove=lambda:removed.append(True)),SimpleNamespace(decode_motion=decode)))
    return worker,model,removed


@pytest.fixture(scope='module')
def complete_fixture(tmp_path_factory):
    tmp=tmp_path_factory.mktemp('main-knockout-full-CPUfixture');pkg=fixture_package(tmp)
    with pytest.MonkeyPatch.context() as mp:
        import os
        mp.setattr(os,'fsync',lambda fd:None)
        worker,model,removed=install_worker(pkg,mp);g=SuffixGuard(context(tmp),pkg,clock=lambda:0.)
        worker.run(pkg,SimpleNamespace(output=tmp/'suffix',device='cpu'),g);g.finish()
    assert removed and model.calls==1800
    return tmp,pkg,g


def test_complete_full_suffix_native_baseline_counts_and_artifacts(complete_fixture):
    tmp,pkg,g=complete_fixture
    assert len(g.state['checks'])==1812 and len(g.state['traces'])==6 and len(g.state['finals'])==2
    assert len(required_artifacts(pkg,tmp/'suffix'))==3626
    for k,n in CAPS.items():assert g.state['counts'][k]['completed']==n and g.state['counts'][k]['completed_calls']==CALL_CAPS[k]
    script('run_mdm_imu_main_knockout_bounded').validate_receipt(read_receipt(tmp/'receipt.json'),context(tmp),pkg,tmp/'suffix')
    for r in pkg.trajectories:
        tid=r['trajectory_id']
        with np.load(tmp/'suffix/starts'/(tid+'.npz')) as z,np.load(r['start']['path']) as old:assert np.array_equal(z['x_899'],old['x_899'])
        with np.load(tmp/'suffix/steps'/(tid+'-899.npz')) as z:
            assert not np.array_equal(z['f00'],z['f01']) and not np.array_equal(z['f10'],z['f11'])
            a,b,c,d=[z[n] for n in ('f00','f10','f01','f11')]
            expected=a+np.float32(1.)*(b-a)+np.float32(0.)*(c-a)+np.float32(1.)*(d-b-c+a)
            assert np.array_equal(expected,z['actual_combined_prediction']) and not np.array_equal(expected,z['f10'])
        with np.load(tmp/'suffix/finals'/(tid+'.npz')) as z:assert np.array_equal(z['normalized_humanml263'],pkg.expected[tid])
        assert read(tmp/'suffix/steps'/(tid+'-0.json'))['noise'] is None



def test_first_old_anchor_failure_stops_queue_and_preserves_raw(tmp_path,monkeypatch):
    pkg=fixture_package(tmp_path);a=pkg.anchors[0]
    with np.load(a['raw']['path']) as z:v={k:z[k] for k in z.files}
    v['f11']+=1;np.savez(a['raw']['path'],**v);a['raw']=artifact(a['raw']['path'])
    worker,model,removed=install_worker(pkg,monkeypatch);g=SuffixGuard(context(tmp_path),pkg,clock=lambda:0.)
    with pytest.raises(ValueError,match='Frozen suffix equation failed'):
        try:worker.run(pkg,SimpleNamespace(output=tmp_path/'suffix',device='cpu'),g)
        except BaseException as e:g.fail(e);raise
    receipt=read_receipt(g.path)
    assert receipt['counts']['mdm']['actual_calls']==1 and model.calls==1 and removed
    assert len(receipt['steps'])==1800 and len(receipt['checks'])==1812 and len(receipt['finals'])==2
    assert all(r['status']=='not_run_after_stop' for r in receipt['finals'])
    assert (tmp_path/'suffix/steps/MK001-899.npz').exists()
    assert any(r['status']=='failed' for r in receipt['checks'])


def test_reservation_exception_and_truncated_journal(tmp_path):
    pkg=SuffixPackage();g=SuffixGuard(context(tmp_path),pkg,clock=lambda:0.);g.state['active']=dict(table='steps',id='MK001-899')
    with pytest.raises(RuntimeError):g.invoke('mdm',4,lambda:(_ for _ in ()).throw(RuntimeError('CPUfixture forward fault')))
    with g.journal.open('ab') as f:f.write(b'{"incomplete":')
    r=read_receipt(g.path)
    assert r['counts']['mdm']['reserved_calls']==r['counts']['mdm']['actual_calls']==1
    assert r['counts']['mdm']['completed_calls']==0 and r['truncated_journal_tail']


def test_deadline_and_invalid_batch_never_call_model(tmp_path):
    g=SuffixGuard(context(tmp_path),SuffixPackage(),clock=lambda:0.)
    with pytest.raises(RuntimeError):g.invoke('mdm',5,lambda:pytest.fail('No forward'))
    g.clock=lambda:1201.
    with pytest.raises(RuntimeError):g.invoke('mdm',4,lambda:pytest.fail('No forward'))
    assert g.state['counts']['mdm']['actual_calls']==0


def test_parent_deadline_includes_preflight_and_failed_evaluation_denominators(tmp_path,monkeypatch):
    parent=script('run_mdm_imu_main_knockout_bounded');pkg=SuffixPackage();manifest=tmp_path/'implementation.json'
    from itm.models.mdm_run_limits import atomic_json
    atomic_json(manifest,dict(package=artifact(pkg.path),expected_artifacts=[],environment=pkg.common['environment']))
    monkeypatch.setattr(parent,'authorization',lambda *a:{})
    monkeypatch.setattr(parent,'SuffixPackage',lambda *a:pkg)
    monkeypatch.setattr(parent,'environment',lambda:pkg.common['environment'])
    now=[0.]
    def preflight(*a,**kw):now[0]=1201.;kw['check']()
    monkeypatch.setattr(parent,'preflight_gpu',preflight)
    monkeypatch.setattr(parent,'supervise',lambda *a,**k:pytest.fail('No worker launch'))
    with pytest.raises(TimeoutError):parent.execute(manifest,tmp_path/'run',tmp_path/'record','GPU-CPUfixture',clock=lambda:now[0])
    s=read(tmp_path/'run/runtime/status.json');assert s['deadline_monotonic']==1200. and len(s['steps'])==1800 and len(s['checks'])==1812
    assert s['counts']['mdm']['actual_calls']==0 and len(s['finals'])==2
    evaluator=script('evaluate_b0_imu_main_knockout');evaluator.evaluate(tmp_path/'run',tmp_path/'evaluation')
    e=read(tmp_path/'evaluation/summary.json');assert e['final_counts']=={'missing':2} and e['trace_counts']=={'missing':6} and e['status']=='inconclusive'


def test_real_prepare_no_GPU_or_model_construction(tmp_path,monkeypatch):
    from test_mdm_sampling_alignment import forbid_gpu_and_models
    forbid_gpu_and_models(monkeypatch);p=script('run_mdm_imu_main_knockout_bounded').prepare(tmp_path/'implementation');m=read(p)
    assert m['model_constructions']==m['model_forwards']==m['GPU_queries']==0
    assert len(m['inputs'])==2 and m['planned_worker_artifacts']==3626 and set(m['snapshots'])=={'F200'}


def test_preferred_device_binding_rejects_other_UUID():
    parent=script('run_mdm_imu_main_knockout_bounded')
    entry=parent.device_preference('GPU-43c2f925-db05-c655-bb46-25196d0627d6')
    assert entry['sha256']==parent.DEVICE_PREFERENCE_SHA
    with pytest.raises(ValueError):parent.device_preference('GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004')
