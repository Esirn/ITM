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
from itm.models.mdm_suffix_switch import SuffixPackage,SuffixGuard,PACKAGE_SHA,CAPS,CALL_CAPS,ledger,read_receipt,replay_suffix,required_artifacts
from test_mdm_prefix_mechanism import script,Model,Encoder,Diffusion,toy_item


def decode(features):return features[:,:66,0,:].permute(0,2,1).reshape(1,3,22,3).clone()


def fixture_package(tmp):
    real=SuffixPackage();paths=deepcopy(real.trajectories);noise={};anchors=[];weight=None
    for trajectory in paths:trajectory['frames']=3
    def inputs(t):
        row=deepcopy(real.slots[t['input_slot_id']]);row['frames']=3
        return toy_item(row['text']),row
    for cid in real.cases:
        pair=[r for r in paths if r['panel_case_id']==cid];first=pair[0]
        model=Model().eval();enc=Encoder().eval()
        with torch.no_grad():model.adapter.weight.fill_(1.)
        weight=dict(imu_encoder=deepcopy(enc.state_dict()),adapters=deepcopy(model.adapter.state_dict()))
        item,row=inputs(first);_,y=script('run_mdm_correction_worker').batch_inputs([item],[row],'cpu')
        y.update(text_embed=model.encode_text([row['text']]),imu_control=torch.arange(3,dtype=torch.float32)[None,:,None].repeat(1,1,4),imu_control_mode='static_delta');y.pop('imu_reference')
        guided=make_text_imu_guidance_model(model,mode='factorized',branch_execution='batched').eval();cache={};seen={}
        handle=model.register_forward_hook(lambda m,args,result:cache.update(branches=tuple(result.chunk(4))))
        class Capture(torch.nn.Module):
            njoints=263;nfeats=1
            def forward(self,x,t,y=None):
                result=guided(x,t,y=y);step=int(t[0])
                if step==899:cache['start']=numpy(x)
                if step in [500,250,50]:
                    z={b:numpy(v) for b,v in zip(('f00','f10','f01','f11'),cache['branches'])}
                    z.update(x_t=numpy(x),actual_combined_prediction=numpy(result),frame_mask=numpy(y['imu_frame_mask']));seen[step]=z
                return result
        capture=Capture().eval()
        with torch.no_grad():features,plans=sample_addressed_ddpm(capture,Diffusion(),y=y,lengths=torch.tensor([3]),case_ids=[cid],sample_seed=12,backbone_id='CPUfixture')
        handle.remove();sp=tmp/(first['trajectory_id']+'-start.npz');np.savez(sp,x_899=cache['start'],frame_mask=np.ones((1,3),bool))
        for trajectory in pair:trajectory['start']=artifact(sp)
        noise[first['noise_id']]=dict(draws=plans[0]['noise_draws'][101:])
        for t,z in seen.items():
            ap=tmp/(first['trajectory_id']+'-'+str(t)+'-anchor.npz');np.savez(ap,**z)
            anchors.append(dict(trajectory_id=first['trajectory_id'],timestep=t,raw=artifact(ap),arrays=['x_t','f00','f10','f01','f11','actual_combined_prediction']))
        ap=tmp/(first['trajectory_id']+'-final.npz');np.savez(ap,normalized_humanml263=numpy(features),motion=numpy(decode(features)),frame_mask=np.ones((1,3),bool))
        anchors.append(dict(trajectory_id=first['trajectory_id'],kind='final',raw=artifact(ap),arrays=['normalized_humanml263','joints','frame_mask']))
    return SimpleNamespace(trajectories=paths,noise=noise,anchors=anchors,cases=real.cases,
        state=lambda sid:deepcopy(weight),inputs=inputs,snapshots={'F200':dict(artifact=dict(path='CPUfixture',sha256='F200'))})


def context(tmp):return dict(version='itm-suffix-context-v1',phase='suffix',package_sha256=PACKAGE_SHA,
    implementation_sha256='CPUfixture',deadline_monotonic=1200.,receipt_path=str(tmp/'receipt.json'),expected_artifacts=[])


def install_worker(pkg,monkeypatch):
    worker=script('run_mdm_suffix_switch_worker');model=Model().eval();enc=Encoder().eval()
    model.adapter.load_state_dict(pkg.state('F200')['adapters']);removed=[]
    monkeypatch.setattr(worker,'load_model',lambda *a:(model,Diffusion(),SimpleNamespace(adapters=model.adapter),enc,torch.nn.Identity().eval(),
        SimpleNamespace(remove=lambda:removed.append(True)),SimpleNamespace(decode_motion=decode)))
    return worker,model,removed


@pytest.fixture(scope='module')
def complete_fixture(tmp_path_factory):
    tmp=tmp_path_factory.mktemp('suffix-full-CPUfixture');pkg=fixture_package(tmp)
    with pytest.MonkeyPatch.context() as mp:
        worker,model,removed=install_worker(pkg,mp);g=SuffixGuard(context(tmp),pkg,clock=lambda:0.)
        worker.run(pkg,SimpleNamespace(output=tmp/'suffix',device='cpu'),g);g.finish()
    assert removed and model.calls==3600
    return tmp,pkg,g


def test_complete_full_suffix_native_baseline_counts_and_artifacts(complete_fixture):
    tmp,pkg,g=complete_fixture
    assert len(g.state['checks'])==7244 and len(g.state['traces'])==12 and len(g.state['finals'])==4
    assert len(required_artifacts(pkg,tmp/'suffix'))==7242
    for k,n in CAPS.items():assert g.state['counts'][k]['completed']==n and g.state['counts'][k]['completed_calls']==CALL_CAPS[k]
    script('run_mdm_suffix_switch_bounded').validate_receipt(read_receipt(tmp/'receipt.json'),context(tmp),pkg,tmp/'suffix')
    for r in pkg.trajectories:
        if r['condition']=='switch_imu_off':
            with np.load(tmp/'suffix/steps'/(r['trajectory_id']+'-899.npz')) as z:
                assert np.array_equal(z['f00'],z['f01']) and np.array_equal(z['f10'],z['f11'])
        with np.load(tmp/'suffix/starts'/(r['trajectory_id']+'.npz')) as z,np.load(r['start']['path']) as old:assert np.array_equal(z['x_899'],old['x_899'])
    for r in pkg.trajectories:
        m=read(tmp/'suffix/steps'/(r['trajectory_id']+'-0.json'));assert m['noise'] is None


def test_first_old_anchor_failure_stops_queue_and_preserves_raw(tmp_path,monkeypatch):
    pkg=fixture_package(tmp_path);a=pkg.anchors[0]
    with np.load(a['raw']['path']) as z:v={k:z[k] for k in z.files}
    v['f11']+=1;np.savez(a['raw']['path'],**v);a['raw']=artifact(a['raw']['path'])
    worker,model,removed=install_worker(pkg,monkeypatch);g=SuffixGuard(context(tmp_path),pkg,clock=lambda:0.)
    with pytest.raises(ValueError,match='Frozen suffix equation failed'):
        try:worker.run(pkg,SimpleNamespace(output=tmp_path/'suffix',device='cpu'),g)
        except BaseException as e:g.fail(e);raise
    receipt=read_receipt(g.path)
    assert receipt['counts']['mdm']['actual_calls']==400 and model.calls==400 and removed
    assert len(receipt['steps'])==3600 and len(receipt['checks'])==7244 and len(receipt['finals'])==4
    assert all(r['status']=='not_run_after_stop' for r in receipt['finals'])
    assert (tmp_path/'suffix/steps/SX001-500.npz').exists()
    assert any(r['status']=='failed' for r in receipt['checks'])


def test_reservation_exception_and_truncated_journal(tmp_path):
    pkg=SuffixPackage();g=SuffixGuard(context(tmp_path),pkg,clock=lambda:0.);g.state['active']=dict(table='steps',id='SX001-899')
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
    parent=script('run_mdm_suffix_switch_bounded');pkg=SuffixPackage();manifest=tmp_path/'implementation.json'
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
    s=read(tmp_path/'run/runtime/status.json');assert s['deadline_monotonic']==1200. and len(s['steps'])==3600 and len(s['checks'])==7244
    assert s['counts']['mdm']['actual_calls']==0 and len(s['finals'])==4
    evaluator=script('evaluate_b0_suffix_switch');evaluator.evaluate(tmp_path/'run',tmp_path/'evaluation')
    e=read(tmp_path/'evaluation/summary.json');assert e['final_counts']=={'missing':4} and e['trace_counts']=={'missing':12} and e['status']=='inconclusive'


def test_real_prepare_no_GPU_or_model_construction(tmp_path,monkeypatch):
    from test_mdm_sampling_alignment import forbid_gpu_and_models
    forbid_gpu_and_models(monkeypatch);p=script('run_mdm_suffix_switch_bounded').prepare(tmp_path/'implementation');m=read(p)
    assert m['model_constructions']==m['model_forwards']==m['GPU_queries']==0
    assert len(m['inputs'])==4 and m['planned_worker_artifacts']==7242 and set(m['snapshots'])=={'F200'}
