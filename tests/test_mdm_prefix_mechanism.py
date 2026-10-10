"""Synthetic CPU posterior/queue/fault tests; never real model predictions."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from itm.data.random_plan import sampling_noise
from itm.models.mdm_correction import artifact,read
from itm.models.mdm_imu_control import make_text_imu_guidance_model
from itm.models.mdm_prefix_mechanism import PrefixPackage,PrefixGuard,PACKAGE_SHA,CAPS,CALL_CAPS,ledger,read_receipt,replay_prefix,required_artifacts
from itm.models.mdm_sampling import sample_addressed_ddpm
from itm.models.mdm_sampling_alignment import input_arrays,save_npz,numpy


def script(name):
    p=Path(__file__).resolve().parents[1]/'scripts'/(name+'.py')
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.adapter=torch.nn.Linear(4,263,bias=False);self.calls=0
        self.njoints=263;self.nfeats=1;self.rot2xyz=None;self.translation=False;self.data_rep='hml_vec';self.cond_mode='text'
    def encode_text(self,text):return torch.ones(1,len(text),4)
    def forward(self,x,t,y=None):
        self.calls+=1
        control=y['imu_control']*y['imu_condition_present'][:,None,None]
        delta=self.adapter(control).transpose(1,2)[:,:,None]
        return .85*x+.01*y['text_embed'].mean((0,2))[:,None,None,None]+.01*delta


class Encoder(torch.nn.Module):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.));self.config=SimpleNamespace(latent_dim=4)
    def forward(self,x,sensor_mask,frame_mask,condition_present=None,timestamps=None,return_presence=False):
        c=x.mean((2,3))[:,:,None].repeat(1,1,4)*self.weight
        return c*condition_present[:,None,None],condition_present


class Diffusion:
    num_timesteps=1000;timestep_map=list(range(1000));rescale_timesteps=False
    def _scale_timesteps(self,t):return t
    def p_mean_variance(self,model,x,t,clip_denoised=False,model_kwargs=None):
        assert clip_denoised is False
        pred=model(x,t,**model_kwargs)
        return dict(mean=.9*x+.1*pred,log_variance=torch.full_like(x,-6.))


def toy_item(text):
    frames=3;motion=np.arange(frames,dtype=np.float32)[:,None]*np.ones((1,263),np.float32)
    return dict(motion=motion,text=text,imu=np.arange(frames,dtype=np.float32)[:,None,None]*np.ones((1,6,12),np.float32),
        reference=np.zeros((frames,6,12),np.float32),sensor_mask=np.ones(6,bool),timestamps=np.arange(frames)/20,
        raw_acceleration=np.zeros((frames,6,3),np.float32),raw_orientation=np.tile(np.eye(3,dtype=np.float32),(frames,6,1,1)),
        original_orientation0=np.tile(np.eye(3,dtype=np.float32),(6,1,1)),gt=np.zeros((frames,22,3),np.float32))


def fixture_package(tmp):
    real=PrefixPackage();trajectories=deepcopy(real.trajectories);states=deepcopy(real.states);probes=deepcopy(real.probes)
    weights={}
    for sid,scale in [('F200',1.),('U200',.5)]:
        model=Model();enc=Encoder()
        with torch.no_grad():model.adapter.weight.fill_(scale)
        weights[sid]=dict(imu_encoder=deepcopy(enc.state_dict()),adapters=deepcopy(model.adapter.state_dict()))
    noise={}
    for i,(key,row) in enumerate(real.noise.items()):
        specs=[]
        for t in [-1]+list(range(999,899,-1)):
            _,meta=sampling_noise((1,263,1,3),backbone_id='CPUfixture',case_id='toy-'+str(i),sample_seed=12,step=t);specs.append(meta)
        noise[key]=dict(draws=specs)
    for t in trajectories:t['frames']=3
    def inputs(t):
        row=deepcopy(real.slots[t['input_slot_id']]);row['frames']=3
        return toy_item(row['text']),row
    pkg=SimpleNamespace(trajectories=trajectories,states=states,probes=probes,noise=noise,
        cases=real.cases,snapshots={k:dict(artifact=dict(path='CPUfixture',sha256=k)) for k in weights},
        state=lambda sid:deepcopy(weights[sid]),inputs=inputs,anchors=[])
    correction=script('run_mdm_correction_worker')
    # Native full-loop fixture produces independent historical t900 anchors.
    with torch.no_grad():
        for trajectory in trajectories:
            model=Model();encoder=Encoder();model.adapter.load_state_dict(weights[trajectory['checkpoint']]['adapters']);encoder.eval();model.eval()
            item,row=inputs(trajectory);_,y=correction.batch_inputs([item],[row],'cpu')
            y.update(text_embed=model.encode_text([row['text']]),imu_control=torch.arange(3,dtype=torch.float32)[None,:,None].repeat(1,1,4),imu_control_mode='static_delta')
            y.pop('imu_reference')
            if trajectory['path_arm']=='T':y.update(imu_control=torch.zeros_like(y['imu_control']),imu_condition_present=torch.tensor([False]))
            guided=make_text_imu_guidance_model(model,mode='factorized',branch_execution='batched');guided.eval();captured={}
            def hook(m,args,result):captured['branches']=tuple(result.chunk(4))
            handle=model.register_forward_hook(hook)
            class Capture(torch.nn.Module):
                njoints=263;nfeats=1
                def forward(self,x,t,y=None):
                    result=guided(x,t,y=y)
                    if int(t[0])==900:
                        captured['arrays']={b:numpy(v) for b,v in zip(('f00','f10','f01','f11'),captured['branches'])}
                        captured['arrays'].update(x_t=numpy(x),actual_combined_prediction=numpy(result),frame_mask=numpy(y['imu_frame_mask']))
                    return result
            capture=Capture().eval();first=noise[trajectory['noise_id']]['draws'][0]
            sample_addressed_ddpm(capture,Diffusion(),y=y,lengths=torch.tensor([3]),case_ids=[first['case_id']],sample_seed=12,backbone_id='CPUfixture')
            handle.remove();path=tmp/(trajectory['trajectory_id']+'-historical.npz');np.savez(path,**captured['arrays'])
            for arm in (['F','U'] if trajectory['path_arm']=='T' else [trajectory['path_arm']]):
                pkg.anchors.append(dict(trajectory_id=trajectory['trajectory_id'],historical_arm=arm,npz=artifact(path),arrays=list(captured['arrays'])[:6]))
    return pkg


def context(tmp):return dict(version='itm-prefix-context-v1',phase='prefix',package_sha256=PACKAGE_SHA,
    implementation_sha256='CPUfixture',deadline_monotonic=600.,receipt_path=str(tmp/'receipt.json'),expected_artifacts=[])


def test_prefix_native_full_loop_first100_states_exact():
    class Plain(torch.nn.Module):
        njoints=263;nfeats=1
        def forward(self,x,t,y=None):return x*.85
    model=Plain().eval();diff=Diffusion();states={}
    class Recording(Diffusion):
        def p_mean_variance(self,m,x,t,**kwargs):
            if int(t[0])>=900:states[int(t[0])]=x.clone()
            return super().p_mean_variance(m,x,t,**kwargs)
    sample_addressed_ddpm(model,Recording(),y={},lengths=torch.tensor([3]),case_ids=['toy'],sample_seed=12,backbone_id='CPUfixture')
    def draw(t):return torch.from_numpy(sampling_noise((1,263,1,3),backbone_id='CPUfixture',case_id='toy',sample_seed=12,step=t)[0])
    observed=[]
    def before(t,x):assert torch.equal(x,states[t]);observed.append(t)
    replay_prefix(model,diff,{},draw,before=before)
    assert observed==list(range(999,899,-1))


@pytest.fixture(scope='module')
def complete_fixture(tmp_path_factory):
    tmp=tmp_path_factory.mktemp('prefix-full-CPUfixture');pkg=fixture_package(tmp);worker=script('run_mdm_prefix_mechanism_worker')
    model=Model().eval();encoder=Encoder().eval();removed=[]
    worker.load_model=lambda *a:(model,Diffusion(),SimpleNamespace(adapters=model.adapter),encoder,torch.nn.Identity().eval(),SimpleNamespace(remove=lambda:removed.append(True)),SimpleNamespace())
    guard=PrefixGuard(context(tmp),pkg,clock=lambda:0.)
    worker.run(pkg,SimpleNamespace(output=tmp/'prefix',device='cpu'),guard);guard.finish()
    assert removed and model.calls==3960
    return tmp,pkg,guard


def test_complete_synthetic_queue_counts_checks_and_artifacts(complete_fixture):
    tmp,pkg,g=complete_fixture;parent=script('run_mdm_prefix_mechanism_bounded')
    for kind,n in CAPS.items():assert g.state['counts'][kind]['completed']==n and g.state['counts'][kind]['completed_calls']==CALL_CAPS[kind]
    assert len(g.state['checks'])==7380
    assert len(required_artifacts(pkg,tmp/'prefix'))==8292
    parent.validate_receipt(read_receipt(tmp/'receipt.json'),context(tmp),pkg,tmp/'prefix')


def test_first_anchor_failure_preserves_raw_and_unrun_denominators(tmp_path):
    pkg=fixture_package(tmp_path);worker=script('run_mdm_prefix_mechanism_worker')
    a=pkg.anchors[0];p=Path(a['npz']['path'])
    with np.load(p) as z:v={k:z[k] for k in z.files}
    v['f11']+=1;np.savez(p,**v);a['npz']=artifact(p)
    model=Model().eval();encoder=Encoder().eval()
    worker.load_model=lambda *a:(model,Diffusion(),SimpleNamespace(adapters=model.adapter),encoder,torch.nn.Identity().eval(),SimpleNamespace(remove=lambda:None),SimpleNamespace())
    g=PrefixGuard(context(tmp_path),pkg,clock=lambda:0.)
    with pytest.raises(ValueError,match='Frozen prefix equation failed'):
        try:worker.run(pkg,SimpleNamespace(output=tmp_path/'prefix',device='cpu'),g)
        except BaseException as e:g.fail(e);raise
    receipt=read_receipt(tmp_path/'receipt.json')
    assert len(receipt['steps'])==3600 and len(receipt['probes'])==360
    assert receipt['counts']['mdm']['actual_calls']==100
    raw=tmp_path/'prefix/steps/PX001-900.npz';assert raw.exists()
    assert sum(r['status']=='not_run_after_stop' for r in receipt['probes'])==360
    assert any(r['status']=='failed' for r in receipt['checks'])


def test_reservation_survives_model_exception_and_journal_tail(tmp_path):
    real=PrefixPackage();g=PrefixGuard(context(tmp_path),real,clock=lambda:0.)
    g.state['active']=dict(table='steps',id='PX001-999')
    with pytest.raises(RuntimeError):g.invoke('mdm',4,lambda:(_ for _ in ()).throw(RuntimeError('fixture forward fault')))
    with g.journal.open('ab') as f:f.write(b'{"incomplete":')
    receipt=read_receipt(g.path)
    assert receipt['counts']['mdm']['reserved_calls']==receipt['counts']['mdm']['actual_calls']==1
    assert receipt['counts']['mdm']['completed_calls']==0 and receipt['truncated_journal_tail']


def test_deadline_and_batch_reject_before_forward(tmp_path):
    real=PrefixPackage();g=PrefixGuard(context(tmp_path),real,clock=lambda:0.)
    with pytest.raises(RuntimeError):g.invoke('mdm',5,lambda:None)
    g.clock=lambda:601.
    with pytest.raises(RuntimeError):g.invoke('mdm',4,lambda:None)
    assert g.state['counts']['mdm']['actual_calls']==0


def test_real_prepare_no_GPU_or_model_construction(tmp_path,monkeypatch):
    from test_mdm_sampling_alignment import forbid_gpu_and_models
    forbid_gpu_and_models(monkeypatch);parent=script('run_mdm_prefix_mechanism_bounded')
    p=parent.prepare(tmp_path/'implementation');m=read(p)
    assert m['model_constructions']==m['model_forwards']==m['GPU_queries']==0
    assert len(m['inputs'])==36 and m['planned_worker_artifacts']==8292
    assert set(m['snapshots'])=={'F200','U200'}


def test_parent_preflight_failure_preserves_complete_denominator(tmp_path,monkeypatch):
    parent=script('run_mdm_prefix_mechanism_bounded');real=PrefixPackage()
    manifest=tmp_path/'manifest.json';manifest.write_text(__import__('json').dumps(dict(package=artifact(real.path),expected_artifacts=[],environment={})))
    monkeypatch.setattr(parent,'authorization',lambda *a:dict(implementation_sha256='CPUfixture'))
    monkeypatch.setattr(parent,'PrefixPackage',lambda *a:real)
    monkeypatch.setattr(parent,'environment',lambda:real.common['environment'])
    m=read(manifest);m['environment']=real.common['environment'];manifest.write_text(__import__('json').dumps(m))
    monkeypatch.setattr(parent,'preflight_gpu',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('CPUfixture preflight stop')))
    monkeypatch.setattr(parent,'supervise',lambda *a,**k:pytest.fail('worker must not start'))
    with pytest.raises(RuntimeError,match='CPUfixture preflight stop'):parent.execute(manifest,tmp_path/'run',tmp_path/'record','GPU-CPUfixture',clock=lambda:0.)
    status=read(tmp_path/'run/runtime/status.json')
    assert len(status['steps'])==3600 and len(status['probes'])==360 and len(status['checks'])==7380
    assert all(r['status']=='not_run_after_stop' for r in status['probes'])
    assert status['counts']['mdm']['actual_calls']==0
