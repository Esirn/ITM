"""CPU fixtures only: no pretrained backbone, CUDA, real training or generation."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from itm.models.mdm_correction import (ARMS, CorrectionPackage, transform_inputs, training_noise, probe_noise,
    isolated_rng, encode_control, ConditionCache, accumulate_update, create_initial_state, tensor_hashes)
from itm.models.mdm_correction_limits import CorrectionGuard
from itm.models.mdm_imu_control import (MDMIMUControlConfig, make_imu_control_encoder, install_imu_control,
                                        configure_control_training)
from itm.models.mdm_run_limits import RunLimitExceeded, sha256, atomic_json

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'docs/itm_rework/protocol/b0_correction_20260926_v1/package.json'


def script(name):
    sys.path.insert(0,str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def small_config(**kwargs):
    return MDMIMUControlConfig(latent_dim=8,encoder_heads=2,encoder_layers=2,encoder_variant='ordered_concat',
                               null_mode='strict',time_encoding='physical_sinusoidal',**kwargs)


def inputs(n=5):
    x=torch.randn(1,n,6,12)
    return dict(imu=x,imu_reference=x.clone(),sensor_mask=torch.tensor([[1,1,0,0,0,0]],dtype=torch.bool),
                imu_frame_mask=torch.ones(1,n,dtype=torch.bool),imu_timestamps=torch.arange(n)[None]/20.,
                imu_condition_present=torch.ones(1,dtype=torch.bool))


def guard(tmp_path,phase='train',arm=ARMS[0],clock=time.monotonic):
    caps={'mdm_examples':480 if phase=='probe' else 600 if phase=='train' else 4000,
          'encoder_examples':480 if phase=='probe' else (1200 if arm==ARMS[1] else 600) if phase=='train' else (2 if arm==ARMS[1] else 1),
          'exposures':600 if phase=='train' else 0}
    return CorrectionGuard(dict(version='itm-correction-context-v1',phase=phase,arm=arm,counters=caps,
        receipt_path=str(tmp_path/'receipt.json'),deadline_monotonic=clock()+60),clock=clock)


def test_reference_dropout_exact_zero_gradient_and_rng_restored():
    encoder=make_imu_control_encoder(small_config()).train();y=inputs()
    before=torch.random.get_rng_state().clone()
    delta,_=encode_control(encoder,y,mode='static_delta',seed=123)
    assert torch.equal(before,torch.random.get_rng_state())
    assert torch.count_nonzero(delta)==0
    adapter=torch.nn.Linear(8,8,bias=False)
    out=adapter(delta);(out-1).square().mean().backward()
    grads=[p.grad for p in list(encoder.parameters())+list(adapter.parameters()) if p.grad is not None]
    assert grads and max(float(g.abs().max()) for g in grads)<=1e-7
    # Non-static reference must retain both branches' gradients and RNG independence.
    encoder.zero_grad();y['imu_reference']=torch.zeros_like(y['imu'])
    d1,_=encode_control(encoder,y,mode='static_delta',seed=123)
    torch.rand(50)
    d2,_=encode_control(encoder,y,mode='static_delta',seed=123)
    torch.testing.assert_close(d1,d2,rtol=0,atol=0)
    d1.square().mean().backward()
    assert sum(float(p.grad.abs().sum()) for p in encoder.parameters() if p.grad is not None)>0


class TinyMDM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer=torch.nn.TransformerEncoderLayer(8,2,16,dropout=0.)
        self.seqTransEncoder=torch.nn.TransformerEncoder(layer,2)
    def forward(self,x,t=None,y=None):
        return self.seqTransEncoder(x)


def test_static_delta_bias_unused_frozen_and_null_matches_raw():
    model=TinyMDM();original=model.seqTransEncoder
    controlled,encoder,handles=install_imu_control(model,small_config(reference_mode='static_delta'))
    configure_control_training(model)
    assert all(not a.bias.requires_grad for a in controlled.adapters)
    for a in controlled.adapters:
        torch.nn.init.normal_(a.weight);a.bias.data.fill_(100)
    model.eval();x=torch.randn(6,1,8);y=inputs()
    with torch.no_grad():
        expected=original(x)
        delta,present=encode_control(encoder,y,mode='static_delta')
        cache=dict(imu_control=delta,imu_condition_present=present,imu_frame_mask=y['imu_frame_mask'],imu_control_mode='static_delta')
        actual=model(x,y=cache)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
        cache['imu_control']=torch.randn_like(delta);cache['imu_condition_present']=torch.zeros(1,dtype=torch.bool)
        torch.testing.assert_close(model(x,y=cache),expected,atol=1e-6,rtol=1e-6)
        with pytest.raises(ValueError,match='identified'):model(x,y=y)
    handles.remove()


def test_accumulation_matches_three_objectives_including_null(tmp_path,monkeypatch):
    g=guard(tmp_path);p=torch.nn.Parameter(torch.tensor([1.,-2.])); q=p.detach().clone().requires_grad_()
    opt=torch.optim.SGD([p],lr=.1)
    ((q.square().mean()+q.sum().square()+torch.tensor(7.))/3).backward()
    expected=q.detach()-.1*q.grad
    monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',lambda *a,**k:pytest.fail('No gradient clipping'))
    accumulate_update([lambda:p.square().mean(),lambda:p.sum().square(),lambda:torch.tensor(7.)],opt,[p],g)
    torch.testing.assert_close(p,expected,atol=1e-6,rtol=1e-5)
    assert g.state['updates_attempted']==g.state['updates_completed']==1


def test_200_steps_600_exposures_and_no_extra_update(tmp_path):
    g=guard(tmp_path);p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.SGD([p],lr=.001)
    def loss():
        g.reserve('mdm_examples');g.reserve('encoder_examples');g.reserve('exposures')
        return p.square()
    for _ in range(200):accumulate_update([loss]*3,opt,[p],g)
    assert g.state['exposures']==g.state['mdm_examples']==600
    before=p.detach().clone()
    with pytest.raises(RunLimitExceeded):accumulate_update([loss]*3,opt,[p],g)
    torch.testing.assert_close(p,before);g.finish()


@pytest.mark.parametrize('fault',['loss','gradient','step'])
def test_fault_during_accumulation_preserves_attempted_and_stops(tmp_path,fault):
    g=guard(tmp_path);p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.SGD([p],lr=.1)
    if fault=='gradient':p.register_hook(lambda grad:grad*float('inf'))
    if fault=='step':
        def bad():
            with torch.no_grad():p.fill_(float('nan'))
        opt.step=bad
    losses=[lambda:p.square(),lambda:p*float('nan') if fault=='loss' else p.square(),lambda:p.square()]
    with pytest.raises(FloatingPointError) as error:accumulate_update(losses,opt,[p],g)
    g.fail(error.value)
    assert g.state['updates_completed']==0 and g.state['updates_attempted']==int(fault=='step')
    assert not g.state['artifacts']


def test_frozen_600_noise_arrays_and_probe_valid_padding_identity():
    p=CorrectionPackage(PACKAGE)
    for row in p.schedule:assert training_noise(row).shape==(1,263,1,row['frames'])
    for entry in p.noise.values():
        valid=probe_noise(entry);padded=probe_noise(entry,True)
        assert np.array_equal(valid,padded[...,:valid.shape[-1]])
    groups=script('run_mdm_correction_worker').probe_groups(p.probes)
    assert len(groups)==416 and sum(map(len,groups))==480
    assert sum(len(g)==3 for g in groups)==32


def test_real_input_transforms_reference_origin_and_tail_order():
    p=CorrectionPackage(PACKAGE)
    base=p.inputs(p.samples[0]);static=p.inputs(p.samples[5]);zero=p.inputs(p.samples[4])
    assert np.array_equal(static['imu'],base['reference'])
    assert np.array_equal(zero['reference'],base['reference']) and not zero['imu'].any()
    row=next(r for r in p.samples if r['condition']=='tail_reversed' and r['motion_id']=='000433' and r['sensor_config']=='head')
    reverse=p.inputs(row);ix=np.r_[0,np.arange(len(base['imu'])-1,0,-1)]
    assert np.array_equal(reverse['raw_acceleration'],base['raw_acceleration'][ix])
    assert np.array_equal(reverse['raw_orientation'],base['raw_orientation'][ix])
    assert np.array_equal(reverse['reference'],base['reference'])
    assert np.array_equal(reverse['timestamps'],base['timestamps'])


def test_cache_rejects_changed_input_checkpoint_reference_mode():
    encoder=make_imu_control_encoder(small_config()).eval(); y=inputs()
    for mutation in ('imu','imu_reference','sensor_mask','imu_frame_mask','imu_timestamps','imu_condition_present','checkpoint','mode'):
        z={k:v.clone() for k,v in y.items()};cache=ConditionCache()
        cache.get(encoder,z,checkpoint_sha='a',mode='none')
        if mutation in z:
            z[mutation].flatten()[0]=~z[mutation].flatten()[0] if z[mutation].dtype==torch.bool else z[mutation].flatten()[0]+1
        with pytest.raises(ValueError,match='identity'):
            cache.get(encoder,z,checkpoint_sha='b' if mutation=='checkpoint' else 'a',mode='static_delta' if mutation=='mode' else 'none')


def test_common_cpu_initialization_is_reproducible_no_forward(tmp_path,monkeypatch):
    from dataclasses import asdict
    pkg=SimpleNamespace(common=dict(control_config=asdict(small_config()),fresh_seed=20260925))
    # Any module forward would be an unexpected compute action in preparation.
    monkeypatch.setattr(torch.nn.Linear,'forward',lambda *a:pytest.fail('initialization must not forward'))
    a=create_initial_state(pkg,tmp_path/'a.pt');b=create_initial_state(pkg,tmp_path/'b.pt')
    assert a['tensor_hashes']==b['tensor_hashes']
    state=torch.load(tmp_path/'a.pt',weights_only=True)
    assert tensor_hashes(state)==state['tensor_hashes']


def test_probe_hooks_once_and_excludes_token_padding():
    worker=script('run_mdm_correction_worker');model=TinyMDM()
    # 8 tiny backbone layers to match the real hook topology, no real MDM.
    model.seqTransEncoder=torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(8,2,16,dropout=0.),8)
    controlled,encoder,handles=install_imu_control(model,small_config());model.eval()
    hooks=worker.ProbeHooks(encoder,controlled);y=inputs();x=torch.randn(6,1,8)
    with torch.no_grad(): model(x,y=y)
    assert len(hooks.values)==21
    assert hooks.values['hidden_0'][1:6,0].shape==(5,8)
    assert hooks.values['encoder_layer_0'][0].shape==(5,8)
    assert worker.reductions(np.array([0.,2.]))['sums_sq']==4
    hooks.close();handles.remove()


def test_partial_final_and_counter_rejection(tmp_path):
    g=guard(tmp_path);g.state.update(updates_completed=199,mdm_examples=600,exposures=600)
    with pytest.raises(RunLimitExceeded,match='final200'):g.finish()
    g.state['encoder_examples']=600
    with pytest.raises(RunLimitExceeded,match='encoder'):g.reserve('encoder_examples')


@pytest.mark.parametrize('fault', [None, 'preflight', 'D_timeout', 'R_partial', 'sample_oom', 'sample_timeout', 'static_mismatch', 'between_samples_timeout'])
def test_top_queue_shared_deadlines_full_denominators_and_failure(tmp_path,monkeypatch,fault):
    """Top coordinator uses CPU stubs; no GPU query, MDM, or real sampling."""
    from itm.models.mdm_correction import PACKAGE_SHA,artifact,environment
    module=script('run_mdm_correction_bounded')
    package=CorrectionPackage(PACKAGE)
    fake_package=SimpleNamespace(path=PACKAGE,probes=package.probes,samples=package.samples)
    monkeypatch.setattr(module,'CorrectionPackage',lambda _:fake_package)
    initial=tmp_path/'CPU_FIXTURE_initial';initial.write_bytes(b'not model weights')
    manifest=tmp_path/'CPU_FIXTURE_implementation.json'
    atomic_json(manifest,dict(version='itm-correction-implementation-v1',package=artifact(PACKAGE),
                             initialization=dict(artifact=artifact(initial)),expected_artifacts=[],environment=environment()))
    run=tmp_path/'CPU_FIXTURE_run';approval=tmp_path/'CPU_FIXTURE_authorization.json'
    atomic_json(approval,dict(authorized=True,approved_by='user',user_instruction_reference='CPU FIXTURE ONLY; not user budget approval',
        package_sha256=PACKAGE_SHA,implementation_sha256=sha256(manifest),run_directory=str(run),gpu_uuid='GPU-fixture'))
    now=[100.];launches=[]
    def preflight(gpu):
        now[0]+=7
        if fault=='preflight':raise RuntimeError('injected preflight unavailable')
        return {'gpu_uuid':gpu,'CPU_fixture':True}
    monkeypatch.setattr(module,'preflight_gpu',preflight)
    def supervise(argv,*,deadline,log_path,env,clock):
        def flag(k):return argv[argv.index(k)+1]
        context=json.loads(Path(flag('--run-context')).read_text());phase=context['phase'];arm=context.get('arm')
        launches.append((phase,arm,deadline));now[0]+=2
        output=Path(flag('--output'));output.parent.mkdir(parents=True,exist_ok=True)
        sample_index=int(flag('--slot')) if phase=='sample' else None
        error = fault=='D_timeout' and phase=='probe' or fault in ('sample_oom','sample_timeout') and sample_index==1
        artifacts=[]
        def save(path,content=b'CPU FIXTURE'):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content);artifacts.append(artifact(path))
        if phase=='probe':
            ids=[r['probe_id'] for r in package.probes]
            for pid in ids if not error else ids[:1]:
                for ext in ('.npz','.json'):save(output/(pid+ext))
        elif phase=='train':
            n=199 if fault=='R_partial' and arm==ARMS[1] else 200
            output.mkdir(exist_ok=False)
            torch.save(dict(correction_arm=arm,package_sha256=PACKAGE_SHA,initialization_sha256=sha256(initial),
                execution=dict(updates_completed=n,updates_attempted=n,final_update_200=n==200)),output/'model.pt')
            artifacts.append(artifact(output/'model.pt'));ids=[]
        else:
            ids=[package.samples[sample_index]['case_id']]
            if not error:
                motion=np.zeros((1,2,22,3),np.float32)
                if fault=='static_mismatch' and sample_index==65:motion+=1
                np.savez(output,motion=motion);artifacts.append(artifact(output));save(output.with_suffix('.json'),b'{}')
        receipt=dict(status='failed' if error else 'completed',phase=phase,context=context,initialization_sha256=sha256(initial),
            mdm_examples=480 if phase=='probe' else 600 if phase=='train' else 4000,
            encoder_examples=0,exposures=600 if phase=='train' else 0,
            updates_completed=n if phase=='train' else 0,updates_attempted=n if phase=='train' else 0,
            denoising_calls=1000 if phase=='sample' else 0,branch_examples=4000 if phase=='sample' else 0,
            completed_slots=(ids[:1] if phase=='probe' else []) if error else ids,artifacts=artifacts)
        atomic_json(context['receipt_path'],receipt)
        if error:
            return dict(status='failed',launched=True,reason='wall_timeout_process_group_killed' if fault.endswith('timeout') else 'child_failed',returncode=1)
        return dict(status='completed',launched=True,reason=None,returncode=0)
    monkeypatch.setattr(module,'supervise',supervise)
    original_atomic=module.atomic_json
    def write_status(path,value):
        original_atomic(path,value)
        if fault=='between_samples_timeout' and Path(path).name=='status.json' and value['samples'][0]['status']=='completed' and value['samples'][1]['status']=='planned':
            now[0]=value['sampling_deadline_monotonic']
    monkeypatch.setattr(module,'atomic_json',write_status)
    if fault:
        with pytest.raises((RuntimeError,ValueError,TimeoutError)):
            module.execute(manifest,run,approval,'GPU-fixture',clock=lambda:now[0])
    else:module.execute(manifest,run,approval,'GPU-fixture',clock=lambda:now[0])
    status=json.loads((run/'runtime/status.json').read_text())
    assert len(status['samples'])==120 and len(status['probes'])==480
    assert status['total_started_monotonic']==100 and status['total_deadline_monotonic']==7300
    assert status['status']==('failed' if fault else 'completed')
    if fault=='preflight':assert not launches
    else:assert launches[0]==('probe',None,707.)
    if fault=='D_timeout':assert len(launches)==1
    if fault=='R_partial':assert [x[0] for x in launches]==['probe','train','train']
    samples=[x for x in launches if x[0]=='sample']
    if samples:
        assert len({x[2] for x in samples})==1
        if fault in ('sample_oom','sample_timeout'):
            assert status['samples'][0]['status']=='completed' and status['samples'][1]['status']=='failed'
            assert all(s['status']=='not_run_after_stop' for s in status['samples'][2:])
            assert (run/status['samples'][0]['output']).exists() and len(samples)==2
    if fault=='static_mismatch':assert status['samples'][65]['status']=='failed' and len(samples)==66
    if fault=='between_samples_timeout':
        assert len(samples)==1 and status['samples'][0]['status']=='completed'
        assert status['samples'][1]['status']=='failed' and status['samples'][1]['launched'] is False
    if not fault:
        assert len(samples)==120
        assert all(s['status']=='completed' for s in status['samples'])


def test_correction_authorization_cannot_reuse_old_or_change_bindings(tmp_path):
    m=script('run_mdm_correction_bounded');p=tmp_path/'manifest';p.write_text('CPU fixture')
    a=tmp_path/'approval';run=tmp_path/'run'
    atomic_json(a,dict(authorized=True,approved_by='user',user_instruction_reference='CPU fixture',
        package_sha256=m.PACKAGE_SHA,implementation_sha256=sha256(p),run_directory=str(run),gpu_uuid='GPU-fixture'))
    m.authorization(a,p,run,'GPU-fixture')
    with pytest.raises(ValueError):m.authorization(a,p,tmp_path/'other','GPU-fixture')
    with pytest.raises(ValueError):m.authorization(a,p,run,'GPU-other')
    p.write_text('changed')
    with pytest.raises(ValueError):m.authorization(a,p,run,'GPU-fixture')


def test_worker_training_two_arms_real_control_flow_cpu_fixture(tmp_path,monkeypatch):
    """Exercise actual worker accumulation/checkpoint loop on tiny synthetic modules."""
    from dataclasses import asdict
    from itm.models.mdm_correction import array_sha,PACKAGE_SHA
    worker=script('run_mdm_correction_worker')
    config=small_config()
    common=dict(control_config=asdict(config),fresh_seed=20260925,
                optimizer=dict(name='AdamW',lr=5e-5,betas=[.9,.999],eps=1e-8,weight_decay=.01),
                imu_normalization={'CPU_fixture':True},training_input_provenance={'CPU_fixture':True})
    rng=np.random.default_rng(4)
    rows=[]
    for j in range(600):
        n=2+j%3;digest=f'{j+1:064x}';noise=np.random.Generator(np.random.PCG64(int(digest,16))).standard_normal((1,263,1,n)).astype(np.float32)
        rows.append(dict(exposure_id=f'E{j+1:04d}',update=j//3+1,motion_id=str(j%3),sensor_config='head',frames=n,
            text_present=bool(j%4),imu_present=bool(j%5),encoder_dropout_seed=j,timestep=j%1000,
            noise=dict(address_sha256=digest,shape=[1,263,1,n],noise_sha256=array_sha(noise))))
    data={}
    for row in rows[:3]:
        n=row['frames'];imu=rng.normal(size=(n,6,12)).astype(np.float32)
        data[row['motion_id']]=dict(motion=rng.normal(size=(n,263)).astype(np.float32),imu=imu,reference=np.zeros_like(imu),
            sensor_mask=np.array([1,1,0,0,0,0],dtype=bool),timestamps=np.arange(n)/20.,text='CPU fixture')
    pkg=SimpleNamespace(common=common,schedule=rows,inputs=lambda r:data[r['motion_id']])
    initial=tmp_path/'initial.pt';create_initial_state(pkg,initial)
    class FixtureControl(torch.nn.Module):
        def __init__(self,mode):
            super().__init__();self.reference_mode=mode
            self.adapters=torch.nn.ModuleList([torch.nn.Linear(8,8) for _ in range(8)])
    class FixtureModel(torch.nn.Module):
        def __init__(self,state):
            super().__init__();cfg=MDMIMUControlConfig(**state['control_config'])
            self.imu_control_encoder=make_imu_control_encoder(cfg)
            self.seqTransEncoder=FixtureControl(cfg.reference_mode)
            self.imu_control_encoder.load_state_dict(state['imu_encoder'])
            self.seqTransEncoder.adapters.load_state_dict(state['adapters'])
        def encode_text(self,text):return torch.zeros(1,len(text),8)
        def forward(self,x,t,y):
            c=y['imu_control'];res=0
            for a in self.seqTransEncoder.adapters:
                res=res+torch.nn.functional.linear(c,a.weight,None if self.seqTransEncoder.reference_mode=='static_delta' else a.bias)
            res=torch.where(y['imu_condition_present'][:,None,None],res,0).mean(-1)[:,None,None,:]
            return x*.5+res
    def load_model(package,state,device):
        model=FixtureModel(state)
        diffusion=SimpleNamespace(q_sample=lambda x,t,noise:x*.8+noise*.2,_scale_timesteps=lambda t:t)
        return model,diffusion,model.seqTransEncoder,model.imu_control_encoder,None,SimpleNamespace(remove=lambda:None),None
    monkeypatch.setattr(worker,'load_model',load_model)
    for arm in ARMS:
        gdir=tmp_path/(arm+'_guard');gdir.mkdir();g=guard(gdir,arm=arm,clock=lambda:0.)
        initial_hash=sha256(initial);g.context['initialization_sha256']=initial_hash
        args=SimpleNamespace(initial=initial,arm=arm,output=tmp_path/arm,device='cpu')
        worker.train(pkg,args,g);g.finish()
        assert g.state['mdm_examples']==g.state['exposures']==600
        assert g.state['encoder_examples']==(600 if arm==ARMS[0] else 1200)
        files=list((tmp_path/arm).glob('*.pt'));assert len(files)==4
        final=torch.load(tmp_path/arm/'model.pt',weights_only=True)
        assert final['execution']['final_update_200'] and final['execution']['updates_completed']==200
        assert final['initial_tensor_hashes']==tensor_hashes(torch.load(initial,weights_only=True))
        if arm==ARMS[1]:assert all(not value.any() for key,value in final['adapters'].items() if key.endswith('bias'))
    for j in range(1,201):
        a=json.loads((tmp_path/ARMS[0]/f'update_{j:03d}.json').read_text())['examples']
        b=json.loads((tmp_path/ARMS[1]/f'update_{j:03d}.json').read_text())['examples']
        assert [(r['exposure_id'],r['actual_noise_sha256']) for r in a]==[(r['exposure_id'],r['actual_noise_sha256']) for r in b]


def test_worker_probe_records_same_forward_activations_and_padding(tmp_path,monkeypatch):
    worker=script('run_mdm_correction_worker')
    from dataclasses import asdict
    from itm.models.mdm_correction import artifact
    from itm.data.random_plan import sampling_noise
    config=small_config();rows=[];noise={};items={}
    rng=np.random.default_rng(3)
    ids=['000027','000506','004728']
    for n,mid in enumerate(ids,2):
        imu=rng.normal(size=(n,6,12)).astype(np.float32)
        items[mid]=dict(motion=rng.normal(size=(n,263)).astype(np.float32),imu=imu,reference=np.zeros_like(imu),
            sensor_mask=np.array([1,1,0,0,0,0],bool),timestamps=np.arange(n)/20.,text='fixture')
        _,valid=sampling_noise([1,263,1,n],backbone_id='CPU',case_id=mid,sample_seed=1,step=50)
        _,padding=sampling_noise([1,263,1,158-n],backbone_id='CPU',case_id=mid+'-padding',sample_seed=1,step=50)
        noise[mid]=dict(valid_noise=valid,padding_noise=padding)
        for condition in ('text_only','paired','physical_static','tail_reversed'):
            rows.append(dict(probe_id=f'D{len(rows):04d}',split='train',motion_id=mid,sensor_config='head',frames=n,
                timestep=50,noise_seed=1,condition=condition,context='exact_length',context_frames=n,noise_key=mid,
                text_present=True,imu_present=condition!='text_only'))
    for condition in ('text_only','paired'):
        for mid in ids:
            row=next(r for r in rows if r['motion_id']==mid and r['condition']==condition)
            rows.append(dict(row,probe_id=f'D{len(rows):04d}',context='padded',context_frames=158))
    path=tmp_path/'fixture.pt';torch.save({'control_config':asdict(config)},path)
    pkg=SimpleNamespace(design={'D':{'checkpoint':artifact(path)}},probes=rows,noise=noise,
                        inputs=lambda r:items[r['motion_id']])
    class FakeMDM(TinyMDM):
        def __init__(self):
            super().__init__();self.seqTransEncoder=torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(8,2,16,dropout=0.),8)
            self.project=torch.nn.Linear(263,8);self.unproject=torch.nn.Linear(8,263)
        def encode_text(self,text):return torch.zeros(1,len(text),8)
        def forward(self,x,t,y):
            h=self.project(x.squeeze(2).permute(2,0,1));h=torch.cat((torch.zeros_like(h[:1]),h),0)
            return self.unproject(self.seqTransEncoder(h)[1:]).permute(1,2,0).unsqueeze(2)
    def load(*a):
        model=FakeMDM();controlled,enc,h=install_imu_control(model,config);model.eval()
        d=SimpleNamespace(q_sample=lambda x,t,noise:x*.8+noise*.2,_scale_timesteps=lambda t:t,num_timesteps=1000,
                          betas=np.ones(1000)*.001,timestep_map=list(range(1000)))
        return model,d,controlled,enc,None,h,None
    monkeypatch.setattr(worker,'load_model',load)
    g=guard(tmp_path,phase='probe',clock=lambda:0.)
    worker.probe(pkg,SimpleNamespace(output=tmp_path/'probes',device='cpu'),g)
    assert g.state['mdm_examples']==g.state['encoder_examples']==18
    assert len(g.state['completed_slots'])==18
    for mid in ids:
        exact=next(r for r in rows if r['motion_id']==mid and r['condition']=='paired' and r['context']=='exact_length')
        padded=next(r for r in rows if r['motion_id']==mid and r['context']=='padded' and r['condition']=='paired')
        with np.load(tmp_path/'probes'/(exact['probe_id']+'.npz')) as a,np.load(tmp_path/'probes'/(padded['probe_id']+'.npz')) as b:
            assert np.array_equal(a['x_t'],b['x_t'])
            assert a['hidden_0'].shape==b['hidden_0'].shape==(exact['frames'],8)
            record=json.loads((tmp_path/'probes'/(exact['probe_id']+'.json')).read_text())
            assert record['reductions']['hidden_0']['count']==exact['frames']*8
            assert record['status']=='completed'


def test_worker_complete_sampling_cpu_fixture_static_equals_null(tmp_path,monkeypatch):
    worker=script('run_mdm_correction_worker')
    from dataclasses import asdict
    from itm.models.mdm_correction import environment,PACKAGE_SHA
    config=small_config(reference_mode='static_delta');rng=np.random.default_rng(10);n=3
    a=rng.normal(size=(n,6,3)).astype(np.float32);o=np.broadcast_to(np.eye(3,dtype=np.float32),(n,6,3,3)).copy()
    norm={'acceleration_mean':np.zeros((6,3)), 'acceleration_std':np.ones((6,3))}
    rows=[dict(arm=ARMS[1],split='val',motion_id='fixture',sensor_config='head',frames=n,condition=c,
               case_id=c,random_plan_case_id='CPU_fixture',sampling_seed=1,text_present=True,
               imu_present=c!='text_only') for c in ('paired','text_only','physical_static')]
    def item(row):
        u,ref,raw_a,raw_o=transform_inputs(a,o,norm,row['condition'])
        return dict(motion=np.zeros((n,263),np.float32),imu=u,reference=ref,raw_acceleration=raw_a,raw_orientation=raw_o,
            original_orientation0=o[0],sensor_mask=np.array([1,1,0,0,0,0],bool),timestamps=np.arange(n)/20.,
            gt=np.zeros((n,22,3),np.float32),text='fixture',metadata={'CPU_fixture':True})
    pkg=SimpleNamespace(samples=rows,inputs=item,common={'backbone':{'sha256':'CPU_fixture'}})
    state=dict(correction_arm=ARMS[1],package_sha256=PACKAGE_SHA,initialization_sha256='CPU_fixture',environment=environment(),
               execution={'final_update_200':True,'updates_completed':200},control_config=asdict(config),training_input_provenance={'CPU_fixture':True})
    checkpoint=tmp_path/'fixture.pt';torch.save(state,checkpoint)
    class FakeModel(torch.nn.Module):
        njoints=263;nfeats=1;rot2xyz=None;translation=True;data_rep='hml_vec';cond_mode='text'
        def __init__(self):
            super().__init__();self.seqTransEncoder=torch.nn.Identity()
        def encode_text(self,text):return torch.ones(1,len(text),8)
        def forward(self,x,t,y=None):
            c=y['imu_control'].mean(-1)[:,None,None,:]
            return x*.99+c*.0001+y['text_embed'].mean((0,2))[:,None,None,None]*.0001
    def load(*a):
        with isolated_rng(1,'cpu'):enc=make_imu_control_encoder(config).eval()
        model=FakeModel().eval()
        class D:
            num_timesteps=1000;timestep_map=list(range(1000))
            def p_mean_variance(self,model,x,t,**kw):
                p=model(x,t,kw['model_kwargs']['y'])
                return dict(mean=p,log_variance=torch.full_like(x,-1000),pred_xstart=p)
        backbone=SimpleNamespace(decode_motion=lambda f:f[:,:66,0].transpose(1,2).reshape(1,n,22,3))
        return model,D(),None,enc,model.seqTransEncoder,SimpleNamespace(remove=lambda:None),backbone
    monkeypatch.setattr(worker,'load_model',load)
    for slot,row in enumerate(rows):
        folder=tmp_path/row['case_id'];folder.mkdir();g=guard(folder,phase='sample',arm=ARMS[1],clock=lambda:0.)
        g.context['initialization_sha256']='CPU_fixture'
        worker.sample(pkg,SimpleNamespace(slot=slot,arm=ARMS[1],checkpoint=checkpoint,device='cpu',output=folder/'sample.npz'),g)
        g.finish();assert g.state['denoising_calls']==1000 and g.state['mdm_examples']==4000 and g.state['encoder_examples']==2
    with np.load(tmp_path/'text_only/sample.npz') as x,np.load(tmp_path/'physical_static/sample.npz') as y:
        assert np.array_equal(x['motion'],y['motion'])
        assert json.loads(str(x['metadata']))['status']=='generated'
        assert json.loads(str(x['metadata']))['random_plans']==json.loads(str(y['metadata']))['random_plans']
