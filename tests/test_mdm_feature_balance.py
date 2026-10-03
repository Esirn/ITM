"""F CPU-only fixtures: independent derivatives, provenance, real tiny workers."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from itm.models.mdm_feature_balance import (FeaturePackage, FeatureGuard, PACKAGE, PACKAGE_SHA, SCHEMA, ARM,
    feature_losses, parameter_contract, audit_training, validate_state, ForwardAccounting)
from itm.models.mdm_correction import tensor_hashes, array_sha, artifact, read, environment, accumulate_update
from itm.models.mdm_imu_control import MDMIMUControlConfig, make_imu_control_encoder
from itm.models.mdm_run_limits import atomic_json, RunLimitExceeded
from itm.backbones.mdm import MDMBackbone
from test_mdm_sampling_alignment import (script, forbid_gpu_and_models, Model, Encoder, Diffusion, fixture_item,
    plan, traces, GUIDANCE, digest)


def guard(path,phase='train',clock=lambda:0.,slots=None,trace_ids=None):
    path.mkdir(parents=True,exist_ok=True)
    return FeatureGuard(dict(version='itm-feature-context-v1',phase=phase,package_sha256=PACKAGE_SHA,
        implementation_sha256='CPU_fixture',run_directory=str(path.parent),deadline_monotonic=600.,
        receipt_path=str(path/'receipt.json'),expected_slots=slots if slots is not None else [f'U{i:03d}' for i in range(1,201)],
        expected_traces=trace_ids or [],checkpoint={'path':'CPU_fixture','sha256':'CPU_fixture'}),clock=clock)


@pytest.fixture(scope='module')
def package():return FeaturePackage()


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_independent_scalar_gradient_true_lengths_and_padding(dtype):
    lengths=[104,136,158];n=max(lengths)
    a=np.random.default_rng(8).normal(size=(3,263,1,n))
    x=torch.tensor(a,dtype=dtype,requires_grad=True);mask=torch.arange(n)[None]<torch.tensor(lengths)[:,None]
    losses=feature_losses(x,torch.zeros_like(x),mask)
    losses['balanced'].mean().backward()
    expected=np.zeros_like(a);values=[]
    for i,l in enumerate(lengths):
        e=x.detach().double().numpy()[i,:,0,:l]
        values.append(.5*(np.square(e[:67]).mean()+np.square(e[67:]).mean()))
        expected[i,:67,0,:l]=e[:67]/(67*l*3)
        expected[i,67:,0,:l]=e[67:]/(196*l*3)
    tol=dict(atol=1e-7,rtol=1e-6) if dtype==torch.float32 else dict(atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(losses['balanced'].detach(),values,**tol)
    np.testing.assert_allclose(x.grad,expected,**tol)
    altered=x.detach().clone();altered[~mask[:,None,None].expand_as(altered)]=float('nan')
    torch.testing.assert_close(feature_losses(altered,torch.zeros_like(x),mask)['balanced'],losses['balanced'],**tol)


def test_finite_differences_boundary_channels_and_invalid_frames():
    x=torch.linspace(-.9,1.1,263*4,dtype=torch.float64).reshape(1,263,1,4).requires_grad_()
    mask=torch.tensor([[True,True,True,False]])
    f=lambda v:feature_losses(v,torch.zeros_like(v),mask)['balanced'].sum()
    f(x).backward()
    for c,t in [(0,0),(66,1),(67,2),(262,2),(5,3)]:
        a=x.detach().clone();b=a.clone();a[0,c,0,t]+=1e-5;b[0,c,0,t]-=1e-5
        assert float((f(a)-f(b))/2e-5)==pytest.approx(float(x.grad[0,c,0,t]),abs=1e-11)
    with pytest.raises(ValueError):feature_losses(x,x,torch.zeros_like(mask))
    with pytest.raises(ValueError):feature_losses(x[:,:262],x[:,:262],mask)


def test_uniform_reduction_original_gradient_adam_update_and_null_denominator(tmp_path):
    original=torch.nn.Parameter(torch.linspace(-1,1,263*7).reshape(1,263,1,7))
    candidate=torch.nn.Parameter(original.detach().clone())
    oa=torch.optim.AdamW([original],lr=5e-5);ob=torch.optim.AdamW([candidate],lr=5e-5)
    g=guard(tmp_path/'g')
    def loss(p,kind):
        if kind=='uniform':return feature_losses(p,torch.zeros_like(p),torch.ones(1,7,dtype=torch.bool))['uniform'].mean()
        return p.square().mean()
    accumulate_update([lambda:loss(original,'old'),lambda:torch.tensor(.2),lambda:2*loss(original,'old')],oa,[original],g)
    h=guard(tmp_path/'h')
    accumulate_update([lambda:loss(candidate,'uniform'),lambda:torch.tensor(.2),lambda:2*loss(candidate,'uniform')],ob,[candidate],h)
    torch.testing.assert_close(original.grad,candidate.grad,atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(original,candidate,atol=1e-7,rtol=1e-6)
    # Two differentiable terms /3, including the no-grad third denominator.
    torch.testing.assert_close(original.grad,2*torch.linspace(-1,1,263*7).reshape_as(original)/(263*7))


def test_prepare_without_model_gpu_or_initialization(tmp_path,monkeypatch,package):
    forbid_gpu_and_models(monkeypatch)
    bounded=script('run_mdm_feature_balance_bounded')
    monkeypatch.setattr(bounded,'preflight_gpu',lambda *a:pytest.fail('GPU preflight'))
    monkeypatch.setattr(bounded,'supervise',lambda *a,**k:pytest.fail('worker launch'))
    m=bounded.prepare(tmp_path/'prepared')
    assert len(m['inputs'])==60 and len(m['training_inputs'])==6
    assert m['initial_tensor_hashes']==package.initial_state()['tensor_hashes']
    assert m['new_checkpoints']==m['new_parameter_initializations']==m['real_model_forwards']==0
    assert [p.name for p in (tmp_path/'prepared').iterdir()]==['implementation.json']


def tiny_training_package():
    cfg=MDMIMUControlConfig(latent_dim=8,encoder_heads=2,encoder_layers=1,reference_mode='static_delta',null_mode='strict')
    enc=make_imu_control_encoder(cfg);adapters=torch.nn.ModuleList([torch.nn.Linear(8,8) for _ in range(2)])
    for a in adapters:torch.nn.init.zeros_(a.weight);torch.nn.init.zeros_(a.bias)
    initial=dict(imu_encoder=enc.state_dict(),adapters=adapters.state_dict(),control_config=asdict(cfg))
    initial['tensor_hashes']=tensor_hashes(initial)
    rows=[]
    for j in range(600):
        n=2+j%3;address=f'{j+1:064x}'
        noise=np.random.Generator(np.random.PCG64(int(address,16))).standard_normal((1,263,1,n)).astype(np.float32)
        rows.append(dict(exposure_id=f'E{j+1:04d}',update=j//3+1,motion_id=str(j%3),sensor_config='head',frames=n,
            condition='paired',text_present=bool(j%4),imu_present=bool(j%5),encoder_dropout_seed=j,timestep=j%1000,
            noise=dict(address_sha256=address,shape=[1,263,1,n],noise_sha256=array_sha(noise))))
    defaults=dict(name='AdamW',lr=5e-5,betas=[.9,.999],eps=1e-8,weight_decay=.01,amsgrad=False,foreach=None,
                  maximize=False,capturable=False,differentiable=False,fused=None)
    pkg=SimpleNamespace(common=dict(control_config=asdict(cfg),optimizer=defaults),schedule=rows,
        initial_state=lambda:deepcopy(initial),identity=lambda s:dict(loss_version='CPU_fixture'),inputs=fixture_item)
    return pkg,initial


class TrainModel(torch.nn.Module):
    def __init__(self,state):
        super().__init__();self.seqTransEncoder=torch.nn.Module()
        self.seqTransEncoder.reference_mode='static_delta'
        self.seqTransEncoder.adapters=torch.nn.ModuleList([torch.nn.Linear(8,8) for _ in range(2)])
        self.imu_control_encoder=make_imu_control_encoder(MDMIMUControlConfig(**state['control_config']))
        self.seqTransEncoder.adapters.load_state_dict(state['adapters']);self.imu_control_encoder.load_state_dict(state['imu_encoder'])
        self.calls=0
    def encode_text(self,text):return torch.zeros(1,len(text),8)
    def forward(self,x,t,y):
        self.calls+=1
        if not y['imu_condition_present'].any():return x*.5
        c=y['imu_control'];r=sum(torch.nn.functional.linear(c,a.weight,None) for a in self.seqTransEncoder.adapters)
        return x*.5+r.mean(-1)[:,None,None,:]


@pytest.mark.parametrize('fault',[None,'oom','nan','timeout','write'])
def test_real_training_loop_200_updates_and_first_failure(tmp_path,monkeypatch,fault):
    worker=script('run_mdm_feature_balance_worker');pkg,initial=tiny_training_package();removed=[];now=[0.]
    models=[]
    def load(pkg,state,device):
        m=TrainModel(state);models.append(m);original=m.forward
        def forward(*a,**k):
            if m.calls==3 and fault=='oom':raise RuntimeError('CPU injected out of memory')
            value=original(*a,**k)
            if m.calls==4 and fault=='nan':value=value*float('nan')
            if m.calls==4 and fault=='timeout':now[0]=601.
            return value
        m.forward=forward
        return m,SimpleNamespace(q_sample=lambda x,t,noise:x*.8+noise*.2,_scale_timesteps=lambda t:t),m.seqTransEncoder,m.imu_control_encoder,None,SimpleNamespace(remove=lambda:removed.append(True)),None
    monkeypatch.setattr(worker,'load_model',load)
    if fault=='write':
        save=worker.save_json
        def fail(path,*args):
            if Path(path).name=='E0004.json':raise OSError('CPU injected write failure')
            return save(path,*args)
        monkeypatch.setattr(worker,'save_json',fail)
    g=guard(tmp_path/'guard',clock=lambda:now[0]);out=tmp_path/'train'
    # All counters remain real; avoid repeated receipt fsync in the 600-step tiny fixture.
    g.persist=lambda:None
    if fault:
        with pytest.raises((RuntimeError,FloatingPointError,RunLimitExceeded,OSError)):
            worker.train(pkg,SimpleNamespace(output=out,device='cpu'),g)
        assert (out/'update_001.json').exists() and not (out/'model.pt').exists()
        assert g.state['updates_completed']==1
    else:
        worker.train(pkg,SimpleNamespace(output=out,device='cpu'),g);g.finish()
        assert g.state['mdm_calls']==g.state['exposures']==600 and g.state['encoder_examples']==1200
        assert len(list(out.glob('*.pt')))==4 and len(list((out/'exposures').glob('*.json')))==600
        state=torch.load(out/'model.pt',weights_only=True)
        assert state['checkpoint_schema']==SCHEMA and state['arm']==ARM
        assert state['training_audit']['trainable']==parameter_contract(initial)
        for row in pkg.schedule:
            entry=read(out/'exposures'/(row['exposure_id']+'.json'))
            assert all(entry[k]==v for k,v in row.items())
            assert entry['loss']==entry['loss_components']['balanced']
            assert entry['loss']==pytest.approx(.5*(entry['loss_components']['root_ric']+entry['loss_components']['remaining']),rel=1e-6)
        assert read(out/'exposures/E0001.json')['backward_skipped']
    assert removed and not models[0]._forward_hooks and not models[0]._forward_pre_hooks


@pytest.mark.parametrize('fault',[None,'trace_write','combination','nan','decode'])
def test_real_sampling_path_1000_calls_traces_and_failures(tmp_path,monkeypatch,fault):
    worker=script('run_mdm_feature_balance_worker');p=plan(3)
    history=tmp_path/'historical.json';atomic_json(history,dict(random_plans=[p]))
    row=dict(slot_index=0,case_id='cpu',frames=3,condition='paired',text_present=True,imu_present=True,
        historical_case_id='old',guidance=GUIDANCE,random_plan_case_id='cpu',sampling_seed=1,
        historical_metadata=artifact(history),historical_random_plan_sha256=digest(p))
    model=Model();encoder=Encoder();model.eval();encoder.eval();removed=[]
    backbone=MDMBackbone(model,Diffusion(),torch.zeros(263),torch.ones(263),lambda v,n:v[...,:66].reshape(1,1,3,22,3))
    if fault=='decode':
        decode=backbone.decode_motion
        backbone.decode_motion=lambda features:decode(features)+.01
    state=dict(initialization_sha256='CPU',control_config={},training_input_provenance={},loss_version='CPU',
        loss_weights=[],loss_weights_sha256='CPU',training_schedule_sha256='CPU',initial_tensor_hashes={},training_audit={},
        initial_package_sha256='CPU',execution={},optimizer_defaults={},environment={},training_run_directory='CPU_fixture')
    monkeypatch.setattr(worker,'load_final',lambda *a:(state,(model,Diffusion(),None,encoder,torch.nn.Identity(),SimpleNamespace(remove=lambda:removed.append(True)),backbone)))
    pkg=SimpleNamespace(samples=[row],traces=traces(),inputs=fixture_item,common={'backbone':{'sha256':'CPU'}})
    g=guard(tmp_path/'g','sample',slots=['cpu'],trace_ids=[r['trace_id'] for r in traces()]);g.persist=lambda:None
    out=tmp_path/'run/samples/cpu.npz'
    if fault=='trace_write':
        import itm.models.mdm_sampling_alignment as alignment
        save=alignment.save_json
        def fail(path,*args):
            if Path(path).name=='T0002.json':raise OSError('CPU write')
            return save(path,*args)
        monkeypatch.setattr(alignment,'save_json',fail)
    if fault=='combination':
        factory=worker.make_text_imu_guidance_model
        def changed(*a,**k):
            v=factory(*a,**k);v.register_forward_hook(lambda m,a,result:result+1);return v
        monkeypatch.setattr(worker,'make_text_imu_guidance_model',changed)
    if fault=='nan':
        original=model.forward
        def changed(*a,**k):return original(*a,**k)*float('nan')
        model.forward=changed
    if fault:
        with pytest.raises((OSError,ValueError,FloatingPointError)):worker.sample(pkg,SimpleNamespace(slot=0,device='cpu',output=out),g)
        assert not out.exists()
        if fault=='trace_write':assert (out.parent.parent/'traces/T0002.npz').exists()
        if fault=='nan':assert g.state['counts']['mdm']['actual']==4 and g.state['counts']['mdm']['completed']==0
        if fault=='decode':
            raw=out.parent.parent/'raw_samples/cpu.npz'
            assert raw.exists() and len(g.state['completed_traces'])==4
            with np.load(raw) as z:assert np.isfinite(z['normalized_humanml263']).all()
    else:
        worker.sample(pkg,SimpleNamespace(slot=0,device='cpu',output=out),g);g.finish()
        assert model.calls==1000 and encoder.calls==2 and g.state['caption_calls']==1
        meta=read(out.with_suffix('.json'));assert meta['random_plans']==[p] and meta['decode_check']['passed']
    assert removed and not model._forward_hooks and not model._forward_pre_hooks and not encoder._forward_hooks


def checkpoint_fixture(package):
    initial=package.initial_state();s=package.identity('CPU_fixture')
    cfg=asdict(MDMIMUControlConfig(**dict(package.common['control_config'],reference_mode='static_delta')))
    s.update(checkpoint_schema=SCHEMA,arm=ARM,control_config=cfg,imu_encoder=deepcopy(initial['imu_encoder']),
        adapters=deepcopy(initial['adapters']),training_audit=dict(trainable=parameter_contract(initial),frozen_unchanged=True),
        execution=dict(updates_completed=200,updates_attempted=200,final_update_200=True,exposures=600,
                       mdm_examples=600,encoder_examples=1200,caption_calls=600))
    s['imu_encoder'][next(iter(s['imu_encoder']))].view(-1)[0]+=.01  # Synthetic fixture, never a runnable model.
    s['tensor_hashes']=tensor_hashes(s)
    defaults=dict(package.common['optimizer']);defaults.pop('name');defaults['betas']=tuple(defaults['betas'])
    s['optimizer_defaults']=defaults
    s['optimizer']=dict(param_groups=[dict(params=list(range(len(parameter_contract(initial)))))],state={0:{'step':torch.tensor(200.)}})
    return s


@pytest.mark.parametrize('field',['schema','loss_version','weights','initial','schedule','implementation','environment',
                                  'parameter_order','intermediate','optimizer','bias','tensor_hash','historical_relabel'])
def test_checkpoint_rejects_wrong_provenance_before_model_construction(package,monkeypatch,field):
    s=checkpoint_fixture(package)
    forbid_gpu_and_models(monkeypatch)
    if field=='schema':s['checkpoint_schema']='itm-correction-control-v1'
    if field=='loss_version':s['loss_version']='wrong'
    if field=='weights':s['loss_weights'][66]=1.
    if field=='initial':s['initialization_sha256']='wrong'
    if field=='schedule':s['training_schedule_sha256']='wrong'
    if field=='implementation':s['implementation_sha256']='wrong'
    if field=='environment':s['environment']['threads']=999
    if field=='parameter_order':s['training_audit']['trainable'].reverse()
    if field=='intermediate':s['execution']['updates_completed']=150
    if field=='optimizer':s['optimizer_defaults']['lr']=.001
    if field=='bias':
        s['adapters']['0.bias'][0]=.1;s['tensor_hashes']=tensor_hashes(s)
    if field=='tensor_hash':s['tensor_hashes']={}
    if field=='historical_relabel':
        old=torch.load(package.design['historical_checkpoint']['path'],weights_only=True,map_location='cpu')
        s.update(imu_encoder=old['imu_encoder'],adapters=old['adapters']);s['tensor_hashes']=tensor_hashes(s)
    with pytest.raises(ValueError):validate_state(s,package,'CPU_fixture')


def test_all48_real_software_equalities_and_no_imu_only_cross_arm_check(tmp_path,package):
    bounded=script('run_mdm_feature_balance_bounded')
    from itm.models.mdm_feature_balance import ledger
    status=ledger(package)
    for row,spec in zip(status['samples'],package.samples):
        p=tmp_path/row['output'];p.parent.mkdir(parents=True,exist_ok=True)
        with np.load(spec['historical_output']['path']) as z:motion=z['motion'].copy()
        if spec['condition']=='imu_only':motion+=100  # F/U IMU-only is explicitly not an equality.
        np.savez(p,motion=motion);row['status']='completed'
    bounded.run_checks(status,package,tmp_path)
    assert len(status['checks'])==48 and all(r['passed'] for r in status['checks'])
    s=deepcopy(status)
    for r in s['checks']:r['status']='planned'
    bad=next(x for x in package.samples if x['condition']=='physical_static')
    p=tmp_path/'samples'/(bad['case_id']+'.npz')
    with np.load(p) as z:v=z['motion'].copy()
    np.savez(p,motion=v+.001)
    with pytest.raises(ValueError):bounded.run_checks(s,package,tmp_path)
    assert p.exists() and any(r['status']=='failed' for r in s['checks'])


@pytest.mark.parametrize('scenario',[None,'preflight','training_timeout','sampling_timeout','oom','missing_receipt',
    'missing_artifact','artifact_hash','wrong_counts','missing_trace','checkpoint','write'])
def test_queue_full_ledgers_shared_deadlines_and_preservation(tmp_path,monkeypatch,package,scenario):
    bounded=script('run_mdm_feature_balance_bounded');run=tmp_path/'run';now=[10.];calls=[]
    manifest=tmp_path/'implementation.json'
    atomic_json(manifest,dict(version='itm-feature-implementation-v1',package=artifact(PACKAGE),
        initial=artifact(package.initial),environment=environment(),expected_artifacts=[]))
    approval=tmp_path/'approval.json'
    atomic_json(approval,dict(authorized=True,approved_by='user',user_instruction_reference='CPU fixture; all GPU/launch functions replaced',
        package_sha256=PACKAGE_SHA,implementation_sha256=artifact(manifest)['sha256'],run_directory=str(run),gpu_uuid='GPU-CPU-fixture'))
    pkg=SimpleNamespace(**vars(package))
    def state(*args):
        if scenario=='checkpoint':raise ValueError('CPU invalid final checkpoint')
        return dict(training_run_directory=str(run))
    pkg.state=state
    monkeypatch.setattr(bounded,'FeaturePackage',lambda *a:pkg)
    def preflight(gpu):
        s=read(run/'runtime/status.json')
        assert tuple(len(s[k]) for k in ('updates','exposures','samples','traces','checks'))==(200,600,60,240,48)
        assert s['training_deadline_monotonic']==610 and s['total_deadline_monotonic']==3610
        if scenario=='preflight':raise RuntimeError('CPU preflight')
        now[0]=20.;return dict(CPU_fixture=True)
    monkeypatch.setattr(bounded,'preflight_gpu',preflight)
    def fake(argv,deadline,log_path,env,clock):
        phase=argv[argv.index('--phase')+1];output=Path(argv[argv.index('--output')+1])
        ctx=read(argv[argv.index('--run-context')+1]);calls.append(ctx)
        if phase=='train':assert deadline==610
        else:assert deadline==3030  # Train completed at30; shared sample deadline never resets.
        train=phase=='train'
        counts={k:dict(reserved=n,actual=n,completed=n,calls=(1000 if k=='mdm' and not train else n),
            completed_calls=(1000 if k=='mdm' and not train else n)) for k,n in dict(mdm=600 if train else 4000,encoder=1200 if train else 2,caption=600 if train else 1).items()}
        c=dict(status='completed',context=ctx,counts=counts,updates_attempted=200 if train else 0,
            updates_completed=200 if train else 0,exposures=600 if train else 0,denoising_calls=0 if train else 1000,
            branch_examples=0 if train else 4000,completed_slots=ctx['expected_slots'],completed_traces=ctx['expected_traces'],artifacts=[])
        for name in sorted(bounded.required_artifacts(ctx,output,package)):
            p=Path(name);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('CPU fixture artifact, not actual output')
            c['artifacts'].append(artifact(p))
        if scenario=='write' and train:raise OSError('CPU parent write')
        if scenario=='missing_receipt' and not train:return dict(status='completed',launched=True,returncode=0)
        if scenario=='missing_artifact' and not train:c['artifacts'].pop()
        if scenario=='artifact_hash' and not train:c['artifacts'][0]['sha256']='wrong'
        if scenario=='wrong_counts' and not train:c['counts']['mdm']['completed']-=4
        if scenario=='missing_trace' and not train:c['completed_traces']=ctx['expected_traces'][:-1]
        failed=scenario=='oom' and not train
        if failed:c.update(status='failed',failure={'message':'CPU OOM'})
        atomic_json(ctx['receipt_path'],c)
        now[0]=30. if train else now[0]+1
        if scenario=='training_timeout' and train:now[0]=611.
        if scenario=='sampling_timeout' and not train:now[0]=3031.
        return dict(status='failed' if failed else 'completed',launched=True,returncode=1 if failed else 0)
    monkeypatch.setattr(bounded,'supervise',fake)
    def checks(status,pkg,run):
        completed={s['case_id'] for s in status['samples'] if s['status']=='completed'}
        for row,spec in zip(status['checks'],pkg.checks):
            if spec['new_case_id'] in completed:row['status']='completed'
    monkeypatch.setattr(bounded,'run_checks',checks) # Actual 48-array comparisons tested separately above.
    if scenario:
        with pytest.raises((ValueError,RuntimeError,TimeoutError,OSError)):
            bounded.execute(manifest,run,approval,'GPU-CPU-fixture',clock=lambda:now[0])
    else:bounded.execute(manifest,run,approval,'GPU-CPU-fixture',clock=lambda:now[0])
    s=read(run/'runtime/status.json')
    assert tuple(len(s[k]) for k in ('updates','exposures','samples','traces','checks'))==(200,600,60,240,48)
    assert all(row['status'] not in ('planned','running') for key in ('updates','exposures','samples','traces','checks') for row in s[key])
    if not scenario:
        assert len(calls)==61 and s['status']=='completed'
        assert s['counts']['mdm']['actual']==240600 and s['counts']['mdm']['calls']==60600
        assert s['counts']['encoder']['actual']==1320 and s['counts']['caption']['calls']==660
    else:
        assert len(calls)<=2 and s['status']=='failed'
        if scenario not in ('preflight',):assert (run/'F/update_001.json').exists()
    with pytest.raises(FileExistsError):bounded.execute(manifest,run,approval,'GPU-CPU-fixture',clock=lambda:now[0])


def test_authorization_and_call_boundaries(tmp_path):
    bounded=script('run_mdm_feature_balance_bounded')
    manifest=tmp_path/'m';manifest.write_text('CPU fixture')
    approval=tmp_path/'a';run=tmp_path/'r'
    a=dict(authorized=True,approved_by='user',user_instruction_reference='CPU fixture',package_sha256=PACKAGE_SHA,
        implementation_sha256=artifact(manifest)['sha256'],run_directory=str(run),gpu_uuid='GPU-CPU')
    atomic_json(approval,a);bounded.authorization(approval,manifest,run,'GPU-CPU')
    for field in ('authorized','package_sha256','implementation_sha256','run_directory','gpu_uuid','user_instruction_reference'):
        atomic_json(approval,dict(a,**{field:None}))
        with pytest.raises(ValueError):bounded.authorization(approval,manifest,run,'GPU-CPU')
    g=guard(tmp_path/'g','sample',slots=['cpu'])
    with pytest.raises(RunLimitExceeded):g.before_update()
    with pytest.raises(RunLimitExceeded):g.before_denoise('cpu',998,4)
    g.before_denoise('cpu',999,4)
    with pytest.raises(RunLimitExceeded):g.enter('mdm',5)
    g.state['counts']['mdm']['reserved']=4000
    with pytest.raises(RunLimitExceeded):g.enter('mdm',4)


@pytest.mark.parametrize('condition',['paired','text_only','imu_only','physical_static','backbone_bypass','numeric_zero','tail_reversed'])
def test_F_trace_hooks_preserve_predictions_rng_cache_and_counters(tmp_path,condition):
    from itm.models.mdm_correction import ConditionCache
    from itm.models.mdm_imu_control import make_text_imu_guidance_model
    from itm.models.mdm_sampling import sample_addressed_ddpm
    from itm.models.mdm_sampling_alignment import TraceGuidance
    worker=script('run_mdm_feature_balance_worker');results=[]
    row=dict(frames=3,condition=condition,text_present=condition!='imu_only',imu_present=condition not in ('text_only','backbone_bypass'))
    for enabled in (False,True):
        model=Model();encoder=Encoder();model.eval();encoder.eval()
        g=guard(tmp_path/str(enabled),'sample',slots=['cpu'],trace_ids=[r['trace_id'] for r in traces()] if enabled else [])
        g.persist=lambda:None
        accounting=ForwardAccounting(model,encoder,g)
        _,y=worker.batch_inputs([fixture_item(row)],[row],'cpu');worker.counted_text(model,y,g)
        cache=ConditionCache()
        y['imu_control'],y['imu_condition_present']=cache.get(encoder,y,checkpoint_sha='CPU',mode='static_delta')
        cached=y['imu_control'].clone();identity=cache.identity
        y.pop('imu_reference');y.update(imu_control_mode='static_delta',text_scale=1.,imu_scale=1.,joint_scale=1.,
            backbone_bypass=torch.tensor([condition=='backbone_bypass']))
        guided=make_text_imu_guidance_model(model,mode='factorized',branch_execution='batched',bypass_encoder=torch.nn.Identity());guided.eval()
        observer=TraceGuidance(guided,model,traces(),tmp_path/'traces',row,g,lambda t:t) if enabled else None
        if observer:observer.eval()
        rng=torch.random.get_rng_state().clone()
        with torch.no_grad():
            output,p=sample_addressed_ddpm(observer or guided,Diffusion(),y=y,lengths=y['lengths'],case_ids=['cpu'],
                sample_seed=1,backbone_id='CPU',run_guard=g)
        assert torch.equal(rng,torch.random.get_rng_state()) and torch.equal(cached,cache.value[0]) and cache.identity==identity
        g.state['completed_slots']=['cpu'];g.finish()
        if observer:observer.close()
        accounting.close();results.append((output,p,g.state['counts']))
        assert not model._forward_hooks and not model._forward_pre_hooks
    torch.testing.assert_close(results[0][0],results[1][0],atol=0,rtol=0)
    assert results[0][1:]==results[1][1:]
