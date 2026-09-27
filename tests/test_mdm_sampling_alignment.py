"""01 alignment CPU fixtures. Never load a pretrained model or query CUDA."""
import ast
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from itm.backbones.mdm import MDMBackbone
from itm.data.random_plan import sampling_noise, VERSION
from itm.models.mdm_correction import (artifact,array_sha,ConditionCache,environment,transform_inputs,probe_noise)
from itm.models.mdm_imu_control import (MDMIMUControlConfig,install_imu_control,make_text_imu_guidance_model,
                                      compose_factorized_text_imu_guidance)
from itm.models.mdm_sampling import sample_addressed_ddpm
from itm.models.mdm_sampling_alignment import (AlignmentPackage,AlignmentGuard,TraceGuidance,PACKAGE,PACKAGE_SHA,
    CHECKPOINT_SHA,INITIAL_SHA,GUIDANCE,BRANCHES,describe,digest,reduced_analysis,combination_check,validate_state)
from itm.models.mdm_run_limits import atomic_json,RunLimitExceeded

ROOT=Path(__file__).resolve().parents[1]


def script(name):
    if str(ROOT/'scripts') not in sys.path:sys.path.insert(0,str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def guard(path,phase='sample',slots=None,traces=None,clock=lambda:0.):
    path.mkdir(parents=True,exist_ok=True)
    slots=slots if slots is not None else ['cpu']
    return AlignmentGuard(dict(version='itm-alignment-context-v1',phase=phase,deadline_monotonic=100.,
        checkpoint_sha256=CHECKPOINT_SHA,receipt_path=str(path/'receipt.json'),
        counters=dict(mdm_examples=480 if phase=='Q' else 4000,encoder_examples=576 if phase=='Q' else 2),
        expected_slots=slots,expected_traces=traces or [],package_sha256=PACKAGE_SHA),clock=clock)


@pytest.fixture(scope='module')
def package():return AlignmentPackage()


def forbid_gpu_and_models(monkeypatch):
    def forbidden(*a,**kw):pytest.fail('CPU preparation called a model or CUDA')
    monkeypatch.setattr(torch.nn.Module,'_call_impl',forbidden)
    for key in ('init','_lazy_init','is_available','device_count','get_device_properties','synchronize'):
        monkeypatch.setattr(torch.cuda,key,forbidden)


def test_prepare_all_frozen_inputs_without_model_or_gpu(tmp_path,monkeypatch):
    bounded=script('run_mdm_sampling_alignment_bounded');forbid_gpu_and_models(monkeypatch)
    manifest=bounded.prepare(tmp_path/'prepared')
    assert (len(manifest['inputs']),len(manifest['Q_inputs']))==(60,480)
    assert manifest['checkpoint']['sha256']==CHECKPOINT_SHA
    assert manifest['new_parameter_initializations']==manifest['new_checkpoints']==0
    assert not list((tmp_path/'prepared').glob('*.pt'))


@pytest.mark.parametrize('field',['checkpoint_arm','initialization','updates','provenance','config','environment'])
def test_reject_wrong_checkpoint_metadata_before_construction(package,field):
    state=package.state()
    if field=='checkpoint_arm':state['correction_arm']='L_exact_length'
    if field=='initialization':state['initialization_sha256']='wrong'
    if field=='updates':state['execution']['updates_completed']=199
    if field=='provenance':state['training_input_provenance']={}
    if field=='config':state['control_config']['reference_mode']='none'
    if field=='environment':state['environment']['threads']=999
    with pytest.raises(ValueError):validate_state(state,package)


def test_real_new_loader_with_upstream_nonreturning_methods(tmp_path,monkeypatch):
    worker=script('run_mdm_sampling_alignment_worker');old=sys.modules['run_mdm_correction_worker']
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.seqTransEncoder=torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(8,2,16),2)
    source=ROOT/'outputs/mdm/runtime_root/model/mdm.py'
    upstream=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='MDM')
    methods=[n for n in upstream.body if isinstance(n,ast.FunctionDef) and n.name in ('_apply','train')]
    cls=ast.ClassDef(name='Legacy',bases=[ast.Name(id='Tiny',ctx=ast.Load())],keywords=[],body=methods,decorator_list=[])
    namespace={'Tiny':Tiny};exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(source),'exec'),namespace)
    model=namespace['Legacy']();model.rot2xyz=SimpleNamespace(smpl_model=torch.nn.Linear(2,2))
    cfg=MDMIMUControlConfig(latent_dim=8,encoder_heads=2,reference_mode='static_delta',null_mode='strict')
    controlled,encoder,_=install_imu_control(Tiny(),cfg)
    state=dict(control_config=asdict(cfg),imu_encoder=encoder.state_dict(),adapters=controlled.adapters.state_dict())
    backbone=tmp_path/'backbone.pt';torch.save(model.state_dict(),backbone)
    options=tmp_path/'args.json';atomic_json(options,dict(predict_xstart=True,mask_frames=False))
    pkg=SimpleNamespace(state=lambda:state,common=dict(backbone=artifact(backbone),
        training_input_provenance=dict(payload=dict(artifacts=dict(mdm_args=artifact(options))))),
        mean=np.zeros(263,np.float32),std=np.ones(263,np.float32))
    diffusion=SimpleNamespace(num_timesteps=1000,timestep_map=list(range(1000)))
    monkeypatch.setitem(sys.modules,'train_mdm_imu_control',SimpleNamespace(_legacy_compatibility=lambda:None,_set_mdm_compatibility_defaults=lambda x:None))
    monkeypatch.setitem(sys.modules,'utils.model_util',SimpleNamespace(create_model_and_diffusion=lambda *a:(model,diffusion),load_model_wo_clip=lambda m,s:m.load_state_dict(s)))
    monkeypatch.setitem(sys.modules,'data_loaders.humanml.scripts.motion_process',SimpleNamespace(recover_from_ric=lambda *a:None))
    runtime=tmp_path/'outputs/mdm/runtime_root';runtime.mkdir(parents=True)
    monkeypatch.setattr(old,'ROOT',tmp_path);monkeypatch.setattr(sys,'path',list(sys.path));monkeypatch.chdir(tmp_path)
    forbid_gpu_and_models(monkeypatch)
    actual,loaded=worker.load_fixed(pkg,'cpu')
    assert actual is state and loaded[0] is model and loaded[1] is diffusion
    assert all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters())
    assert not model.rot2xyz.smpl_model.training
    for k,v in loaded[3].state_dict().items():assert torch.equal(v,state['imu_encoder'][k])


@pytest.mark.parametrize('scale',[1e-4,1.,1e5])
def test_actual_four_branch_composition_matches_frozen_bounds(scale):
    values=torch.tensor(np.random.default_rng(2).normal(size=(4,2,263,1,3))*scale,dtype=torch.float32)
    for wt in (1.,2.5):
        result=compose_factorized_text_imu_guidance(*values,text_scale=wt,imu_scale=1.,joint_scale=1.)
        checks=combination_check(values,result,wt)
        assert all(r['max_error_over_bound']<=1 for r in checks.values())
        with pytest.raises(ValueError):combination_check(values,result+scale,wt)
    helpful=reduced_analysis(np.zeros(2),-np.ones(2),np.ones(2),np.zeros(2))
    harmful=reduced_analysis(np.zeros(2),np.ones(2),np.ones(2),np.zeros(2))
    assert helpful['legacy_minus_unit']<0<harmful['legacy_minus_unit']


class Encoder(torch.nn.Module):
    def __init__(self):super().__init__();self.config=SimpleNamespace(latent_dim=8);self.calls=0
    def forward(self,imu,mask,frame_mask,condition_present,timestamps,return_presence):
        self.calls+=len(imu)
        return imu[:,:,0,:8]*condition_present[:,None,None],condition_present


class Model(torch.nn.Module):
    njoints=263;nfeats=1;rot2xyz=None;translation=True;data_rep='hml_vec';cond_mode='text'
    def __init__(self):
        super().__init__();self.seqTransEncoder=torch.nn.Identity();self.calls=0;self.batch5=[]
    def encode_text(self,text):return torch.ones(1,len(text),8)
    def forward(self,x,t,y=None):
        self.calls+=1
        if len(x)==5:
            self.batch5.append((x.clone(),y['text_embed'].clone(),y['imu_condition_present'].clone(),y['imu_control'].clone()))
        c=y['imu_control'].mean(-1)[:,None,None,:]
        c=c*y['imu_condition_present'][:,None,None,None]
        if y.get('_raw_backbone_bypass',False):c=torch.zeros_like(c)
        return .99*x+.0001*c+.0001*y['text_embed'].mean((0,2))[:,None,None,None]


class Diffusion:
    num_timesteps=1000;timestep_map=list(range(1000));betas=np.full(1000,.001)
    def _scale_timesteps(self,t):return t
    def q_sample(self,x,t,noise):return .8*x+.2*noise
    def p_mean_variance(self,model,x,t,**kwargs):
        p=model(x,t,y=kwargs['model_kwargs']['y'])
        return dict(mean=p,log_variance=torch.full_like(x,-1000),pred_xstart=p)


def fixture_item(row):
    n=row['frames'];rng=np.random.default_rng(4)
    a=rng.normal(size=(n,6,3)).astype(np.float32)
    o=np.broadcast_to(np.eye(3,dtype=np.float32),(n,6,3,3)).copy()
    norm=dict(acceleration_mean=np.zeros((6,3)),acceleration_std=np.ones((6,3)))
    u,ref,a,o=transform_inputs(a,o,norm,row.get('input_transform',row['condition']))
    return dict(motion=np.zeros((n,263),np.float32),imu=u,reference=ref,raw_acceleration=a,raw_orientation=o,
        original_orientation0=o[0],sensor_mask=np.array([1,1,0,0,0,0],bool),timestamps=np.arange(n)/20.,
        gt=np.zeros((n,22,3),np.float32),text='CPU fixture',metadata={'CPU_fixture':True})


def plan(n,case='cpu'):
    draws=[sampling_noise([1,263,1,n],backbone_id='CPU',case_id=case,sample_seed=1,step=s)[1] for s in [-1,*range(999,0,-1)]]
    return dict(random_plan_id=digest(draws),version=VERSION,sampler='ddpm',steps=1000,clip_denoised=False,
                noise_draws=draws,timestep_map=list(range(1000)))


def traces(case='cpu',n=3):
    return [dict(trace_id=f'T{i:04d}',case_id=case,timestep=t,shape=[1,263,1,n],status='frozen_not_executed')
            for i,t in enumerate((900,500,250,50),1)]


@pytest.mark.parametrize('condition',['paired','text_only','imu_only','physical_static','backbone_bypass','numeric_zero'])
def test_trace_on_off_actual_sampler_bitwise_rng_cache_and_counts(tmp_path,condition):
    worker=script('run_mdm_sampling_alignment_worker')
    row=dict(frames=3,condition=condition,text_present=condition!='imu_only',imu_present=condition not in ('text_only','backbone_bypass'))
    outputs=[];plans=[]
    for enabled in (False,True):
        model=Model();encoder=Encoder();model.eval();encoder.eval()
        _,y=worker.batch_inputs([fixture_item(row)],[row],'cpu');worker.embed_text(model,y)
        cache=ConditionCache();g=guard(tmp_path/str(enabled),traces=[s['trace_id'] for s in traces()] if enabled else [])
        # Counter semantics are exercised; durable receipt writes have separate queue/fault tests.
        g.persist=lambda:None
        y['imu_control'],y['imu_condition_present']=cache.get(encoder,y,checkpoint_sha=CHECKPOINT_SHA,mode='static_delta',guard=g)
        cached=y['imu_control'].clone();identity=cache.identity
        y.pop('imu_reference');y.update(imu_control_mode='static_delta',text_scale=1.,imu_scale=1.,joint_scale=1.,
                                      backbone_bypass=torch.tensor([condition=='backbone_bypass']))
        guided=make_text_imu_guidance_model(model,mode='factorized',branch_execution='batched',bypass_encoder=torch.nn.Identity());guided.eval()
        observer=None
        if enabled:
            observer=TraceGuidance(guided,model,traces(),tmp_path/'traces',row,g,lambda t:t);observer.eval()
        rng=torch.random.get_rng_state().clone()
        with torch.no_grad():
            output,p=sample_addressed_ddpm(observer or guided,Diffusion(),y=y,lengths=y['lengths'],case_ids=['cpu'],sample_seed=1,backbone_id='CPU',run_guard=g)
        assert torch.equal(rng,torch.random.get_rng_state()) and torch.equal(cached,cache.value[0]) and identity==cache.identity
        assert encoder.calls==g.state['encoder_examples']==2 and model.calls==g.state['mdm_calls']==1000
        g.state['completed_slots']=['cpu'];g.finish()
        if observer:observer.close()
        assert not model._forward_hooks
        outputs.append(output);plans.append(p)
    assert torch.equal(*outputs) and plans[0]==plans[1]
    for slot in traces():
        meta=json.loads((tmp_path/'traces'/(slot['trace_id']+'.json')).read_text())
        with np.load(tmp_path/'traces'/(slot['trace_id']+'.npz')) as a:
            assert meta['actual_return_sha256']==array_sha(a['actual_combined_prediction'])


@pytest.mark.parametrize('bad_static',[False,True])
def test_worker_Q_all_96_batch5_groups_and_derived_counts(tmp_path,monkeypatch,bad_static):
    worker=script('run_mdm_sampling_alignment_worker');rows=[];groups=[];noise={}
    for j in range(96):
        key=f'cpu{j}';t=(50,250,500,900)[j%4]
        valid=sampling_noise([1,263,1,3],backbone_id='CPU',case_id=key,sample_seed=1,step=t)[1]
        noise[key]=dict(valid_noise=valid,padding_noise=None)
        ids=[]
        for i,(branch,c) in enumerate(zip(BRANCHES,('unconditional','text_only','paired','physical_static','tail_reversed'))):
            pid=f'Q{len(rows)+1:04d}';ids.append(pid)
            rows.append(dict(probe_id=pid,branch=branch,condition=c,input_transform='paired' if i<2 else c,
                frames=3,text_present=i>0,imu_present=i>1,text='CPU fixture'))
        groups.append(dict(group_index=j,group_id=key,probe_ids=ids,frames=3,timestep=t))
    model=Model();encoder=Encoder();model.eval();encoder.eval();removed=[]
    if bad_static:
        forward=model.forward
        def corrupt(*args,**kwargs):
            result=forward(*args,**kwargs);result[3]+=0.1;return result
        monkeypatch.setattr(model,'forward',corrupt)
    monkeypatch.setattr(worker,'load_fixed',lambda *a:({},(model,Diffusion(),None,encoder,None,SimpleNamespace(remove=lambda:removed.append(True)),None)))
    pkg=SimpleNamespace(groups=groups,probes=rows,noise=noise,inputs=fixture_item)
    g=guard(tmp_path/'guard','Q',slots=[r['probe_id'] for r in rows])
    if bad_static:
        with pytest.raises(ValueError,match='static/text'):
            worker.Q(pkg,SimpleNamespace(output=tmp_path/'Q',device='cpu'),g)
        assert removed and g.state['mdm_examples']==5 and not g.state['completed_slots']
        assert (tmp_path/'Q/Q0001.npz').exists() and not (tmp_path/'Q/Q0001.json').exists()
        return
    worker.Q(pkg,SimpleNamespace(output=tmp_path/'Q',device='cpu'),g);g.finish()
    assert model.calls==96 and encoder.calls==576 and removed
    for x,text,present,control in model.batch5:
        assert torch.equal(x,x[:1].repeat(5,1,1,1))
        assert present.tolist()==[False,False,True,True,True] and not text[:,0].any()
        assert not control[:2].any() and not control[3].any() and control[2].any()
    derived=json.loads((tmp_path/'Q/derived.json').read_text())
    assert [len(derived[k]) for k in ('conditions','within','difference_of_differences')]==[288,384,192]
    assert len(list((tmp_path/'Q').glob('Q*.npz')))==480


@pytest.mark.parametrize('fault',[None,'trace_write','combination'])
def test_worker_full_sample_includes_all_traces_and_decode(tmp_path,monkeypatch,fault):
    worker=script('run_mdm_sampling_alignment_worker');p=plan(3)
    history=tmp_path/'historical.json';atomic_json(history,dict(random_plans=[p]))
    row=dict(slot_index=0,case_id='cpu',frames=3,condition='paired',text_present=True,imu_present=True,
        comparison_case_id='old_cpu',guidance=GUIDANCE,random_plan_case_id='cpu',sampling_seed=1,
        historical_metadata=artifact(history),historical_random_plan_sha256=digest(p))
    model=Model();encoder=Encoder();model.eval();encoder.eval();removed=[]
    recover=lambda v,n:v[...,:66].reshape(1,1,3,22,3)
    backbone=MDMBackbone(model,Diffusion(),torch.zeros(263),torch.ones(263),recover)
    state=dict(initialization_sha256=INITIAL_SHA,control_config={'reference_mode':'static_delta'},training_input_provenance={'CPU_fixture':True})
    monkeypatch.setattr(worker,'load_fixed',lambda *a:(state,(model,Diffusion(),None,encoder,torch.nn.Identity(),SimpleNamespace(remove=lambda:removed.append(True)),backbone)))
    pkg=SimpleNamespace(samples=[row],traces=traces(),inputs=fixture_item,common={'backbone':{'sha256':'CPU'}})
    g=guard(tmp_path/'guard',traces=[r['trace_id'] for r in traces()])
    g.persist=lambda:None
    output=tmp_path/'run/samples/cpu.npz'
    if fault=='trace_write':
        import itm.models.mdm_sampling_alignment as alignment
        save=alignment.save_json
        def fail_write(path,*args):
            if Path(path).name=='T0002.json':raise OSError('CPU injected trace write failure')
            return save(path,*args)
        monkeypatch.setattr(alignment,'save_json',fail_write)
    if fault=='combination':
        factory=worker.make_text_imu_guidance_model
        def altered(*args,**kwargs):
            guided=factory(*args,**kwargs)
            guided.register_forward_hook(lambda m,a,out:out+1.)
            return guided
        monkeypatch.setattr(worker,'make_text_imu_guidance_model',altered)
    if fault:
        with pytest.raises((OSError,ValueError)):
            worker.sample(pkg,SimpleNamespace(slot=0,device='cpu',output=output),g)
        assert removed and not model._forward_hooks and not output.exists()
        if fault=='trace_write':
            assert g.state['completed_traces']==['T0001'] and (tmp_path/'run/traces/T0002.npz').exists()
        else:
            meta=read_json(tmp_path/'run/traces/T0001.json')
            assert meta['status']=='failed' and not g.state['completed_traces']
        return
    worker.sample(pkg,SimpleNamespace(slot=0,device='cpu',output=output),g);g.finish()
    assert removed and not model._forward_hooks and model.calls==1000 and encoder.calls==2
    with np.load(output) as arrays:
        meta=json.loads(str(arrays['metadata']));assert meta['trace_ids']==g.state['completed_traces']
        assert meta['guidance']==GUIDANCE and meta['random_plans']==[p] and meta['decode_check']['passed']
    assert len(list((tmp_path/'run/traces').glob('*.npz')))==4


def test_guard_rejects_training_extra_forward_missing_trace_and_nonfinite(tmp_path):
    g=guard(tmp_path/'a',traces=['T1'])
    with pytest.raises(RunLimitExceeded):g.before_update()
    with pytest.raises(RunLimitExceeded):g.before_denoise('cpu',999,3)
    with pytest.raises(FloatingPointError):g.ensure_finite(torch.tensor(float('nan')),'fixture')
    g.state.update(mdm_examples=4000,mdm_calls=1000,denoising_calls=1000,branch_examples=4000,completed_slots=['cpu'])
    with pytest.raises(RunLimitExceeded,match='trace'):g.finish()
    with pytest.raises(RunLimitExceeded):g.before_denoise('cpu',-1,4)
    q=guard(tmp_path/'q','Q',slots=[])
    with pytest.raises(ValueError):q.before_q(['Q1'])
    q.state['mdm_examples']=480
    with pytest.raises(RunLimitExceeded):q.before_q(['Q1']*5)
    deadline=guard(tmp_path/'late',clock=lambda:101.)
    with pytest.raises(RunLimitExceeded):deadline.check()


@pytest.mark.parametrize('scenario',['success','preflight','Q_timeout','sample_oom','trace_missing','trace_write','consistency','between_deadline'])
def test_queue_full_denominators_and_failure_retention(tmp_path,monkeypatch,scenario):
    bounded=script('run_mdm_sampling_alignment_bounded');now=[0.];calls=[]
    checkpoint=tmp_path/'cpu-fixture.pt';checkpoint.write_bytes(b'CPU fixture, not model weights')
    manifest_path=tmp_path/'implementation.json'
    atomic_json(manifest_path,dict(version='itm-alignment-implementation-v1',package=artifact(PACKAGE),
        checkpoint=artifact(checkpoint),environment=environment(),expected_artifacts=[]))
    run=tmp_path/'run';approval=tmp_path/'approval.json';gpu='GPU-alignment-CPU-fixture'
    atomic_json(approval,dict(authorized=True,approved_by='user',user_instruction_reference='CPU fixture only; all launch/preflight tools replaced',
        package_sha256=PACKAGE_SHA,implementation_sha256=artifact(manifest_path)['sha256'],run_directory=str(run),gpu_uuid=gpu))
    monkeypatch.setattr(bounded,'AlignmentPackage',lambda *a:SimpleNamespace(path=PACKAGE,checkpoint=checkpoint,common={'environment':environment()}))
    def preflight(gpu):
        if scenario=='preflight':raise ValueError('CPU simulated occupied GPU')
        return {'CPU_fixture':True}
    monkeypatch.setattr(bounded,'preflight_gpu',preflight)
    def consistency(status,slot,run):
        if slot['spec']['condition'] not in ('backbone_bypass','physical_static','imu_only'):return None
        return dict(status='failed' if scenario=='consistency' else 'passed',max_abs_m=1. if scenario=='consistency' else 0.)
    monkeypatch.setattr(bounded,'full_check',consistency)
    original_atomic=bounded.atomic_json
    def persist(path,value):
        if scenario=='between_deadline' and Path(path).name=='status.json' and value['samples'][1]['status']=='running':
            now[0]=value['sampling_deadline_monotonic']+1
        return original_atomic(path,value)
    monkeypatch.setattr(bounded,'atomic_json',persist)
    def supervise(argv,**kwargs):
        phase=argv[argv.index('--phase')+1];output=Path(argv[argv.index('--output')+1]);calls.append(phase)
        context=read_json(Path(argv[argv.index('--run-context')+1]));slot=int(argv[argv.index('--slot')+1]) if '--slot' in argv else None
        now[0]+=1
        receipt=dict(version='itm-alignment-receipt-v1',phase=phase,status='completed',context=context,
            checkpoint_sha256=CHECKPOINT_SHA,mdm_examples=480 if phase=='Q' else 4000,mdm_calls=96 if phase=='Q' else 1000,
            encoder_examples=576 if phase=='Q' else 2,updates_attempted=0,updates_completed=0,
            denoising_calls=0 if phase=='Q' else 1000,branch_examples=0 if phase=='Q' else 4000,
            completed_slots=list(context['expected_slots']),completed_traces=list(context['expected_traces']),artifacts=[],CPU_fixture=True)
        def publish(path):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'CPU fixture artifact')
            receipt['artifacts'].append(artifact(path))
        if phase=='Q':
            for pid in context['expected_slots']:
                for ext in ('.npz','.json'):publish(output/(pid+ext))
            publish(output/'derived.json')
            if scenario=='Q_timeout':
                receipt.update(status='failed',completed_slots=context['expected_slots'][:5],
                    active_probe_ids=context['expected_slots'][5:10],active_probe_forward_started=True,
                    failure={'kind':'limit'})
        else:
            if scenario=='sample_oom' and slot==1:
                receipt.update(status='failed',completed_slots=[],completed_traces=[],failure={'kind':'oom'})
            else:
                publish(output);publish(output.with_suffix('.json'))
                for tid in context['expected_traces']:
                    if scenario=='trace_missing' and slot==0 and tid==context['expected_traces'][-1]:continue
                    for ext in ('.npz','.json'):publish(run/'traces'/(tid+ext))
                if scenario=='trace_missing' and slot==0:receipt['completed_traces']=context['expected_traces'][:-1]
                if scenario=='trace_write' and slot==1:
                    receipt.update(status='failed',completed_slots=[],completed_traces=context['expected_traces'][:1],
                        active_trace_ids=context['expected_traces'][1:2],failure={'kind':'exception'})
        atomic_json(context['receipt_path'],receipt)
        return dict(status='completed' if receipt['status']=='completed' else 'failed',reason='CPU_fixture',launched=True,
                    returncode=0 if receipt['status']=='completed' else 1)
    monkeypatch.setattr(bounded,'supervise',supervise)
    if scenario=='success':bounded.execute(manifest_path,run,approval,gpu,clock=lambda:now[0])
    else:
        with pytest.raises((ValueError,RuntimeError,TimeoutError)):
            bounded.execute(manifest_path,run,approval,gpu,clock=lambda:now[0])
    status=read_json(run/'runtime/status.json')
    assert (len(status['probes']),len(status['samples']),len(status['traces']))==(480,60,240)
    assert len(status['consistency_checks'])==24
    if scenario=='success':
        assert status['status']=='completed' and len(calls)==61
        assert all(r['status']=='completed' for r in status['traces'])
    else:
        assert status['status']=='failed'
        if scenario=='preflight':assert not calls and all(r['status']=='not_run_after_stop' for r in status['probes'])
        if scenario=='Q_timeout':
            assert calls==['Q'] and sum(r['status']=='completed' for r in status['probes'])==5
            assert sum(r['status']=='failed' for r in status['probes'])==5
        if scenario in ('sample_oom','trace_write','between_deadline'):
            assert status['samples'][0]['status']=='completed' and status['samples'][1]['status']=='failed'
            assert (run/status['samples'][0]['output']).exists()
        if scenario=='between_deadline':assert calls==['Q','sample'] and not status['samples'][1]['launched']
        if scenario=='trace_missing':
            assert status['samples'][0]['output_present'] and not status['samples'][0]['diagnostic_complete']
            assert status['traces'][3]['status']=='failed'
        if scenario=='trace_write':
            assert status['traces'][4]['status']=='completed' and status['traces'][5]['status']=='failed'
        assert any(r['status']=='not_run_after_stop' for r in status['samples'])


def read_json(path):return json.loads(Path(path).read_text())


def test_all_frozen_full_sample_comparisons_and_missing_counterpart(tmp_path,package):
    bounded=script('run_mdm_sampling_alignment_bounded')
    motion=np.zeros((1,3,22,3),np.float32)
    historical=tmp_path/'historical.npz';np.savez(historical,motion=motion)
    rows=deepcopy(package.samples)
    status=dict(samples=[])
    for row in rows:
        row['historical_output']=artifact(historical)
        output=row['case_id']+'.npz';np.savez(tmp_path/output,motion=motion)
        status['samples'].append(dict(case_id=row['case_id'],spec=row,status='completed',output=output))
    counts={}
    for slot in status['samples']:
        result=bounded.full_check(status,slot,tmp_path)
        kind=slot['spec']['condition']
        if result is None:continue
        counts[kind]=counts.get(kind,0)+1
        assert result['status']=='passed' and result['bitwise'] and result['max_abs_m']==result['rms_m']==0
        assert result['count']==motion.size
        if kind=='imu_only':assert result['reference']==slot['spec']['comparison_case_id']
        bad=motion.copy();bad.flat[0]=2e-5;np.savez(tmp_path/slot['output'],motion=bad)
        assert bounded.full_check(status,slot,tmp_path)['status']=='failed'
        np.savez(tmp_path/slot['output'],motion=motion)
    assert counts==dict(backbone_bypass=6,physical_static=12,imu_only=6)
    for slot in status['samples']:
        if slot['spec']['condition']=='text_only':slot['status']='failed'
    for slot in status['samples']:
        if slot['spec']['condition'] in ('backbone_bypass','physical_static'):
            assert bounded.full_check(status,slot,tmp_path)['status']=='not_comparable'


def test_new_authorization_rejects_old_package_manifest_directory_and_gpu(tmp_path):
    bounded=script('run_mdm_sampling_alignment_bounded');manifest=tmp_path/'m.json';manifest.write_text('{}')
    run=tmp_path/'run';path=tmp_path/'approval.json'
    value=dict(authorized=True,approved_by='user',user_instruction_reference='CPU fixture',package_sha256=PACKAGE_SHA,
        implementation_sha256=artifact(manifest)['sha256'],run_directory=str(run),gpu_uuid='GPU-fixture')
    atomic_json(path,value);assert bounded.authorization(path,manifest,run,'GPU-fixture')==value
    for key in value:
        bad=dict(value);bad[key]=None;atomic_json(path,bad)
        with pytest.raises(ValueError):bounded.authorization(path,manifest,run,'GPU-fixture')
