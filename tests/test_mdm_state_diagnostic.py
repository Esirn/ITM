"""CPU fixtures for saved-state V. No pretrained forward, CUDA query or real launch."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from itm.models.mdm_correction import artifact,read,environment
from itm.models.mdm_sampling_alignment import input_arrays,describe
from itm.models.mdm_state_diagnostic import (StatePackage,StateGuard,StateCache,PACKAGE,PACKAGE_SHA,CHECKPOINT_SHA,
    CAPS,CALL_CAPS,ledger)
from itm.models.mdm_run_limits import atomic_json,RunLimitExceeded
from test_mdm_sampling_alignment import script,Model,Encoder,fixture_item,forbid_gpu_and_models


def guard(tmp_path,clock=lambda:0.):
    tmp_path.mkdir(parents=True,exist_ok=True)
    return StateGuard(dict(version='itm-state-context-v1',phase='V',package_sha256=PACKAGE_SHA,
        checkpoint_sha256=CHECKPOINT_SHA,implementation_sha256='CPU_fixture',receipt_path=str(tmp_path/'receipt.json'),
        deadline_monotonic=300.),clock=clock)


@pytest.fixture(scope='module')
def package():return StatePackage()


def test_prepare_real_inputs_states_cpu_only(tmp_path,monkeypatch,package):
    forbid_gpu_and_models(monkeypatch)
    bounded=script('run_mdm_state_diagnostic_bounded')
    monkeypatch.setattr(bounded,'preflight_gpu',lambda *a:pytest.fail('prepare preflight'))
    monkeypatch.setattr(bounded,'supervise',lambda *a,**k:pytest.fail('prepare execute'))
    manifest=bounded.prepare(tmp_path/'prepared')
    assert len(manifest['inputs'])==12 and len(manifest['states'])==48
    assert sum(len(v['branches']) for v in manifest['inputs'])==48
    assert manifest['checkpoint']['sha256']==CHECKPOINT_SHA
    assert list((tmp_path/'prepared').iterdir())==[tmp_path/'prepared/implementation.json']


@pytest.mark.parametrize('field',['state_descriptor','target_descriptor','frame_mask','historical_T','historical_P'])
def test_saved_identity_rejected(package,field):
    group=deepcopy(package.groups[0]);group[field]['sha256']='wrong'
    with pytest.raises(ValueError,match='identity'):package.saved(group)


def test_input_caption_and_reference_identity_rejected(package):
    case=deepcopy(package.cases[0]);case['caption']='wrong'
    with pytest.raises(ValueError,match='caption'):package.inputs(case)
    case=deepcopy(package.cases[0]);case['branches'][0]['arrays']['normalized_reference_imu']['sha256']='wrong'
    with pytest.raises(ValueError,match='reference input'):package.inputs(case)


class VModel(Model):
    def __init__(self):
        super().__init__();self.caption_calls=[];self.seen=[]
        self.clip_model=torch.nn.Linear(2,2).half()  # Upstream native CLIP convention.
    def encode_text(self,text):self.caption_calls.append(list(text));return super().encode_text(text)
    def forward(self,x,t,y=None):
        assert not torch.is_grad_enabled() and not self.training
        self.seen.append((x.clone(),t.clone(),y['imu_condition_present'].clone(),y['imu_control'].clone(),y['text_embed'].clone()))
        return super().forward(x,t,y=y)


def fixture_package(package):
    cases=deepcopy(package.cases);groups=deepcopy(package.groups);slots=deepcopy(package.slots)
    for c in cases:
        c['frames']=3;c['caption']='CPU fixture';c['sample_spec'].update(frames=3,text='CPU fixture')
    for g in groups:g.update(frames=3,input_batch_shape=[4,263,1,3])
    for s in slots:s.update(frames=3,output_shape=[263,1,3])
    def inputs(case):
        rows=[dict(frames=3,condition=b['condition'],text_present=True,imu_present=b['imu_present']) for b in case['branches']]
        items=[fixture_item(r) for r in rows]
        return items,rows,[{k:describe(v) for k,v in input_arrays(i,r).items()} for i,r in zip(items,rows)]
    def saved(group):
        item=fixture_item(dict(frames=3,condition='paired'))
        xt=np.linspace(-1,1,789,dtype=np.float32).reshape(1,263,1,3)
        x=torch.from_numpy(xt);control=torch.from_numpy(item['imu'][:,0,:8]-item['reference'][:,0,:8])
        null=.99*x+.0001*torch.zeros(1,1,1,3)+.0001
        paired=.99*x+.0001*control.mean(-1)[None,None,None,:]+.0001
        return dict(x_t=xt,target=item['motion'].T[:,None],frame_mask=np.ones((1,3),bool),f10=null.numpy(),f11=paired.numpy())
    return SimpleNamespace(cases=cases,groups=groups,slots=slots,checks=deepcopy(package.checks),inputs=inputs,saved=saved,
                           common={'imu_normalization':{'CPU_fixture':True}})


def forbidden(*a,**kw):pytest.fail('V attempted forbidden noise/diffusion/CFG/decode/train work')


@pytest.mark.parametrize('fault',[None,'anchor','nan','oom','write','timeout'])
def test_actual_worker_48_states_counts_cache_and_failure_retention(tmp_path,monkeypatch,package,fault):
    worker=script('run_mdm_state_diagnostic_worker');pkg=fixture_package(package)
    model=VModel();encoder=Encoder();model.eval();encoder.eval();removed=[];now=[0.]
    diffusion=SimpleNamespace(_scale_timesteps=lambda t:t,q_sample=forbidden,p_mean_variance=forbidden,p_sample_loop=forbidden)
    monkeypatch.setattr(worker,'load_fixed',lambda *a:({},(model,diffusion,None,encoder,None,SimpleNamespace(remove=lambda:removed.append(True)),SimpleNamespace(decode_motion=forbidden))))
    original=model.forward
    def forward(*a,**kw):
        if fault=='oom' and model.calls==1:raise RuntimeError('CPU injected CUDA out of memory')
        result=original(*a,**kw)
        if fault=='anchor' and model.calls==2:result[1]+=0.1
        if fault=='nan' and model.calls==2:result[1,0,0,0]=float('nan')
        if fault=='timeout' and model.calls==2:now[0]=301.
        return result
    monkeypatch.setattr(model,'forward',forward)
    # No stochastic or sampling API may be used by V after fixture setup.
    monkeypatch.setattr(torch,'randn',forbidden);monkeypatch.setattr(torch,'randn_like',forbidden)
    import itm.models.mdm_sampling as sampler
    import itm.models.mdm_imu_control as control
    monkeypatch.setattr(sampler,'sample_addressed_ddpm',forbidden)
    monkeypatch.setattr(control,'make_text_imu_guidance_model',forbidden)
    if fault=='write':
        save=worker.save_json
        def fail(path,*args):
            if Path(path).name=='V002-P.json':raise OSError('CPU injected metadata write')
            return save(path,*args)
        monkeypatch.setattr(worker,'save_json',fail)
    g=guard(tmp_path/'guard',clock=lambda:now[0]);args=SimpleNamespace(output=tmp_path/'V',device='cpu')
    if fault:
        with pytest.raises((ValueError,RuntimeError,FloatingPointError,OSError,RunLimitExceeded)):
            try:worker.diagnose(pkg,args,g)
            except BaseException as e:g.fail(e);raise
        assert removed and g.state['groups'][0]['status']=='completed' and g.state['groups'][1]['status']=='failed'
        assert all(v['status']=='not_run_after_stop' for v in g.state['groups'][2:])
        assert len(g.state['branches'])==192 and len(g.state['checks'])==144
        assert (tmp_path/'V/branches/V001-P.npz').exists()
        if fault=='anchor':
            assert all((tmp_path/'V/branches'/('V002-'+b+'.npz')).exists() for b in 'TPSR')
            assert g.state['branches'][5]['diagnostic_only'] and g.state['branches'][5]['executed']
            assert read(tmp_path/'V/groups/V002.json')['status']=='failed'
        if fault=='write':assert (tmp_path/'V/branches/V002-P.npz').exists()
        if fault in ('nan','oom','timeout'):
            assert g.state['counts']['mdm']['reserved']==g.state['counts']['mdm']['actual']==8
            assert g.state['counts']['mdm']['completed']==4
        return
    worker.diagnose(pkg,args,g);g.finish()
    assert removed and model.calls==48 and encoder.calls==72 and len(model.caption_calls)==12
    assert all(len(v)==1 for v in model.caption_calls)
    assert all(v['status']=='completed' for k in ('groups','branches','checks') for v in g.state[k])
    for i,(x,t,present,c,text) in enumerate(model.seen):
        assert torch.equal(x,x[:1].repeat(4,1,1,1)) and x.shape==(4,263,1,3)
        assert t.tolist()==[pkg.groups[i]['timestep']]*4 and present.tolist()==[False,True,True,True]
        assert not c[0].any() and not c[2].any() and c[1].any()
        assert torch.equal(text,text[:,:1].repeat(1,4,1))
        if i%4:assert torch.equal(c,model.seen[i-1][3])
    assert len(list((tmp_path/'V/branches').glob('*.npz')))==192
    assert len(list((tmp_path/'V/checks').glob('*.json')))==144
    assert all(v['finite_count']==v['count']==789 for v in g.state['checks'])


@pytest.mark.parametrize('mutation',['caption','input','tensor','mode','checkpoint'])
def test_cache_identity_fail_closed(tmp_path,package,mutation):
    worker=script('run_mdm_state_diagnostic_worker');pkg=fixture_package(package);case=deepcopy(pkg.cases[0]);items,rows,desc=pkg.inputs(case)
    model=VModel();encoder=Encoder();model.eval();encoder.eval();g=guard(tmp_path/'g');cache=StateCache()
    with torch.no_grad():entry,_=cache.get(case,items,rows,desc,pkg,model,encoder,g,'cpu',worker.batch_inputs)
    if mutation=='caption':case['caption']='different'
    if mutation=='input':items[1]['imu'][0,0,0]+=1
    if mutation=='tensor':entry['text'].add_(1)
    if mutation=='mode':encoder.train()
    if mutation=='checkpoint':
        import itm.models.mdm_state_diagnostic as module
        old=module.CHECKPOINT_SHA;module.CHECKPOINT_SHA='wrong'
    try:
        with pytest.raises(ValueError):cache.get(case,items,rows,desc,pkg,model,encoder,g,'cpu',worker.batch_inputs)
    finally:
        if mutation=='checkpoint':module.CHECKPOINT_SHA=old
    assert len(model.caption_calls)==1 and encoder.calls==6


def test_reservation_precedes_actual_call_and_forbidden_work(tmp_path):
    now=[0.];g=guard(tmp_path/'g',clock=lambda:now[0]);original=g.persist
    def persist():
        original()
        if g.state['counts']['caption']['reserved']==1:now[0]=301.
    g.persist=persist
    with pytest.raises(RunLimitExceeded):g.invoke('caption',1,forbidden)
    assert g.state['counts']['caption']['reserved']==1 and g.state['counts']['caption']['actual']==0
    with pytest.raises(RunLimitExceeded):g.before_update()
    with pytest.raises(RunLimitExceeded):g.before_denoise('x',1,4)
    h=guard(tmp_path/'h')
    with pytest.raises(RunLimitExceeded):h.invoke('mdm',5,forbidden)
    h.state['counts']['encoder']['reserved']=72
    with pytest.raises(RunLimitExceeded):h.invoke('encoder',1,forbidden)
    with pytest.raises(RunLimitExceeded):h.finish()


def test_new_entry_uses_real_nonreturning_upstream_loader(tmp_path,monkeypatch):
    import test_mdm_sampling_alignment as fixture
    worker=script('run_mdm_state_diagnostic_worker');original=fixture.script
    monkeypatch.setattr(fixture,'script',lambda name:worker if name=='run_mdm_sampling_alignment_worker' else original(name))
    fixture.test_real_new_loader_with_upstream_nonreturning_methods(tmp_path,monkeypatch)


@pytest.mark.parametrize('scenario',['success','preflight','timeout','receipt_missing','artifact_missing','artifact_hash','counter','deadline'])
def test_parent_single_deadline_ledger_and_retention(tmp_path,monkeypatch,scenario):
    bounded=script('run_mdm_state_diagnostic_bounded');clock=[0.];calls=[]
    checkpoint=tmp_path/'CPU-not-weights.pt';checkpoint.write_bytes(b'CPU fixture')
    manifest=tmp_path/'implementation.json';run=tmp_path/'run';approval=tmp_path/'approval.json';gpu='GPU-V-CPU-fixture'
    atomic_json(manifest,dict(version='itm-state-implementation-v1',package=artifact(PACKAGE),checkpoint=artifact(checkpoint),
        environment=environment(),expected_artifacts=[]))
    atomic_json(approval,dict(authorized=True,approved_by='user',package_sha256=PACKAGE_SHA,
        implementation_sha256=artifact(manifest)['sha256'],run_directory=str(run),gpu_uuid=gpu,
        user_instruction_reference='Temporary CPU fixture only; launch/preflight replaced'))
    monkeypatch.setattr(bounded,'StatePackage',lambda *a:SimpleNamespace(path=PACKAGE,checkpoint=checkpoint,common={'environment':environment()}))
    def preflight(gpu):
        s=read(run/'runtime/status.json');assert [len(s[k]) for k in ('groups','branches','checks')]==[48,192,144]
        assert all(r['status']=='planned' for k in ('groups','branches','checks') for r in s[k])
        if scenario=='preflight':raise ValueError('CPU preflight failure')
        clock[0]=301. if scenario=='deadline' else 1.
        return {'CPU_fixture':True}
    monkeypatch.setattr(bounded,'preflight_gpu',preflight)
    def supervise(argv,**kw):
        calls.append(argv);assert kw['deadline']==300.
        context=read(argv[argv.index('--run-context')+1]);assert context['deadline_monotonic']==300.
        output=Path(argv[argv.index('--output')+1])
        child=dict(status='completed',context=context,updates_attempted=0,updates_completed=0,denoising_calls=0,
            counts={k:dict(reserved=v,actual=v,completed=v,reserved_calls=CALL_CAPS[k],actual_calls=CALL_CAPS[k],completed_calls=CALL_CAPS[k]) for k,v in CAPS.items()},artifacts=[],**ledger())
        for key in ('groups','branches','checks'):
            for row in child[key]:row['status']='completed'
        def publish(path):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'CPU placeholder; worker arrays tested separately')
            child['artifacts'].append(artifact(path))
        for row in child['groups']:
            publish(output/'states'/(row['group_id']+'.npz'));publish(output/'groups'/(row['group_id']+'.json'))
        for row in child['branches']:
            for ext in ('.npz','.json'):publish(output/'branches'/(row['slot_id']+ext))
        for row in child['checks']:publish(output/'checks'/(row['check_id']+'.json'))
        for i in range(1,13):
            publish(output/'inputs'/f'C{i:02d}.json')
            for b in 'TPSR':publish(output/'inputs'/f'C{i:02d}-{b}.npz')
        if scenario=='timeout':
            child['status']='running'
            for key,width in (('groups',1),('branches',4),('checks',3)):
                for i,row in enumerate(child[key]):row['status']='completed' if i<width else 'running' if i<2*width else 'planned'
        if scenario=='counter':child['counts']['mdm']['actual']=188
        target=output/'branches/V001-P.npz'
        if scenario=='artifact_missing':child['artifacts']=[a for a in child['artifacts'] if a['path']!=str(target)];target.unlink()
        if scenario=='artifact_hash':target.write_bytes(b'corrupted CPU output')
        if scenario!='receipt_missing':atomic_json(context['receipt_path'],child)
        clock[0]=302. if scenario=='timeout' else 2.
        return dict(status='failed' if scenario=='timeout' else 'completed',launched=True,reason='CPU simulated watchdog' if scenario=='timeout' else 'CPU fixture')
    monkeypatch.setattr(bounded,'supervise',supervise)
    if scenario=='success':bounded.execute(manifest,run,approval,gpu,clock=lambda:clock[0])
    else:
        with pytest.raises((ValueError,TimeoutError,RuntimeError)):
            bounded.execute(manifest,run,approval,gpu,clock=lambda:clock[0])
    s=read(run/'runtime/status.json')
    assert [len(s[k]) for k in ('groups','branches','checks')]==[48,192,144]
    if scenario=='success':assert s['status']=='completed' and len(calls)==1
    else:
        assert s['status']=='failed'
        if scenario in ('preflight','deadline'):
            assert not calls and all(r['status']=='not_run_after_stop' for r in s['groups'])
        if scenario=='timeout':
            assert s['groups'][0]['status']=='completed' and s['groups'][1]['status']=='failed'
            assert s['groups'][2]['status']=='not_run_after_stop' and (run/'V/branches/V001-P.npz').exists()
        if scenario in ('artifact_missing','artifact_hash'):
            assert s['groups'][0]['status']=='failed' and s['branches'][1]['diagnostic_only']
        assert len(calls)<=1


def test_authorization_and_context_identity_rejection(tmp_path):
    bounded=script('run_mdm_state_diagnostic_bounded');m=tmp_path/'manifest';m.write_text('{}')
    p=tmp_path/'approval.json';run=tmp_path/'run'
    good=dict(authorized=True,approved_by='user',package_sha256=PACKAGE_SHA,implementation_sha256=artifact(m)['sha256'],
              run_directory=str(run),gpu_uuid='CPU',user_instruction_reference='CPU fixture only')
    atomic_json(p,good);assert bounded.authorization(p,m,run,'CPU')==good
    for key in good:
        bad=dict(good);bad[key]=None;atomic_json(p,bad)
        with pytest.raises(ValueError):bounded.authorization(p,m,run,'CPU')
    g=guard(tmp_path/'context')
    for key in ('package_sha256','checkpoint_sha256'):
        context=dict(g.context);context[key]='wrong'
        with pytest.raises(ValueError):StateGuard(context)
