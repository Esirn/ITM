"""CPU-only real-plan fixtures for caches, first-failure stops and supervision."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from itm.models.mdm_correction import artifact,read,environment
from itm.models.mdm_sampling_alignment import describe,input_arrays
from itm.models.mdm_run_limits import atomic_json,RunLimitExceeded
from itm.models.mdm_training_sufficiency import (TrainingPackage,TrainingGuard,SnapshotCache,PACKAGE,PACKAGE_SHA,
                                               CAPS,CALL_CAPS,BRANCHES,ledger,read_receipt)
from test_mdm_sampling_alignment import script,fixture_item,forbid_gpu_and_models,Encoder


@pytest.fixture(scope='module')
def package():return TrainingPackage()


class FixtureEncoder(Encoder):
    def __init__(self):
        super().__init__();self.register_buffer('gain',torch.tensor(1.))
    def forward(self,*args,**kwargs):
        result,present=super().forward(*args,**kwargs)
        return result*self.gain,present


class FixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.adapter=torch.nn.Module();self.adapter.register_buffer('gain',torch.tensor(0.))
        self.clip_model=torch.nn.Linear(2,2).half();self.calls=0;self.caption_calls=0;self.seen=[]
    def encode_text(self,text):
        assert len(text)==1;self.caption_calls+=1;return torch.ones(1,1,8)
    def forward(self,x,t,y):
        assert not torch.is_grad_enabled() and not self.training
        self.calls+=1;self.seen.append((x.clone(),y['imu_control'].clone(),float(self.adapter.gain)))
        c=y['imu_control'].mean(-1)[:,None,None,:]*y['imu_condition_present'][:,None,None,None]
        return .99*x+.0001*self.adapter.gain*c+.0001*y['text_embed'].mean((0,2))[:,None,None,None]


def cpu_package(package,tmp_path):
    cases=deepcopy(package.cases);states=deepcopy(package.states);jobs=deepcopy(package.jobs);checks=deepcopy(package.checks)
    xt=np.linspace(-1,1,789,dtype=np.float32).reshape(1,263,1,3)
    saved=dict(x_t=xt,target=np.zeros((263,1,3),np.float32),frame_mask=np.ones((1,3),bool))
    for c in cases:c['frames']=3;c['caption']='CPU fixture'
    for s in states:s.update(frames=3,arrays={k:describe(v) for k,v in saved.items()})
    for j in jobs:j['input_batch_shape']=[4,263,1,3]
    def inputs(case):
        rows=[dict(frames=3,condition=b['condition'],text_present=True,imu_present=b['imu_present']) for b in case['branches']]
        items=[fixture_item(r) for r in rows]
        return items,rows,[{k:describe(v) for k,v in input_arrays(i,r).items()} for i,r in zip(items,rows)]
    snapshots=deepcopy(package.snapshots)
    def state(snapshot):
        index=[r['snapshot_id'] for r in snapshots].index(snapshot['snapshot_id'])
        return dict(control_config={'reference_mode':'static_delta'},imu_encoder=dict(gain=torch.tensor(float(index+1))),
                    adapters=dict(gain=torch.tensor(float(index))))
    anchors={}
    for spec in checks:
        ref=spec['reference']
        if ref['kind']!='saved_array':continue
        sid=spec['snapshot_id'];key=(sid,spec['state_id'])
        if key not in anchors:
            index=[r['snapshot_id'] for r in snapshots].index(sid);items,_,_=inputs(cases[0])
            controls=[np.zeros((3,8),np.float32)]+[(i['imu'][:,0,:8]-i['reference'][:,0,:8])*(index+1) for i in items[1:]]
            values={b:(.99*xt[0]+.0001*index*controls[i].mean(-1)[None,None,:]+.0001).astype(np.float32) for i,b in enumerate(BRANCHES)}
            path=tmp_path/(sid+'-'+spec['state_id']+'.npz');np.savez_compressed(path,**values);anchors[key]=path
        p=anchors[key]
        with np.load(p,allow_pickle=False) as z:desc=describe(z[spec['branch']])
        spec['reference']=dict(kind='saved_array',snapshot_id=sid,branch=spec['branch'],artifact=artifact(p),array=spec['branch'],descriptor=desc)
    return SimpleNamespace(states=states,jobs=jobs,checks=checks,cases=cases,snapshots=snapshots,
                           common=dict(imu_normalization={'CPU_fixture':True},backbone={'CPU_fixture':True}),
                           inputs=inputs,saved=lambda row:deepcopy(saved),state=state)


def guard(tmp_path,pkg,clock=lambda:0.):
    return TrainingGuard(dict(version='itm-sufficiency-context-v1',phase='diagnostic',package_sha256=PACKAGE_SHA,
                             implementation_sha256='CPU_fixture',deadline_monotonic=600.,receipt_path=str(tmp_path/'receipt.json'),
                             expected_artifacts=[]),pkg,clock=clock)


@pytest.mark.parametrize('fault',[None,'anchor','nan','oom','write','timeout'])
def test_full_worker_plan_and_first_failure(tmp_path,monkeypatch,package,fault):
    worker=script('run_mdm_training_sufficiency_worker');pkg=cpu_package(package,tmp_path)
    model=FixtureModel();encoder=FixtureEncoder();model.eval();encoder.eval();removed=[];clock=[0.]
    diffusion=SimpleNamespace(num_timesteps=1000,_scale_timesteps=lambda t:t,q_sample=lambda *a:pytest.fail('q_sample'))
    monkeypatch.setattr(worker,'load_model',lambda *a:(model,diffusion,SimpleNamespace(adapters=model.adapter),encoder,
        torch.nn.Identity().eval(),SimpleNamespace(remove=lambda:removed.append(True)),SimpleNamespace()))
    old=model.forward
    def forward(*args,**kwargs):
        if fault=='oom' and model.calls==1:raise RuntimeError('CPU injected CUDA out of memory')
        value=old(*args,**kwargs)
        if model.calls==2:
            if fault=='anchor':value[1]+=0.1
            if fault=='nan':value[1,0,0,0]=float('nan')
            if fault=='timeout':clock[0]=601.
        return value
    monkeypatch.setattr(model,'forward',forward)
    def forbidden(*args,**kwargs):pytest.fail('Diagnostic attempted new noise/sampling/CFG')
    monkeypatch.setattr(torch,'randn',forbidden);monkeypatch.setattr(torch,'randn_like',forbidden)
    if fault=='write':
        save=worker.save_json
        def fail(path,*args):
            if Path(path).name==pkg.jobs[1]['job_id']+'.json':raise OSError('CPU injected metadata write')
            return save(path,*args)
        monkeypatch.setattr(worker,'save_json',fail)
    g=guard(tmp_path/'g',pkg,clock=lambda:clock[0]);args=SimpleNamespace(output=tmp_path/'diagnostic',device='cpu')
    if fault:
        with pytest.raises((ValueError,RuntimeError,FloatingPointError,OSError,RunLimitExceeded)):
            try:worker.diagnose(pkg,args,g)
            except BaseException as e:g.fail(e);raise
        assert removed and g.state['jobs'][0]['status']=='completed' and g.state['jobs'][1]['status']=='failed'
        assert all(r['status']=='not_run_after_stop' for r in g.state['jobs'][2:])
        assert [len(g.state[k]) for k in ('states','jobs','branches','checks')]==[192,1152,4608,2880]
        assert (args.output/'predictions'/(pkg.jobs[0]['job_id']+'.npz')).exists()
        if fault in ('anchor','write'):
            assert (args.output/'predictions'/(pkg.jobs[1]['job_id']+'.npz')).exists()
            assert g.state['jobs'][1]['raw_saved'] and g.state['jobs'][1]['diagnostic_only']
        if fault in ('nan','oom','timeout'):
            assert g.state['counts']['mdm']['reserved']==g.state['counts']['mdm']['actual']==8
            assert g.state['counts']['mdm']['completed']==4
        return
    worker.diagnose(pkg,args,g);g.finish()
    assert removed and model.calls==1152 and encoder.calls==432 and model.caption_calls==12
    assert all(r['status']=='completed' for k in ('states','jobs','branches','checks') for r in g.state[k])
    for i,(xt,control,gain) in enumerate(model.seen):
        assert torch.equal(xt,xt[:1].repeat(4,1,1,1))
        assert not control[0].any() and not control[2].any()
        assert gain==i//192
    # Same case/state in six snapshots: controls must actually reflect new E.
    assert torch.equal(model.seen[192][1],2*model.seen[0][1])
    assert torch.equal(model.seen[384][1],3*model.seen[0][1])
    assert len(list((args.output/'predictions').glob('*.npz')))==1152


@pytest.mark.parametrize('mutation',['caption','input','tensor','mode','checkpoint'])
def test_cache_fails_closed_and_resets_controls(tmp_path,package,mutation):
    worker=script('run_mdm_training_sufficiency_worker');pkg=cpu_package(package,tmp_path)
    case=deepcopy(pkg.cases[0]);items,rows,desc=pkg.inputs(case);model=FixtureModel();encoder=FixtureEncoder()
    model.eval();encoder.eval();g=guard(tmp_path/'g',pkg);cache=SnapshotCache();cache.begin('initial')
    with torch.no_grad():entry,_,_=cache.get(case,items,rows,desc,pkg,model,encoder,g,'cpu',worker.batch_inputs)
    if mutation=='caption':case['caption']='wrong'
    if mutation=='input':items[1]['imu'][0,0,0]+=1
    if mutation=='tensor':entry['control'].add_(1)
    if mutation=='mode':encoder.train()
    if mutation=='checkpoint':cache.checkpoint='wrong' # Forbidden reuse without begin/reset.
    with pytest.raises(ValueError):cache.get(case,items,rows,desc,pkg,model,encoder,g,'cpu',worker.batch_inputs)
    assert encoder.calls==6 and model.caption_calls==1


def test_reservation_before_actual_and_no_forbidden_steps(tmp_path,package):
    clock=[0.];g=guard(tmp_path,package,clock=lambda:clock[0]);old=g.persist
    def persist():
        old()
        if g.state['counts']['caption']['reserved']:clock[0]=601.
    g.persist=persist
    with pytest.raises(RunLimitExceeded):g.invoke('caption',1,lambda:pytest.fail('Unreserved late call'))
    assert g.state['counts']['caption']['reserved']==1 and g.state['counts']['caption']['actual']==0
    for operation in (lambda:g.before_update(),lambda:g.before_denoise('x',1,4),lambda:g.optimizer_step(None,None,None)):
        with pytest.raises(RunLimitExceeded):operation()


def test_journal_replay_retains_reservations_raw_and_partial_tail(tmp_path,package):
    g=guard(tmp_path,package);jid=package.jobs[0]['job_id']
    g.state['active_job']=jid;g.state['jobs'][0]['status']='running';g.persist()
    g.invoke('caption',1,lambda:torch.ones(1,1,8))
    g.state['jobs'][0].update(raw_saved=True,raw_artifact={'CPU_fixture':True});g.persist()
    # The full snapshot remains at initialization; only events contain progress.
    assert read(g.path)['counts']['caption']['actual']==0
    restored=read_receipt(g.path)
    assert restored['counts']['caption']['reserved']==restored['counts']['caption']['actual']==1
    assert restored['jobs'][0]['raw_saved']
    with g.journal.open('ab') as f:f.write(b'{"sequence":')
    restored=read_receipt(g.path)
    assert restored['truncated_journal_tail'] and restored['counts']['caption']['completed']==1


def test_journal_rejects_sequence_gap(tmp_path,package):
    g=guard(tmp_path,package)
    g.invoke('caption',1,lambda:torch.ones(1,1,8))
    lines=g.journal.read_text().splitlines(keepends=True)
    g.journal.write_text(''.join(lines[1:]))
    with pytest.raises(ValueError,match='sequence gap'):read_receipt(g.path)


def test_prepare_real_inputs_and_snapshots_without_forward_or_GPU(tmp_path,monkeypatch):
    bounded=script('run_mdm_training_sufficiency_bounded');forbid_gpu_and_models(monkeypatch)
    manifest=read(bounded.prepare(tmp_path/'implementation'))
    assert len(manifest['states'])==192 and len(manifest['inputs'])==12 and len(manifest['snapshots'])==6
    assert manifest['model_forwards']==manifest['GPU_queries']==0


@pytest.mark.parametrize('scenario',['success','preflight','timeout','receipt_missing','artifact_missing','artifact_hash','counter','deadline','source_hash'])
def test_parent_deadline_denominators_and_one_launch(tmp_path,monkeypatch,package,scenario):
    bounded=script('run_mdm_training_sufficiency_bounded');clock=[0.];calls=[]
    pkg=SimpleNamespace(path=PACKAGE,states=package.states,jobs=package.jobs,checks=package.checks,cases=package.cases,
                        snapshots=package.snapshots,common=dict(environment=environment()))
    monkeypatch.setattr(bounded,'TrainingPackage',lambda *a,**kw:pkg)
    manifest=tmp_path/'implementation.json';record=tmp_path/'record.json';run=tmp_path/'run'
    expected=[]
    if scenario=='source_hash':
        source=tmp_path/'source.py';source.write_text('original');expected=[artifact(source)];source.write_text('changed')
    atomic_json(manifest,dict(version='itm-sufficiency-implementation-v1',package=artifact(PACKAGE),environment=environment(),
                              snapshots=[dict(snapshot_id=r['snapshot_id'],artifact=r['artifact']) for r in package.snapshots],expected_artifacts=expected))
    acceptance=tmp_path/'acceptance.json';atomic_json(acceptance,dict(status='passed',package_sha256=PACKAGE_SHA,
                                                                   implementation_sha256=artifact(manifest)['sha256'],artifacts=[]))
    bounded.record_execution(record,manifest,run,'GPU-abc',acceptance)
    def preflight(gpu):
        status=read(run/'runtime/status.json')
        assert [len(status[k]) for k in ('states','jobs','branches','checks')]==[192,1152,4608,2880]
        assert all(r['status']=='planned' for r in status['jobs'])
        if scenario=='preflight':raise ValueError('CPU preflight failure')
        clock[0]=601. if scenario=='deadline' else 1.;return {'CPU_fixture':True}
    monkeypatch.setattr(bounded,'preflight_gpu',preflight)
    def supervise(argv,**kwargs):
        calls.append(argv);assert kwargs['deadline']==600.
        context=read(run/'runtime/diagnostic.context.json')
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='GPU-abc'
        if scenario=='receipt_missing':return dict(status='completed')
        child=dict(version='itm-sufficiency-receipt-v1',status='completed',
                   context={k:v for k,v in context.items() if k!='expected_artifacts'},**ledger(package),
                   updates_attempted=0,updates_completed=0,denoising_calls=0,peak_allocated_bytes=0,
                   counts={k:dict(reserved=cap,actual=cap,completed=cap,reserved_calls=CALL_CAPS[k],actual_calls=CALL_CAPS[k],completed_calls=CALL_CAPS[k]) for k,cap in CAPS.items()},artifacts=[])
        for key in ('states','jobs','branches','checks'):
            for row in child[key]:row['status']='completed'
        if scenario=='timeout':
            child['status']='running';child['jobs'][1]['status']='running'
            for r in child['jobs'][2:]:r['status']='planned'
        for filename in sorted(bounded.required_artifacts(package,run/'diagnostic')):
            p=Path(filename);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'CPU fixture')
            child['artifacts'].append(artifact(p))
        if scenario=='artifact_missing':child['artifacts'].pop()
        if scenario=='artifact_hash':child['artifacts'][0]['sha256']='wrong'
        if scenario=='counter':child['counts']['mdm']['completed']-=4
        atomic_json(Path(context['receipt_path']),child)
        return dict(status='failed' if scenario=='timeout' else 'completed')
    monkeypatch.setattr(bounded,'supervise',supervise)
    if scenario=='success':
        result=bounded.execute(manifest,run,record,'GPU-abc',clock=lambda:clock[0]);assert result['status']=='completed'
    else:
        with pytest.raises((RuntimeError,ValueError,TimeoutError)):
            bounded.execute(manifest,run,record,'GPU-abc',clock=lambda:clock[0])
        status=read(run/'runtime/status.json');assert status['status']=='failed'
        assert [len(status[k]) for k in ('states','jobs','branches','checks')]==[192,1152,4608,2880]
        assert not any(r['status'] in ('planned','running') for r in status['jobs'])
    assert len(calls)==(0 if scenario in ('preflight','deadline','source_hash') else 1)


def test_evaluation_Q_replicates_then_equal_sources_and_missing():
    evaluator=script('evaluate_b0_training_sufficiency')
    rows=[dict(row_id=str(i)+source,motion_id=source,key='fixed',status='valid',temporal={'mse':value})
          for source,values in [('a',(0.,10.)),('b',(100.,100.)),('c',(10.,10.))] for i,value in enumerate(values)]
    report=evaluator.aggregate(rows,('key',),'temporal')[0]
    assert report['metrics']['mse']['mean']==pytest.approx((5+100+10)/3)
    rows[0].update(status='missing',reason='fixture')
    report=evaluator.aggregate(rows,('key',),'temporal')[0]
    assert report['planned_sources']==3 and report['valid_sources']==2
    assert report['metrics']['mse']['mean']==55 and report['metrics']['mse']['valid_n']==2
    assert report['sources'][0]['metrics']['mse'] is None


def test_saved_only_evaluator_keeps_all_denominators_for_preflight_failure(tmp_path,package,monkeypatch):
    evaluator=script('evaluate_b0_training_sufficiency');run=tmp_path/'run';(run/'runtime').mkdir(parents=True)
    manifest=tmp_path/'implementation.json';atomic_json(manifest,dict(CPU_fixture=True))
    status=dict(status='failed',implementation=artifact(manifest),execution_record=dict(implementation_sha256=artifact(manifest)['sha256']),
                **ledger(package),started_monotonic=0.,finished_monotonic=1.,failure={'message':'CPU preflight fixture'},
                counts={k:dict(reserved=0,actual=0,completed=0,reserved_calls=0,actual_calls=0,completed_calls=0) for k in CAPS})
    for key in ('states','jobs','branches','checks'):
        for row in status[key]:row['status']='not_run_after_stop'
    atomic_json(run/'runtime/status.json',status)
    forbid_gpu_and_models(monkeypatch)
    summary=evaluator.evaluate(run,tmp_path/'evaluation')
    assert summary['branch_statuses']=={'missing':4608}
    assert summary['within_statuses']==summary['checkpoint_statuses']=={'not_comparable':4608}
    assert summary['decision']=='inconclusive_incomplete_or_failed_software'
