#!/usr/bin/env python
"""Frozen single-state CLIP batch-size / MDM batch-size path isolation.

Separate bounded experiment, never resumes the stopped main queue. Same U200,
same saved x_t, no new noise/training/sampling; original elementwise bounds.
"""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from itm.models.mdm_correction import ROOT,read,verify,artifact,encode_control,environment
from itm.models.mdm_sampling_alignment import numpy,describe,difference,save_npz,save_json
from itm.models.mdm_run_limits import RunGuard,RunLimitExceeded,atomic_json,sha256,GIB
from itm.models.mdm_state_diagnostic import CountedEncoder
from itm.models.mdm_training_sufficiency import TrainingPackage
from run_mdm_correction_worker import load_model,batch_inputs
from run_mdm_b0_bounded import supervise
from itm_shared_gpu_preflight import preflight_gpu

CAPS=dict(mdm=(4,18),encoder=(6,6),caption=(2,6))
OLD=ROOT/'outputs/itm_rework/method/sampling_alignment_authorized_20260927_v1/Q'
NEW=ROOT/'outputs/itm_rework/method/training_sufficiency_authorized_20261008_v2'


def initial_ledger():
    return dict(jobs=[dict(job_id=b,status='planned',executed=False) for b in ('A','B','C','D')],
                checks=[dict(check_id=b+'-'+str(i),status='planned') for b,n in (('A',5),('C',4)) for i in range(n)])


class ProbeGuard(RunGuard):
    def __init__(self,context):
        super().__init__(dict(context,phase='sample'))
        self.context=context
        self.state.update(**initial_ledger(),counts={k:dict(reserved_calls=0,reserved_examples=0,actual_calls=0,actual_examples=0,completed_calls=0,completed_examples=0) for k in CAPS})
        self.persist()
    def before_update(self):raise RunLimitExceeded('Path probe forbids training')
    def before_denoise(self,*args):raise RunLimitExceeded('Path probe forbids denoising')
    def invoke(self,kind,n,function):
        self.check();c=self.state['counts'][kind];calls,examples=CAPS[kind]
        if c['reserved_calls']+1>calls or c['reserved_examples']+n>examples:raise RunLimitExceeded('Path probe call cap')
        c['reserved_calls']+=1;c['reserved_examples']+=n;self.persist();self.check()
        c['actual_calls']+=1;c['actual_examples']+=n;self.persist()
        result=function();self.ensure_finite(result,kind);self.check()
        c['completed_calls']+=1;c['completed_examples']+=n;self.persist();return result
    def finish(self):
        for kind,(calls,examples) in CAPS.items():
            c=self.state['counts'][kind]
            if c!=dict(reserved_calls=calls,reserved_examples=examples,actual_calls=calls,actual_examples=examples,completed_calls=calls,completed_examples=examples):
                raise RunLimitExceeded('Incomplete path probe count')
        if any(r['status']!='completed' for k in ('jobs','checks') for r in self.state[k]):raise RunLimitExceeded('Incomplete path probe ledger')
        self.check();self.state['status']='completed';self.persist()
    def fail(self,error):
        for key in ('jobs','checks'):
            for row in self.state[key]:
                if row['status']=='planned':row['status']='not_run_after_stop'
                elif row['status']=='running':row['status']='failed'
        super().fail(error)


def text_paths(single,native,presence):
    if single.shape!=(1,1,native.shape[-1]) or native.shape[0:2]!=(1,5) or presence.tolist()!=[False,True,True,True,True]:
        raise ValueError('Frozen caption/branch shape changed')
    return (torch.where(presence[None,:,None],single.repeat(1,5,1),0),
            torch.where(presence[None,:,None],native,0))


def worker(plan,context,output):
    g=ProbeGuard(context);handles=None
    try:
        for entry in plan['expected_artifacts']:verify(entry);g.check()
        package=TrainingPackage();g.configure_cuda(torch,torch.device('cuda:0'))
        state=package.state(package.snapshots[-1]);model,diffusion,controlled,encoder,original,handles,backbone=load_model(package,state,'cuda:0')
        for p in model.parameters():p.requires_grad_(False)
        out=Path(output);out.mkdir(parents=True,exist_ok=False)
        rows=package.alignment.probes[:5];items=[package.alignment.inputs(r) for r in rows]
        saved=package.saved(package.states[0])
        if any(i['text']!=plan['caption'] or not np.array_equal(i['motion'].T[:,None],saved['target']) for i in items):
            raise ValueError('Saved target/caption changed')
        with torch.no_grad():
            _,y5=batch_inputs(items,rows,'cuda:0')
            one=g.invoke('caption',1,lambda:model.encode_text([plan['caption']]))
            five=g.invoke('caption',5,lambda:model.encode_text([plan['caption']]*5))
            if one.dtype!=torch.float32 or five.dtype!=torch.float32:raise ValueError('Native CLIP return dtype changed')
            broadcast,native=text_paths(one,five,y5['text_condition_present'])
            counted=CountedEncoder(encoder,g);controls=[torch.zeros((1,plan['frames'],encoder.config.latent_dim),device='cuda:0') for _ in range(2)]
            for item,row in zip(items[2:],rows[2:]):
                _,y=batch_inputs([item],[row],'cuda:0');control,present=encode_control(counted,y,mode='static_delta')
                if not present.all():raise ValueError('Control presence changed')
                controls.append(control)
            y5.update(imu_control=torch.cat(controls),imu_control_mode='static_delta');y5.pop('imu_reference')
            _,y4=batch_inputs(items[1:],rows[1:],'cuda:0');y4.update(imu_control=y5['imu_control'][1:],imu_control_mode='static_delta');y4.pop('imu_reference')
            save_npz(out/'captions.npz',dict(batch1=numpy(one),batch5=numpy(five)),g)
            predictions={};reports=[]
            for index,(name,batch,text) in enumerate((('A',5,native),('B',5,broadcast),('C',4,broadcast[:,1:]),('D',4,native[:,1:]))):
                row=g.state['jobs'][index];row['status']='running';row['executed']=True;g.persist()
                y=dict(y5 if batch==5 else y4,text_embed=text)
                xt=torch.from_numpy(saved['x_t']).to('cuda:0').repeat(batch,1,1,1)
                t=torch.full((batch,),plan['timestep'],device='cuda:0',dtype=torch.long)
                scaled=diffusion._scale_timesteps(t)
                if numpy(scaled).tolist()!=[plan['timestep']]*batch:raise ValueError('Timestep identity changed')
                p=numpy(g.invoke('mdm',batch,lambda:model(xt,scaled,y=y)));predictions[name]=p
                save_npz(out/(name+'.npz'),dict(prediction=p),g);row['raw_artifact']=artifact(out/(name+'.npz'));g.persist()
                if name in ('A','C'):
                    if name=='A':
                        refs=[]
                        for pid in plan['old_probe_ids']:
                            with np.load(OLD/(pid+'.npz'),allow_pickle=False) as z:refs.append(z['predicted_x0'].copy())
                    else:
                        with np.load(NEW/'diagnostic/predictions/U200-S001.npz',allow_pickle=False) as z:refs=[z[b].copy() for b in ('T','P','S','R')]
                    for i,ref in enumerate(refs):
                        report=dict(difference(p[i],ref,enforce=False),check_id=name+'-'+str(i));reports.append(report)
                        r=next(r for r in g.state['checks'] if r['check_id']==report['check_id']);r['status']='completed' if report['passed'] else 'failed';g.persist()
                        if not report['passed']:raise ValueError('Original 1e-6 path anchor failed: '+report['check_id'])
                row['status']='completed';g.persist()
            deltas={}
            for label,a,b in (('B_minus_A_caption_effect',predictions['B'][1:],predictions['A'][1:]),
                              ('D_minus_A_MDM_batch_effect',predictions['D'],predictions['A'][1:]),
                              ('C_minus_B_MDM_batch_effect',predictions['C'],predictions['B'][1:]),
                              ('C_minus_D_caption_effect',predictions['C'],predictions['D'])):
                deltas[label]=difference(a,b,enforce=False)
            save_json(out/'result.json',dict(checks=reports,comparisons=deltas,
                       caption_batch1_vs_batch5=difference(numpy(one).repeat(5,axis=1),numpy(five),enforce=False),
                       caption_batch5_duplicate_spread=difference(numpy(five[:,:1]).repeat(5,axis=1),numpy(five),enforce=False),
                       interpretation='Controlled numerical path diagnosis, no motion/convergence claim.'),g)
        g.finish()
    except BaseException as error:g.fail(error);raise
    finally:
        if handles is not None:handles.remove()


def prepare(directory):
    package=TrainingPackage();row=package.states[0]
    if row['state_id']!='S001' or row['timestep']!=50 or row['motion_id']!='000506':raise ValueError('Fixed source changed')
    probes=package.alignment.probes[:5];caption=probes[1]['text']
    paths={Path(__file__).resolve(),ROOT/'tests/test_b0_caption_path_probe.py',ROOT/'scripts/itm_shared_gpu_preflight.py',
           ROOT/'docs/itm_rework/GPU_SHARED_EXECUTION_POLICY_20261008.json',package.path,*package.files.values(),*package.source_paths,
           NEW/'runtime/status.json',NEW/'diagnostic/predictions/U200-S001.npz',NEW/'diagnostic/jobs/U200-S001.json'}
    manifest=read(ROOT/'outputs/itm_rework/method/training_sufficiency_implementation_20261008_v2/implementation.json')
    for entry in manifest['expected_artifacts']:paths.add(verify(entry))
    for p in probes:paths.update((OLD/(p['probe_id']+'.npz'),OLD/(p['probe_id']+'.json')))
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    plan=dict(version='itm-caption-path-probe-v1',frames=row['frames'],timestep=50,caption=caption,
              state=row,checkpoint=package.snapshots[-1],old_probe_ids=[r['probe_id'] for r in probes],
              paths={'A':'native CLIP batch5, MDM batch5','B':'CLIP batch1 broadcast, MDM batch5',
                     'C':'CLIP batch1 broadcast, MDM batch4','D':'native CLIP batch5 slots1:5, MDM batch4'},
              budget=dict(seconds=120,GPUs=1,memory_GiB=24,calls=CAPS,new_training=0,new_noise=0,new_sampling=0,decode=0),
              anchors='A versus old Q0001..Q0005; C versus saved U200-S001 T/P/S/R. Original atol=rtol=1e-6; no relaxation.',
              **initial_ledger(),expected_artifacts=[artifact(p) for p in sorted(paths)],environment=environment())
    atomic_json(directory/'plan.json',plan);return directory/'plan.json'


def execute(plan_path,run,gpu):
    start=time.monotonic();deadline=start+120;plan_path=Path(plan_path).resolve();plan=read(plan_path)
    if plan['version']!='itm-caption-path-probe-v1' or plan['budget']['seconds']!=120:raise ValueError('Wrong path probe plan')
    run=Path(run).resolve();run.mkdir(parents=True,exist_ok=False);runtime=run/'runtime';runtime.mkdir()
    status=dict(status='initializing',plan=artifact(plan_path),start=start,deadline=deadline,**initial_ledger(),
                authorization='Standing user GPU delegation plus explicit low-load sharing instruction 2026-10-08; separate single-state path isolation, not main-queue retry.')
    def check():
        if time.monotonic()>=deadline:raise TimeoutError('Path probe absolute 120-second deadline')
    cp=runtime/'context.json';receipt=runtime/'receipt.json'
    try:
        for entry in plan['expected_artifacts']:verify(entry);check()
        if environment()!=plan['environment']:raise ValueError('Bound environment changed')
        atomic_json(runtime/'status.json',status)
        with (Path('/tmp')/('itm-b0-'+gpu+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            status['preflight']=preflight_gpu(gpu,evidence_dir=runtime/'preflight',check=check)
            context=dict(phase='sample',deadline_monotonic=deadline,gpu_uuid=gpu,memory_bytes=24*GIB,receipt_path=str(receipt),plan=artifact(plan_path))
            atomic_json(cp,context);status['status']='running';atomic_json(runtime/'status.json',status)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTORCH_CUDA_ALLOC_CONF='backend:native',PYTORCH_ALLOC_CONF='backend:native')
            result=supervise([sys.executable,str(Path(__file__).resolve()),'--worker','--plan',str(plan_path),'--context',str(cp),'--run-dir',str(run)],deadline=deadline,log_path=runtime/'worker.log',env=env)
            status['worker_result']=result
            if receipt.exists():status['receipt']=artifact(receipt)
            if result['status']!='completed':raise RuntimeError('Path probe worker failed')
            child=read(receipt)
            if child['status']!='completed':raise ValueError('Incomplete path probe receipt')
            for a in child['artifacts']:verify(a);check()
            status.update(status='completed',finish=time.monotonic(),counts=child['counts'],peak_allocated_bytes=child['peak_allocated_bytes'])
    except BaseException as error:status.update(status='failed',finish=time.monotonic(),failure=str(error));raise
    finally:atomic_json(runtime/'status.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-dir',type=Path);p.add_argument('--plan',type=Path)
    p.add_argument('--run-dir',type=Path);p.add_argument('--gpu-uuid');p.add_argument('--worker',action='store_true');p.add_argument('--context',type=Path)
    a=p.parse_args()
    if a.prepare_dir:print(prepare(a.prepare_dir))
    elif a.worker:worker(read(a.plan),read(a.context),a.run_dir/'probe')
    else:execute(a.plan,a.run_dir,a.gpu_uuid)
