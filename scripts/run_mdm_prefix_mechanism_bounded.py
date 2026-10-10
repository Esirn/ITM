#!/usr/bin/env python
"""CPU preparation and delegated execution of one frozen 600-second prefix queue."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from itm.models.mdm_correction import ROOT,read,verify,artifact,environment
from itm.models.mdm_run_limits import atomic_json,sha256,GIB
from itm.models.mdm_prefix_mechanism import PrefixPackage,PACKAGE,PACKAGE_SHA,CAPS,CALL_CAPS,TABLES,ledger,stop_rows,read_receipt,required_artifacts
from itm.models.mdm_training_sufficiency import compact_json
from itm.models.mdm_sampling_alignment import input_arrays,describe
from run_mdm_b0_bounded import supervise
from run_mdm_native_training_sufficiency_bounded import POLICY,POLICY_SHA,SHARED_POLICY,shared_policy
from itm_shared_gpu_preflight import preflight_gpu,LIMITS


def prepare(directory):
    package=PrefixPackage()
    if environment()!=package.common['environment']:raise ValueError('Frozen environment mismatch')
    for sid in ('F200','U200'):package.state(sid)
    inputs=[]
    for r in package.trajectories:
        item,row=package.inputs(r)
        inputs.append(dict(trajectory_id=r['trajectory_id'],input_arrays={k:describe(v) for k,v in input_arrays(item,row).items()}))
    paths={package.path,*package.files.values(),*package.source_paths,POLICY,SHARED_POLICY,Path(__file__).resolve()}
    for folder in ('src/itm/models','src/itm/backbones','src/itm/data','src/itm/metrics','outputs/mdm/runtime_root'):
        paths.update(p.resolve() for p in (ROOT/folder).rglob('*.py'))
    for name in ('run_mdm_prefix_mechanism_worker.py','run_mdm_correction_worker.py','run_mdm_b0_bounded.py',
                 'run_mdm_native_training_sufficiency_bounded.py','evaluate_b0_prefix_mechanism.py','itm_shared_gpu_preflight.py','train_mdm_imu_control.py'):
        paths.add(ROOT/'scripts'/name)
    paths.add(ROOT/'tests/test_mdm_prefix_mechanism.py')
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    manifest=dict(version='itm-prefix-implementation-v1',package=artifact(package.path),environment=environment(),
        resource_policy=shared_policy(),snapshots={sid:package.snapshots[sid]['artifact'] for sid in ('F200','U200')},
        expected_artifacts=[artifact(p) for p in sorted(paths)],inputs=inputs,budgets=package.budgets,
        planned_worker_artifacts=len(required_artifacts(package,directory/'placeholder_output')),
        model_constructions=0,model_forwards=0,GPU_queries=0,new_training=0,new_motion_decodes=0)
    atomic_json(directory/'implementation.json',manifest);return directory/'implementation.json'


def record_execution(path,manifest,run,gpu,acceptance):
    if sha256(POLICY)!=POLICY_SHA or read(POLICY).get('authorized') is not True:raise ValueError('Standing delegation changed')
    manifest=Path(manifest).resolve();value=read(manifest);acceptance=Path(acceptance).resolve();test=read(acceptance)
    if value['package']['sha256']!=PACKAGE_SHA or value['version']!='itm-prefix-implementation-v1':raise ValueError('Wrong prefix implementation')
    if test.get('status')!='passed' or test.get('package_sha256')!=PACKAGE_SHA or test.get('implementation_sha256')!=sha256(manifest):raise ValueError('Bound CPU acceptance required')
    for e in test['artifacts']:verify(e)
    path=Path(path).resolve()
    if path.exists() or Path(run).exists():raise FileExistsError('Fresh execution record/run required')
    if not gpu.startswith('GPU-'):raise ValueError('Physical GPU UUID required')
    value=dict(version='itm-prefix-execution-v1',authorized=True,approved_by='user_standing_delegation',
        policy=artifact(POLICY),resource_policy=shared_policy(),resource_gates=LIMITS,
        package_sha256=PACKAGE_SHA,implementation=artifact(manifest),implementation_sha256=sha256(manifest),
        CPU_acceptance=artifact(acceptance),run_directory=str(Path(run).resolve()),gpu_uuid=gpu,
        absolute_seconds=600,memory_bytes=24*GIB,calls=CALL_CAPS,examples=CAPS,
        scope='One new bounded replay of original t999..900 prefixes and fixed-state probes; no training/full decode/retry',
        user_instruction='以后使用GPU不用我手动授权，你自行决定；低负载可共享GPU')
    atomic_json(path,value);return value


def authorization(path,manifest,run,gpu):
    value=read(path)
    expected=dict(version='itm-prefix-execution-v1',authorized=True,approved_by='user_standing_delegation',
        package_sha256=PACKAGE_SHA,implementation_sha256=sha256(manifest),run_directory=str(Path(run).resolve()),
        gpu_uuid=gpu,absolute_seconds=600,memory_bytes=24*GIB,calls=CALL_CAPS,examples=CAPS)
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Concrete prefix execution identity changed')
    if sha256(verify(value['policy']))!=POLICY_SHA or value['resource_policy']!=shared_policy() or value['resource_gates']!=LIMITS:raise ValueError('Delegated policy changed')
    acceptance=read(verify(value['CPU_acceptance']))
    if acceptance.get('status')!='passed' or acceptance['implementation_sha256']!=sha256(manifest):raise ValueError('CPU acceptance identity changed')
    for e in acceptance['artifacts']:verify(e)
    verify(value['implementation']);return value


def validate_receipt(receipt,context,package,out,check=lambda:None):
    if receipt['status']!='completed' or receipt['context']!={k:v for k,v in context.items() if k!='expected_artifacts'}:raise ValueError('Incomplete receipt/context')
    for k,cap in CAPS.items():
        c=receipt['counts'][k]
        if any(c[f]!=cap for f in ('reserved','actual','completed')) or any(c[f]!=CALL_CAPS[k] for f in ('reserved_calls','actual_calls','completed_calls')):raise ValueError('Incomplete prefix count')
    if receipt['updates_attempted'] or receipt['updates_completed'] or receipt['denoising_calls']!=3600 or receipt['branch_examples']!=15840:raise ValueError('Forbidden/incomplete work')
    planned=ledger(package)
    for key in TABLES:
        if [r['id'] for r in receipt[key]]!=[r['id'] for r in planned[key]] or any(r['status']!='completed' for r in receipt[key]):raise ValueError('Incomplete prefix ledger')
    produced={a['path'] for a in receipt['artifacts']}
    missing=required_artifacts(package,out)-produced
    if missing:raise ValueError('Missing required prefix artifact: '+sorted(missing)[0])
    for e in receipt['artifacts']:verify(e);check()


def execute(manifest_path,run,record,gpu,*,clock=time.monotonic):
    started=clock();deadline=started+600;run=Path(run).resolve();manifest_path=Path(manifest_path).resolve()
    approval=authorization(record,manifest_path,run,gpu);manifest=read(manifest_path)
    package_path=verify(manifest['package']);files={Path(a['path']).stem:verify(a) for a in read(package_path)['artifacts']}
    package=SimpleNamespace(trajectories=read(files['trajectories']),states=read(files['states']),probes=read(files['probes']),anchors=read(files['historical_anchors']))
    run.mkdir(parents=True,exist_ok=False);runtime=run/'runtime';runtime.mkdir()
    status=dict(version='itm-prefix-queue-v1',status='initializing',execution_record=approval,implementation=artifact(manifest_path),
        started_monotonic=started,deadline_monotonic=deadline,**ledger(package),
        counts={k:dict(reserved=0,actual=0,completed=0,reserved_calls=0,actual_calls=0,completed_calls=0) for k in CAPS})
    sp=runtime/'status.json';compact_json(sp,status);receipt=None;child=None
    def check():
        if clock()>=deadline:raise TimeoutError('Prefix 600-second absolute deadline')
    try:
        expected=manifest['expected_artifacts']+[artifact(manifest_path)]
        for e in expected:verify(e);check()
        package=PrefixPackage(package_path);check()
        if environment()!=manifest['environment'] or environment()!=package.common['environment']:raise ValueError('Frozen environment changed')
        with (Path('/tmp')/('itm-b0-'+gpu+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check()
            status['gpu_preflight']=preflight_gpu(gpu,evidence_dir=runtime/'preflight',check=check);check()
            receipt=runtime/'prefix.receipt.json'
            context=dict(version='itm-prefix-context-v1',phase='prefix',package_sha256=PACKAGE_SHA,
                implementation_sha256=sha256(manifest_path),deadline_monotonic=deadline,receipt_path=str(receipt),
                gpu_uuid=gpu,memory_bytes=24*GIB,expected_artifacts=expected)
            cp=runtime/'prefix.context.json';atomic_json(cp,context)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                PYTORCH_CUDA_ALLOC_CONF='backend:native',PYTORCH_ALLOC_CONF='backend:native',OMP_NUM_THREADS='2')
            for k in ('RANK','LOCAL_RANK','WORLD_SIZE'):env.pop(k,None)
            argv=[sys.executable,str(ROOT/'scripts/run_mdm_prefix_mechanism_worker.py'),'--package',str(package.path),'--run-context',str(cp),'--output',str(run/'prefix')]
            status['status']='prefix';compact_json(sp,status)
            result=supervise(argv,deadline=deadline,log_path=runtime/'prefix.log',env=env,clock=clock);status['worker_result']=result
            if receipt.exists():child=read_receipt(receipt)
            if result['status']!='completed':raise RuntimeError('Prefix child stopped: '+str(result))
            if child is None:raise ValueError('Prefix receipt missing')
            validate_receipt(child,context,package,run/'prefix',check);check()
            for key in (*TABLES,'counts','peak_allocated_bytes'):status[key]=child[key]
            status.update(status='completed',receipt=artifact(receipt),receipt_journal=artifact(receipt.with_suffix('.journal.jsonl')),finished_monotonic=clock())
            compact_json(sp,status)
    except BaseException as error:
        if child is None and receipt is not None and receipt.exists():
            try:child=read_receipt(receipt)
            except Exception as reconciliation:status['receipt_reconciliation_error']=str(reconciliation)
        if child:
            for key in (*TABLES,'counts','peak_allocated_bytes'):
                if key in child:status[key]=child[key]
            status['receipt']=artifact(receipt)
            jp=receipt.with_suffix('.journal.jsonl')
            if jp.exists():status['receipt_journal']=artifact(jp)
        stop_rows(status);status.update(status='failed',finished_monotonic=clock(),failure=dict(type=type(error).__name__,message=str(error)))
        compact_json(sp,status);raise
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-dir',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--record-execution',action='store_true')
    for n in ('implementation','run-dir','execution-record','CPU-acceptance'):p.add_argument('--'+n,type=Path)
    p.add_argument('--gpu-uuid');a=p.parse_args()
    if a.prepare_dir:
        if a.execute or a.record_execution:raise ValueError('Prepare must remain CPU')
        print(prepare(a.prepare_dir));return
    if not all([a.implementation,a.run_dir,a.execution_record,a.gpu_uuid]):raise ValueError('Concrete identities required')
    if a.record_execution:record_execution(a.execution_record,a.implementation,a.run_dir,a.gpu_uuid,a.CPU_acceptance)
    elif a.execute:execute(a.implementation,a.run_dir,a.execution_record,a.gpu_uuid)
    else:raise ValueError('Explicit execution/record required')


if __name__=='__main__':main()
