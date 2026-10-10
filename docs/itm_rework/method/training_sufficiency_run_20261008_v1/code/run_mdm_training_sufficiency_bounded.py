#!/usr/bin/env python
"""CPU preparation/execution record, or one 600-second frozen diagnostic."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from itm.models.mdm_correction import ROOT,artifact,read,verify,environment
from itm.models.mdm_run_limits import atomic_json,sha256,GIB
from itm.models.mdm_training_sufficiency import (TrainingPackage,PACKAGE,PACKAGE_SHA,CAPS,CALL_CAPS,
                                               ledger,stop_rows,compact_json,SNAPSHOTS,BRANCHES,read_receipt)
from run_mdm_b0_bounded import preflight_gpu,supervise

POLICY=ROOT/'docs/itm_rework/GPU_EXECUTION_POLICY_20261008.json'
POLICY_SHA='6d63492234840f349b7ff67ef222499ca20987a9f9c75fb49b32e7444fc91179'


def prepare(directory):
    package=TrainingPackage()
    if environment()!=package.common['environment']:raise ValueError('Frozen environment mismatch')
    for snapshot in package.snapshots:package.state(snapshot)
    inputs=[dict(cache_id=c['cache_id'],descriptors=package.inputs(c)[2]) for c in package.cases]
    for row in package.states:package.saved(row)
    paths={package.path,*package.files.values(),*package.source_paths,POLICY,Path(__file__).resolve()}
    for folder in ('src/itm/models','src/itm/backbones','src/itm/data','src/itm/metrics','outputs/mdm/runtime_root'):
        paths.update(p.resolve() for p in (ROOT/folder).rglob('*.py'))
    for name in ('run_mdm_training_sufficiency_worker.py','run_mdm_correction_worker.py',
                 'run_mdm_b0_bounded.py','train_mdm_imu_control.py','evaluate_b0_training_sufficiency.py'):
        paths.add(ROOT/'scripts'/name)
    paths.add(ROOT/'tests/test_mdm_training_sufficiency.py')
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    manifest=dict(version='itm-sufficiency-implementation-v1',package=artifact(package.path),
                  snapshots=[dict(snapshot_id=r['snapshot_id'],artifact=r['artifact']) for r in package.snapshots],
                  environment=environment(),expected_artifacts=[artifact(p) for p in sorted(paths)],
                  inputs=inputs,states=package.states,plan=dict(MDM_calls=1152,MDM_examples=4608,encoder_calls=432,caption_calls=12),
                  model_constructions=0,model_forwards=0,GPU_queries=0,new_training=0,new_samples=0)
    atomic_json(directory/'implementation.json',manifest)
    return directory/'implementation.json'


def record_execution(path,manifest,run,gpu,acceptance):
    if sha256(POLICY)!=POLICY_SHA or read(POLICY).get('authorized') is not True:
        raise ValueError('Standing GPU delegation identity missing')
    if not gpu.startswith('GPU-'):raise ValueError('Concrete physical GPU UUID required')
    manifest=Path(manifest).resolve();value=read(manifest)
    if value['version']!='itm-sufficiency-implementation-v1' or value['package']['sha256']!=PACKAGE_SHA:
        raise ValueError('Wrong prepared implementation')
    test_record=read(acceptance)
    if (test_record.get('status')!='passed' or test_record.get('package_sha256')!=PACKAGE_SHA
            or test_record.get('implementation_sha256')!=sha256(manifest)):
        raise ValueError('Completed bound CPU acceptance required')
    for entry in test_record['artifacts']:verify(entry)
    path=Path(path).resolve()
    if path.exists() or Path(run).exists():raise FileExistsError('Execution record/run must be fresh')
    value=dict(version='itm-sufficiency-execution-record-v1',authorized=True,
               approved_by='user_standing_delegation',policy=artifact(POLICY),
               package_sha256=PACKAGE_SHA,implementation=artifact(manifest),
               implementation_sha256=sha256(manifest),CPU_acceptance=artifact(acceptance),
               run_directory=str(Path(run).resolve()),gpu_uuid=gpu,
               absolute_seconds=600,memory_bytes=24*GIB,calls=CALL_CAPS,examples=CAPS,
               scope='One fresh fixed-state diagnostic only, no training/new samples/new noise/retry.',
               user_instruction='以后使用GPU不用我手动授权，你自行决定')
    atomic_json(path,value);return value


def authorization(path,manifest,run,gpu):
    value=read(path)
    expected=dict(version='itm-sufficiency-execution-record-v1',authorized=True,
                  approved_by='user_standing_delegation',package_sha256=PACKAGE_SHA,
                  implementation_sha256=sha256(manifest),run_directory=str(Path(run).resolve()),gpu_uuid=gpu,
                  absolute_seconds=600,memory_bytes=24*GIB,calls=CALL_CAPS,examples=CAPS)
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Concrete delegated execution record mismatch')
    if sha256(verify(value['policy']))!=POLICY_SHA:raise ValueError('Standing policy identity changed')
    acceptance=read(verify(value['CPU_acceptance']))
    if acceptance['status']!='passed' or acceptance['implementation_sha256']!=sha256(manifest):
        raise ValueError('CPU acceptance identity changed')
    for entry in acceptance['artifacts']:verify(entry)
    verify(value['implementation'])
    return value


def required_artifacts(package,output):
    output=Path(output).resolve();paths=set()
    for row in package.states:paths.add(output/'states'/(row['state_id']+'.npz'))
    for job in package.jobs:
        paths.update((output/'predictions'/(job['job_id']+'.npz'),output/'jobs'/(job['job_id']+'.json')))
    for case in package.cases:
        cid=case['cache_id'];paths.add(output/'inputs'/(cid+'.json'))
        paths.update(output/'inputs'/(cid+'-'+b+'.npz') for b in BRANCHES)
        paths.add(output/'cache'/('text-'+cid+'.npz'))
        paths.update(output/'cache'/(sid+'-'+cid+ext) for sid in SNAPSHOTS for ext in ('.npz','.json'))
    return {str(p) for p in paths}


def validate_receipt(receipt,context,package,output,check):
    if receipt['status']!='completed' or receipt['context']!={k:v for k,v in context.items() if k!='expected_artifacts'}:
        raise ValueError('Receipt/context incomplete')
    if receipt['updates_attempted'] or receipt['updates_completed'] or receipt['denoising_calls']:
        raise ValueError('Forbidden diagnostic work')
    for kind,cap in CAPS.items():
        c=receipt['counts'][kind]
        if (any(c[k]!=cap for k in ('reserved','actual','completed'))
                or any(c[k]!=CALL_CAPS[kind] for k in ('reserved_calls','actual_calls','completed_calls'))):
            raise ValueError('Incomplete diagnostic count')
    initial=ledger(package)
    for key,idkey in (('states','state_id'),('jobs','job_id'),('branches','branch_id'),('checks','check_id')):
        if ([r[idkey] for r in receipt[key]]!=[r[idkey] for r in initial[key]]
                or any(r['status']!='completed' for r in receipt[key])):raise ValueError('Incomplete diagnostic ledger')
    produced={a['path'] for a in receipt['artifacts']}
    missing=required_artifacts(package,output)-produced
    if missing:
        receipt['missing_artifacts']=sorted(missing);raise ValueError('Required diagnostic artifacts missing')
    for entry in receipt['artifacts']:verify(entry);check()


def execute(manifest_path,run,record,gpu,*,clock=time.monotonic):
    start=clock();deadline=start+600
    manifest_path=Path(manifest_path).resolve();run=Path(run).resolve()
    approval=authorization(record,manifest_path,run,gpu);manifest=read(manifest_path)
    if manifest['version']!='itm-sufficiency-implementation-v1' or manifest['package']['sha256']!=PACKAGE_SHA:
        raise ValueError('Wrong diagnostic implementation')
    # Read only the immutable plan before creating its full failure ledger.
    # Runtime/data source verification happens inside the recorded attempt.
    package_path=verify(manifest['package'])
    plan_files={Path(a['path']).stem:a for a in read(package_path)['artifacts']}
    package=SimpleNamespace(states=read(verify(plan_files['states'])),jobs=read(verify(plan_files['jobs'])),
                            checks=read(verify(plan_files['software_checks'])))
    run.mkdir(parents=True,exist_ok=False);runtime=run/'runtime';runtime.mkdir()
    status=dict(version='itm-sufficiency-queue-v1',status='initializing',execution_record=approval,
                implementation=artifact(manifest_path),started_monotonic=start,deadline_monotonic=deadline,
                **ledger(package),counts={k:dict(reserved=0,actual=0,completed=0,
                  reserved_calls=0,actual_calls=0,completed_calls=0) for k in CAPS})
    path=runtime/'status.json';compact_json(path,status)
    def check():
        if clock()>=deadline:raise TimeoutError('Diagnostic 600-second absolute deadline')
    receipt=None;child=None
    try:
        expected=manifest['expected_artifacts']+[artifact(manifest_path)]
        for entry in expected:verify(entry);check()
        package=TrainingPackage(package_path);check()
        if environment()!=manifest['environment'] or environment()!=package.common['environment']:
            raise ValueError('Prepared environment changed')
        if manifest['snapshots']!=[dict(snapshot_id=r['snapshot_id'],artifact=r['artifact']) for r in package.snapshots]:
            raise ValueError('Prepared snapshot identities changed')
        with (Path('/tmp')/('itm-b0-'+gpu+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            check();status['gpu_preflight']=preflight_gpu(gpu);check()
            receipt=runtime/'diagnostic.receipt.json'
            context=dict(version='itm-sufficiency-context-v1',phase='diagnostic',package_sha256=PACKAGE_SHA,
                         implementation_sha256=sha256(manifest_path),deadline_monotonic=deadline,
                         receipt_path=str(receipt),gpu_uuid=gpu,memory_bytes=24*GIB,expected_artifacts=expected)
            cp=runtime/'diagnostic.context.json';atomic_json(cp,context)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                 PYTORCH_CUDA_ALLOC_CONF='backend:native',PYTORCH_ALLOC_CONF='backend:native')
            for key in ('RANK','LOCAL_RANK','WORLD_SIZE'):env.pop(key,None)
            argv=[sys.executable,str(ROOT/'scripts/run_mdm_training_sufficiency_worker.py'),'--package',str(package.path),
                  '--run-context',str(cp),'--output',str(run/'diagnostic')]
            status['status']='diagnostic';compact_json(path,status)
            result=supervise(argv,deadline=deadline,log_path=runtime/'diagnostic.log',env=env,clock=clock)
            status['worker_result']=result
            if receipt.exists():child=read_receipt(receipt)
            if result['status']!='completed':raise RuntimeError('Diagnostic child stopped: '+str(result))
            if child is None:raise ValueError('Diagnostic receipt missing')
            validate_receipt(child,context,package,run/'diagnostic',check);check()
            for key in ('states','jobs','branches','checks','counts','peak_allocated_bytes'):status[key]=child[key]
            status.update(status='completed',receipt=artifact(receipt),finished_monotonic=clock())
            journal=receipt.with_suffix('.journal.jsonl')
            if journal.exists():status['receipt_journal']=artifact(journal)
            compact_json(path,status)
    except BaseException as error:
        if child is None and receipt is not None and receipt.exists():
            try:child=read_receipt(receipt)
            except Exception as reconciliation_error:
                status['receipt_reconciliation_error']=str(reconciliation_error)
        if child:
            for key in ('states','jobs','branches','checks','counts','peak_allocated_bytes'):
                if key in child:status[key]=child[key]
                else:status['receipt_incomplete']=True
            status['receipt']=artifact(receipt)
            journal=receipt.with_suffix('.journal.jsonl')
            if journal.exists():status['receipt_journal']=artifact(journal)
            if child.get('missing_artifacts'):status['missing_artifacts']=child['missing_artifacts']
            # A supervisor/validation failure invalidates every otherwise-completed
            # output as primary evidence, while retaining execution/raw metadata.
            for key in ('jobs','branches'):
                for row in status[key]:
                    if row['status']=='completed':row['diagnostic_only']=True
        stop_rows(status)
        status.update(status='failed',finished_monotonic=clock(),failure=dict(type=type(error).__name__,message=str(error)))
        compact_json(path,status);raise
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-dir',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--record-execution',action='store_true')
    for name in ('implementation','run-dir','execution-record','CPU-acceptance'):p.add_argument('--'+name,type=Path)
    p.add_argument('--gpu-uuid');a=p.parse_args()
    if a.prepare_dir:
        if a.execute or a.record_execution:raise ValueError('CPU preparation cannot execute')
        print(prepare(a.prepare_dir));return
    if not all((a.implementation,a.run_dir,a.execution_record,a.gpu_uuid)):raise ValueError('Concrete identities required')
    if a.record_execution:
        if a.execute or not a.CPU_acceptance:raise ValueError('CPU acceptance required for record')
        record_execution(a.execution_record,a.implementation,a.run_dir,a.gpu_uuid,a.CPU_acceptance);return
    if not a.execute:raise ValueError('Explicit execute flag required')
    execute(a.implementation,a.run_dir,a.execution_record,a.gpu_uuid)


if __name__=='__main__':main()
