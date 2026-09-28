#!/usr/bin/env python
"""CPU V preparation or one newly authorized 300-second saved-state diagnostic."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
from itm.models.mdm_correction import ROOT,artifact,read,verify,environment
from itm.models.mdm_run_limits import atomic_json,sha256,GIB
from itm.models.mdm_sampling_alignment import describe
from itm.models.mdm_state_diagnostic import StatePackage,PACKAGE,PACKAGE_SHA,CHECKPOINT_SHA,CAPS,CALL_CAPS,ledger,stop_rows
from run_mdm_b0_bounded import preflight_gpu,supervise


def prepare(directory):
    package=StatePackage();package.state()
    if environment()!=package.common['environment']:raise ValueError('V frozen environment mismatch')
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    paths={package.path,*package.files.values(),*package.source_paths,package.checkpoint}
    for folder in ('src/itm/models','src/itm/backbones','src/itm/data','outputs/mdm/runtime_root'):
        paths.update(p.resolve() for p in (ROOT/folder).rglob('*.py'))
    paths.update(ROOT/'scripts'/name for name in ('run_mdm_state_diagnostic_bounded.py','run_mdm_state_diagnostic_worker.py',
        'run_mdm_sampling_alignment_worker.py','run_mdm_correction_worker.py','run_mdm_b0_bounded.py','train_mdm_imu_control.py'))
    inputs=[dict(cache_id=c['cache_id'],branches=package.inputs(c)[2]) for c in package.cases]
    states=[dict(group_id=g['group_id'],arrays={k:describe(v) for k,v in package.saved(g).items()}) for g in package.groups]
    manifest=dict(version='itm-state-implementation-v1',package=artifact(package.path),checkpoint=artifact(package.checkpoint),
        environment=environment(),expected_artifacts=[artifact(p) for p in sorted(paths)],inputs=inputs,states=states,
        real_model_forwards=0,training_updates=0,samples=0,GPU_queries=0,new_parameter_initializations=0,new_checkpoints=0)
    atomic_json(directory/'implementation.json',manifest);print(directory/'implementation.json');return manifest


def authorization(path,manifest,run,gpu):
    value=read(path)
    expected=dict(authorized=True,approved_by='user',package_sha256=PACKAGE_SHA,implementation_sha256=sha256(manifest),
                  run_directory=str(Path(run).resolve()),gpu_uuid=gpu)
    if any(value.get(k)!=v for k,v in expected.items()) or not value.get('user_instruction_reference'):
        raise ValueError('New explicit V authorization required')
    return value


def validate_receipt(receipt,context,output,check):
    if receipt['status']!='completed' or receipt['context']!=context:raise ValueError('V receipt/context incomplete')
    if receipt['updates_attempted'] or receipt['updates_completed'] or receipt['denoising_calls']:raise ValueError('V forbidden work')
    def invalidate(paths):
        affected=set()
        for path in paths:
            path=Path(path);stem=path.stem
            if stem.startswith('V'):affected.add(stem.split('-')[0])
            elif path.parent.name=='inputs' and stem.startswith('C'):
                number=int(stem[1:3]);affected.update(f'V{i:03d}' for i in range(4*number-3,4*number+1))
        for key in ('groups','branches'):
            for row in receipt[key]:
                if row['group_id'] in affected:row.update(status='failed',diagnostic_only=True)
    for kind,cap in CAPS.items():
        c=receipt['counts'][kind]
        if any(c[k]!=cap for k in ('reserved','actual','completed')) or any(c[k]!=CALL_CAPS[kind] for k in ('reserved_calls','actual_calls','completed_calls')):
            invalidate([output/'groups'/(r['group_id']+'.json') for r in receipt['groups']])
            raise ValueError('V count mismatch')
    initial=ledger()
    for key,idkey in (('groups','group_id'),('branches','slot_id'),('checks','check_id')):
        if [v[idkey] for v in receipt[key]]!=[v[idkey] for v in initial[key]] or any(v['status']!='completed' for v in receipt[key]):
            raise ValueError('V ledger incomplete')
    produced={a['path'] for a in receipt['artifacts']}
    required=set()
    for row in initial['groups']:
        required.update((str(output/'states'/(row['group_id']+'.npz')),str(output/'groups'/(row['group_id']+'.json'))))
    for row in initial['branches']:
        required.update(str(output/'branches'/(row['slot_id']+ext)) for ext in ('.npz','.json'))
    required.update(str(output/'checks'/(row['check_id']+'.json')) for row in initial['checks'])
    for n in range(1,13):
        required.add(str(output/'inputs'/f'C{n:02d}.json'))
        required.update(str(output/'inputs'/(f'C{n:02d}-'+b+'.npz')) for b in ('T','P','S','R'))
    missing=required-produced
    if missing:
        receipt['missing_artifacts']=sorted(missing)
        invalidate(missing)
        for key,idkey,folder in (('groups','group_id','groups'),('branches','slot_id','branches'),('checks','check_id','checks')):
            for row in receipt[key]:
                if any(str(output/folder/(row[idkey]+ext)) in missing for ext in ('.npz','.json')):
                    row.update(status='failed',diagnostic_only=True)
        raise ValueError('V required artifacts missing')
    for entry in receipt['artifacts']:
        try:verify(entry)
        except (ValueError,OSError):
            invalidate([entry['path']]);raise
        check()


def execute(manifest_path,run,approval_path,gpu,*,clock=time.monotonic):
    start=clock();deadline=start+300
    manifest_path=Path(manifest_path).resolve();run=Path(run).resolve()
    approval=authorization(approval_path,manifest_path,run,gpu);manifest=read(manifest_path)
    if manifest['version']!='itm-state-implementation-v1' or manifest['package']['sha256']!=PACKAGE_SHA:raise ValueError('Wrong V implementation')
    run.mkdir(parents=True,exist_ok=False);runtime=run/'runtime';runtime.mkdir()
    status=dict(version='itm-state-queue-v1',status='initializing',authorization=approval,implementation=artifact(manifest_path),
                started_monotonic=start,deadline_monotonic=deadline,**ledger(),
                counts={k:dict(reserved=0,actual=0,completed=0,reserved_calls=0,actual_calls=0,completed_calls=0) for k in CAPS})
    status_path=runtime/'status.json';atomic_json(status_path,status)
    def check():
        if clock()>=deadline:raise TimeoutError('V 300-second absolute deadline')
    receipt=None;child=None
    try:
        expected=manifest['expected_artifacts']+[artifact(manifest_path)]
        for entry in expected:verify(entry);check()
        package=StatePackage(verify(manifest['package']));check()
        if manifest['checkpoint']!=artifact(package.checkpoint) or environment()!=manifest['environment'] or environment()!=package.common['environment']:
            raise ValueError('Prepared V checkpoint/environment changed')
        with (Path('/tmp')/('itm-b0-'+gpu+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            status['gpu_preflight']=preflight_gpu(gpu);check()
            receipt=runtime/'V.receipt.json'
            context=dict(version='itm-state-context-v1',phase='V',package_sha256=PACKAGE_SHA,checkpoint_sha256=CHECKPOINT_SHA,
                implementation_sha256=sha256(manifest_path),deadline_monotonic=deadline,receipt_path=str(receipt),gpu_uuid=gpu,
                memory_bytes=24*GIB,expected_artifacts=expected)
            context_path=runtime/'V.context.json';atomic_json(context_path,context)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                PYTORCH_CUDA_ALLOC_CONF='backend:native',PYTORCH_ALLOC_CONF='backend:native')
            for key in ('RANK','LOCAL_RANK','WORLD_SIZE'):env.pop(key,None)
            argv=[sys.executable,str(ROOT/'scripts/run_mdm_state_diagnostic_worker.py'),'--package',str(package.path),
                  '--run-context',str(context_path),'--output',str(run/'V')]
            check();status['status']='V';atomic_json(status_path,status)
            result=supervise(argv,deadline=deadline,log_path=runtime/'V.log',env=env,clock=clock)
            status['worker_result']=result
            if receipt.exists():
                child=read(receipt)
                for key in ('groups','branches','checks','counts'):status[key]=child[key]
                status['receipt']=artifact(receipt)
            if result['status']!='completed':raise RuntimeError('V child stopped: '+str(result))
            check()
            if child is None:raise ValueError('Missing V receipt')
            validate_receipt(child,context,run/'V',check);check()
            status.update(status='completed',finished_monotonic=clock());atomic_json(status_path,status)
    except BaseException as error:
        # Reconcile a killed worker's last durable ledger. Never rerun it.
        if child is None and receipt is not None and receipt.exists():child=read(receipt)
        if child:
            for key in ('groups','branches','checks','counts'):status[key]=child[key]
            if child.get('missing_artifacts'):status['missing_artifacts']=child['missing_artifacts']
        stop_rows(status)
        status.update(status='failed',finished_monotonic=clock(),failure=dict(type=type(error).__name__,message=str(error)))
        atomic_json(status_path,status);raise
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-dir',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--implementation',type=Path)
    p.add_argument('--run-dir',type=Path);p.add_argument('--authorization-record',type=Path);p.add_argument('--gpu-uuid')
    args=p.parse_args()
    if args.prepare_dir:
        if args.execute:raise ValueError('V prepare cannot execute')
        prepare(args.prepare_dir);return
    if not args.execute or not all((args.implementation,args.run_dir,args.authorization_record,args.gpu_uuid)):raise ValueError('New V authorization required')
    execute(args.implementation,args.run_dir,args.authorization_record,args.gpu_uuid)
if __name__=='__main__':main()
