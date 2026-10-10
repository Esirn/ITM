#!/usr/bin/env python
"""CPU preparation and delegated execution of one frozen 1200-second suffix queue."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from itm.models.mdm_correction import ROOT,read,verify,artifact,environment
from itm.models.mdm_run_limits import atomic_json,sha256,GIB
from itm.models.mdm_suffix_history_factorial import SuffixPackage,PACKAGE,PACKAGE_SHA,CAPS,CALL_CAPS,TABLES,ledger,stop_rows,read_receipt,required_artifacts
from itm.models.mdm_training_sufficiency import compact_json
from itm.models.mdm_sampling_alignment import input_arrays,describe
from run_mdm_b0_bounded import supervise
from run_mdm_native_training_sufficiency_bounded import POLICY,POLICY_SHA
from itm_contended_gpu_preflight import preflight_gpu,LIMITS

SHARED_POLICY=ROOT/'docs/itm_rework/GPU_CONTENDED_EXECUTION_POLICY_20261008.json'
RESOURCE_POLICY_SHA='eb89f0f66c86a15368a3d0a4bf5db35027b380a642f2c43c04c59624cbd30d72'
ABSOLUTE_SECONDS=1200
DEVICE_PREFERENCE=ROOT/'docs/itm_rework/GPU_DEVICE_PREFERENCE_20261008.json'
DEVICE_PREFERENCE_SHA='bd156929862abef44824948c47ec0cc051cb0043c1afbd6c1db826ebeb9871a0'


def device_preference(gpu=None):
    value=read(DEVICE_PREFERENCE)
    if (sha256(DEVICE_PREFERENCE)!=DEVICE_PREFERENCE_SHA or value.get('authorized') is not True
        or value.get('preferred_physical_index')!=1 or value.get('preferred_gpu_uuid')!='GPU-43c2f925-db05-c655-bb46-25196d0627d6'):
        raise ValueError('User preferred device identity changed')
    if gpu is not None and gpu!=value['preferred_gpu_uuid']:raise ValueError('This queue binds preferred physical cuda:1')
    return artifact(DEVICE_PREFERENCE)

def shared_policy():
    value=read(SHARED_POLICY)
    if (sha256(SHARED_POLICY)!=RESOURCE_POLICY_SHA or value.get('authorized') is not True
            or value.get('resource_gates')!=LIMITS):
        raise ValueError('Explicit contended GPU policy changed')
    return artifact(SHARED_POLICY)


def prepare(directory):
    package=SuffixPackage()
    if environment()!=package.common['environment']:raise ValueError('Frozen environment mismatch')
    for sid in ('F200',):package.state(sid)
    inputs=[]
    for r in package.trajectories:
        item,row=package.inputs(r)
        inputs.append(dict(trajectory_id=r['trajectory_id'],input_arrays={k:describe(v) for k,v in input_arrays(item,row).items()}))
    paths={package.path,*package.files.values(),*package.source_paths,POLICY,SHARED_POLICY,DEVICE_PREFERENCE,Path(__file__).resolve()}
    for folder in ('src/itm/models','src/itm/backbones','src/itm/data','src/itm/metrics','outputs/mdm/runtime_root'):
        paths.update(p.resolve() for p in (ROOT/folder).rglob('*.py'))
    for name in ('run_mdm_suffix_history_factorial_worker.py','run_mdm_correction_worker.py','run_mdm_b0_bounded.py',
                 'run_mdm_native_training_sufficiency_bounded.py','evaluate_b0_suffix_history_factorial.py','verify_b0_suffix_history_factorial_results.py',
                 'render_b0_suffix_history_factorial.py','render_b0_feature_balance_review.py','itm_contended_gpu_preflight.py',
                 'train_mdm_imu_control.py','run_mdm_prefix_mechanism_worker.py'):
        paths.add(ROOT/'scripts'/name)
    paths.add(ROOT/'tests/test_mdm_suffix_history_factorial.py')
    paths.add(ROOT/'tests/test_contended_prefix_preflight.py')
    paths.add(ROOT/'scripts/run_mdm_suffix_history_factorial_bounded.py')
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    manifest=dict(version='itm-history-suffix-implementation-v1',package=artifact(package.path),environment=environment(),
        resource_policy=shared_policy(),device_preference=device_preference(),effective_absolute_seconds=ABSOLUTE_SECONDS,effective_resource_gates=LIMITS,resource_override='Latest user preference selects physical cuda:1; scientific package unchanged',snapshots={sid:package.snapshots[sid]['artifact'] for sid in ('F200',)},
        expected_artifacts=[artifact(p) for p in sorted(paths)],inputs=inputs,budgets=package.budgets,
        planned_worker_artifacts=len(required_artifacts(package,directory/'placeholder_output')),
        model_constructions=0,model_forwards=0,GPU_queries=0,new_training=0,new_motion_decodes=0)
    atomic_json(directory/'implementation.json',manifest);return directory/'implementation.json'


def record_execution(path,manifest,run,gpu,acceptance):
    preference=device_preference(gpu)
    if sha256(POLICY)!=POLICY_SHA or read(POLICY).get('authorized') is not True:raise ValueError('Standing delegation changed')
    manifest=Path(manifest).resolve();value=read(manifest);acceptance=Path(acceptance).resolve();test=read(acceptance)
    if value['package']['sha256']!=PACKAGE_SHA or value['version']!='itm-history-suffix-implementation-v1' or value.get('effective_absolute_seconds')!=ABSOLUTE_SECONDS:raise ValueError('Wrong suffix implementation')
    if test.get('status')!='passed' or test.get('package_sha256')!=PACKAGE_SHA or test.get('implementation_sha256')!=sha256(manifest):raise ValueError('Bound CPU acceptance required')
    for e in test['artifacts']:verify(e)
    path=Path(path).resolve()
    if path.exists() or Path(run).exists():raise FileExistsError('Fresh execution record/run required')
    if not gpu.startswith('GPU-'):raise ValueError('Physical GPU UUID required')
    value=dict(version='itm-history-suffix-long-execution-v1',authorized=True,approved_by='user_standing_delegation',
        policy=artifact(POLICY),resource_policy=shared_policy(),device_preference=preference,resource_gates=LIMITS,
        package_sha256=PACKAGE_SHA,implementation=artifact(manifest),implementation_sha256=sha256(manifest),
        CPU_acceptance=artifact(acceptance),run_directory=str(Path(run).resolve()),gpu_uuid=gpu,
        absolute_seconds=ABSOLUTE_SECONDS,memory_bytes=24*GIB,calls=CALL_CAPS,examples=CAPS,
        scope='One new bounded replay of text-prefix x899-started keep-text/switch-on four paths, original decoding; no training/new seed/retry',
        user_instruction=read(SHARED_POLICY)['user_instruction'])
    atomic_json(path,value);return value


def authorization(path,manifest,run,gpu):
    preference=device_preference(gpu)
    value=read(path)
    expected=dict(version='itm-history-suffix-long-execution-v1',authorized=True,approved_by='user_standing_delegation',
        package_sha256=PACKAGE_SHA,implementation_sha256=sha256(manifest),run_directory=str(Path(run).resolve()),
        gpu_uuid=gpu,absolute_seconds=ABSOLUTE_SECONDS,memory_bytes=24*GIB,calls=CALL_CAPS,examples=CAPS)
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Concrete suffix execution identity changed')
    if value.get('device_preference')!=preference:raise ValueError('Concrete preferred device policy changed')
    if sha256(verify(value['policy']))!=POLICY_SHA or value['resource_policy']!=shared_policy() or value['resource_gates']!=LIMITS:raise ValueError('Delegated policy changed')
    acceptance=read(verify(value['CPU_acceptance']))
    if acceptance.get('status')!='passed' or acceptance['implementation_sha256']!=sha256(manifest):raise ValueError('CPU acceptance identity changed')
    for e in acceptance['artifacts']:verify(e)
    verify(value['implementation']);return value


def validate_receipt(receipt,context,package,out,check=lambda:None):
    if receipt['status']!='completed' or receipt['context']!={k:v for k,v in context.items() if k!='expected_artifacts'}:raise ValueError('Incomplete receipt/context')
    for k,cap in CAPS.items():
        c=receipt['counts'][k]
        if any(c[f]!=cap for f in ('reserved','actual','completed')) or any(c[f]!=CALL_CAPS[k] for f in ('reserved_calls','actual_calls','completed_calls')):raise ValueError('Incomplete suffix count')
    if receipt['updates_attempted'] or receipt['updates_completed'] or receipt['denoising_calls']!=3600 or receipt['branch_examples']!=14400:raise ValueError('Forbidden/incomplete work')
    planned=ledger(package)
    for key in TABLES:
        if [r['id'] for r in receipt[key]]!=[r['id'] for r in planned[key]] or any(r['status']!='completed' for r in receipt[key]):raise ValueError('Incomplete suffix ledger')
    produced={a['path'] for a in receipt['artifacts']}
    missing=required_artifacts(package,out)-produced
    if missing:raise ValueError('Missing required suffix artifact: '+sorted(missing)[0])
    for e in receipt['artifacts']:verify(e);check()


def execute(manifest_path,run,record,gpu,*,clock=time.monotonic):
    started=clock();deadline=started+ABSOLUTE_SECONDS;run=Path(run).resolve();manifest_path=Path(manifest_path).resolve()
    approval=authorization(record,manifest_path,run,gpu);manifest=read(manifest_path)
    package_path=verify(manifest['package']);files={Path(a['path']).stem:verify(a) for a in read(package_path)['artifacts']}
    package=SimpleNamespace(trajectories=read(files['trajectories']),anchors=read(files['historical_anchors']),cases={s['panel_case_id']:s for s in read(files['starts'])})
    run.mkdir(parents=True,exist_ok=False);runtime=run/'runtime';runtime.mkdir()
    status=dict(version='itm-history-suffix-queue-v1',status='initializing',execution_record=approval,implementation=artifact(manifest_path),
        started_monotonic=started,deadline_monotonic=deadline,**ledger(package),
        counts={k:dict(reserved=0,actual=0,completed=0,reserved_calls=0,actual_calls=0,completed_calls=0) for k in CAPS})
    sp=runtime/'status.json';compact_json(sp,status);receipt=None;child=None
    def check():
        if clock()>=deadline:raise TimeoutError('Suffix 1200-second absolute deadline')
    try:
        expected=manifest['expected_artifacts']+[artifact(manifest_path)]
        for e in expected:verify(e);check()
        package=SuffixPackage(package_path);check()
        if environment()!=manifest['environment'] or environment()!=package.common['environment']:raise ValueError('Frozen environment changed')
        with (Path('/tmp')/('itm-b0-'+gpu+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check()
            status['gpu_preflight']=preflight_gpu(gpu,evidence_dir=runtime/'preflight',check=check);check()
            receipt=runtime/'suffix.receipt.json'
            context=dict(version='itm-history-suffix-context-v1',phase='history_suffix',package_sha256=PACKAGE_SHA,
                implementation_sha256=sha256(manifest_path),deadline_monotonic=deadline,receipt_path=str(receipt),
                gpu_uuid=gpu,memory_bytes=24*GIB,expected_artifacts=expected)
            cp=runtime/'suffix.context.json';atomic_json(cp,context)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                PYTORCH_CUDA_ALLOC_CONF='backend:native',PYTORCH_ALLOC_CONF='backend:native',OMP_NUM_THREADS='2')
            for k in ('RANK','LOCAL_RANK','WORLD_SIZE'):env.pop(k,None)
            argv=[sys.executable,str(ROOT/'scripts/run_mdm_suffix_history_factorial_worker.py'),'--package',str(package.path),'--run-context',str(cp),'--output',str(run/'suffix')]
            status['status']='suffix';compact_json(sp,status)
            result=supervise(argv,deadline=deadline,log_path=runtime/'suffix.log',env=env,clock=clock);status['worker_result']=result
            if receipt.exists():child=read_receipt(receipt)
            if result['status']!='completed':raise RuntimeError('Suffix child stopped: '+str(result))
            if child is None:raise ValueError('Suffix receipt missing')
            validate_receipt(child,context,package,run/'suffix',check);check()
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
