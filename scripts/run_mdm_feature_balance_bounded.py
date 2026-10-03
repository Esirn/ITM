#!/usr/bin/env python
"""CPU F preparation or a newly authorized single train/60-sample queue."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
from itm.models.mdm_correction import ROOT, read, verify, artifact, environment
from itm.models.mdm_sampling_alignment import input_arrays, describe, difference
from itm.models.mdm_feature_balance import FeaturePackage, PACKAGE, PACKAGE_SHA, ledger, stop_rows, parameter_contract
from itm.models.mdm_run_limits import atomic_json, sha256, GIB
from run_mdm_b0_bounded import preflight_gpu, supervise


def prepare(directory):
    package=FeaturePackage();initial=package.initial_state()
    if environment()!=package.common['environment']:raise ValueError('F preparation environment mismatch')
    paths={package.path,*package.files.values(),*package.source_paths,package.initial}
    for folder in ('src/itm/models','src/itm/data','src/itm/backbones','outputs/mdm/runtime_root'):
        paths.update(p.resolve() for p in (ROOT/folder).rglob('*.py'))
    paths.update(ROOT/'scripts'/n for n in ('run_mdm_feature_balance_bounded.py','run_mdm_feature_balance_worker.py',
        'run_mdm_correction_worker.py','run_mdm_b0_bounded.py','train_mdm_imu_control.py'))
    def inputs(row):return {k:describe(v) for k,v in input_arrays(package.inputs(row),dict({'condition':'paired'},**row)).items()}
    samples=[dict(case_id=r['case_id'],arrays=inputs(r)) for r in package.samples]
    train=[dict(exposure_id=r['exposure_id'],arrays=inputs(r)) for r in package.schedule[:6]]
    manifest=dict(version='itm-feature-implementation-v1',package=artifact(package.path),initial=artifact(package.initial),
        environment=environment(),loss=package.loss,parameter_contract=parameter_contract(initial),
        initial_tensor_hashes=initial['tensor_hashes'],training_schedule=artifact(package.files['training_schedule']),
        inputs=samples,training_inputs=train,expected_artifacts=[artifact(p) for p in sorted(paths)],
        real_model_forwards=0,training_updates=0,samples=0,GPU_queries=0,new_parameter_initializations=0,new_checkpoints=0)
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    atomic_json(directory/'implementation.json',manifest);print(directory/'implementation.json');return manifest


def authorization(path,manifest,run,gpu):
    value=read(path)
    expected=dict(authorized=True,approved_by='user',package_sha256=PACKAGE_SHA,
        implementation_sha256=sha256(manifest),run_directory=str(Path(run).resolve()),gpu_uuid=gpu)
    if any(value.get(k)!=v for k,v in expected.items()) or not value.get('user_instruction_reference'):
        raise ValueError('New explicit F authorization required')
    return value


def required_artifacts(context,output,package):
    output=Path(output)
    if context['phase']=='train':
        return {str(output/f'update_{i:03d}.json') for i in range(1,201)} | {
            str(output/'exposures'/(r['exposure_id']+'.json')) for r in package.schedule} | {
            str(output/name) for name in ('update_050.pt','update_100.pt','update_150.pt','model.pt')}
    case=context['expected_slots'][0]
    return {str(output),str(output.with_suffix('.json')),str(output.parent.parent/'raw_samples'/(case+'.npz'))} | {
        str(output.parent.parent/'traces'/(r['trace_id']+ext)) for r in package.traces if r['case_id']==case for ext in ('.json','.npz')}


def validate_receipt(child,context,output,package,check=lambda:None):
    if child['status']!='completed' or child['context']!=context:raise ValueError('F receipt/context incomplete')
    train=context['phase']=='train'
    for kind,cap in dict(mdm=600 if train else 4000,encoder=1200 if train else 2,caption=600 if train else 1).items():
        c=child['counts'][kind];calls=1000 if kind=='mdm' and not train else cap
        if any(c[k]!=cap for k in ('reserved','actual','completed')) or c['calls']!=calls or c['completed_calls']!=calls:
            raise ValueError('F receipt counter mismatch')
    if (child['updates_attempted'],child['updates_completed'],child['exposures'],child['denoising_calls'],child['branch_examples']) != (
        (200,200,600,0,0) if train else (0,0,0,1000,4000)):
        raise ValueError('F receipt work budget mismatch')
    if child['completed_slots']!=context['expected_slots'] or child['completed_traces']!=context['expected_traces']:
        raise ValueError('F receipt IDs incomplete')
    produced={r['path'] for r in child['artifacts']}
    if not required_artifacts(context,output,package).issubset(produced):raise ValueError('F required artifacts missing')
    for entry in child['artifacts']:verify(entry);check()


def run_checks(status,package,run):
    samples={r['case_id']:r for r in status['samples']}
    for row,spec in zip(status['checks'],package.checks):
        if row['status']!='planned' or samples[spec['new_case_id']]['status']!='completed':continue
        if spec['reference_arm']=='F':
            reference=samples[spec['reference_case_id']]
            if reference['status']!='completed':continue
            path=run/reference['output']
        else:path=verify(spec['reference_output'])
        row['status']='running'
        with np.load(run/samples[spec['new_case_id']]['output'],allow_pickle=False) as a,np.load(path,allow_pickle=False) as b:
            result=difference(a['motion'],b['motion'],atol=spec['max_abs_m'],rtol=0,enforce=False)
        row.update(result,finite_count=result['count'],status='completed' if result['passed'] else 'failed')
        atomic_json(run/'checks'/(row['check_id']+'.json'),dict(spec,result=row))
        if not result['passed']:
            samples[spec['new_case_id']].update(status='failed',diagnostic_only=True)
            raise ValueError('Frozen F complete-motion software equality failed')


def reconcile(status,child,context):
    if context['phase']=='train':
        completed=set(child.get('completed_slots',[]))
        for row in status['updates']:
            if f"U{row['update']:03d}" in completed:row['status']='completed'
            elif row['update']==child.get('active_update'):row['status']='failed'
        n=child.get('exposures',0)
        for i,row in enumerate(status['exposures']):
            if row['update']<=child.get('updates_completed',0):row['status']='completed'
            elif i<n:row['status']='failed';row['executed']=True
    else:
        for row in status['traces']:
            if row['trace_id'] in child.get('completed_traces',[]):row['status']='completed'
            elif row['trace_id'] in child.get('active_trace_ids',[]):row['status']='failed'


def execute(manifest_path,run,approval_path,gpu,*,clock=time.monotonic):
    start=clock();total_deadline=start+3600;train_deadline=start+600
    manifest_path=Path(manifest_path).resolve();run=Path(run).resolve()
    approval=authorization(approval_path,manifest_path,run,gpu);manifest=read(manifest_path)
    if manifest['version']!='itm-feature-implementation-v1' or manifest['package']['sha256']!=PACKAGE_SHA:
        raise ValueError('Wrong F implementation')
    run.mkdir(parents=True,exist_ok=False);runtime=run/'runtime';runtime.mkdir()
    raw=SimpleNamespace(schedule=read(PACKAGE.parent/'training_schedule.json')['rows'],samples=read(PACKAGE.parent/'sampling_slots.json'),
        traces=read(PACKAGE.parent/'trace_slots.json'),checks=read(PACKAGE.parent/'software_checks.json'))
    status=dict(version='itm-feature-queue-v1',status='initializing',authorization=approval,implementation=artifact(manifest_path),
        started_monotonic=start,total_deadline_monotonic=total_deadline,training_deadline_monotonic=train_deadline,
        completed_receipts=[],job_counts={},**ledger(raw))
    status_path=runtime/'status.json';atomic_json(status_path,status)
    def check(deadline):
        if clock()>=min(deadline,total_deadline):raise TimeoutError('F absolute/shared deadline exhausted')
    expected=manifest['expected_artifacts']+[artifact(manifest_path)]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        PYTORCH_CUDA_ALLOC_CONF='backend:native',PYTORCH_ALLOC_CONF='backend:native')
    for k in ('RANK','LOCAL_RANK','WORLD_SIZE'):env.pop(k,None)
    checkpoint=None
    def job(phase,name,deadline,output,slot=None):
        status['active_job']=name;status['active_result']=dict(status='not_launched',launched=False)
        status.pop('active_receipt',None)
        check(deadline);receipt=runtime/(name+'.receipt.json')
        context=dict(version='itm-feature-context-v1',phase=phase,package_sha256=PACKAGE_SHA,
            implementation_sha256=sha256(manifest_path),run_directory=str(run),deadline_monotonic=deadline,
            receipt_path=str(receipt),gpu_uuid=gpu,memory_bytes=24*GIB,expected_artifacts=expected,
            expected_slots=[f'U{i:03d}' for i in range(1,201)] if phase=='train' else [package.samples[slot]['case_id']],
            expected_traces=[] if phase=='train' else [r['trace_id'] for r in package.traces if r['case_id']==package.samples[slot]['case_id']])
        if checkpoint is not None:context['checkpoint']=checkpoint
        context_path=runtime/(name+'.context.json');atomic_json(context_path,context)
        argv=[sys.executable,str(ROOT/'scripts/run_mdm_feature_balance_worker.py'),'--package',str(package.path),
            '--run-context',str(context_path),'--phase',phase,'--output',str(output)]
        if slot is not None:argv+=['--slot',str(slot)]
        status['active_job']=name;atomic_json(status_path,status)
        result=supervise(argv,deadline=deadline,log_path=runtime/(name+'.log'),env=env,clock=clock)
        status['active_result']=result
        child=read(receipt) if receipt.exists() else None
        if child:
            reconcile(status,child,context);status['active_receipt']=artifact(receipt)
            status['job_counts'][name]=child['counts']
            status['counts']={k:{field:sum(v[k][field] for v in status['job_counts'].values())
                for field in ('reserved','actual','completed','calls','completed_calls')} for k in ('mdm','encoder','caption')}
        atomic_json(status_path,status)
        if result['status']!='completed':raise RuntimeError('F worker stopped: '+str(result))
        if child is None:raise ValueError('F worker receipt missing')
        check(deadline)
        try:validate_receipt(child,context,output,package,lambda:check(deadline))
        except BaseException:
            # Completed-but-unverifiable worker outputs never masquerade as accepted.
            for key in ('updates','exposures') if phase=='train' else ('traces',):
                for row in status[key]:
                    if phase=='train' or row['case_id']==package.samples[slot]['case_id']:
                        if row['status']=='completed':row.update(status='failed',diagnostic_only=True)
            raise
        status['completed_receipts'].append(artifact(receipt));return child
    try:
        for entry in expected:verify(entry);check(train_deadline)
        package=FeaturePackage(verify(manifest['package']))
        if environment()!=manifest['environment'] or manifest['environment']!=package.common['environment']:
            raise ValueError('F bound environment mismatch')
        if manifest['initial']!=artifact(package.initial):raise ValueError('F bound initialization mismatch')
        with (Path('/tmp')/('itm-b0-'+gpu+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            check(train_deadline);status['gpu_preflight']=preflight_gpu(gpu);check(train_deadline)
            status['status']='train';atomic_json(status_path,status)
            job('train','train',train_deadline,run/'F')
            checkpoint=artifact(run/'F/model.pt');state=package.state(checkpoint,sha256(manifest_path))
            if state['training_run_directory']!=str(run):raise ValueError('F checkpoint from another run')
            status['checkpoint']=checkpoint;check(train_deadline)
            deadline=min(clock()+3000,total_deadline);status['sampling_deadline_monotonic']=deadline
            status['status']='sample'
            for i,slot in enumerate(status['samples']):
                slot['status']='running';atomic_json(status_path,status)
                child=job('sample',f'sample_{i:03d}',deadline,run/slot['output'],i)
                slot.update(status='completed',artifacts=child['artifacts']);run_checks(status,package,run)
                check(deadline);atomic_json(status_path,status)
            if any(row['status']!='completed' for key in ('updates','exposures','samples','traces','checks') for row in status[key]):
                raise ValueError('F final ledger incomplete')
            check(deadline);status.update(status='completed',finished_monotonic=clock());atomic_json(status_path,status)
    except BaseException as error:
        stop_rows(status)
        status.update(status='failed',finished_monotonic=clock(),failure=dict(type=type(error).__name__,message=str(error)))
        atomic_json(status_path,status);raise
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-dir',type=Path);p.add_argument('--execute',action='store_true')
    p.add_argument('--implementation',type=Path);p.add_argument('--run-dir',type=Path)
    p.add_argument('--authorization-record',type=Path);p.add_argument('--gpu-uuid')
    args=p.parse_args()
    if args.prepare_dir:
        if args.execute:raise ValueError('F preparation cannot execute')
        prepare(args.prepare_dir);return
    if not args.execute or not all((args.implementation,args.run_dir,args.authorization_record,args.gpu_uuid)):
        raise ValueError('F requires new bound authorization')
    execute(args.implementation,args.run_dir,args.authorization_record,args.gpu_uuid)

if __name__=='__main__':main()
