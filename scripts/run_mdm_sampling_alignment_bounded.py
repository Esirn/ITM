#!/usr/bin/env python
"""CPU-only immutable preparation, or one newly authorized Q/S1 queue."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

import numpy as np

from itm.models.mdm_correction import ROOT, artifact, environment, read, verify
from itm.models.mdm_run_limits import atomic_json, sha256, GIB
from itm.models.mdm_sampling_alignment import (AlignmentPackage, PACKAGE, PACKAGE_SHA, CHECKPOINT_SHA,
                                              input_arrays, describe, difference)
from run_mdm_b0_bounded import preflight_gpu, supervise


def prepare(directory):
    package=AlignmentPackage();package.state() # Read existing tensors on CPU; no model construction.
    if environment()!=package.common['environment']:raise ValueError('Frozen preparation environment mismatch')
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    paths={package.path,*package.files.values(),*package.source_paths,package.checkpoint}
    for folder in ('src/itm/models','src/itm/backbones','src/itm/data','outputs/mdm/runtime_root'):
        paths.update((ROOT/folder).rglob('*.py'))
    paths.update(ROOT/'scripts'/name for name in ('run_mdm_sampling_alignment_bounded.py',
        'run_mdm_sampling_alignment_worker.py','run_mdm_correction_worker.py','train_mdm_imu_control.py','run_mdm_b0_bounded.py'))
    inputs=[]
    for row in package.samples:
        item=package.inputs(row)
        inputs.append(dict(case_id=row['case_id'],arrays={k:describe(v) for k,v in input_arrays(item,row).items()}))
    # Q includes train reversal which is not a historical full-sample condition.
    q_inputs=[]
    for row in package.probes:
        item=package.inputs(row)
        q_inputs.append(dict(probe_id=row['probe_id'],arrays={k:describe(v) for k,v in input_arrays(item,row).items()}))
    manifest=dict(version='itm-alignment-implementation-v1',package=artifact(package.path),
        checkpoint=artifact(package.checkpoint),environment=environment(),expected_artifacts=[artifact(p) for p in sorted(paths)],
        inputs=inputs,Q_inputs=q_inputs,real_model_forwards=0,real_updates=0,samples=0,gpu_queries=0,
        new_parameter_initializations=0,new_checkpoints=0)
    atomic_json(directory/'implementation.json',manifest);print(directory/'implementation.json')
    return manifest


def authorization(path,manifest,run,gpu):
    value=read(path)
    expected=dict(authorized=True,approved_by='user',package_sha256=PACKAGE_SHA,
        implementation_sha256=sha256(manifest),run_directory=str(Path(run).resolve()),gpu_uuid=gpu)
    if any(value.get(k)!=v for k,v in expected.items()) or not value.get('user_instruction_reference'):
        raise ValueError('New explicit alignment authorization required; old approval cannot be reused')
    return value


def mark_stop(status):
    for key in ('stages','probes','samples','traces'):
        for row in status[key]:
            if row['status']=='planned':row.update(status='not_run_after_stop',launched=False)
            elif row['status']=='running':row.update(status='failed')


def validate_receipt(path,context,check=lambda:None):
    receipt=read(path);phase=context['phase']
    if receipt['status']!='completed' or receipt['phase']!=phase or receipt['context']!=context:
        raise ValueError('Incomplete/wrong receipt context')
    if receipt['checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('Wrong receipt checkpoint')
    if receipt['mdm_examples']!=(480 if phase=='Q' else 4000) or receipt['mdm_calls']!=(96 if phase=='Q' else 1000):
        raise ValueError('Wrong actual forward counts')
    if receipt['encoder_examples']>context['counters']['encoder_examples'] or receipt['updates_attempted'] or receipt['updates_completed']:
        raise ValueError('Encoder/update budget mismatch')
    if receipt['completed_slots']!=context['expected_slots'] or receipt['completed_traces']!=context['expected_traces']:
        raise ValueError('Incomplete slot/trace identities')
    if phase=='sample' and (receipt['denoising_calls']!=1000 or receipt['branch_examples']!=4000):
        raise ValueError('Incomplete full sampling')
    for entry in receipt['artifacts']:verify(entry);check()
    return receipt


def full_check(status,slot,run):
    row=slot['spec'];kind=row['condition']
    if kind not in ('backbone_bypass','physical_static','imu_only'):return None
    if kind=='imu_only':
        reference=verify(row['historical_output']);reference_id=row['comparison_case_id']
    else:
        matches=[r for r in status['samples'] if r['spec']['data_split']==row['data_split'] and
            r['spec']['motion_id']==row['motion_id'] and r['spec']['sensor_config']==row['sensor_config'] and
            r['spec']['condition']=='text_only']
        if len(matches)!=1 or matches[0]['status']!='completed':
            return dict(status='not_comparable',reason='counterpart_not_completed')
        reference=run/matches[0]['output'];reference_id=matches[0]['case_id']
    with np.load(run/slot['output'],allow_pickle=False) as a,np.load(reference,allow_pickle=False) as b:
        result=difference(a['motion'],b['motion'],atol=1e-5,rtol=0,enforce=False)
    return dict(status='passed' if result['passed'] else 'failed',reference=reference_id,
                max_abs_m=result['max_abs'],rms_m=result['rms'],bitwise=result['bitwise'],count=result['count'])


def execute(manifest_path,run,approval_path,gpu,*,clock=time.monotonic):
    start=clock();total_deadline=start+3600
    manifest_path=Path(manifest_path).resolve();run=Path(run).resolve()
    approval=authorization(approval_path,manifest_path,run,gpu);manifest=read(manifest_path)
    if manifest['version']!='itm-alignment-implementation-v1' or manifest['package']['sha256']!=PACKAGE_SHA:
        raise ValueError('Exact alignment implementation required')
    run.mkdir(parents=True,exist_ok=False);runtime=run/'runtime';runtime.mkdir()
    # The pinned protocol ledger exists before preflight/child loading.
    if sha256(PACKAGE)!=PACKAGE_SHA:raise ValueError('Package changed')
    rows=read(PACKAGE.parent/'sampling_slots.json');qrows=read(PACKAGE.parent/'probe_slots.json')
    traces=read(PACKAGE.parent/'trace_slots.json')
    status=dict(version='itm-alignment-queue-v1',status='initializing',authorization=approval,
        implementation=artifact(manifest_path),total_started_monotonic=start,total_deadline_monotonic=total_deadline,
        stages=[dict(stage=k,status='planned') for k in ('Q','S1')],
        probes=[dict(probe_id=r['probe_id'],status='planned') for r in qrows],
        traces=[dict(trace_id=r['trace_id'],case_id=r['case_id'],status='planned') for r in traces],
        samples=[dict(case_id=r['case_id'],spec=r,status='planned',output='samples/'+r['case_id']+'.npz') for r in rows],
        consistency_checks=[dict(case_id=r['case_id'],status='not_comparable',reason='slot_not_completed') for r in rows
                            if r['condition'] in ('backbone_bypass','physical_static','imu_only')])
    status_path=runtime/'status.json';atomic_json(status_path,status)
    def check(deadline=total_deadline):
        if clock()>=min(deadline,total_deadline):raise TimeoutError('Alignment absolute/shared deadline exhausted')
    expected=manifest['expected_artifacts']+[artifact(manifest_path)]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        PYTORCH_CUDA_ALLOC_CONF='backend:native',PYTORCH_ALLOC_CONF='backend:native')
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE'):env.pop(key,None)
    def job(name,phase,deadline,output,slot=None):
        status['active_result']=dict(name=name,status='not_launched',launched=False,reason='before_launch')
        status.pop('active_receipt',None);check(deadline)
        receipt=runtime/(name+'.receipt.json');context_path=runtime/(name+'.context.json')
        slot_rows=qrows if phase=='Q' else [rows[slot]]
        trace_rows=[] if phase=='Q' else [r for r in traces if r['case_id']==rows[slot]['case_id']]
        context=dict(version='itm-alignment-context-v1',phase=phase,package_sha256=PACKAGE_SHA,
            checkpoint_sha256=CHECKPOINT_SHA,deadline_monotonic=deadline,receipt_path=str(receipt),gpu_uuid=gpu,
            memory_bytes=24*GIB,counters=dict(mdm_examples=480 if phase=='Q' else 4000,encoder_examples=576 if phase=='Q' else 2),
            expected_slots=[r['probe_id'] if phase=='Q' else r['case_id'] for r in slot_rows],
            expected_traces=[r['trace_id'] for r in trace_rows],expected_artifacts=expected,
            max_denoising_calls=1000,max_branch_examples=4000)
        atomic_json(context_path,context)
        argv=[sys.executable,str(ROOT/'scripts/run_mdm_sampling_alignment_worker.py'),'--package',str(package.path),
              '--run-context',str(context_path),'--phase',phase,'--output',str(output)]
        if slot is not None:argv+=['--slot',str(slot)]
        result=supervise(argv,deadline=deadline,log_path=runtime/(name+'.log'),env=env,clock=clock)
        status['active_result']=dict(name=name,**result)
        if receipt.exists():
            child=read(receipt);status['active_receipt']=artifact(receipt)
            for key,id_key,completed,active in (
                    ('probes','probe_id','completed_slots','active_probe_ids'),
                    ('traces','trace_id','completed_traces','active_trace_ids')):
                if key=='probes' and phase!='Q':continue
                for row in status[key]:
                    if row[id_key] in child.get(completed,[]):row.update(status='completed',launched=True)
                    elif row[id_key] in child.get(active,[]):
                        row.update(status='failed',launched=child.get('active_probe_forward_started',False) if key=='probes' else True,
                                   failure=child.get('failure'))
            if phase=='sample':
                status['samples'][slot]['output_present']=output.exists()
                status['samples'][slot]['diagnostic_complete']=child.get('completed_traces',[])==context['expected_traces']
                if result['status']=='completed' and child.get('status')=='completed':
                    for row in status['traces']:
                        if row['trace_id'] in context['expected_traces'] and row['trace_id'] not in child.get('completed_traces',[]):
                            row.update(status='failed',failure=dict(type='MissingTrace',message='Completed sampler omitted required trace'))
        atomic_json(status_path,status)
        if result['status']!='completed':raise RuntimeError('Alignment child stopped: '+str(result))
        check(deadline);child=validate_receipt(receipt,context,lambda:check(deadline))
        produced={a['path'] for a in child['artifacts']}
        if phase=='Q':
            required={str(output/(r['probe_id']+ext)) for r in qrows for ext in ('.npz','.json')}
            required.add(str(output/'derived.json'))
        else:
            required={str(output),str(output.with_suffix('.json'))}
            required.update(str(run/'traces'/(r['trace_id']+ext)) for r in trace_rows for ext in ('.npz','.json'))
        if not required.issubset(produced):
            missing=required-produced
            for row in status['traces'] if phase=='sample' else status['probes']:
                key='trace_id' if phase=='sample' else 'probe_id'
                folder=run/'traces' if phase=='sample' else output
                if any(str(folder/(row[key]+ext)) in missing for ext in ('.npz','.json')):
                    row.update(status='failed',failure=dict(type='MissingArtifact',message='Required output absent from receipt'))
                    if phase=='sample':status['samples'][slot]['diagnostic_complete']=False
            raise ValueError('Missing frozen artifacts, including traces')
        check(deadline)
        return child,artifact(receipt)
    try:
        for entry in expected:verify(entry);check()
        package=AlignmentPackage(verify(manifest['package']));check()
        if environment()!=manifest['environment'] or manifest['environment']!=package.common['environment']:
            raise ValueError('Prepared/frozen environment mismatch')
        if manifest['checkpoint']!=artifact(package.checkpoint):raise ValueError('Prepared fixed checkpoint mismatch')
        with (Path('/tmp')/('itm-b0-'+gpu+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            status['gpu_preflight']=preflight_gpu(gpu);check()
            deadline=min(total_deadline,clock()+600)
            status['stages'][0].update(status='running',deadline_monotonic=deadline);status['status']='Q';atomic_json(status_path,status)
            _,rec=job('Q','Q',deadline,run/'Q')
            status['stages'][0].update(status='completed',receipt=rec)
            deadline=min(total_deadline,clock()+2400);status['sampling_deadline_monotonic']=deadline
            status['stages'][1].update(status='running',deadline_monotonic=deadline);status['status']='S1'
            for index,slot in enumerate(status['samples']):
                slot.update(status='running',launched=False);atomic_json(status_path,status)
                child,rec=job(f'sample_{index:03d}','sample',deadline,run/slot['output'],index)
                slot.update(status='completed',launched=True,receipt=rec,artifacts=child['artifacts'],diagnostic_complete=True)
                try:
                    result=full_check(status,slot,run)
                except BaseException as error:
                    slot.update(status='failed',failure=dict(type=type(error).__name__,message=str(error)))
                    raise
                if result is not None:
                    slot['consistency']=result
                    next(r for r in status['consistency_checks'] if r['case_id']==slot['case_id']).update(result)
                    if result['status']!='passed':
                        slot['status']='failed';atomic_json(status_path,status)
                        raise ValueError('Frozen sample consistency failed: '+str(result))
                check(deadline);atomic_json(status_path,status)
            if any(r['status']!='completed' for r in status['traces']):raise ValueError('Incomplete trace ledger')
            status['stages'][1]['status']='completed';check()
            status.update(status='completed',finished_monotonic=clock());atomic_json(status_path,status)
    except BaseException as error:
        status.update(status='failed',failure=dict(type=type(error).__name__,message=str(error)),finished_monotonic=clock())
        for slot in status['samples']:
            if slot['status']=='running':
                slot.update(launched=status.get('active_result',{}).get('launched',False),failure=status['failure'],
                            output_present=(run/slot['output']).exists())
        mark_stop(status);atomic_json(status_path,status);raise
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-dir',type=Path);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--implementation',type=Path);parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--authorization-record',type=Path);parser.add_argument('--gpu-uuid')
    args=parser.parse_args()
    if args.prepare_dir:
        if args.execute:raise ValueError('Prepare cannot execute')
        prepare(args.prepare_dir);return
    if not args.execute or not all((args.implementation,args.run_dir,args.authorization_record,args.gpu_uuid)):
        raise ValueError('New explicit authorization and bound implementation/directory/GPU required')
    execute(args.implementation,args.run_dir,args.authorization_record,args.gpu_uuid)


if __name__=='__main__':main()
