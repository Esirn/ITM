#!/usr/bin/env python
"""Prepare immutable D/L/R implementation inputs or run one newly authorized queue."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from itm.models.mdm_correction import (ROOT, ARMS, PACKAGE_SHA, CorrectionPackage, read, verify,
    artifact, create_initial_state, environment)
from itm.models.mdm_run_limits import atomic_json, sha256, GIB
from run_mdm_b0_bounded import preflight_gpu, supervise

PACKAGE = ROOT/'docs/itm_rework/protocol/b0_correction_20260926_v1/package.json'


def prepare(directory):
    """Parameter initialization/data/hash validation only; never load a backbone."""
    package = CorrectionPackage(PACKAGE)
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=False)
    initialization = create_initial_state(package, directory/'initial.pt')
    sources = read(package.files['sources'])
    # Owner02 sources are historical immutable evidence, and unchanged by this implementation.
    source_entries = sources if isinstance(sources, list) else sources['artifacts']
    paths = {PACKAGE, *package.files.values()}
    for entry in source_entries: paths.add(verify(entry))
    for split in ('train','val'):
        for mid in package.data.plans[split]:
            for entry in package.data.admissions[(split,mid)]['artifacts']: paths.add(verify(entry))
    paths.update(verify(a) for a in package.common['training_input_provenance']['payload']['artifacts'].values())
    paths.update((ROOT/'src/itm/models').glob('*.py'))
    paths.update((ROOT/'src/itm/backbones').glob('*.py'))
    paths.update((ROOT/'src/itm/data').glob('*.py'))
    paths.update((ROOT/'outputs/mdm/runtime_root').rglob('*.py'))
    paths.update(ROOT/'scripts'/name for name in ('run_mdm_correction_bounded.py','run_mdm_correction_worker.py',
                                                'run_mdm_b0_bounded.py','train_mdm_imu_control.py'))
    # Fully prepare all inputs on CPU to expose contract/shape errors before authorization.
    inputs=[]
    for row in package.samples[:60]:
        item=package.inputs(row)
        inputs.append(dict(case_id=row['case_id'], frames=len(item['motion']), input_metadata=item['metadata']))
    manifest = dict(version='itm-correction-implementation-v1', package=artifact(PACKAGE),
                    initialization=initialization, environment=environment(),
                    expected_artifacts=[artifact(p) for p in sorted(paths)],
                    inputs=inputs, real_model_forwards=0, real_updates=0, samples=0, gpu_queries=0)
    atomic_json(directory/'implementation.json', manifest)
    print(directory/'implementation.json')
    return manifest


def authorization(path, manifest_path, run, gpu):
    value = read(path)
    expected = dict(authorized=True, approved_by='user', package_sha256=PACKAGE_SHA,
                    implementation_sha256=sha256(manifest_path), run_directory=str(Path(run).resolve()), gpu_uuid=gpu)
    if any(value.get(k)!=v for k,v in expected.items()) or not value.get('user_instruction_reference'):
        raise ValueError('New user authorization bound to correction package/implementation/directory/GPU required')
    return value


def mark_stop(status):
    for rows in (status['stages'],status['probes'],status['samples']):
        for row in rows:
            if row['status']=='planned': row.update(status='not_run_after_stop',launched=False)
            elif row['status']=='running': row.update(status='failed')


def validate_receipt(path, phase, arm, initialization_sha):
    child = read(path)
    if child['status'] != 'completed' or child['phase'] != phase or child['context'].get('arm') != arm:
        raise ValueError('Correction child receipt incomplete or wrong identity')
    if child['initialization_sha256'] != initialization_sha:
        raise ValueError('Child initialization identity mismatch')
    expected = 480 if phase=='probe' else 600 if phase=='train' else 4000
    if child['mdm_examples'] != expected or child['encoder_examples'] > child['context']['counters']['encoder_examples']:
        raise ValueError('Correction child forward count mismatch')
    if phase=='probe' and len(set(child['completed_slots'])) != 480:
        raise ValueError('Incomplete probe slots')
    if phase=='train' and (child['updates_completed']!=200 or child['updates_attempted']!=200 or child['exposures']!=600):
        raise ValueError('Partial correction training cannot sample')
    if phase=='sample' and (child['denoising_calls']!=1000 or child['branch_examples']!=4000 or len(child['completed_slots'])!=1):
        raise ValueError('Incomplete correction sample')
    for entry in child['artifacts']: verify(entry)
    return child


def final_checkpoint(path, arm, init_sha):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state['correction_arm'] != arm or state['package_sha256'] != PACKAGE_SHA or
        state['initialization_sha256'] != init_sha or state['execution']['updates_completed'] != 200 or
        state['execution']['updates_attempted'] != 200 or state['execution']['final_update_200'] is not True):
        raise ValueError('Only this arm final200 qualifies for sampling')
    return artifact(path)


def sample_consistency(status, slot, run):
    """No extra forwards. Frozen null/bypass/R-static mismatch stops subsequent queue."""
    import numpy as np
    row=slot['spec']
    if row['condition'] not in ('backbone_bypass','physical_static'): return None
    if row['condition']=='physical_static' and row['arm']!=ARMS[1]: return None
    candidates=[r for r in status['samples'] if r['spec']['arm']==row['arm'] and
                r['spec']['motion_id']==row['motion_id'] and r['spec']['sensor_config']==row['sensor_config'] and
                r['spec']['split']==row['split'] and r['spec']['condition']=='text_only']
    reference=candidates[0]
    if reference['status']!='completed': raise ValueError('Missing earlier frozen text-only comparison')
    with np.load(run/slot['output'],allow_pickle=False) as a, np.load(run/reference['output'],allow_pickle=False) as b:
        x,y=a['motion'],b['motion']; difference=float(np.max(np.abs(x-y)))
        result=dict(reference=reference['case_id'],max_abs_m=difference,bitwise=bool(np.array_equal(x,y)))
    if difference>1e-5: raise ValueError('Frozen null/R-static coordinate tolerance exceeded: '+str(result))
    return result


def execute(manifest_path, run, approval_path, gpu, *, clock=time.monotonic):
    start=clock() # includes package/implementation verification and GPU preflight
    total_deadline=start+7200
    manifest_path=Path(manifest_path).resolve(); run=Path(run).resolve()
    approval=authorization(approval_path,manifest_path,run,gpu)
    manifest=read(manifest_path)
    if manifest['version']!='itm-correction-implementation-v1' or manifest['package']['sha256']!=PACKAGE_SHA:
        raise ValueError('Wrong correction implementation manifest')
    run.mkdir(parents=True,exist_ok=False); runtime=run/'runtime'; runtime.mkdir()
    # Read the fixed, bound slot identities to preserve full denominators even on preflight failure.
    frozen_probes=read(PACKAGE.parent/'probe_slots.json')
    frozen_samples=read(PACKAGE.parent/'sampling_slots.json')
    status=dict(version='itm-correction-queue-v1',status='initializing',authorization=approval,
        implementation=artifact(manifest_path),total_started_monotonic=start,total_deadline_monotonic=total_deadline,
        stages=[dict(stage=k,status='planned') for k in ('D_probe','L_train','R_train','L_sample_60','R_sample_60')],
        probes=[dict(probe_id=r['probe_id'],status='planned') for r in frozen_probes],
        samples=[dict(case_id=r['case_id'],spec=r,status='planned',output='samples/'+r['case_id']+'.npz') for r in frozen_samples])
    path=runtime/'status.json';atomic_json(path,status)
    def check(deadline=total_deadline):
        if clock()>=min(deadline,total_deadline): raise TimeoutError('Correction shared deadline exhausted')
    initial=Path(manifest['initialization']['artifact']['path']).resolve()
    init_sha=manifest['initialization']['artifact']['sha256']
    expected=manifest['expected_artifacts']+[artifact(manifest_path),manifest['initialization']['artifact']]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        PYTORCH_CUDA_ALLOC_CONF='backend:native',PYTORCH_ALLOC_CONF='backend:native')
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE'): env.pop(key,None)
    def job(name,phase,arm,deadline,output,extra=(),additional=()):
        status['active_result']=dict(name=name,status='not_launched',launched=False,reason='before_launch')
        status.pop('active_receipt',None)
        check(deadline)
        receipt=runtime/(name+'.receipt.json');context=runtime/(name+'.context.json')
        caps={'mdm_examples':480 if phase=='probe' else 600 if phase=='train' else 4000,
              'encoder_examples':480 if phase=='probe' else (1200 if arm==ARMS[1] else 600) if phase=='train' else (2 if arm==ARMS[1] else 1),
              'exposures':600 if phase=='train' else 0}
        atomic_json(context,dict(version='itm-correction-context-v1',phase=phase,arm=arm,deadline_monotonic=deadline,
            receipt_path=str(receipt),gpu_uuid=gpu,memory_bytes=24*GIB,counters=caps,
            initialization_sha256=init_sha,expected_artifacts=expected+list(additional),
            max_updates=200,max_denoising_calls=1000,max_branch_examples=4000))
        argv=[sys.executable,str(ROOT/'scripts/run_mdm_correction_worker.py'),'--package',str(package.path),
              '--run-context',str(context),'--phase',phase,'--initial',str(initial),'--output',str(output)]
        if arm: argv+=['--arm',arm]
        argv+=list(extra)
        result=supervise(argv,deadline=deadline,log_path=runtime/(name+'.log'),env=env,clock=clock)
        status['active_result']=dict(name=name,**result)
        if receipt.exists():
            child=read(receipt)
            status['active_receipt']=artifact(receipt)
            if phase=='probe':
                completed=set(child['completed_slots'])
                for r in status['probes']:
                    if r['probe_id'] in completed:r['status']='completed'
                    elif r['probe_id'] in child.get('active_probe_ids',[]):
                        r.update(status='failed',launched=child.get('active_probe_forward_started',False),failure=child.get('failure'))
        atomic_json(path,status)
        if result['status']!='completed': raise RuntimeError('Correction child stopped: '+str(result))
        check(deadline)
        child=validate_receipt(receipt,phase,arm,init_sha)
        check(deadline)
        return child,artifact(receipt)
    try:
        package=CorrectionPackage(verify(manifest['package']))
        check()
        for entry in expected: verify(entry);check()
        if environment()!=manifest['environment']:raise ValueError('Prepared runtime environment mismatch')
        with (Path('/tmp')/('itm-b0-'+gpu+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            status['gpu_preflight']=preflight_gpu(gpu);check()
            stage=status['stages'][0];stage.update(status='running',deadline_monotonic=min(total_deadline,clock()+600))
            status['status']='D_probe';atomic_json(path,status)
            child,rec=job('D_probe','probe',None,stage['deadline_monotonic'],run/'D')
            if set(child['completed_slots'])!={r['probe_id'] for r in package.probes}:raise ValueError('Wrong probe identity set')
            produced={a['path'] for a in child['artifacts']}
            required={str(run/'D'/(r['probe_id']+ext)) for r in package.probes for ext in ('.npz','.json')}
            if not required.issubset(produced):raise ValueError('Missing probe artifacts')
            stage.update(status='completed',receipt=rec)
            train_deadline=min(total_deadline,clock()+1200);status['training_deadline_monotonic']=train_deadline
            finals={}
            for index,arm in enumerate(ARMS,1):
                stage=status['stages'][index];deadline=min(train_deadline,clock()+600)
                stage.update(status='running',deadline_monotonic=deadline);status['status']=stage['stage'];atomic_json(path,status)
                _,rec=job(stage['stage'],'train',arm,deadline,run/arm)
                finals[arm]=final_checkpoint(run/arm/'model.pt',arm,init_sha);check(deadline)
                stage.update(status='completed',receipt=rec,checkpoint=finals[arm]);atomic_json(path,status)
            sampling_deadline=min(total_deadline,clock()+5400);status['sampling_deadline_monotonic']=sampling_deadline
            for index,slot in enumerate(status['samples']):
                arm=slot['spec']['arm'];stage=status['stages'][3 if arm==ARMS[0] else 4]
                stage.update(status='running',deadline_monotonic=sampling_deadline)
                status['status']=stage['stage'];slot.update(status='running',launched=False);atomic_json(path,status)
                child,rec=job(f'sample_{index:03d}','sample',arm,sampling_deadline,run/slot['output'],
                    ['--slot',str(index),'--checkpoint',finals[arm]['path']],additional=[finals[arm]])
                if child['completed_slots']!=[slot['case_id']]:raise ValueError('Wrong completed sample identity')
                expected_paths={str(run/slot['output']),str((run/slot['output']).with_suffix('.json'))}
                if not expected_paths.issubset({a['path'] for a in child['artifacts']}):raise ValueError('Sample artifacts missing')
                slot.update(status='completed',launched=True,receipt=rec,artifacts=child['artifacts'])
                try:
                    slot['consistency']=sample_consistency(status,slot,run);check(sampling_deadline)
                except BaseException as error:
                    slot.update(status='failed',failure=dict(type=type(error).__name__,message=str(error)))
                    raise
                if index in (59,119):stage['status']='completed'
                atomic_json(path,status)
            status.update(status='completed',finished_monotonic=clock());check();atomic_json(path,status)
    except BaseException as error:
        status.update(status='failed',failure=dict(type=type(error).__name__,message=str(error)),finished_monotonic=clock())
        # A launched but failed job remains distinguishable from unlaunched slots.
        if status.get('active_result'):
            for slot in status['samples']:
                if slot['status']=='running':slot.update(launched=status['active_result']['launched'],result=status['active_result'],failure=status['failure'])
        mark_stop(status);atomic_json(path,status);raise
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-dir',type=Path)
    parser.add_argument('--implementation',type=Path)
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--authorization-record',type=Path)
    parser.add_argument('--gpu-uuid')
    args=parser.parse_args()
    if args.prepare_dir:
        if args.execute:raise ValueError('Preparation cannot execute')
        prepare(args.prepare_dir);return 0
    if not args.execute or not all((args.implementation,args.run_dir,args.authorization_record,args.gpu_uuid)):
        raise ValueError('Execution requires implementation, new authorization, unique run directory and GPU')
    execute(args.implementation,args.run_dir,args.authorization_record,args.gpu_uuid)
    return 0


if __name__=='__main__':raise SystemExit(main())
