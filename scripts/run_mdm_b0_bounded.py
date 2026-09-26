#!/usr/bin/env python
"""Prepare or execute owner-02's exact B0 queue with shared, nonrenewable limits.

Preparation is CPU/file-only. Execution requires an external authorization record
bound to the immutable budget and execution plan; this tool never creates one.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from itm.models.mdm_run_limits import GIB, atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def verify_artifact(entry):
    path = Path(entry['path'])
    path = path if path.is_absolute() else ROOT/path
    if sha256(path) != entry['sha256']:
        raise ValueError(f'Identity mismatch: {path}')
    return path.resolve()


def flag(argv, name):
    if argv.count(name) != 1:
        raise ValueError(f'Expected exactly one {name}')
    return argv[argv.index(name)+1]


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan['version'] != 'canonical-B0-execution-plan-v1':
        raise ValueError('Unsupported B0 execution plan')
    for name in ('budget', 'sampling_plan', 'training_input_report'):
        verify_artifact(plan[name])
    cases = json.loads(verify_artifact(plan['sampling_plan']).read_text())
    queue = plan['queue']
    if len(queue) != 36 or [r['case_id'] for r in queue] != [c['case_id'] for c in cases]:
        raise ValueError('Exactly the fixed 36 ordered slots are required')
    for row, case in zip(queue, cases):
        if json.loads(verify_artifact(row['spec']).read_text()) != [case]:
            raise ValueError('Sampling spec differs from the fixed slot')
        if flag(row['argv_template'], '--spec') != str(verify_artifact(row['spec'])):
            raise ValueError('Sampling argv does not consume the bound spec')
        if row['steps'] != 1000 or row['maximum_branch_example_forwards'] != 4000:
            raise ValueError('B0 sampling limits differ')
    for argv, script in [(plan['train_argv_template'], 'train_mdm_imu_control.py')] + [
            (r['argv_template'], 'sample_mdm_imu_control.py') for r in queue]:
        if Path(argv[1]).resolve() != ROOT/'scripts'/script or flag(argv, '--device') != '<AUTHORIZED_GPU>':
            raise ValueError('Unexpected entrypoint/device template')
    expected = {'--epochs': '100', '--batch-size': '3', '--seed': '20260925', '--sensor-configs': 'head,wrists',
                '--encoder-variant': 'ordered_concat', '--adapter-variant': 'additive', '--null-mode': 'strict',
                '--backbone-mode': 'frozen_eval', '--time-encoding': 'physical_sinusoidal'}
    for key, value in expected.items():
        if flag(plan['train_argv_template'], key) != value:
            raise ValueError(f'Training template differs from B0: {key}')
    return plan


def authorization(path, plan_path, budget_sha, *, run_directory, gpu_uuid):
    record = json.loads(Path(path).read_text())
    if (record.get('authorized') is not True or record.get('approved_by') != 'user'
            or not record.get('user_instruction_reference') or record.get('budget_sha256') != budget_sha
            or record.get('execution_plan_sha256') != sha256(plan_path)
            or record.get('run_directory') != str(Path(run_directory).resolve())
            or record.get('gpu_uuid') != gpu_uuid):
        raise ValueError('Explicit user authorization bound to this budget and plan is required')
    return record


def preflight_gpu(uuid):
    if not uuid.startswith('GPU-') or any(c not in 'GPU-0123456789abcdefABCDEF' for c in uuid):
        raise ValueError('Name exactly one physical GPU UUID')
    result = subprocess.run(['nvidia-smi', '--id='+uuid, '--query-gpu=uuid,memory.total,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10)
    rows = [line.split(',') for line in result.stdout.strip().splitlines()]
    if len(rows) != 1 or rows[0][0].strip() != uuid or int(rows[0][2]) < 4096:
        raise RuntimeError('GPU identity/free-memory preflight failed')
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, check=True, timeout=10)
    if any(line.split(',')[0].strip() == uuid for line in apps.stdout.splitlines()):
        raise RuntimeError('Selected GPU already has a compute process')
    return {'gpu_uuid': uuid, 'total_MiB': int(rows[0][1]), 'free_MiB': int(rows[0][2]), 'compute_processes': 0}


def supervise(argv, *, deadline, log_path, env, clock=time.monotonic):
    """Hard wall limit includes process import, model load, compute and file writes."""
    remaining = deadline-clock()
    if remaining <= 0:
        return {'status': 'failed', 'reason': 'shared_deadline_exhausted_before_launch', 'launched': False}
    with Path(log_path).open('xb') as log:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=max(0., deadline-clock()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            return {'status': 'failed', 'reason': 'wall_timeout_process_group_killed', 'launched': True,
                    'pid': process.pid, 'returncode': process.returncode}
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
    if clock() >= deadline:
        return {'status': 'failed', 'reason': 'wall_deadline_reached', 'launched': True, 'returncode': code}
    return {'status': 'completed' if code == 0 else 'failed', 'reason': None if code == 0 else 'child_failed',
            'launched': True, 'returncode': code}


def mark_remaining(slots):
    for slot in slots:
        if slot['status'] == 'planned':
            slot['status'] = 'not_run_after_stop'
        elif slot['status'] == 'running':
            slot.update(status='failed', reason='supervisor_stopped_before_completion')


def run_jobs(jobs, *, root, deadline, gpu_uuid, env, status, status_path, clock=time.monotonic):
    """One deadline for the whole sampling queue, never one duration per job."""
    for index, job in enumerate(jobs):
        slot = status['slots'][index]
        slot['status'] = 'running'
        atomic_json(status_path, status)
        receipt = root/f'sample_{index:02d}.receipt.json'
        context_path = root/f'sample_{index:02d}.context.json'
        context = dict(phase='sample', deadline_monotonic=deadline, gpu_uuid=gpu_uuid,
                       receipt_path=str(receipt), memory_bytes=24*GIB,
                       expected_artifacts=status.get('required_artifacts', [])+job.get('input_artifacts', []),
                       max_denoising_calls=min(1000, 36000-status['denoising_calls']),
                       max_branch_examples=min(4000, 144000-status['branch_examples']))
        atomic_json(context_path, context)
        result = supervise(job['argv']+['--run-context', str(context_path)], deadline=deadline,
                           log_path=root/f'sample_{index:02d}.log', env=env, clock=clock)
        child = json.loads(receipt.read_text()) if receipt.exists() else None
        if child:
            status['denoising_calls'] += child['denoising_calls']
            status['branch_examples'] += child['branch_examples']
            slot['receipt'] = {'path': str(receipt), 'sha256': sha256(receipt)}
            slot['artifacts'] = child['artifacts']
        if result['status'] == 'completed':
            try:
                validate_sample_receipt(child, job['output'])
            except Exception as error:
                result = {'status': 'failed', 'reason': str(error), 'launched': True}
        slot.update(result)
        if result['status'] != 'completed':
            status['status'] = 'failed'
            mark_remaining(status['slots'])
            atomic_json(status_path, status)
            return False
        atomic_json(status_path, status)
    return True


def validate_sample_receipt(receipt, output):
    if (not receipt or receipt['status'] != 'completed' or receipt['denoising_calls'] != 1000
            or receipt['branch_examples'] != 4000):
        raise ValueError('Sample receipt incomplete or wrong DDPM count')
    artifacts = {entry['path']: entry for entry in receipt['artifacts']}
    for path in (Path(output), Path(output).with_suffix('.json')):
        verify_artifact(artifacts[str(path.resolve())])


def validate_final_checkpoint(path, receipt):
    import torch
    if receipt['status'] != 'completed' or receipt['updates_completed'] != 200 or receipt['updates_attempted'] != 200:
        raise ValueError('Training did not complete exactly 200 updates')
    artifacts = {entry['path']: entry for entry in receipt['artifacts']}
    verify_artifact(artifacts[str(Path(path).resolve())])
    state = torch.load(path, map_location='cpu', weights_only=True)
    execution = state.get('execution') or {}
    if execution.get('final_update_200') is not True or execution.get('updates_completed') != 200:
        raise ValueError('Partial checkpoint is not the final B0 model')


def resolve_argv(template, proposal, run):
    argv = [sys.executable] + [str(run)+part[len(proposal):] if part.startswith(proposal+'/') else part
                              for part in template[1:]]
    argv[argv.index('--device')+1] = 'cuda:0'
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--authorization-record', type=Path)
    parser.add_argument('--gpu-uuid')
    args = parser.parse_args()
    plan = load_plan(args.plan)
    run = args.run_dir.resolve()
    if args.execute:
        if not args.authorization_record or not args.gpu_uuid:
            raise ValueError('Execution requires authorization record and a named GPU')
        approval = authorization(args.authorization_record, args.plan, plan['budget']['sha256'],
                                 run_directory=run, gpu_uuid=args.gpu_uuid)
    else:
        approval = None
    run.mkdir(parents=True, exist_ok=False)
    runtime = run/'runtime'; runtime.mkdir()
    train = resolve_argv(plan['train_argv_template'], plan['run_directory_proposal'], run)
    jobs = [dict(case_id=row['case_id'], argv=resolve_argv(row['argv_template'], plan['run_directory_proposal'], run))
            for row in plan['queue']]
    for job, row in zip(jobs, plan['queue']):
        job['output'] = flag(job['argv'], '--output')
        job['input_artifacts'] = [row['spec']]
    report = json.loads(verify_artifact(plan['training_input_report']).read_text())
    required = list(report['training_input_provenance']['payload']['artifacts'].values())
    required += [dict(path=str(ROOT/p), sha256=sha256(ROOT/p)) for p in (
        'scripts/train_mdm_imu_control.py', 'scripts/sample_mdm_imu_control.py',
        'src/itm/models/mdm_run_limits.py', 'src/itm/models/mdm_sampling.py')]
    status_path = runtime/'status.json'
    status = dict(version='itm-b0-bounded-run-v1', status='prepared_not_executed', authorization=approval,
                  plan={'path': str(args.plan.resolve()), 'sha256': sha256(args.plan)},
                  required_artifacts=required,
                  budget=plan['budget'], train_argv=train, denoising_calls=0, branch_examples=0,
                  slots=[dict(case_id=j['case_id'], status='planned', argv=j['argv'], output=j['output']) for j in jobs],
                  caps={'updates': 200, 'train_seconds': 3600, 'shared_sampling_seconds': 10800,
                        'total_seconds': 14400, 'allocated_memory_bytes': 24*GIB, 'GPUs': 1})
    atomic_json(status_path, status)
    if not args.execute:
        print(f'Prepared only: {status_path}; no GPU preflight or child launched')
        return 0
    # A prepared directory is never resumed/reused: each invocation requires a new directory.
    try:
        lock_path = Path('/tmp')/('itm-b0-'+args.gpu_uuid+'.lock')
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            status['gpu_preflight'] = preflight_gpu(args.gpu_uuid)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       PYTORCH_CUDA_ALLOC_CONF='backend:native', PYTORCH_ALLOC_CONF='backend:native')
            for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'):
                env.pop(key, None)
            start = time.monotonic()
            total_deadline = start+14400
            train_deadline = min(start+3600, total_deadline)
            status.update(status='training', total_deadline_monotonic=total_deadline,
                          train_deadline_monotonic=train_deadline)
            atomic_json(status_path, status)
            receipt = runtime/'train.receipt.json'
            context_path = runtime/'train.context.json'
            atomic_json(context_path, dict(phase='train', deadline_monotonic=train_deadline, gpu_uuid=args.gpu_uuid,
                                           receipt_path=str(receipt), memory_bytes=24*GIB, max_updates=200,
                                           expected_artifacts=required))
            result = supervise(train+['--run-context', str(context_path)], deadline=train_deadline,
                               log_path=runtime/'train.log', env=env)
            status['training_result'] = result
            if receipt.exists():
                status['training_receipt'] = {'path': str(receipt), 'sha256': sha256(receipt)}
            if result['status'] != 'completed':
                raise RuntimeError('Training stopped: '+str(result['reason']))
            validate_final_checkpoint(flag(train, '--output'), json.loads(receipt.read_text()))
            final = Path(flag(train, '--output'))
            status['required_artifacts'] = required+[dict(path=str(final), sha256=sha256(final))]
            sampling_deadline = min(time.monotonic()+10800, total_deadline)
            status.update(status='sampling', sampling_deadline_monotonic=sampling_deadline)
            atomic_json(status_path, status)
            if not run_jobs(jobs, root=runtime, deadline=sampling_deadline, gpu_uuid=args.gpu_uuid,
                            env=env, status=status, status_path=status_path):
                return 1
            status['status'] = 'completed'
            atomic_json(status_path, status)
            return 0
    except BaseException as error:
        status.update(status='failed', failure={'type': type(error).__name__, 'message': str(error)})
        mark_remaining(status['slots'])
        atomic_json(status_path, status)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
