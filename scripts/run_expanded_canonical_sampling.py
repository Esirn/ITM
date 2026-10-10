#!/usr/bin/env python
"""Complete addressed-DDPM samples for a fixed expanded-development comparison."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from itm.models.mdm_correction import read, verify, artifact, encode_control
from itm.models.mdm_imu_control import make_text_imu_guidance_model
from itm.models.mdm_sampling import sample_addressed_ddpm
from itm_contended_gpu_preflight import preflight_gpu
from run_expanded_canonical_training import ROOT, write, Guard
from run_mdm_correction_worker import load_model, batch_inputs


def sampling_rows(records):
    val = [r for r in records if r['split'] == 'val'][:8]
    anchors = [r for r in records if r['split'] == 'train' and r['motion_id'] in ('000027', '000506', '004728')]
    if len(val) != 8 or len(anchors) != 3: raise ValueError('Fixed 8 validation/3 training anchor sources required')
    rows = []
    for r in val + anchors:
        for sensor in ('head', 'wrists'):
            conditions = ['paired', 'mdm_text_g2_5'] if r['split'] == 'val' else ['paired']
            for condition in conditions:
                rows.append(dict(motion_id=r['motion_id'], split=r['split'], sensor_config=sensor,
                    condition=condition, frames=r['frames'], text_present=True, imu_present=condition == 'paired',
                    case_id=f'{r["split"]}-{r["motion_id"]}-{sensor}-{condition}',
                    noise_group=f'expanded-{r["split"]}-{r["motion_id"]}-{sensor}',
                    text_scale=1. if condition == 'paired' else 2.5, imu_scale=1., joint_scale=1.))
    return rows


def prepare(training_plan, training_run, output):
    if output.exists(): raise FileExistsError(output)
    source_plan = read(training_plan); status = read(training_run / 'queue_status.json')
    if status['status'] != 'completed' or status['worker']['updates'] != 320: raise ValueError('Complete final training required')
    checkpoint = status['worker']['main_checkpoint']; verify(checkpoint)
    dataset = read(verify(source_plan['dataset'])); rows = sampling_rows(dataset['records'])
    sources = [Path(__file__).resolve(), training_plan, training_run / 'queue_status.json',
               ROOT / 'tests/test_expanded_canonical_sampling.py']
    plan = dict(version='expanded-canonical-sampling-v1', training_plan=artifact(training_plan),
        checkpoint=checkpoint, dataset=source_plan['dataset'], common=source_plan['common'],
        input_artifacts=source_plan['input_artifacts'], rows=rows, seed=20261010,
        selection='First 8 validation records in the pre-generated source-seeded data order, including 3 original development anchors; no output-based selection. Training anchors separate.',
        method='expanded U final epoch20, uniform loss, factorized unit guidance',
        text_baseline='official frozen MDM backbone through strict original-encoder bypass, text CFG2.5; same true-length DDPM/noise protocol, not a published benchmark number',
        counts=dict(samples=38, validation_samples=32, training_anchor_samples=6, mdm_calls=38000,
                    encoder_calls=44, caption_calls=11, decodes=38),
        absolute_seconds=3600, memory_bytes=24 * 1024**3,
        preferred_gpu_uuid=source_plan['preferred_gpu_uuid'],
        expected_artifacts=source_plan['expected_artifacts'] + [artifact(p) for p in sources],
        scope='development comparison with K=1; no formal test, text-editing or real-device claim')
    output.mkdir(parents=True); write(output / 'plan.json', plan); print(plan['counts'])


class SamplingGuard(Guard):
    def __init__(self, context):
        super().__init__(context)
        self.state.update(samples=0, decodes=0, denoising_steps=0)

    def before_denoise(self, case, step, examples):
        self.check(); self.state.update(active_case=case, active_step=step)
        self.state['denoising_steps'] += 1

    def ensure_finite(self, value, stage):
        if isinstance(value, dict):
            for v in value.values(): self.ensure_finite(v, stage)
        elif torch.is_tensor(value) and not torch.isfinite(value).all():
            raise FloatingPointError(stage)


def worker(plan_path, context_path, output):
    plan = read(plan_path); context = read(context_path); guard = SamplingGuard(context)
    try:
        if artifact(plan_path) != context['plan']: raise ValueError('Sampling plan changed')
        for a in plan['expected_artifacts'] + plan['input_artifacts']: verify(a)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != context['gpu_uuid']: raise ValueError('One UUID-isolated GPU required')
        torch.set_num_threads(2); torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
        dataset = read(verify(plan['dataset'])); records = {r['motion_id']: r for r in dataset['records']}
        data = {}
        for mid in {r['motion_id'] for r in plan['rows']}:
            with np.load(verify(records[mid]['input']), allow_pickle=False) as z:
                data[mid] = {k: z[k].copy() for k in z.files}
            data[mid]['text'] = records[mid]['text']
        mean, std = [np.load(verify(dataset[k])).astype(np.float32) for k in ('motion_mean', 'motion_std')]
        state = torch.load(verify(plan['checkpoint']), map_location='cpu', weights_only=True)
        package = SimpleNamespace(common=plan['common'], mean=mean, std=std)
        model, diffusion, controlled, encoder, original, handles, backbone = load_model(package, state, 'cuda:0')
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        mdm_counter = model.register_forward_pre_hook(lambda module, args: guard.count('mdm_calls', plan['counts']['mdm_calls']))
        encoder_counter = encoder.register_forward_pre_hook(lambda module, args: guard.count('encoder_calls', plan['counts']['encoder_calls']))
        guided = make_text_imu_guidance_model(model, mode='factorized', branch_execution='batched', bypass_encoder=original)
        guided.eval(); out = Path(output); out.mkdir(parents=True, exist_ok=False)
        texts, input_cache = {}, {}
        guard.state['phase'] = 'caption_cache'
        with torch.no_grad():
            for mid in sorted(data):
                guard.count('caption_calls', plan['counts']['caption_calls'])
                texts[mid] = model.encode_text([data[mid]['text']]).detach()
            guard.state['phase'] = 'full_DDPM_sampling'; guard.persist()
            for row in plan['rows']:
                a = dict(data[row['motion_id']]); active = np.zeros(6, dtype=bool)
                active[[4] if row['sensor_config'] == 'head' else [0, 1]] = True
                a['sensor_mask'] = active
                _, y = batch_inputs([a], [row], 'cuda:0')
                y['text_embed'] = texts[row['motion_id']]
                if row['imu_present']:
                    y['imu_control'], y['imu_condition_present'] = encode_control(encoder, y, mode='static_delta')
                else:
                    y['imu_control'] = torch.zeros((1, row['frames'], 512), device='cuda:0')
                y['imu_control_mode'] = 'static_delta'; y.pop('imu_reference')
                y.update(text_scale=row['text_scale'], imu_scale=1., joint_scale=1.,
                         backbone_bypass=torch.tensor([not row['imu_present']], device='cuda:0'))
                features, noises = sample_addressed_ddpm(guided, diffusion, y=y, lengths=y['lengths'],
                    case_ids=[row['noise_group']], sample_seed=plan['seed'], backbone_id=plan['common']['backbone']['sha256'], run_guard=guard)
                np.savez_compressed(out / (row['case_id'] + '.raw.npz'), normalized_humanml263=features.cpu().numpy())
                joints = backbone.decode_motion(features); guard.ensure_finite(joints, 'Decoded joints')
                guard.state['decodes'] += 1
                np.savez_compressed(out / (row['case_id'] + '.npz'), normalized_humanml263=features.cpu().numpy(),
                    motion=joints.cpu().numpy(), gt=a['gt'][None], frame_mask=y['imu_frame_mask'].cpu().numpy(),
                    acceleration=a['acceleration'][None], orientation=a['orientation'][None])
                write(out / (row['case_id'] + '.json'), dict(row=row, checkpoint=plan['checkpoint'],
                    input=records[row['motion_id']]['input'], random_plans=noises,
                    raw=artifact(out / (row['case_id'] + '.raw.npz')), final=artifact(out / (row['case_id'] + '.npz'))))
                guard.state['samples'] += 1; guard.persist()
                print('sampled', guard.state['samples'], '/', plan['counts']['samples'], row['case_id'], flush=True)
        for k in ('samples', 'mdm_calls', 'encoder_calls', 'caption_calls', 'decodes'):
            if guard.state[k] != plan['counts'][k]: raise ValueError('Incomplete sampling population: ' + k)
        guard.state.update(status='completed', phase='completed'); guard.persist()
        mdm_counter.remove(); encoder_counter.remove(); handles.remove()
    except BaseException as error:
        guard.state.update(status='failed', failure=dict(type=type(error).__name__, message=str(error))); guard.persist(); raise


def execute(plan_path, run):
    plan = read(plan_path)
    if run.exists(): raise FileExistsError(run)
    for a in plan['expected_artifacts'] + plan['input_artifacts']: verify(a)
    if read(ROOT / 'docs/itm_rework/GPU_EXECUTION_POLICY_20261008.json').get('authorized') is not True:
        raise ValueError('Standing GPU delegation required')
    run.mkdir(parents=True); started = time.monotonic(); deadline = started + plan['absolute_seconds']; uuid = plan['preferred_gpu_uuid']
    context = dict(plan=artifact(plan_path), gpu_uuid=uuid, deadline_monotonic=deadline,
                   memory_bytes=plan['memory_bytes'], status_path=str(run / 'worker_status.json'),
                   authorized_by='user_standing_delegation', planned=plan['counts'])
    write(run / 'execution.json', context)
    try:
        with (Path('/tmp') / ('itm-b0-' + uuid + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            def check():
                if time.monotonic() >= deadline: raise TimeoutError('Sampling parent deadline')
            preflight_gpu(uuid, evidence_dir=run / 'preflight', check=check)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONPATH=str(ROOT / 'src') + ':' + str(ROOT / 'scripts'), OMP_NUM_THREADS='2')
            env.pop('PYTORCH_CUDA_ALLOC_CONF', None); env.pop('PYTORCH_ALLOC_CONF', None)
            command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--plan', str(plan_path),
                       '--context', str(run / 'execution.json'), '--output', str(run / 'artifacts')]
            with (run / 'worker.log').open('x') as log:
                child = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                try: code = child.wait(timeout=max(.01, deadline - time.monotonic()))
                except BaseException: child.kill(); child.wait(); raise
            if code: raise RuntimeError(f'Sampling worker failed: {code}')
            status = read(run / 'worker_status.json')
            if status['status'] != 'completed': raise ValueError('Incomplete worker')
            write(run / 'queue_status.json', dict(status='completed', elapsed_seconds=time.monotonic() - started, worker=status))
    except BaseException as error:
        write(run / 'queue_status.json', dict(status='failed', elapsed_seconds=time.monotonic() - started,
                                            failure=dict(type=type(error).__name__, message=str(error)), planned=plan['counts'])); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('prepare'); q.add_argument('--training-plan', type=Path, required=True); q.add_argument('--training-run', type=Path, required=True); q.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('execute'); q.add_argument('--plan', type=Path, required=True); q.add_argument('--run', type=Path, required=True)
    q = sub.add_parser('worker'); q.add_argument('--plan', type=Path, required=True); q.add_argument('--context', type=Path, required=True); q.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.action == 'prepare': prepare(a.training_plan.resolve(), a.training_run.resolve(), a.output.resolve())
    elif a.action == 'execute': execute(a.plan.resolve(), a.run.resolve())
    else: worker(a.plan.resolve(), a.context.resolve(), a.output.resolve())
