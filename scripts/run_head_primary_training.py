#!/usr/bin/env python
"""One bounded head-supervised U training; reuse existing baseline results."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from itm.models.mdm_correction import artifact, read, verify, array_sha, encode_control
from itm.models.mdm_imu_control import configure_control_training
from itm.models.sensor_relative_supervision import sensor_relative_auxiliary_loss
from itm.models.torch_temporal_baseline import TemporalBaselineConfig, make_temporal_model
from run_mdm_correction_worker import load_model, batch_inputs

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def schedule(records, epochs, batch_size, seed):
    train = [r for r in records if r['split'] == 'train']
    if any(not r['training_admissible'] for r in train) or not train:
        raise ValueError('Only training-admitted train records may contribute updates')
    if {r['source_group_id'] for r in train} & {r['source_group_id'] for r in records if r['split'] != 'train'}:
        raise ValueError('Training/validation source overlap')
    population = [(r['motion_id'], c) for r in train for c in ('head', 'wrists')]
    if len(population) % batch_size:
        raise ValueError('Equal accumulation groups required')
    rng = np.random.RandomState(seed); result = []
    for epoch in range(1, epochs + 1):
        for index in rng.permutation(len(population)):
            mid, config = population[index]
            result.append(dict(exposure=len(result) + 1, epoch=epoch, motion_id=mid, sensor_config=config,
                               timestep=int(rng.randint(1000)), noise_seed=int(rng.randint(2**31 - 1)),
                               encoder_seed=int(rng.randint(2**31 - 1)),
                               text_present=bool(rng.rand() >= .1), imu_present=bool(rng.rand() >= .1)))
    return result


def sequence_mse(prediction, target, mask):
    """Equal sequence weights; padded values cannot change the denominator."""
    error = torch.where(mask[..., None], (prediction - target).square(), 0)
    return (error.sum((1, 2)) / (mask.sum(1) * target.shape[-1])).mean()


def training_objective(prediction, target, row, mean, std, scales, weights):
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[:3] != (1,263,1):
        raise ValueError('One true-length MDM sequence required')
    base_loss = (prediction-target).square().mean()
    active = row['sensor_config'] == 'head' and row['imu_present']
    if active:
        auxiliary, parts = sensor_relative_auxiliary_loss(
            prediction[0,:,0,:].T, target[0,:,0,:].T, mean, std, 'head', scales, True,
            position_weight=weights['position'], velocity_weight=weights['velocity'])
    else:
        auxiliary = prediction.sum()*0
        parts = dict(position=auxiliary, velocity=auxiliary)
    return base_loss+auxiliary, dict(base=base_loss, auxiliary=auxiliary, **parts), active


def prepare(base_path, design_path, output):
    if output.exists(): raise FileExistsError(output)
    plan, design = read(base_path), read(design_path)
    if design['candidate'] != 'U_head_relative_pv_v1': raise ValueError('Fixed head candidate required')
    for a in plan['expected_artifacts']+plan['input_artifacts']+design['bound_artifacts']: verify(a)
    if len(plan['exposures']) != 25600 or plan['epochs'] != 100:
        raise ValueError('Original100epoch schedule required')
    active = sum(r['sensor_config']=='head' and r['imu_present'] for r in plan['exposures'])
    if active != design['exposure_counts']['active_head_auxiliary']: raise ValueError('Head exposure population changed')
    plan.update(version='head-primary-training-v1', candidate=design['candidate'],
        head_design=artifact(design_path), original_training_plan=artifact(base_path),
        scales=design['scales'], auxiliary_weights=dict(position=design['position_weight'],velocity=design['velocity_weight']),
        objective=design['objective']+' Auxiliary enabled only for head with IMU present.',
        baselines=[], scope='Head-priority complete training, existing baselines reused, no formal test.',
        comparison='Only head auxiliary position/velocity supervision changes; original schedule/initialization/optimizer retained.')
    for key in ('baseline_updates','baseline_train_examples','baseline_validation_calls'):plan['counts'][key]=0
    plan['counts']['auxiliary_exposures']=active
    plan['expected_artifacts'] += [artifact(p) for p in [base_path,design_path,Path(__file__).resolve(),
        ROOT/'tests/test_head_primary_training.py',ROOT/'src/itm/models/sensor_relative_supervision.py',
        ROOT/'src/itm/metrics/sensor_relative.py']]+[design['scales']]
    output.mkdir(parents=True);write(output/'plan.json',plan);print(plan['counts'],flush=True)


class Guard:
    def __init__(self, context):
        self.context = context
        self.path = Path(context['status_path'])
        self.state = dict(status='running', phase='initialization', updates=0, exposures=0,
                          mdm_calls=0, encoder_calls=0, caption_calls=0, auxiliary_exposures=0, baseline_updates=0,
                          baseline_train_examples=0, baseline_validation_calls=0, peak_allocated_bytes=0)
        self.persist()

    def persist(self): write(self.path, self.state)

    def check(self):
        if time.monotonic() >= self.context['deadline_monotonic']:
            raise TimeoutError('Expanded training whole-attempt deadline')
        if torch.cuda.is_initialized():
            peak = torch.cuda.max_memory_allocated()
            self.state['peak_allocated_bytes'] = max(self.state['peak_allocated_bytes'], peak)
            if peak > self.context['memory_bytes']:
                raise MemoryError('Expanded training allocation cap')

    def count(self, kind, limit):
        self.check()
        self.state[kind] += 1
        if self.state[kind] > limit: raise RuntimeError('Forward budget exceeded: ' + kind)


def run_worker(plan_path, context_path, output):
    plan = read(plan_path); context = read(context_path); guard = Guard(context)
    try:
        if artifact(plan_path) != context['plan']: raise ValueError('Plan changed')
        for a in plan['expected_artifacts'] + plan['input_artifacts']: verify(a)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != context['gpu_uuid']:
            raise ValueError('One UUID-isolated GPU required')
        torch.set_num_threads(2); torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(plan['seed']); np.random.seed(plan['seed'])
        device = 'cuda:0'; dataset = read(verify(plan['dataset']))
        data = {}
        for r in dataset['records']:
            with np.load(verify(r['input']), allow_pickle=False) as z:
                data[r['motion_id']] = {k: z[k].copy() for k in z.files}
            data[r['motion_id']].update(text=r['text'], record=r)
        mean, std = [np.load(verify(dataset[k])).astype(np.float32) for k in ('motion_mean', 'motion_std')]
        scales = read(verify(plan['scales']))
        package = SimpleNamespace(common=plan['common'], mean=mean, std=std)
        initial = torch.load(verify(plan['initial']), map_location='cpu', weights_only=True)
        initial['control_config'] = plan['control_config']
        model, diffusion, controlled, encoder, original, handles, backbone = load_model(package, initial, device)
        mdm_counter = model.register_forward_pre_hook(lambda module, args: guard.count('mdm_calls', plan['counts']['mdm_calls']))
        encoder_counter = encoder.register_forward_pre_hook(lambda module, args: guard.count('encoder_calls', plan['counts']['encoder_calls']))
        configure_control_training(model)
        frozen = {n: array_sha(p) for n, p in model.named_parameters() if not p.requires_grad}
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, **dict(plan['optimizer'], betas=tuple(plan['optimizer']['betas'])))
        out = Path(output); out.mkdir(parents=True, exist_ok=False)
        texts = {}
        guard.state['phase'] = 'caption_cache'
        with torch.no_grad():
            for r in dataset['records']:
                guard.count('caption_calls', plan['counts']['caption_calls'])
                texts[r['motion_id']] = model.encode_text([r['text']]).detach()
                if not torch.isfinite(texts[r['motion_id']]).all(): raise FloatingPointError('Caption')
        guard.state['phase'] = 'expanded_U_training'; guard.persist()

        def item(mid, config):
            a = dict(data[mid]); active = np.zeros(6, dtype=bool)
            active[[4] if config == 'head' else [0, 1]] = True
            a['sensor_mask'] = active
            return a

        for offset in range(0, len(plan['exposures']), plan['batch_size']):
            optimizer.zero_grad(set_to_none=True); logs = []
            group = plan['exposures'][offset:offset + plan['batch_size']]
            for row in group:
                guard.check(); a = item(row['motion_id'], row['sensor_config'])
                x0, y = batch_inputs([a], [row], device)
                y['text_embed'] = torch.where(y['text_condition_present'][None, :, None], texts[row['motion_id']], 0)
                y['imu_control'], y['imu_condition_present'] = encode_control(encoder, y, mode='static_delta', seed=row['encoder_seed'])
                y['imu_control_mode'] = 'static_delta'
                noise = np.random.RandomState(row['noise_seed']).standard_normal(tuple(x0.shape)).astype(np.float32)
                t = torch.tensor([row['timestep']], device=device)
                xt = diffusion.q_sample(x0, t, noise=torch.from_numpy(noise).to(device))
                prediction = model(xt, diffusion._scale_timesteps(t), y=y)
                loss, parts, auxiliary_active = training_objective(prediction, x0, row, mean, std, scales, plan['auxiliary_weights'])
                if auxiliary_active: guard.count('auxiliary_exposures', plan['counts']['auxiliary_exposures'])
                if prediction.dtype != torch.float32 or not torch.isfinite(loss): raise FloatingPointError('Training')
                if loss.requires_grad: (loss / len(group)).backward()
                logs.append(dict(exposure=row['exposure'], loss=float(loss.detach()), frames=len(a['motion']), noise_sha256=array_sha(noise), auxiliary_active=auxiliary_active, loss_parts={k:float(v.detach()) for k,v in parts.items()}))
                guard.state['exposures'] += 1
                guard.check()
            grad = torch.nn.utils.clip_grad_norm_(trainable, 1.)
            if not torch.isfinite(grad): raise FloatingPointError('Gradient')
            optimizer.step(); guard.state['updates'] += 1; guard.persist()
            write(out / 'updates' / f'{guard.state["updates"]:04d}.json', dict(epoch=group[0]['epoch'], exposures=logs,
                                                                          loss=np.mean([r['loss'] for r in logs]), grad_norm=float(grad)))
            if guard.state['updates'] % 16 == 0:
                print('expanded U epoch', group[0]['epoch'], 'updates', guard.state['updates'], 'loss', np.mean([r['loss'] for r in logs]), flush=True)
        for n, p in model.named_parameters():
            if n in frozen and array_sha(p) != frozen[n]: raise ValueError('Frozen parameter changed: ' + n)
        torch.save(dict(checkpoint_schema='expanded-canonical-control-v1', control_config=plan['control_config'],
                        imu_encoder=encoder.state_dict(), adapters=controlled.adapters.state_dict(), optimizer=optimizer.state_dict(),
                        plan=artifact(plan_path), dataset=plan['dataset'], imu_normalization=dataset['normalization'],
                        execution=dict(guard.state), candidate=plan['candidate'], scales=plan['scales'], auxiliary_weights=plan['auxiliary_weights'], semantic_admission=False), out / 'expanded_U.pt')
        del optimizer
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        for k in ('updates', 'exposures', 'mdm_calls', 'encoder_calls', 'caption_calls', 'auxiliary_exposures',
                  'baseline_updates', 'baseline_train_examples', 'baseline_validation_calls'):
            if guard.state[k] != plan['counts'][k]: raise ValueError('Incomplete planned population: ' + k)
        guard.state.update(status='completed', phase='completed', main_checkpoint=artifact(out / 'expanded_U.pt'),
                           complete_sampling_performed=False, frozen_backbone_verified=True)
        guard.persist(); mdm_counter.remove(); encoder_counter.remove(); handles.remove()
    except BaseException as error:
        guard.state.update(status='failed', failure=dict(type=type(error).__name__, message=str(error)))
        guard.persist(); raise


def execute(plan_path, run):
    import fcntl
    from itm_contended_gpu_preflight import preflight_gpu
    plan = read(plan_path)
    if run.exists(): raise FileExistsError(run)
    for a in plan['expected_artifacts'] + plan['input_artifacts']: verify(a)
    policy = read(ROOT / 'docs/itm_rework/GPU_EXECUTION_POLICY_20261008.json')
    if policy.get('authorized') is not True: raise ValueError('Standing GPU delegation required')
    run.mkdir(parents=True); started = time.monotonic(); deadline = started + plan['absolute_seconds']
    uuid = plan['preferred_gpu_uuid']
    context = dict(plan=artifact(plan_path), gpu_uuid=uuid, started_monotonic=started,
                   deadline_monotonic=deadline, memory_bytes=plan['memory_bytes'],
                   status_path=str(run / 'worker_status.json'), authorization='user_standing_delegation',
                   planned=plan['counts'])
    write(run / 'execution.json', context)
    try:
        with (Path('/tmp') / ('itm-b0-' + uuid + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            def check():
                if time.monotonic() >= deadline: raise TimeoutError('Parent deadline')
            preflight_gpu(uuid, evidence_dir=run / 'preflight', check=check)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONPATH=str(ROOT / 'src') + ':' + str(ROOT / 'scripts'), OMP_NUM_THREADS='2')
            env.pop('PYTORCH_CUDA_ALLOC_CONF', None); env.pop('PYTORCH_ALLOC_CONF', None)
            command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--plan', str(plan_path),
                       '--context', str(run / 'execution.json'), '--output', str(run / 'artifacts')]
            with (run / 'worker.log').open('x') as log:
                child = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                try: code = child.wait(timeout=max(.01, deadline - time.monotonic()))
                except BaseException:
                    child.kill(); child.wait(); raise
            if code: raise RuntimeError(f'Expanded worker failed: {code}')
            status = read(run / 'worker_status.json')
            if status['status'] != 'completed': raise ValueError('Incomplete worker')
            write(run / 'queue_status.json', dict(status='completed', elapsed_seconds=time.monotonic() - started, worker=status))
    except BaseException as error:
        write(run / 'queue_status.json', dict(status='failed', elapsed_seconds=time.monotonic() - started,
                                            failure=dict(type=type(error).__name__, message=str(error)), planned=plan['counts']))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('prepare'); q.add_argument('--base', type=Path, required=True); q.add_argument('--design', type=Path, required=True); q.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('execute'); q.add_argument('--plan', type=Path, required=True); q.add_argument('--run', type=Path, required=True)
    q = sub.add_parser('worker'); q.add_argument('--plan', type=Path, required=True); q.add_argument('--context', type=Path, required=True); q.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.action == 'prepare': prepare(a.base.resolve(), a.design.resolve(), a.output.resolve())
    elif a.action == 'execute': execute(a.plan.resolve(), a.run.resolve())
    else: run_worker(a.plan.resolve(), a.context.resolve(), a.output.resolve())
