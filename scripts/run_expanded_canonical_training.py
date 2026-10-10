#!/usr/bin/env python
"""One bounded expanded U training and two common-input temporal baselines."""
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


def prepare(dataset_path, output):
    if output.exists(): raise FileExistsError(output)
    dataset = read(dataset_path)
    if dataset['version'] != 'expanded-canonical-development-inputs-v1' or dataset['test_inputs_used']:
        raise ValueError('Expanded development inputs required')
    records = dataset['records']; exposures = schedule(records, 20, 16, 20261009)
    common = read(ROOT / 'docs/itm_rework/protocol/b0_correction_20260926_v1/design.json')['common']
    initial = ROOT / 'outputs/itm_rework/method/correction_implementation_20260927_v1/initial.pt'
    for a in [common['backbone'], *common['motion_normalization'],
              common['training_input_provenance']['payload']['artifacts']['mdm_args']]: verify(a)
    for r in records:
        path = verify(r['input'])
        with np.load(path, allow_pickle=False) as z:
            if z['motion'].shape != (r['frames'], 263) or z['imu'].shape != (r['frames'], 6, 12):
                raise ValueError('Expanded input shape mismatch')
            if not z['frame_mask'].all() or z['frame_mask'].dtype != np.bool_:
                raise ValueError('Exact true lengths required')
    dependencies = [dataset_path, initial, Path(__file__).resolve(),
                    ROOT / 'tests/test_expanded_canonical_training.py',
                    ROOT / 'scripts/run_mdm_correction_worker.py', ROOT / 'scripts/train_mdm_imu_control.py',
                    ROOT / 'scripts/itm_contended_gpu_preflight.py', ROOT / 'scripts/prepare_expanded_canonical_inputs.py',
                    ROOT / 'docs/itm_rework/GPU_EXECUTION_POLICY_20261008.json',
                    ROOT / 'docs/itm_rework/GPU_CONTENDED_EXECUTION_POLICY_20261008.json',
                    ROOT / 'docs/itm_rework/GPU_DEVICE_PREFERENCE_20261008.json']
    for folder in ('src/itm/models', 'src/itm/backbones', 'src/itm/data', 'outputs/mdm/runtime_root'):
        dependencies.extend((ROOT / folder).rglob('*.py'))
    output.mkdir(parents=True)
    state = torch.load(initial, map_location='cpu', weights_only=True)
    state_config = dict(state['control_config'], reference_mode='static_delta')
    plan = dict(version='expanded-canonical-training-v1', dataset=artifact(dataset_path), initial=artifact(initial),
        common=common, control_config=state_config, seed=20261009, epochs=20, batch_size=16,
        accumulation='16 independent true-length graphs per update; no padded MDM attention',
        exposures=exposures, optimizer=dict(lr=5e-5, weight_decay=.01, betas=[.9, .999], eps=1e-8),
        objective='uniform normalized x0 MSE over 263 dimensions; equal sequence weights',
        snapshot_selection='final epoch 20 only; no validation selection',
        baseline_config=asdict(TemporalBaselineConfig(model_dim=256, num_layers=2, num_heads=4,
                                                     feedforward_dim=512, dropout=.1)),
        baseline_optimizer=dict(lr=3e-4, weight_decay=1e-4),
        baselines=['simple_text_imu', 'simple_imu_only'],
        baseline_input='same normalized observed IMU; ordered six slots plus active flags; same native CLIP captions; no target trajectories as inputs',
        counts=dict(train_groups=128, val_groups=32, updates=320, exposures=5120,
                    mdm_calls=5120, encoder_calls=10240, caption_calls=160,
                    baseline_updates=640, baseline_train_examples=10240, baseline_validation_calls=128),
        absolute_seconds=3600, memory_bytes=24 * 1024**3,
        preferred_gpu_uuid='GPU-43c2f925-db05-c655-bb46-25196d0627d6',
        input_artifacts=[r['input'] for r in records],
        expected_artifacts=[artifact(p) for p in sorted(set(dependencies))],
        scope='expanded development training and deterministic baselines; full diffusion sampling is a separately bounded next stage',
        scientific_limits=['No formal test results.', 'New dataset captions not manually certified.',
                           'Simple IMU-only baseline is not official IMUPoser.', 'No strict generated IMU closure.'])
    write(output / 'plan.json', plan)
    print(plan['counts'], flush=True)


class Guard:
    def __init__(self, context):
        self.context = context
        self.path = Path(context['status_path'])
        self.state = dict(status='running', phase='initialization', updates=0, exposures=0,
                          mdm_calls=0, encoder_calls=0, caption_calls=0, baseline_updates=0,
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
                loss = (prediction - x0).square().mean()
                if prediction.dtype != torch.float32 or not torch.isfinite(loss): raise FloatingPointError('Training')
                if loss.requires_grad: (loss / len(group)).backward()
                logs.append(dict(exposure=row['exposure'], loss=float(loss.detach()), frames=len(a['motion']), noise_sha256=array_sha(noise)))
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
                        execution=dict(guard.state), semantic_admission=False), out / 'expanded_U.pt')
        del optimizer
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        for name in plan['baselines']:
            guard.state['phase'] = name; guard.persist()
            config = TemporalBaselineConfig(**dict(plan['baseline_config'], include_text=name == 'simple_text_imu'))
            torch.manual_seed(plan['seed'])
            net = make_temporal_model(78, 512 if config.include_text else 0, 263, config).to(device)
            optimizer = torch.optim.AdamW(net.parameters(), **plan['baseline_optimizer'])

            def collate(rows):
                values = [item(r['motion_id'], r['sensor_config']) for r in rows]
                length = max(len(a['motion']) for a in values)
                def padded(a):
                    result = np.zeros((length,) + a.shape[1:], dtype=np.float32); result[:len(a)] = a; return result
                sequences, text = [], []
                for a in values:
                    clean = np.where(a['sensor_mask'][None, :, None], a['imu'], 0)
                    seq = np.concatenate((clean.reshape(len(clean), 72), np.broadcast_to(a['sensor_mask'], (len(clean), 6))), axis=-1)
                    sequences.append(padded(seq))
                    text.append(texts[a['record']['motion_id']].reshape(-1).float() if config.include_text else torch.empty(0, device=device))
                mask = torch.arange(length, device=device)[None] < torch.tensor([len(a['motion']) for a in values], device=device)[:, None]
                return (torch.from_numpy(np.stack(sequences)).to(device), torch.stack(text),
                        torch.from_numpy(np.stack([padded(a['motion']) for a in values])).to(device), mask)

            net.train()
            for offset in range(0, len(plan['exposures']), plan['batch_size']):
                rows = plan['exposures'][offset:offset + plan['batch_size']]
                seq, text, target, mask = collate(rows)
                optimizer.zero_grad(set_to_none=True); prediction = net(seq, text, mask)
                loss = sequence_mse(prediction, target, mask)
                if not torch.isfinite(loss): raise FloatingPointError(name)
                loss.backward(); grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
                if not torch.isfinite(grad): raise FloatingPointError(name + ' gradient')
                optimizer.step(); guard.state['baseline_updates'] += 1
                guard.state['baseline_train_examples'] += len(rows); guard.check()
                if (offset // plan['batch_size'] + 1) % 16 == 0:
                    guard.persist(); print(name, 'epoch', rows[0]['epoch'], 'loss', float(loss.detach()), flush=True)
            torch.save(dict(model_state=net.state_dict(), config=asdict(config), plan=artifact(plan_path)), out / (name + '.pt'))
            net.eval()
            with torch.no_grad():
                for r in dataset['records']:
                    if r['split'] != 'val': continue
                    for sensor in ('head', 'wrists'):
                        seq, text, target, mask = collate([dict(motion_id=r['motion_id'], sensor_config=sensor)])
                        pred = net(seq, text, mask)
                        if not torch.isfinite(pred).all(): raise FloatingPointError('Baseline validation')
                        normalized = pred.permute(0, 2, 1)[:, :, None]
                        joints = backbone.decode_motion(normalized)
                        directory = out / 'validation' / name; directory.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(directory / f'{r["motion_id"]}-{sensor}.npz',
                            normalized_humanml263=normalized.cpu().numpy(), motion=joints.cpu().numpy(),
                            frame_mask=mask.cpu().numpy(), gt=data[r['motion_id']]['gt'][None])
                        guard.state['baseline_validation_calls'] += 1; guard.check()
            del optimizer, net
        for k in ('updates', 'exposures', 'mdm_calls', 'encoder_calls', 'caption_calls',
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
    q = sub.add_parser('prepare'); q.add_argument('--dataset', type=Path, required=True); q.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('execute'); q.add_argument('--plan', type=Path, required=True); q.add_argument('--run', type=Path, required=True)
    q = sub.add_parser('worker'); q.add_argument('--plan', type=Path, required=True); q.add_argument('--context', type=Path, required=True); q.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.action == 'prepare': prepare(a.dataset.resolve(), a.output.resolve())
    elif a.action == 'execute': execute(a.plan.resolve(), a.run.resolve())
    else: run_worker(a.plan.resolve(), a.context.resolve(), a.output.resolve())
