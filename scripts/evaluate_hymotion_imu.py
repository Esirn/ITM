#!/usr/bin/env python3
"""Same-seed Text-only/paired/zero/shuffled HY-Motion joint-IMU evaluation."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch
import yaml

from itm.data.hymotion import forward_kinematics, rotation_6d_to_matrix, joint_imu_features, joint_anchor_features, SENSOR_CONFIGS
from itm.models.hymotion_imu_adapter import HYMotionIMUConfig, make_hymotion_imu_adapter, control_hymotion, place_hymotion_rotary_tables
from train_hymotion_imu import digest, write_json


def resample_imu(imu, frames):
    source_times = np.linspace(0, 1, len(imu))
    target_times = np.linspace(0, 1, frames)
    result = np.empty((frames, 6, 12), dtype=np.float32)
    for slot in range(6):
        for axis in range(3):
            result[:, slot, axis] = np.interp(target_times, source_times, imu[:, slot, axis])
        rotation = Rotation.from_matrix(imu[:, slot, 3:].reshape(-1, 3, 3))
        result[:, slot, 3:] = Slerp(source_times, rotation)(target_times).as_matrix().reshape(-1, 9)
    return result


def scores(joints, rotations, target, slots, acc_std):
    predicted = joint_imu_features(joints, rotations)[:, slots]
    target = target[:, slots]
    acc_error = predicted[..., :3] - target[..., :3]
    relative = np.swapaxes(predicted[..., 3:].reshape(-1, len(slots), 3, 3), -1, -2) @ target[..., 3:].reshape(-1, len(slots), 3, 3)
    angles = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    jerk = np.diff(joints, n=3, axis=0) * 30**3
    return dict(acceleration_rmse_m_s2=float(np.sqrt(np.mean(acc_error**2))),
                orientation_mean_deg=float(np.degrees(angles).mean()),
                imu_score=float(0.5 * (np.mean((acc_error / acc_std)**2) + np.mean(angles**2))),
                jerk_mean_m_s3=float(np.linalg.norm(jerk, axis=-1).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--data-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--count', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20260909)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--end-index', type=int)
    parser.add_argument('--control-strength', type=float, default=1.0)
    parser.add_argument('--control-timestep-cutoff', type=float, default=1.0)
    parser.add_argument('--temporal-velocity-smoothing', type=float, default=0.0)
    parser.add_argument('--control-delta-smoothing', type=float, default=0.0)
    parser.add_argument('--control-delta-smoothing-passes', type=int, default=1)
    parser.add_argument('--control-delta-projection', choices=('all', 'sensor_chains'), default='all')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.training_run, args.data_run, args.output = args.training_run.resolve(), args.data_run.resolve(), args.output.resolve()
    if args.resume and not args.output.is_dir():
        raise FileNotFoundError('Resume output directory does not exist')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Choose a new evaluation output directory')
    train = json.loads((args.training_run / 'run.json').read_text())
    if train.get('motion_padding_frames') != 360:
        raise ValueError('Training must use the official 360-frame motion prefix')
    data = json.loads((args.data_run / 'run.json').read_text())
    rows = data['train_records'][:args.count]
    if len(rows) != args.count or args.count < 2:
        raise ValueError('Insufficient evaluation records (at least two required)')
    end_index = args.count if args.end_index is None else args.end_index
    if not 0 <= args.start_index < end_index <= args.count:
        raise ValueError('Invalid evaluation shard bounds')
    if not 0 <= args.control_strength <= 1:
        raise ValueError('control strength must be within [0, 1]')
    if not 0 <= args.control_timestep_cutoff <= 1:
        raise ValueError('control timestep cutoff must be within [0, 1]')
    if not 0 <= args.temporal_velocity_smoothing <= 1:
        raise ValueError('temporal velocity smoothing must be within [0, 1]')
    if not 0 <= args.control_delta_smoothing <= 1:
        raise ValueError('control delta smoothing must be within [0, 1]')
    if not 1 <= args.control_delta_smoothing_passes <= 16:
        raise ValueError('control delta smoothing passes must be within [1, 16]')
    results = []
    completed = set()
    if args.resume:
        metrics_path = args.output / 'metrics.jsonl'
        if not metrics_path.is_file():
            raise FileNotFoundError('Resume metrics.jsonl does not exist')
        row_indices = {row['motion_id']: index for index, row in enumerate(rows)}
        for line in metrics_path.read_text().splitlines():
            entry = json.loads(line)
            key = (entry.get('motion_id'), entry.get('sensor'), entry.get('mode'))
            index = row_indices.get(key[0], -1)
            if (key in completed or not args.start_index <= index < end_index
                    or key[1] not in SENSOR_CONFIGS
                    or key[2] not in ('text_only', 'paired', 'zero', 'shuffled')):
                raise ValueError(f'Invalid or duplicate resume entry: {key}')
            row = rows[index]
            other = rows[(index + 1) % len(rows)]
            expected = dict(seed=args.seed + index, frames=row['num_frames'], caption=row['caption'],
                            shuffled_motion_id=other['motion_id'], control_strength=args.control_strength,
                            control_timestep_cutoff=args.control_timestep_cutoff,
                            temporal_velocity_smoothing=args.temporal_velocity_smoothing,
                            control_delta_smoothing=args.control_delta_smoothing,
                            control_delta_smoothing_passes=args.control_delta_smoothing_passes,
                            control_delta_projection=args.control_delta_projection)
            if any(entry.get(name) != value for name, value in expected.items()):
                raise ValueError(f'Resume protocol mismatch: {key}')
            path = args.output / f'{key[0]}_{key[1]}_{key[2]}.npz'
            with np.load(path, allow_pickle=False) as motion:
                if json.loads(motion['metadata'].item()) != entry:
                    raise ValueError(f'Resume output metadata mismatch: {key}')
            completed.add(key)
            results.append(entry)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        if torch.cuda.mem_get_info(device)[0] < 6 * 1024**3:
            raise RuntimeError('Need 6 GiB free for cached-text controlled sampling')
    runtime = Path(train['arguments']['runtime_root'])
    root = Path(train['arguments']['hymotion_root'])
    sys.path.insert(0, str(runtime))
    os.chdir(root)
    from hymotion.utils.loaders import load_object
    model_dir = root / 'ckpts/tencent/HY-Motion-1.0-Lite'
    if digest(model_dir / 'latest.ckpt') != train['checkpoint_sha256']:
        raise ValueError('Backbone checkpoint hash differs from training')
    cfg = yaml.safe_load((model_dir / 'config.yml').read_text())
    pipeline = load_object(cfg['train_pipeline'], cfg['train_pipeline_args'],
                           network_module=cfg['network_module'], network_module_args=cfg['network_module_args'])
    pipeline.load_in_demo(str(model_dir / 'latest.ckpt'), build_text_encoder=False)
    pipeline.to(device).requires_grad_(False).eval()
    saved = torch.load(args.training_run / 'adapter.pt', map_location='cpu', weights_only=True)
    adapter = make_hymotion_imu_adapter(HYMotionIMUConfig(**saved['adapter_config'])).to(device).eval()
    controlled = control_hymotion(pipeline.motion_transformer, adapter).eval()

    class Bridge(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.controlled = controlled
            self.condition = None
            self.sensor_mask = None

        def forward(self, **kwargs):
            if self.condition is None:
                return self.controlled(
                    temporal_velocity_smoothing=args.temporal_velocity_smoothing,
                    control_delta_smoothing=args.control_delta_smoothing,
                    control_delta_smoothing_passes=args.control_delta_smoothing_passes,
                    control_delta_projection=args.control_delta_projection,
                    **kwargs)
            batch = kwargs['x'].shape[0]
            return self.controlled(imu=self.condition.expand(batch, -1, -1, -1),
                                   sensor_mask=self.sensor_mask.expand(batch, -1),
                                   imu_keep=torch.ones(batch, dtype=torch.bool, device=device),
                                   control_strength=args.control_strength,
                                   control_timestep_cutoff=args.control_timestep_cutoff,
                                   temporal_velocity_smoothing=args.temporal_velocity_smoothing,
                                   control_delta_smoothing=args.control_delta_smoothing,
                                   control_delta_smoothing_passes=args.control_delta_smoothing_passes,
                                   control_delta_projection=args.control_delta_projection,
                                   **kwargs)

    bridge = Bridge().eval()
    pipeline.motion_transformer = bridge
    args.output.mkdir(parents=True, exist_ok=args.resume)
    mean, std = np.array(train['acc_mean']), np.array(train['acc_std'])
    condition_representation = train.get('condition_representation', 'imu')
    start = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for index, row in enumerate(rows):
        if not args.start_index <= index < end_index:
            continue
        with np.load(args.data_run / 'records' / f"{row['motion_id']}.npz") as cache:
            target, target_latent = cache['imu'], cache['latent']
        other = rows[(index + 1) % len(rows)]
        with np.load(args.data_run / 'records' / f"{other['motion_id']}.npz") as cache:
            other_imu, other_latent = cache['imu'], cache['latent']
        shuffled = resample_imu(other_imu, len(target))
        if condition_representation == 'anchors':
            condition_target = joint_anchor_features(target_latent, target)
            other_anchor = joint_anchor_features(other_latent, other_imu)
            source_times, target_times = np.linspace(0, 1, len(other_anchor)), np.linspace(0, 1, len(target))
            positions = np.stack([np.interp(target_times, source_times, other_anchor[:, slot, axis])
                                  for slot in range(6) for axis in range(3)], axis=-1).reshape(len(target), 6, 3)
            condition_shuffled = np.concatenate((positions, shuffled[..., 3:]), axis=-1)
        else:
            condition_target, condition_shuffled = target, shuffled
        text = {k: v.to(device) for k, v in torch.load(args.data_run / 'records' / f"{row['motion_id']}.pt", weights_only=True).items()}
        frames = len(target)
        # Real-checkpoint zero-init equivalence, before loading trained weights.
        if index == args.start_index:
            inputs = dict(x=torch.randn(1, 360, 201, device=device),
                          ctxt_input=text['text_ctxt_raw'], vtxt_input=text['text_vec_raw'],
                          timesteps=torch.tensor([0.5], device=device),
                          x_mask_temporal=torch.arange(360, device=device)[None] < frames,
                          ctxt_mask_temporal=torch.arange(text['text_ctxt_raw'].shape[1], device=device)[None] < text['text_ctxt_raw_length'][:, None])
            with torch.no_grad():
                baseline = controlled(**inputs)
                short_inputs = dict(inputs, x=inputs['x'][:, :frames],
                                    x_mask_temporal=inputs['x_mask_temporal'][:, :frames])
                short_output = controlled(**short_inputs)
                prefix_difference = (baseline[:, :frames] - short_output).abs().max().item()
                padded_target = torch.zeros(1, 360, 6, 12, device=device)
                padded_target[0, :frames] = torch.from_numpy(target).to(device)
                initial = controlled(imu=padded_target,
                                     sensor_mask=torch.ones(1, 6, dtype=torch.bool, device=device), **inputs)
            equality = (baseline - initial).abs().max().item()
            if equality != 0:
                raise RuntimeError(f'Zero-init changed official backbone output: {equality}')
            tables = place_hymotion_rotary_tables(controlled.backbone, device)
            with torch.no_grad():
                placed = controlled(**inputs)
            rope_error = (baseline - placed).abs().max().item()
            if rope_error != 0:
                raise RuntimeError(f'RoPE table placement changed output: {rope_error}')
            adapter.load_state_dict(saved['adapter'], strict=True)
            write_json(args.output / 'integration.json', dict(zero_init_max_absolute_difference=equality,
                       rope_table_placement_max_absolute_difference=rope_error, rotary_modules_placed=tables,
                       variable_vs_360_prefix_max_absolute_difference=prefix_difference,
                       adapter_sha256=digest(args.training_run / 'adapter.pt'), motion_padding_frames=360))
        for sensor, slots_tuple in SENSOR_CONFIGS.items():
            slots = list(slots_tuple)
            bridge.sensor_mask = torch.zeros(1, 6, dtype=torch.bool, device=device)
            bridge.sensor_mask[:, slots] = True
            for mode in ('text_only', 'paired', 'zero', 'shuffled'):
                if (row['motion_id'], sensor, mode) in completed:
                    continue
                if mode == 'text_only':
                    bridge.condition = None
                else:
                    imu = (condition_shuffled if mode == 'shuffled' else condition_target).copy()
                    imu[..., :3] = (imu[..., :3] - mean) / std
                    if mode == 'zero':
                        imu[:] = 0
                    padded = np.zeros((1, 360, 6, 12), dtype=np.float32)
                    padded[0, :frames] = imu
                    bridge.condition = torch.from_numpy(padded).to(device)
                with torch.no_grad():
                    generated = pipeline.generate(row['caption'], [args.seed + index], frames / 30,
                                                  cfg_scale=5, hidden_state_dict=text, length=frames)
                local = rotation_6d_to_matrix(generated['rot6d'].numpy()[0])
                translation = generated['transl'].numpy()[0]
                joints, global_rotation = forward_kinematics(local, translation, pipeline.body_model.j_template.cpu().numpy()[:22])
                metrics = scores(joints, global_rotation, target, slots, std)
                entry = dict(motion_id=row['motion_id'], sensor=sensor, mode=mode,
                             seed=args.seed + index, frames=frames, caption=row['caption'],
                             shuffled_motion_id=other['motion_id'], control_strength=args.control_strength,
                             control_timestep_cutoff=args.control_timestep_cutoff,
                             temporal_velocity_smoothing=args.temporal_velocity_smoothing,
                             control_delta_smoothing=args.control_delta_smoothing,
                             control_delta_smoothing_passes=args.control_delta_smoothing_passes,
                             control_delta_projection=args.control_delta_projection,
                             **metrics)
                np.savez_compressed(args.output / f"{row['motion_id']}_{sensor}_{mode}.npz",
                                    joints=joints, rotations=global_rotation,
                                    latent_denorm=generated['latent_denorm'].numpy(), metadata=np.array(json.dumps(entry)))
                results.append(entry)
                with (args.output / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(entry) + '\n')
                print(json.dumps(entry), flush=True)
    evaluated_count = end_index - args.start_index
    summary = dict(records=evaluated_count, total_records=len(rows), start_index=args.start_index, end_index=end_index,
                   training_run=str(args.training_run), data_run=str(args.data_run),
                   adapter_sha256=digest(args.training_run / 'adapter.pt'),
                   resumed_outputs=len(completed),
                   device=str(device), seconds=time.perf_counter() - start,
                   peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                   imu_score_definition='0.5 * (train-std-normalized acceleration MSE + rotation geodesic radians MSE)',
                   cfg='text CFG=5; same IMU in conditional/unconditional branches',
                   control_strength=args.control_strength,
                   control_timestep_cutoff=args.control_timestep_cutoff,
                   temporal_velocity_smoothing=args.temporal_velocity_smoothing,
                   control_delta_smoothing=args.control_delta_smoothing,
                   control_delta_smoothing_passes=args.control_delta_smoothing_passes,
                   control_delta_projection=args.control_delta_projection,
                   zero_definition='all normalized IMU features zero, active sensor mask retained',
                   shuffled_definition='cyclic other-record assignment, linear acceleration and SO(3) Slerp time resampling',
                   semantic_gate='Not evaluated; visual/semantic review required', full_training_allowed=False)
    for sensor in SENSOR_CONFIGS:
        groups = {mode: [r for r in results if r['sensor'] == sensor and r['mode'] == mode]
                  for mode in ('text_only', 'paired', 'zero', 'shuffled')}
        shuffled_wins = sum(a['imu_score'] < b['imu_score']
                            for a, b in zip(groups['paired'], groups['shuffled']))
        zero_wins = sum(a['imu_score'] < b['imu_score']
                        for a, b in zip(groups['paired'], groups['zero']))
        means = {mode: {metric: float(np.mean([r[metric] for r in values]))
                        for metric in ('imu_score', 'acceleration_rmse_m_s2',
                                       'orientation_mean_deg', 'jerk_mean_m_s3')}
                 for mode, values in groups.items()}
        summary[sensor] = dict(
            paired_beats_shuffled_fraction=shuffled_wins / evaluated_count,
            paired_beats_zero_fraction=zero_wins / evaluated_count,
            paired_to_text_only_jerk_ratio=(means['paired']['jerk_mean_m_s3']
                                            / means['text_only']['jerk_mean_m_s3']),
            means=means)
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
