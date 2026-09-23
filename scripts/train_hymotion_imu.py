#!/usr/bin/env python3
"""Real AMASS/HumanML Text+IMU flow-matching smoke and pilot training.

Uses an explicitly reconstructed WoodenMesh representation, not an official
HY-Motion training dataset. Text encoding and adapter training run sequentially.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import yaml

from itm.data.hymotion import (BODY_PARENTS, forward_kinematics, joint_imu_features,
                               joint_anchor_features, SENSOR_CONFIGS, SENSOR_JOINTS)
from itm.models.hymotion_imu_adapter import HYMotionIMUConfig, make_hymotion_imu_adapter, control_hymotion, pad_hymotion_inputs


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def select_records(manifest, texts, count, seed):
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    rows = sorted((r for r in rows if 20 <= r['num_frames'] <= 360), key=lambda r: r['motion_id'])
    random.Random(seed).shuffle(rows)
    selected = []
    for row in rows:
        captions = []
        for line in (texts / f"{row['motion_id']}.txt").read_text().splitlines():
            parts = line.split('#')
            if len(parts) == 4 and float(parts[2]) == 0 and float(parts[3]) == 0:
                captions.append(parts[0])
        if captions:
            selected.append(dict(row, caption=captions[0]))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f'Only {len(selected)} eligible full-caption records; requested {count}')
    return selected


def prepared_extension_start(prepared, rows, checkpoint_sha256):
    base_rows = prepared.get('train_records', [])
    if (prepared.get('checkpoint_sha256') != checkpoint_sha256
            or not base_rows or len(base_rows) >= len(rows)
            or rows[:len(base_rows)] != base_rows):
        raise ValueError('Prepared extension must be an exact nonempty prefix with the same checkpoint')
    return len(base_rows)


def reconstruct(row, body):
    with np.load(row['imu_path'], allow_pickle=False) as cache:
        rotations = cache['local_rotation'][:, :22].copy()
        translation = cache['translation'].copy()
    # Standard cache is already Y-up. Canonicalize initial facing to +Z.
    facing = rotations[0, 0] @ np.array([0, 0, 1], dtype=np.float32)
    yaw = np.arctan2(facing[0], facing[2])
    c, s = np.cos(yaw), np.sin(yaw)
    canonical = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=np.float32)
    rotations[:, 0] = canonical @ rotations[:, 0]
    template = body.j_template.cpu().numpy()[:22]
    translation = (translation - translation[:1]) @ canonical.T - template[0]
    rot6d = rotations[..., :, :2].reshape(len(rotations), 22, 6)
    min_y = float('inf')
    body_device = body.j_template.device
    with torch.no_grad():
        for start in range(0, len(rotations), 30):
            result = body({
                'rot6d': torch.from_numpy(rot6d[start:start + 30]).to(body_device),
                'trans': torch.from_numpy(translation[start:start + 30]).to(body_device),
            })
            min_y = min(min_y, result['vertices'][..., 1].min().item())
    translation[:, 1] -= min_y
    joints, global_rotation = forward_kinematics(rotations, translation, template)
    root_relative = joints - joints[:, :1]
    latent = np.concatenate((translation, rot6d.reshape(len(rotations), 132), root_relative.reshape(len(rotations), 66)), axis=-1)
    imu = joint_imu_features(joints, global_rotation)
    if not np.isfinite(latent).all() or not np.isfinite(imu).all():
        raise ValueError(f"Nonfinite reconstructed record: {row['motion_id']}")
    return latent, imu


def physical_position_jerk_loss(endpoint, target_latent, frame_mask, mean, std):
    """Match physical world-axis joint jerk between two normalized trajectories."""
    endpoint = endpoint[..., 135:] * std[..., 135:] + mean[..., 135:]
    target = target_latent[..., 135:] * std[..., 135:] + mean[..., 135:]
    losses = []
    for predicted_positions, target_positions, valid in zip(endpoint, target, frame_mask):
        frames = int(valid.sum().item())
        if frames < 4:
            continue
        predicted_jerk = torch.diff(predicted_positions[:frames], n=3, dim=0) * 30**3
        target_jerk = torch.diff(target_positions[:frames], n=3, dim=0) * 30**3
        losses.append((predicted_jerk - target_jerk).square().mean())
    if not losses:
        return endpoint.new_zeros(())
    return torch.stack(losses).mean()


def position_jerk_matching_loss(prediction, noisy_latent, target_latent, timestep, frame_mask, mean, std):
    """Match physical world-axis joint jerk for the one-step flow endpoint."""
    endpoint = noisy_latent + (1 - timestep[:, None, None]) * prediction
    return physical_position_jerk_loss(endpoint, target_latent, frame_mask, mean, std)


def physical_sensor_acceleration(positions, fps=30.0, smooth_window=5):
    """Differentiable counterpart of joint_imu_features acceleration synthesis."""
    if positions.ndim != 4 or positions.shape[-1] != 3 or smooth_window % 2 == 0:
        raise ValueError('expected positions [B,T,S,3] and an odd smoothing window')
    acceleration = torch.zeros_like(positions)
    if positions.shape[1] >= 3:
        acceleration[:, 1:-1] = (positions[:, :-2] + positions[:, 2:]
                                  - 2 * positions[:, 1:-1]) * fps**2
    if smooth_window <= 1:
        return acceleration
    radius = smooth_window // 2
    values = acceleration.permute(0, 2, 3, 1).reshape(-1, 1, positions.shape[1])
    values = torch.nn.functional.pad(values, (radius, radius), mode='replicate')
    values = torch.nn.functional.avg_pool1d(values, smooth_window, stride=1)
    return values.reshape(positions.shape[0], positions.shape[2], 3, positions.shape[1]).permute(0, 3, 1, 2)


def position_acceleration_matching_loss(prediction, noisy_latent, timestep, target_imu,
                                        sensor_mask, frame_mask, mean, std, condition_active):
    """Match generated sensor-joint acceleration to the raw paired IMU condition."""
    endpoint = noisy_latent + (1 - timestep[:, None, None]) * prediction
    endpoint = endpoint * std + mean
    translation = endpoint[..., :3].unsqueeze(-2)
    joints = endpoint[..., 135:].reshape(*endpoint.shape[:2], 22, 3) + translation
    sensor_positions = joints[:, :, list(SENSOR_JOINTS)]
    losses = []
    for batch_index, valid in enumerate(frame_mask):
        frames = int(valid.sum().item())
        active = sensor_mask[batch_index] & condition_active[batch_index]
        if frames < 3 or not active.any():
            continue
        predicted = physical_sensor_acceleration(sensor_positions[batch_index:batch_index + 1, :frames])[0]
        losses.append(torch.nn.functional.smooth_l1_loss(
            predicted[:, active], target_imu[batch_index, :frames, active, :3], beta=1.0))
    return torch.stack(losses).mean() if losses else endpoint.new_zeros(())


def torch_rotation_6d_to_matrix(value):
    """Differentiable counterpart of itm.data.hymotion.rotation_6d_to_matrix."""
    columns = value.reshape(*value.shape[:-1], 3, 2)
    first = torch.nn.functional.normalize(columns[..., 0], dim=-1, eps=1e-8)
    second = columns[..., 1] - (first * columns[..., 1]).sum(dim=-1, keepdim=True) * first
    second = torch.nn.functional.normalize(second, dim=-1, eps=1e-8)
    return torch.stack((first, second, torch.cross(first, second, dim=-1)), dim=-1)


def torch_global_rotations(local_rotation, parents=BODY_PARENTS):
    """Accumulate local rotations along the 22-joint WoodenMesh hierarchy."""
    if local_rotation.shape[-3:] != (len(parents), 3, 3):
        raise ValueError('rotation dimensions do not match the skeleton')
    global_rotation = [local_rotation[..., 0, :, :]]
    for joint in range(1, len(parents)):
        parent = int(parents[joint])
        if not 0 <= parent < joint:
            raise ValueError('parents must precede their children')
        global_rotation.append(global_rotation[parent] @ local_rotation[..., joint, :, :])
    return torch.stack(global_rotation, dim=-3)


def fk_orientation_matching_loss(prediction, noisy_latent, timestep, target_imu,
                                 sensor_mask, frame_mask, mean, std, condition_active):
    """Match FK-derived global sensor orientation to the paired raw IMU."""
    endpoint = noisy_latent + (1 - timestep[:, None, None]) * prediction
    endpoint = endpoint * std + mean
    local = torch_rotation_6d_to_matrix(endpoint[..., 3:135].reshape(*endpoint.shape[:2], 22, 6))
    predicted = torch_global_rotations(local)[:, :, list(SENSOR_JOINTS)]
    target = target_imu[..., 3:].reshape(*target_imu.shape[:3], 3, 3)
    losses = []
    for batch_index, valid in enumerate(frame_mask):
        frames = int(valid.sum().item())
        active = sensor_mask[batch_index] & condition_active[batch_index]
        if frames < 1 or not active.any():
            continue
        losses.append(torch.nn.functional.smooth_l1_loss(
            predicted[batch_index, :frames, active], target[batch_index, :frames, active], beta=0.1))
    return torch.stack(losses).mean() if losses else endpoint.new_zeros(())


def unrolled_position_jerk_loss(model, noise, target_latent, steps, ctxt, vec, frame_mask, ctxt_mask,
                                imu, sensor_mask, imu_keep, mean, std):
    """Differentiate through a short Euler flow rollout from the noise endpoint."""
    state = noise
    for index in range(steps):
        timestep = torch.full((state.shape[0],), (index + 0.5) / steps, device=state.device)
        velocity = model(x=state, ctxt_input=ctxt, vtxt_input=vec, timesteps=timestep,
                         x_mask_temporal=frame_mask, ctxt_mask_temporal=ctxt_mask,
                         imu=imu, sensor_mask=sensor_mask, imu_keep=imu_keep)
        state = state + velocity / steps
    return physical_position_jerk_loss(state, target_latent, frame_mask, mean, std)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hymotion-root', type=Path, required=True)
    parser.add_argument('--runtime-root', type=Path, default=Path('outputs/hymotion/runtime_root'))
    parser.add_argument('--manifest', type=Path, default=Path('outputs/manifests_full/train_standard_imu.jsonl'))
    parser.add_argument('--texts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--text-drop-probability', type=float, default=0.1)
    parser.add_argument('--position-jerk-loss-weight', type=float, default=0.0)
    parser.add_argument('--position-acceleration-loss-weight', type=float, default=0.0)
    parser.add_argument('--fk-orientation-loss-weight', type=float, default=0.0)
    parser.add_argument('--control-residual-curvature-loss-weight', type=float, default=0.0)
    parser.add_argument('--unrolled-steps', type=int, default=0)
    parser.add_argument('--unrolled-jerk-loss-weight', type=float, default=0.0)
    parser.add_argument('--control-token-smoothing-passes', type=int, default=0)
    parser.add_argument('--control-token-stride', type=int, default=1)
    parser.add_argument('--control-injection-start', type=int, default=0)
    parser.add_argument('--control-injection-end', type=int)
    parser.add_argument('--control-timestep-power', type=float, default=0.0)
    parser.add_argument('--control-mode', choices=('hidden', 'output'), default='hidden')
    parser.add_argument('--output-velocity-smoothing-passes', type=int, default=0)
    parser.add_argument('--zero-imu-drop-probability', type=float, default=0.0)
    parser.add_argument('--condition-representation', choices=('imu', 'anchors'), default='imu')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--prepared-data', type=Path)
    parser.add_argument('--extend-prepared-data', type=Path)
    args = parser.parse_args()
    if (args.count < 1 or args.epochs < 1
            or (args.max_steps is not None and args.max_steps < 1)
            or not 0 <= args.text_drop_probability <= 1
            or not np.isfinite(args.position_jerk_loss_weight) or args.position_jerk_loss_weight < 0
            or not np.isfinite(args.position_acceleration_loss_weight)
            or args.position_acceleration_loss_weight < 0
            or not np.isfinite(args.fk_orientation_loss_weight)
            or args.fk_orientation_loss_weight < 0
            or not np.isfinite(args.control_residual_curvature_loss_weight)
            or args.control_residual_curvature_loss_weight < 0):
        raise ValueError('Invalid count, epochs, dropout, or position jerk loss weight')
    if (args.unrolled_steps < 0 or not np.isfinite(args.unrolled_jerk_loss_weight)
            or args.unrolled_jerk_loss_weight < 0
            or (args.unrolled_jerk_loss_weight and args.unrolled_steps < 2)):
        raise ValueError('Invalid unrolled steps or jerk loss weight')
    injection_end = 18 if args.control_injection_end is None else args.control_injection_end
    if (args.control_token_smoothing_passes < 0 or args.control_token_stride < 1
            or not 0 <= args.output_velocity_smoothing_passes <= 16
            or not 0 <= args.control_injection_start < injection_end <= 18
            or not np.isfinite(args.control_timestep_power) or args.control_timestep_power < 0):
        raise ValueError('Invalid control token smoothing configuration')
    if not 0 <= args.zero_imu_drop_probability <= 1:
        raise ValueError('Invalid zero IMU drop probability')
    if args.extend_prepared_data is not None and (not args.prepare_only or args.prepared_data is not None):
        raise ValueError('--extend-prepared-data requires --prepare-only and is incompatible with --prepared-data')
    args.hymotion_root = args.hymotion_root.resolve()
    args.runtime_root = args.runtime_root.resolve()
    args.texts, args.manifest, args.output = args.texts.resolve(), args.manifest.resolve(), args.output.resolve()
    if args.prepared_data is not None:
        args.prepared_data = args.prepared_data.resolve()
    if args.extend_prepared_data is not None:
        args.extend_prepared_data = args.extend_prepared_data.resolve()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory to preserve run provenance')
    device = torch.device(args.device)
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info(device)
        required = (6 if args.prepared_data is not None else
                    18 if args.prepare_only else 23)
        if free < required * 1024**3:
            raise RuntimeError(f'Need at least {required} GiB free; got {free / 1024**3:.2f}')
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rows = select_records(args.manifest, args.texts, args.count, args.seed)
    args.output.mkdir(parents=True)
    cache_dir = args.output / 'records'
    sys.path.insert(0, str(args.runtime_root))
    os.chdir(args.hymotion_root)
    from hymotion.utils.loaders import load_object
    model_dir = args.hymotion_root / 'ckpts/tencent/HY-Motion-1.0-Lite'
    config = yaml.safe_load((model_dir / 'config.yml').read_text())
    pipeline = load_object(config['train_pipeline'], config['train_pipeline_args'],
                           network_module=config['network_module'], network_module_args=config['network_module_args'])
    pipeline.load_in_demo(str(model_dir / 'latest.ckpt'), build_text_encoder=False)
    pipeline.requires_grad_(False).eval()
    run = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               source_commit=json.loads((args.runtime_root / 'workspace.json').read_text())['commit'],
               checkpoint_sha256=digest(model_dir / 'latest.ckpt'),
               representation='ITM reconstruction: translation + column6D root/local rotations + world-axis root-relative WoodenMesh joints',
               representation_status='Not verified against official training preprocessing',
               sensors='joint-mounted acceleration/orientation; no gravity, no hardware calibration',
               loss='flow velocity MSE; optional control-residual curvature, paired FK orientation, sensor acceleration, and position jerk matching; independent text/IMU dropout',
               batch_size=1, train_records=rows)
    run['motion_padding_frames'] = 360
    run['condition_representation'] = args.condition_representation
    run['loss_mask'] = 'valid motion frames only; text RoPE prefix always 360'
    write_json(args.output / 'run.json', run)
    # Geometry and real text encoding run sequentially, before training.
    extension_start = 0
    if args.extend_prepared_data is not None:
        prepared = json.loads((args.extend_prepared_data / 'run.json').read_text())
        extension_start = prepared_extension_start(prepared, rows, run['checkpoint_sha256'])
        cache_dir.mkdir()
        for row in rows[:extension_start]:
            for suffix in ('.npz', '.pt'):
                source = args.extend_prepared_data / 'records' / f"{row['motion_id']}{suffix}"
                if not source.is_file():
                    raise FileNotFoundError(source)
                (cache_dir / source.name).symlink_to(source.resolve())
    if args.prepared_data is None:
        if args.extend_prepared_data is None:
            cache_dir.mkdir()
        pipeline.body_model.to(device)
        for index, row in enumerate(rows[extension_start:], start=extension_start):
            latent, imu = reconstruct(row, pipeline.body_model)
            np.savez_compressed(cache_dir / f"{row['motion_id']}.npz", latent=latent, imu=imu)
            print(f'geometry {index + 1}/{len(rows)} {row["motion_id"]}', flush=True)
        values = []
        for row in rows:
            with np.load(cache_dir / f"{row['motion_id']}.npz") as record:
                condition = (joint_anchor_features(record['latent'], record['imu'])
                             if args.condition_representation == 'anchors' else record['imu'])
                values.append(condition[..., :3])
        all_acc = np.concatenate(values, axis=0)
        acc_mean = all_acc.mean((0, 1))
        acc_std = np.maximum(all_acc.std((0, 1)), 1e-4)
    else:
        prepared = json.loads((args.prepared_data / 'run.json').read_text())
        if prepared['train_records'] != rows or prepared['checkpoint_sha256'] != run['checkpoint_sha256']:
            raise ValueError('Prepared cache records/checkpoint do not match requested training data')
        cache_dir.symlink_to(args.prepared_data / 'records', target_is_directory=True)
        if args.condition_representation == 'anchors':
            values = []
            for row in rows:
                with np.load(args.prepared_data / 'records' / f"{row['motion_id']}.npz") as record:
                    values.append(joint_anchor_features(record['latent'], record['imu'])[..., :3])
            all_acc = np.concatenate(values, axis=0)
            acc_mean = all_acc.mean((0, 1))
            acc_std = np.maximum(all_acc.std((0, 1)), 1e-4)
        else:
            acc_mean = np.array(prepared['acc_mean'], dtype=np.float32)
            acc_std = np.array(prepared['acc_std'], dtype=np.float32)
    run['acc_mean'], run['acc_std'] = acc_mean.tolist(), acc_std.tolist()
    if args.prepare_only:
        # Cache preparation never samples motion. Keep the large frozen
        # transformer on CPU while retaining the exact text encoder weights.
        pipeline.text_encoder = load_object(
            pipeline._text_encoder_module, pipeline._text_encoder_cfg)
        pipeline.text_encoder.to(device).requires_grad_(False).eval()
        run['preparation_memory_strategy'] = 'text encoder on target device; motion transformer retained on CPU'
    else:
        pipeline.to(device)
    if args.prepared_data is None:
        for index, row in enumerate(rows[extension_start:], start=extension_start):
            with torch.no_grad():
                text = pipeline.encode_text({'text': [row['caption']]})
            torch.save({k: v.detach().cpu() for k, v in text.items()}, cache_dir / f"{row['motion_id']}.pt")
            print(f'text {index + 1}/{len(rows)} {row["motion_id"]}', flush=True)
        del text
        del pipeline.text_encoder
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    if args.prepare_only:
        run['status'] = 'prepared only; no training'
        write_json(args.output / 'run.json', run)
        return
    # Cache creation must not change adapter initialization or training noise.
    torch.manual_seed(args.seed)
    adapter_cfg = HYMotionIMUConfig(control_token_smoothing_passes=args.control_token_smoothing_passes,
                                    control_token_stride=args.control_token_stride,
                                    control_injection_start=args.control_injection_start,
                                    control_injection_end=args.control_injection_end,
                                    control_timestep_power=args.control_timestep_power,
                                    control_mode=args.control_mode,
                                    output_velocity_smoothing_passes=args.output_velocity_smoothing_passes)
    adapter = make_hymotion_imu_adapter(adapter_cfg).to(device)
    model = control_hymotion(pipeline.motion_transformer, adapter).train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=0.01)
    run['adapter_config'] = adapter_cfg.to_dict()
    run['trainable_parameters'] = sum(p.numel() for p in adapter.parameters())
    run['frozen_backbone_parameters'] = sum(p.numel() for p in pipeline.motion_transformer.parameters())
    write_json(args.output / 'run.json', run)
    losses = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        order = list(range(len(rows)))
        random.Random(args.seed + epoch).shuffle(order)
        for step, index in enumerate(order):
            if args.max_steps is not None and len(losses) >= args.max_steps:
                break
            row = rows[index]
            with np.load(cache_dir / f"{row['motion_id']}.npz") as record:
                latent = torch.from_numpy(record['latent']).to(device)[None]
                raw_imu = record['imu'].copy()
                imu = raw_imu.copy()
                if args.condition_representation == 'anchors':
                    imu = joint_anchor_features(record['latent'], imu)
            imu[..., :3] = (imu[..., :3] - acc_mean) / acc_std
            imu = torch.from_numpy(imu).to(device)[None]
            raw_imu = torch.from_numpy(raw_imu).to(device)[None]
            std = pipeline.std
            latent = torch.where(std >= 1e-3, (latent - pipeline.mean) / std.clamp_min(1e-3), 0)
            frames = latent.shape[1]
            latent, imu, frame_mask = pad_hymotion_inputs(latent, imu)
            raw_imu = torch.nn.functional.pad(raw_imu, (0, 0, 0, 0, 0, 360 - frames))
            text = {k: v.to(device) for k, v in torch.load(cache_dir / f"{row['motion_id']}.pt", weights_only=True).items()}
            ctxt, vec = text['text_ctxt_raw'], text['text_vec_raw']
            drop_text = bool(torch.rand((), device=device) < args.text_drop_probability)
            if drop_text:
                ctxt = pipeline.null_ctxt_input.expand_as(ctxt)
                vec = pipeline.null_vtxt_feat.expand_as(vec)
            ctxt_mask = torch.arange(ctxt.shape[1], device=device)[None] < text['text_ctxt_raw_length'][:, None]
            sensor = 'head' if (step + epoch) % 2 == 0 else 'wrists'
            sensor_mask = torch.zeros(1, 6, dtype=torch.bool, device=device)
            sensor_mask[:, list(SENSOR_CONFIGS[sensor])] = True
            noise = torch.randn_like(latent)
            t = torch.rand(1, device=device)
            x = noise * (1 - t[:, None, None]) + latent * t[:, None, None]
            keep_imu = torch.rand(1, device=device) >= adapter_cfg.imu_drop_probability
            if args.zero_imu_drop_probability:
                zero_imu = ((torch.rand(1, device=device) < args.zero_imu_drop_probability)
                            & keep_imu)
            else:
                zero_imu = torch.zeros(1, dtype=torch.bool, device=device)
            if zero_imu.item():
                imu = torch.zeros_like(imu)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x=x, ctxt_input=ctxt, vtxt_input=vec, timesteps=t,
                               x_mask_temporal=frame_mask, ctxt_mask_temporal=ctxt_mask,
                               imu=imu, sensor_mask=sensor_mask, imu_keep=keep_imu)
            residual_curvature_loss = model.take_control_residual_curvature_loss()
            error = (prediction[frame_mask] - (latent - noise)[frame_mask]).square()
            flow_loss = error.mean()
            jerk_loss = position_jerk_matching_loss(prediction, x, latent, t, frame_mask,
                                                     pipeline.mean, std)
            acceleration_loss = position_acceleration_matching_loss(
                prediction, x, t, raw_imu, sensor_mask, frame_mask, pipeline.mean, std,
                keep_imu & ~zero_imu)
            orientation_loss = fk_orientation_matching_loss(
                prediction, x, t, raw_imu, sensor_mask, frame_mask, pipeline.mean, std,
                keep_imu & ~zero_imu)
            unrolled_jerk_loss = prediction.new_zeros(())
            if args.unrolled_jerk_loss_weight:
                unrolled_jerk_loss = unrolled_position_jerk_loss(
                    model, noise, latent, args.unrolled_steps, ctxt, vec, frame_mask, ctxt_mask,
                    imu, sensor_mask, keep_imu, pipeline.mean, std)
            loss = (flow_loss + args.position_jerk_loss_weight * jerk_loss
                    + args.position_acceleration_loss_weight * acceleration_loss
                    + args.fk_orientation_loss_weight * orientation_loss
                    + args.control_residual_curvature_loss_weight * residual_curvature_loss
                    + args.unrolled_jerk_loss_weight * unrolled_jerk_loss)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss for {row["motion_id"]}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
            if any(p.grad is not None for p in pipeline.motion_transformer.parameters()):
                raise RuntimeError('Frozen backbone received gradients')
            optimizer.step()
            entry = dict(epoch=epoch + 1, step=step + 1, motion_id=row['motion_id'], frames=frames,
                         sensor=sensor, text_dropped=drop_text, imu_dropped=not keep_imu.item(),
                         zero_imu_dropped=zero_imu.item(),
                         loss=loss.item(), flow_loss=flow_loss.item(), jerk_loss=jerk_loss.item(),
                         position_jerk_loss_weight=args.position_jerk_loss_weight,
                         acceleration_loss=acceleration_loss.item(),
                         position_acceleration_loss_weight=args.position_acceleration_loss_weight,
                         fk_orientation_loss=orientation_loss.item(),
                         fk_orientation_loss_weight=args.fk_orientation_loss_weight,
                         control_residual_curvature_loss=residual_curvature_loss.item(),
                         control_residual_curvature_loss_weight=args.control_residual_curvature_loss_weight,
                         unrolled_jerk_loss=unrolled_jerk_loss.item(), unrolled_steps=args.unrolled_steps,
                         unrolled_jerk_loss_weight=args.unrolled_jerk_loss_weight,
                         translation_loss=error[..., :3].mean().item(),
                         rotation_loss=error[..., 3:135].mean().item(), position_loss=error[..., 135:].mean().item(),
                         grad_norm=float(norm))
            losses.append(entry)
            with (args.output / 'losses.jsonl').open('a') as f:
                f.write(json.dumps(entry) + '\n')
            print(json.dumps(entry), flush=True)
        if args.max_steps is not None and len(losses) >= args.max_steps:
            break
    torch.save(dict(adapter=adapter.state_dict(), adapter_config=adapter_cfg.to_dict(), run=run), args.output / 'adapter.pt')
    run.update(training_seconds=time.perf_counter() - started,
               peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
               completed_steps=len(losses), status='training complete; controlled generation not yet evaluated',
               adapter_checkpoint_sha256=digest(args.output / 'adapter.pt'))
    write_json(args.output / 'run.json', run)
    print(json.dumps({k: v for k, v in run.items() if k != 'train_records'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
