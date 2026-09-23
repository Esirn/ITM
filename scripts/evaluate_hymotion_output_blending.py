#!/usr/bin/env python3
"""Evaluate SO(3) output-space blending between Text-only and IMU control."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from evaluate_hymotion_imu import scores
from itm.data.hymotion import BODY_PARENTS, SENSOR_CONFIGS, forward_kinematics
from train_hymotion_imu import write_json


MODES = ('paired', 'zero', 'shuffled')


def blend_rotations(source, target, alpha):
    """Geodesically interpolate batched rotation matrices."""
    relative = np.swapaxes(source, -1, -2) @ target
    tangent = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec()
    delta = Rotation.from_rotvec(tangent * alpha).as_matrix().reshape(relative.shape)
    return source @ delta


def load_motion(path, template):
    with np.load(path, allow_pickle=False) as value:
        joints = value['joints'].copy()
        global_rotation = value['rotations'].copy()
    local_rotation = global_rotation.copy()
    for joint, parent in enumerate(BODY_PARENTS):
        if parent >= 0:
            local_rotation[:, joint] = (
                np.swapaxes(global_rotation[:, parent], -1, -2) @ global_rotation[:, joint]
            )
    translation = joints[:, 0] - template[0]
    return translation, local_rotation, joints, global_rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--data-run', type=Path, required=True)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--wooden-assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--alphas', type=float, nargs='+', default=(0, 0.1, 0.2, 0.35, 0.5, 0.75, 1))
    args = parser.parse_args()
    if args.output.exists() or not args.alphas or any(not 0 <= value <= 1 for value in args.alphas):
        raise ValueError('output must be fresh and alphas must be within [0, 1]')
    metrics = [json.loads(line) for line in (args.evaluation / 'metrics.jsonl').read_text().splitlines()]
    keyed = {(row['motion_id'], row['sensor'], row['mode']): row for row in metrics}
    motion_ids = sorted({row['motion_id'] for row in metrics})
    train = json.loads((args.training_run / 'run.json').read_text())
    data = json.loads((args.data_run / 'run.json').read_text())
    data_by_id = {row['motion_id']: row for row in data['train_records']}
    template = np.fromfile(args.wooden_assets / 'j_template.bin', np.float32).reshape(-1, 3)[:22]
    acc_std = np.asarray(train['acc_std'])
    results = []
    endpoint_errors = []
    for motion_id in motion_ids:
        row = data_by_id[motion_id]
        with np.load(args.data_run / 'records' / f'{motion_id}.npz', allow_pickle=False) as target_file:
            target_imu = target_file['imu']
        for sensor, slots in SENSOR_CONFIGS.items():
            entries = {mode: keyed[motion_id, sensor, mode] for mode in ('text_only', *MODES)}
            paths = {mode: Path(entry['output_file']) for mode, entry in entries.items()}
            text_translation, text_local, text_joints, text_global = load_motion(
                paths['text_only'], template)
            rebuilt_joints, rebuilt_global = forward_kinematics(
                text_local, text_translation, template)
            endpoint_errors.append(max(float(np.abs(rebuilt_joints - text_joints).max()),
                                       float(np.abs(rebuilt_global - text_global).max())))
            text_metrics = scores(text_joints, text_global, target_imu, list(slots), acc_std)
            for alpha in args.alphas:
                results.append(dict(motion_id=motion_id, sensor=sensor, mode='text_only', alpha=alpha,
                                    frames=row['num_frames'], caption=row['caption'], **text_metrics))
                for mode in MODES:
                    controlled_translation, controlled_local, controlled_joints, controlled_global = (
                        load_motion(paths[mode], template)
                    )
                    local = blend_rotations(text_local, controlled_local, alpha)
                    translation = text_translation + alpha * (
                        controlled_translation - text_translation)
                    joints, global_rotation = forward_kinematics(local, translation, template)
                    if alpha == 0:
                        endpoint_errors.append(max(float(np.abs(joints - text_joints).max()),
                                                   float(np.abs(global_rotation - text_global).max())))
                    if alpha == 1:
                        endpoint_errors.append(max(float(np.abs(joints - controlled_joints).max()),
                                                   float(np.abs(global_rotation - controlled_global).max())))
                    results.append(dict(motion_id=motion_id, sensor=sensor, mode=mode, alpha=alpha,
                                        frames=row['num_frames'], caption=row['caption'],
                                        **scores(joints, global_rotation, target_imu, list(slots), acc_std)))
    summary = dict(records=len(motion_ids), endpoint_max_absolute_error=max(endpoint_errors), alphas={})
    for alpha in args.alphas:
        alpha_rows = [row for row in results if row['alpha'] == alpha]
        alpha_summary = {}
        for sensor in SENSOR_CONFIGS:
            grouped = {mode: {row['motion_id']: row for row in alpha_rows
                              if row['sensor'] == sensor and row['mode'] == mode}
                       for mode in ('text_only', *MODES)}
            ids = sorted(grouped['paired'])
            paired_shuffled = np.mean([grouped['paired'][key]['imu_score']
                                       < grouped['shuffled'][key]['imu_score'] for key in ids])
            paired_zero = np.mean([grouped['paired'][key]['imu_score']
                                   < grouped['zero'][key]['imu_score'] for key in ids])
            means = {mode: {metric: float(np.mean([value[metric] for value in group.values()]))
                            for metric in ('imu_score', 'jerk_mean_m_s3')}
                     for mode, group in grouped.items()}
            alpha_summary[sensor] = dict(
                paired_beats_shuffled_fraction=float(paired_shuffled),
                paired_beats_zero_fraction=float(paired_zero),
                paired_to_text_only_jerk_ratio=(means['paired']['jerk_mean_m_s3']
                                                / means['text_only']['jerk_mean_m_s3']),
                means=means)
        summary['alphas'][str(alpha)] = alpha_summary
    args.output.mkdir(parents=True)
    (args.output / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in results))
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
