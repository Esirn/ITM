#!/usr/bin/env python3
"""CPU diagnostic for root-translation smoothing of saved HY-Motion outputs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from itm.data.hymotion import BODY_PARENTS, SENSOR_CONFIGS, rotation_6d_to_matrix
from evaluate_hymotion_imu import scores


def smooth_root(root, iterations):
    """Repeated binomial filtering with endpoint replication."""
    result = root.copy()
    for _ in range(iterations):
        padded = np.concatenate((result[:1], result, result[-1:]), axis=0)
        result = (padded[:-2] + 2 * padded[1:-1] + padded[2:]) / 4
    return result


def smooth_local_motion(joints, rotations, iterations):
    """Filter local 6D rotations and rebuild a fixed-bone-length skeleton."""
    if iterations == 0:
        return joints, rotations
    local = rotations.copy()
    for joint, parent in enumerate(BODY_PARENTS):
        if parent >= 0:
            local[:, joint] = np.swapaxes(rotations[:, parent], -1, -2) @ rotations[:, joint]
    local6d = local[..., :, :2].reshape(len(local), 22, 6)
    for _ in range(iterations):
        padded = np.concatenate((local6d[:1], local6d, local6d[-1:]), axis=0)
        local6d = (padded[:-2] + 2 * padded[1:-1] + padded[2:]) / 4
    local = rotation_6d_to_matrix(local6d)
    result_joints = np.empty_like(joints)
    result_rotations = np.empty_like(rotations)
    root = smooth_root(joints[:, 0], iterations)
    result_joints[:, 0] = root
    result_rotations[:, 0] = local[:, 0]
    offsets = np.zeros((22, 3), dtype=joints.dtype)
    for joint, parent in enumerate(BODY_PARENTS):
        if parent >= 0:
            offsets[joint] = rotations[0, parent].T @ (joints[0, joint] - joints[0, parent])
            result_rotations[:, joint] = result_rotations[:, parent] @ local[:, joint]
            result_joints[:, joint] = result_joints[:, parent] + np.einsum('tij,j->ti', result_rotations[:, parent], offsets[joint])
    return result_joints, result_rotations


def aggregate(records):
    result = {}
    for sensor in SENSOR_CONFIGS:
        groups = defaultdict(list)
        for record in records:
            if record['sensor'] == sensor:
                groups[record['mode']].append(record)
        paired = {r['motion_id']: r['imu_score'] for r in groups['paired']}
        shuffled = {r['motion_id']: r['imu_score'] for r in groups['shuffled']}
        result[sensor] = dict(
            paired_beats_shuffled_fraction=sum(paired[key] < shuffled[key] for key in paired) / len(paired),
            paired_to_text_only_jerk_ratio=(np.mean([r['jerk_mean_m_s3'] for r in groups['paired']]) /
                                             np.mean([r['jerk_mean_m_s3'] for r in groups['text_only']])),
            means={mode: {metric: float(np.mean([r[metric] for r in values]))
                          for metric in ('imu_score', 'acceleration_rmse_m_s2', 'orientation_mean_deg', 'jerk_mean_m_s3')}
                   for mode, values in groups.items()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--data-run', type=Path, required=True)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, nargs='+', default=[0, 1, 2, 4])
    parser.add_argument('--smooth-local-rotations', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or any(value < 0 for value in args.iterations):
        raise ValueError('Output must be new and smoothing iterations must be nonnegative')
    train = json.loads((args.training_run / 'run.json').read_text())
    acc_std = np.array(train['acc_std'])
    files = sorted(args.source.glob('*.npz'))
    if not files:
        raise ValueError('No source outputs found')
    report = dict(source=str(args.source.resolve()), iterations={})
    for iterations in args.iterations:
        records = []
        for path in files:
            with np.load(path, allow_pickle=False) as saved:
                metadata = json.loads(saved['metadata'].item())
                joints, rotations = saved['joints'], saved['rotations']
            with np.load(args.data_run / 'records' / f"{metadata['motion_id']}.npz", allow_pickle=False) as target_file:
                target = target_file['imu']
            if args.smooth_local_rotations:
                smoothed_joints, smoothed_rotations = smooth_local_motion(joints, rotations, iterations)
            else:
                root = joints[:, 0]
                offset = smooth_root(root, iterations) - root
                smoothed_joints, smoothed_rotations = joints + offset[:, None], rotations
            metric = scores(smoothed_joints, smoothed_rotations, target, list(SENSOR_CONFIGS[metadata['sensor']]), acc_std)
            records.append(dict(metadata, root_smoothing_iterations=iterations, **metric))
        report['iterations'][str(iterations)] = aggregate(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
