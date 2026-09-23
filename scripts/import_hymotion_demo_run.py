#!/usr/bin/env python3
"""Import a four-way HY-Motion evaluation case into the existing ITM demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from itm.data.hymotion import SENSOR_CONFIGS


def resample(values: np.ndarray, frames: int) -> np.ndarray:
    if len(values) == frames:
        return values
    source = np.linspace(0, len(values) - 1, len(values), dtype=np.float32)
    target = np.linspace(0, len(values) - 1, frames, dtype=np.float32)
    return np.stack([np.interp(target, source, values[:, index]) for index in range(values.shape[1])], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--data-run', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, default=Path('outputs/demo_runs'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--motion-id', required=True)
    parser.add_argument('--sensor', choices=('head', 'wrists'), required=True)
    parser.add_argument('--max-frames', type=int, default=196)
    args = parser.parse_args()
    if not 3 <= args.max_frames <= 196:
        raise ValueError('max-frames must be in 3..196 for the existing SMPL preview')
    run_dir = args.run_root / args.run_id
    if run_dir.exists():
        raise FileExistsError(f'Run already exists: {run_dir}')

    panels, metadata = [], None
    for mode, label, text_source, imu_source in (
        ('text_only', 'Text-only', 'A', None),
        ('paired', 'Text + paired IMU', 'A', 'A'),
        ('zero', 'Text + zero IMU', 'A', 'Zero'),
        ('shuffled', 'Text + shuffled IMU', 'A', 'B'),
    ):
        path = args.evaluation / f'{args.motion_id}_{args.sensor}_{mode}.npz'
        with np.load(path, allow_pickle=False) as data:
            joints = np.asarray(data['joints'], dtype=np.float32)
            metadata = json.loads(data['metadata'].item())
        if joints.shape[1:] != (22, 3):
            raise ValueError(f'Unexpected joint shape in {path}: {joints.shape}')
        flattened = joints.reshape(len(joints), -1)
        motion = resample(flattened, args.max_frames).reshape(args.max_frames, 22, 3)
        panels.append({'label': label, 'kind': 'generated', 'text_source': text_source,
                       'imu_source': imu_source, 'caption': metadata['caption'], 'motion': motion.tolist()})

    with np.load(args.data_run / 'records' / f'{args.motion_id}.npz', allow_pickle=False) as data:
        imu = np.asarray(data['imu'], dtype=np.float32)
        latent = np.asarray(data['latent'], dtype=np.float32)
    if latent.shape[1] != 201:
        raise ValueError(f'Unexpected latent shape: {latent.shape}')
    ground_truth = latent[:, 135:].reshape(len(latent), 22, 3) + latent[:, None, :3]
    target_motion = resample(ground_truth.reshape(len(ground_truth), -1), args.max_frames).reshape(args.max_frames, 22, 3)
    panels.insert(0, {'label': 'Ground truth (text + target IMU)', 'kind': 'ground_truth',
                      'text_source': 'A', 'imu_source': 'A', 'caption': metadata['caption'],
                      'motion': target_motion.tolist()})
    slot = SENSOR_CONFIGS[args.sensor][0]
    acceleration = resample(imu[:, slot, :3], args.max_frames)
    run_dir.mkdir(parents=True)
    request = {'mode': 'four_way', 'experiment': 'hymotion_import', 'split': 'val',
               'sample_a': args.motion_id, 'sample_b': metadata['shuffled_motion_id'],
               'sensor_config': args.sensor, 'seed': metadata['seed'], 'text_scale': 5,
               'imu_scale': 1, 'source_evaluation': str(args.evaluation.resolve()),
               'viewer_resampled_frames': args.max_frames}
    result = {'run_id': args.run_id, 'request': request,
              'samples': {'A': {'motion_id': args.motion_id, 'caption': metadata['caption'], 'frames': len(imu)},
                          'B': {'motion_id': metadata['shuffled_motion_id']}},
              'fps': 30, 'frame_count': args.max_frames, 'panels': panels,
              'imu': {'A': {'sensor_config': args.sensor, 'acceleration': acceleration.tolist(),
                            'orientation': resample(imu[:, slot, 3:6], args.max_frames).tolist()}}}
    (run_dir / 'request.json').write_text(json.dumps(request, indent=2), encoding='utf-8')
    (run_dir / 'result.json').write_text(json.dumps(result), encoding='utf-8')
    print(run_dir)


if __name__ == '__main__':
    main()
