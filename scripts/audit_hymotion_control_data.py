#!/usr/bin/env python3
"""Check reconstructed control targets and cached real text embeddings."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from itm.data.hymotion import forward_kinematics, rotation_6d_to_matrix, joint_imu_features
from train_hymotion_imu import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for folder in args.runs:
        run = json.loads((folder / 'run.json').read_text())
        assets = Path(run['arguments']['hymotion_root']) / 'scripts/gradio/static/assets/dump_wooden'
        template = np.fromfile(assets / 'j_template.bin', np.float32).reshape(-1, 3)[:22]
        max_imu_error, max_position_error, count = 0.0, 0.0, 0
        for row in run['train_records']:
            with np.load(folder / 'records' / f"{row['motion_id']}.npz", allow_pickle=False) as data:
                latent, imu = data['latent'], data['imu']
            if latent.shape != (row['num_frames'], 201) or imu.shape != (row['num_frames'], 6, 12):
                raise ValueError(f"Incorrect shape: {row['motion_id']}")
            if not np.isfinite(latent).all() or not np.isfinite(imu).all():
                raise ValueError('Nonfinite motion/IMU cache')
            rotations = rotation_6d_to_matrix(latent[:, 3:135].reshape(-1, 22, 6))
            joints, global_rotation = forward_kinematics(rotations, latent[:, :3], template)
            imu_error = float(np.abs(joint_imu_features(joints, global_rotation) - imu).max())
            position_error = float(np.abs((joints - joints[:, :1]).reshape(-1, 66) - latent[:, 135:]).max())
            max_imu_error, max_position_error = max(max_imu_error, imu_error), max(max_position_error, position_error)
            # Float32 FK perturbations are amplified by second differencing.
            if imu_error > 0.02 or position_error > 1e-4:
                raise ValueError(f"Inconsistent target/IMU round trip: {row['motion_id']}")
            text = torch.load(folder / 'records' / f"{row['motion_id']}.pt", map_location='cpu', weights_only=True)
            vec, ctxt, length = text['text_vec_raw'], text['text_ctxt_raw'], text['text_ctxt_raw_length']
            if vec.shape != (1, 1, 768) or ctxt.ndim != 3 or ctxt.shape[0] != 1 or ctxt.shape[-1] != 4096:
                raise ValueError('Invalid real text embedding shape')
            if length.shape != (1,) or not 0 < length.item() <= ctxt.shape[1]:
                raise ValueError('Invalid text length')
            if not torch.isfinite(vec).all() or not torch.isfinite(ctxt).all() or not vec.any() or not ctxt.any():
                raise ValueError('Nonfinite or zero-filled text embeddings')
            count += 1
        reports.append(dict(run=str(folder.resolve()), records=count, max_imu_roundtrip_error=max_imu_error,
                            max_root_relative_position_error_m=max_position_error, real_text_shapes_and_finite_nonzero=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, dict(runs=reports, passed=True))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
