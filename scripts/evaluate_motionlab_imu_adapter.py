#!/usr/bin/env python3
"""Evaluate IMU-to-MotionLab trajectory prediction on aligned test motions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from itm.models.motionlab_imu_adapter import (
    MotionLabIMUAdapterConfig,
    active_joint_mask,
    make_motionlab_imu_adapter,
    masked_trajectory_losses,
)
from itm.models.torch_frame_baseline import require_torch
from train_motionlab_imu_adapter import PairedIMUTrajectoryDataset, collate, SENSOR_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/manifests_full/test.jsonl"))
    parser.add_argument(
        "--imu-manifest", type=Path, default=Path("outputs/manifests_full/test_standard_imu.jsonl")
    )
    parser.add_argument(
        "--motionlab-root", type=Path, default=Path("/home/a200/mount/a40/relatedworks/MotionLab")
    )
    parser.add_argument("--sensor-configs", default="head,wrists")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def evaluate_config(args, checkpoint, name, slots, mean, std, model, device):
    torch = require_torch()
    dataset = PairedIMUTrajectoryDataset(
        args.manifest,
        args.imu_manifest,
        {name: slots},
        mean,
        std,
        normalization=checkpoint["imu_normalization"],
        limit=args.max_records,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )
    sums = {"trajectory_loss": 0.0, "velocity_loss": 0.0}
    distance_sum = 0.0
    distance_count = 0
    samples = 0
    mean_tensor = torch.from_numpy(mean).to(device).view(1, 1, 22, 3)
    std_tensor = torch.from_numpy(std).to(device).view(1, 1, 22, 3)
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            prediction = model(batch["imu"], batch["sensor_mask"], batch["frame_mask"])
            joints_mask = active_joint_mask(batch["sensor_mask"])
            trajectory, velocity = masked_trajectory_losses(
                prediction, batch["target"], batch["frame_mask"], joints_mask
            )
            size = len(prediction)
            sums["trajectory_loss"] += float(trajectory) * size
            sums["velocity_loss"] += float(velocity) * size
            pred_joints = prediction.view(size, -1, 22, 3) * std_tensor + mean_tensor
            target_joints = batch["target"].view(size, -1, 22, 3) * std_tensor + mean_tensor
            valid = batch["frame_mask"][:, :, None] & joints_mask[:, None, :, 0]
            distances = torch.linalg.norm(pred_joints - target_joints, dim=-1)
            distance_sum += float((distances * valid).sum())
            distance_count += int(valid.sum())
            samples += size
    return {
        "samples": len(dataset),
        "trajectory_mse_normalized": sums["trajectory_loss"] / samples,
        "velocity_mse_normalized": sums["velocity_loss"] / samples,
        "active_joint_position_error_m": distance_sum / distance_count,
    }


def main() -> int:
    args = parse_args()
    torch = require_torch()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = make_motionlab_imu_adapter(MotionLabIMUAdapterConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device).eval()
    root = args.motionlab_root.resolve()
    mean = np.load(root / "datasets/all/mean_motion.npy").astype(np.float32)
    std = np.load(root / "datasets/all/std_motion.npy").astype(np.float32)
    names = [name.strip() for name in args.sensor_configs.split(",") if name.strip()]
    results = {
        "checkpoint": str(args.checkpoint.resolve()),
        "test_manifest": str(args.manifest.resolve()),
        "results": {
            name: evaluate_config(
                args, checkpoint, name, SENSOR_CONFIGS[name], mean, std, model, device
            )
            for name in names
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
