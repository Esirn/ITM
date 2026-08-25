#!/usr/bin/env python3
"""Predict MotionLab trajectory hints from standard 12D IMU sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from itm.data.manifest import read_jsonl
from itm.data.standard_imu import resample_standard_imu, sensor_mask
from itm.models.motionlab_imu_adapter import (
    MotionLabIMUAdapterConfig,
    active_joint_mask,
    make_motionlab_imu_adapter,
)
from itm.models.torch_frame_baseline import require_torch


SENSOR_CONFIGS = {"head": (4,), "wrists": (0, 1)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--imu-manifest", type=Path, required=True)
    parser.add_argument("--motion-ids", required=True, help="comma-separated IDs")
    parser.add_argument("--sensor-config", choices=sorted(SENSOR_CONFIGS), required=True)
    parser.add_argument(
        "--motionlab-root", type=Path, default=Path("/home/a200/mount/a40/relatedworks/MotionLab")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch = require_torch()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = MotionLabIMUAdapterConfig(**checkpoint["config"])
    model = make_motionlab_imu_adapter(config)
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device).eval()

    records = {str(record["motion_id"]): record for record in read_jsonl(args.imu_manifest)}
    motion_ids = [value.strip() for value in args.motion_ids.split(",") if value.strip()]
    missing = [motion_id for motion_id in motion_ids if motion_id not in records]
    if missing:
        raise KeyError(f"motion IDs absent from IMU manifest: {missing}")
    normalization = checkpoint["imu_normalization"]
    acceleration_mean = np.asarray(normalization["acceleration_mean"], dtype=np.float32)
    acceleration_std = np.asarray(normalization["acceleration_std"], dtype=np.float32)
    sequences = []
    for motion_id in motion_ids:
        with np.load(records[motion_id]["imu_path"]) as data:
            acceleration, orientation = resample_standard_imu(
                data["acceleration"],
                data["orientation"],
                source_fps=float(data["fps"]),
                target_fps=20.0,
            )
        acceleration = (acceleration - acceleration_mean[None]) / acceleration_std[None]
        sequences.append(
            np.concatenate(
                [acceleration, orientation.reshape(len(orientation), 6, 9)], axis=-1
            ).astype(np.float32)[: config.max_frames]
        )
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int64)
    frames = int(lengths.max())
    imu = np.zeros((len(sequences), frames, 6, 12), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        imu[index, : len(sequence)] = sequence
    slots = SENSOR_CONFIGS[args.sensor_config]
    sensors = np.stack([sensor_mask(slots, sensor_count=6)] * len(sequences))
    frame_mask = np.arange(frames)[None] < lengths[:, None]
    with torch.inference_mode():
        prediction = model(
            torch.from_numpy(imu).to(device),
            torch.from_numpy(sensors).to(device),
            torch.from_numpy(frame_mask).to(device),
        ).view(len(sequences), frames, 22, 3)
        joint_mask = active_joint_mask(torch.from_numpy(sensors).to(device))
    root = args.motionlab_root.resolve()
    mean = torch.from_numpy(np.load(root / "datasets/all/mean_motion.npy")).to(prediction)
    std = torch.from_numpy(np.load(root / "datasets/all/std_motion.npy")).to(prediction)
    joints = prediction * std[None, None] + mean[None, None]
    mask = joint_mask[:, None].expand(-1, frames, -1, -1)
    mask = mask & torch.from_numpy(frame_mask).to(device)[:, :, None, None]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        joints=joints.cpu().numpy(),
        mask=mask.cpu().numpy(),
        lengths=lengths,
        motion_ids=np.asarray(motion_ids),
    )
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "imu_manifest": str(args.imu_manifest.resolve()),
        "motion_ids": motion_ids,
        "sensor_config": args.sensor_config,
        "sensor_slots": list(slots),
        "lengths": lengths.tolist(),
        "output": str(args.output.resolve()),
    }
    metadata_path = args.metadata or args.output.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
