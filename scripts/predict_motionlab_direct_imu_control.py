#!/usr/bin/env python3
"""Encode standard IMU as learned MotionLab direct-control tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from itm.data.manifest import read_jsonl
from itm.data.standard_imu import resample_standard_imu, sensor_mask
from itm.models.motionlab_imu_adapter import (
    MotionLabIMUAdapterConfig,
    make_motionlab_imu_adapter,
)
from itm.models.torch_frame_baseline import require_torch


SENSOR_CONFIGS = {"head": (4,), "wrists": (0, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--imu-manifest", type=Path, required=True)
    parser.add_argument("--motion-ids", required=True)
    parser.add_argument("--sensor-config", choices=sorted(SENSOR_CONFIGS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    torch = require_torch()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("control_space") != "learned MotionLab 66D hint tokens":
        raise ValueError("checkpoint is not a direct MotionLab control adapter")
    model = make_motionlab_imu_adapter(MotionLabIMUAdapterConfig())
    model.load_state_dict(checkpoint["adapter"])
    device = torch.device(args.device)
    model.to(device).eval()
    records = {str(item["motion_id"]): item for item in read_jsonl(args.imu_manifest)}
    ids = [value.strip() for value in args.motion_ids.split(",") if value.strip()]
    normalization = checkpoint["imu_normalization"]
    mean = np.asarray(normalization["acceleration_mean"], dtype=np.float32)
    std = np.asarray(normalization["acceleration_std"], dtype=np.float32)
    sequences = []
    for motion_id in ids:
        with np.load(records[motion_id]["imu_path"]) as data:
            acceleration, orientation = resample_standard_imu(
                data["acceleration"], data["orientation"],
                source_fps=float(data["fps"]), target_fps=20.0,
            )
        acceleration = (acceleration - mean[None]) / std[None]
        sequences.append(np.concatenate(
            [acceleration, orientation.reshape(len(orientation), 6, 9)], axis=-1
        ).astype(np.float32)[:196])
    lengths = np.asarray([len(item) for item in sequences], dtype=np.int64)
    frames = int(lengths.max())
    imu = np.zeros((len(ids), frames, 6, 12), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        imu[index, :len(sequence)] = sequence
    mask = np.arange(frames)[None] < lengths[:, None]
    slots = SENSOR_CONFIGS[args.sensor_config]
    sensors = np.stack([sensor_mask(slots, sensor_count=6)] * len(ids))
    with torch.inference_mode():
        tokens = model(
            torch.from_numpy(imu).to(device),
            torch.from_numpy(sensors).to(device),
            torch.from_numpy(mask).to(device),
        ).cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, tokens=tokens, mask=mask, lengths=lengths, motion_ids=np.asarray(ids))
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()), "motion_ids": ids,
        "sensor_config": args.sensor_config, "sensor_slots": list(slots),
        "lengths": lengths.tolist(), "control_space": checkpoint["control_space"],
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
