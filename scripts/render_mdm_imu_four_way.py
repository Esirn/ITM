#!/usr/bin/env python
"""Render GT, official MDM, flexible IMUPoser, and ITM in one comparison."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np

from itm.data.standard_imu import imuposer_features, sensor_mask
from itm.models.flexible_imu_poser import FlexibleIMUPoserConfig, make_flexible_imu_poser, r6d_to_rotation_matrix
from itm.models.torch_frame_baseline import require_torch
from itm.visualization.skeleton import save_control_comparison


SENSOR_CONFIGS = {"head": (4,), "wrists": (0, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--imuposer-checkpoint", required=True)
    parser.add_argument("--sensor-config", choices=tuple(SENSOR_CONFIGS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imuposer-root", default="/home/a200/0relatedworks/IMUPoser")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    torch = require_torch()
    _legacy_compatibility()
    with np.load(args.results) as data:
        generated = data["motion"]
        gt = data["gt"][0]
        acceleration = data["acceleration"][0]
        orientation = data["orientation"][0]
        metadata = json.loads(str(data["metadata"]))
    slots = SENSOR_CONFIGS[args.sensor_config]
    features = imuposer_features(
        acceleration,
        orientation,
        sensor_mask(slots, sensor_count=6),
    )
    checkpoint = torch.load(args.imuposer_checkpoint, map_location="cpu", weights_only=True)
    poser = make_flexible_imu_poser(FlexibleIMUPoserConfig(**checkpoint["model_config"]))
    poser.load_state_dict(checkpoint["model_state"])
    poser.to(args.device).eval()
    sys.path.insert(0, str(Path(args.imuposer_root) / "src"))
    from imuposer.smpl.parametricModel import ParametricModel

    body_model = ParametricModel(
        Path(args.imuposer_root) / "src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl",
        device=torch.device(args.device),
    )
    with torch.inference_mode():
        prediction = poser(
            torch.from_numpy(features)[None].to(args.device),
            torch.tensor([len(features)]),
        )[0]
        rotation = r6d_to_rotation_matrix(prediction).reshape(-1, 24, 3, 3)
        _, poser_joints = body_model.forward_kinematics(rotation)
    motions = [gt[:, :22], generated[0], poser_joints[:, :22].cpu().numpy(), generated[2]]
    display_slot = slots[0]
    captions = [
        "paired ground truth",
        metadata["cases"][0]["text"],
        f"{args.sensor_config} IMU only",
        metadata["cases"][2]["text"],
    ]
    save_control_comparison(
        args.output,
        motions,
        ["Ground truth", "Official MDM Text-only", f"Flexible IMUPoser: {args.sensor_config}", "ITM Text + IMU"],
        captions,
        [acceleration[:, display_slot]] * 4,
        [orientation[:, display_slot, :, 0]] * 4,
        [f"input {args.sensor_config} | slot {display_slot}"] * 4,
    )
    Path(args.output).with_suffix(".json").write_text(json.dumps({
        **metadata,
        "imuposer_checkpoint": str(Path(args.imuposer_checkpoint).resolve()),
        "sensor_config": args.sensor_config,
    }, indent=2), encoding="utf-8")
    print(f"visualization: {Path(args.output).resolve()}")
    return 0


def _legacy_compatibility():
    aliases = {"bool": bool, "int": int, "float": float, "complex": complex, "object": object, "unicode": str, "str": str}
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec


if __name__ == "__main__":
    raise SystemExit(main())
