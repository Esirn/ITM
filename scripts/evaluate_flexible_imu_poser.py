#!/usr/bin/env python
"""Evaluate flexible IMUPoser with SMPL FK and temporal quality metrics."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np

from itm.data.manifest import read_jsonl
from itm.data.standard_imu import imuposer_features, sensor_mask
from itm.metrics.motion_quality import (
    foot_skating,
    jerk_ratio,
    root_relative_mpjpe,
    rotation_geodesic_error,
)
from itm.models.flexible_imu_poser import (
    FlexibleIMUPoserConfig,
    make_flexible_imu_poser,
    r6d_to_rotation_matrix,
)
from itm.models.torch_frame_baseline import require_torch
from itm.visualization.skeleton import save_motion_comparison


SENSOR_CONFIGS = {"head": (4,), "wrists": (0, 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imuposer-root", default="/home/a200/0relatedworks/IMUPoser")
    parser.add_argument("--visualization-dir")
    parser.add_argument("--visualize-records", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch = require_torch()
    _legacy_compatibility()
    sys.path.insert(0, str(Path(args.imuposer_root) / "src"))
    from imuposer.smpl.parametricModel import ParametricModel

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = make_flexible_imu_poser(
        FlexibleIMUPoserConfig(**checkpoint["model_config"])
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    body_model = ParametricModel(
        Path(args.imuposer_root) / "src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl",
        device=torch.device(args.device),
    )
    records = list(read_jsonl(args.cache_manifest))
    summary = {}
    for name, slots in SENSOR_CONFIGS.items():
        values = []
        for record_index, record in enumerate(records):
            with np.load(record["imu_path"]) as data:
                features = imuposer_features(
                    data["acceleration"],
                    data["orientation"],
                    sensor_mask(slots, sensor_count=6),
                )
                target_rotation = data["local_rotation"].astype(np.float32)
                target_joints = data["joints"].astype(np.float32)
                shape = data["shape"].astype(np.float32)
                fps = float(data["fps"])
                display_acceleration = data["acceleration"].astype(np.float32)
                display_orientation = data["orientation"].astype(np.float32)
            with torch.inference_mode():
                prediction = model(
                    torch.from_numpy(features)[None].to(args.device),
                    torch.tensor([len(features)]),
                )[0]
                pred_rotation = r6d_to_rotation_matrix(prediction).reshape(-1, 24, 3, 3)
                _, pred_joints = body_model.forward_kinematics(
                    pred_rotation, torch.from_numpy(shape).to(args.device)
                )
            pred_rotation_np = pred_rotation.cpu().numpy()
            pred_joints_np = pred_joints[:, :24].cpu().numpy()
            values.append(
                {
                    "motion_id": record["motion_id"],
                    "mpjpe_m": root_relative_mpjpe(pred_joints_np, target_joints),
                    "rotation_rad": rotation_geodesic_error(
                        pred_rotation_np, target_rotation
                    ),
                    "jerk_ratio": jerk_ratio(pred_joints_np, target_joints, fps=fps),
                    "foot_skating_mps": foot_skating(pred_joints_np, fps=fps),
                }
            )
            if args.visualization_dir and record_index < args.visualize_records:
                display_slot = slots[0]
                save_motion_comparison(
                    Path(args.visualization_dir) / f"{name}_{record['motion_id']}.gif",
                    [target_joints[:, :22], pred_joints_np[:, :22]],
                    ["Ground truth SMPL", f"IMUPoser: {name}"],
                    f"IMU-only baseline | {name}",
                    fps=round(fps),
                    imu_acceleration=display_acceleration[:, display_slot],
                    imu_orientation=display_orientation[:, display_slot, :, 0],
                    imu_title=f"Model sensors: {name} | displayed slot {display_slot}",
                )
        summary[name] = {
            "num_records": len(values),
            **{
                key: float(np.mean([item[key] for item in values]))
                for key in ("mpjpe_m", "rotation_rad", "jerk_ratio", "foot_skating_mps")
            },
            "records": values,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in stats.items() if k != "records"} for name, stats in summary.items()}, indent=2))
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
