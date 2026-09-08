#!/usr/bin/env python3
"""Export joint-mounted virtual IMU signals from a HY-Motion result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from itm.data.standard_imu import moving_average, second_difference


SENSORS = {"head": (15,), "wrists": (20, 21)}


def joint_acceleration(joints: np.ndarray, fps: float, smooth_window: int = 5) -> np.ndarray:
    return moving_average(second_difference(joints, fps=fps), smooth_window)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--sensor-config", choices=tuple(SENSORS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.results) as result:
        metadata = json.loads(result["metadata"].item())
        joints = np.asarray(result["keypoints3d"][0, :, :22], dtype=np.float32)
        rotations = np.asarray(result["global_rotations_mat"][0], dtype=np.float32)
    indices = np.asarray(SENSORS[args.sensor_config], dtype=np.int64)
    acceleration = joint_acceleration(joints, float(metadata["fps"]))[:, indices]
    orientation = rotations[:, indices]
    export_metadata = {
        "source": str(args.results),
        "sensor_config": args.sensor_config,
        "joint_indices": indices.tolist(),
        "fps": metadata["fps"],
        "mounting": "joint-mounted proxy; no sensor-to-bone calibration",
        "acceleration": "second finite difference of generated joint position in m/s^2",
        "acceleration_smooth_window": 5,
        "orientation": "global WoodenMesh joint rotation matrix",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        acceleration=acceleration,
        orientation=orientation,
        metadata=np.array(json.dumps(export_metadata)),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(export_metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(export_metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
