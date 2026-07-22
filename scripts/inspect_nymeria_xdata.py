#!/usr/bin/env python3
"""Print the body-motion and IMU fields stored in Nymeria xdata.npz."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SEGMENT_NAMES = (
    "Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head",
    "RightShoulder", "RightUpperArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftUpperArm", "LeftForeArm", "LeftHand",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToe",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToe",
)

SENSOR_NAMES = (
    "Pelvis", "T8", "Head", "RightShoulder", "RightUpperArm",
    "RightForeArm", "RightHand", "LeftShoulder", "LeftUpperArm",
    "LeftForeArm", "LeftHand", "RightUpperLeg", "RightLowerLeg",
    "RightFoot", "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
)


def resolve_npz(path: Path) -> Path:
    if path.is_dir():
        path = path / "body" / "xdata.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Sequence directory or xdata.npz")
    parser.add_argument("--frame", type=int, default=1000)
    args = parser.parse_args()

    path = resolve_npz(args.path)
    data = np.load(path, allow_pickle=False)
    frame_count = int(data["frameCount"][0])
    frame = min(max(args.frame, 0), frame_count - 1)
    segment_count = int(data["segmentCount"][0])
    sensor_count = int(data["sensorCount"][0])

    segment_positions = data["segment_tXYZ"].reshape(frame_count, segment_count, 3)
    segment_orientations = data["segment_qWXYZ"].reshape(
        frame_count, segment_count, 4
    )
    sensor_orientations = data["sensor_qWXYZ"].reshape(
        frame_count, sensor_count, 4
    )
    sensor_accelerations = data["sensor_freeAcceleration"].reshape(
        frame_count, sensor_count, 3
    )

    print(f"File: {path}")
    print(f"Frames: {frame_count}, FPS: {int(data['frameRate'][0])}")
    print("\nAll NPZ fields:")
    for key in data.files:
        print(f"  {key:28s} shape={str(data[key].shape):18s} dtype={data[key].dtype}")

    print("\nBody-motion ground truth (Xsens solved body segments):")
    print(f"  segment_tXYZ:  {segment_positions.shape}  # 23 x 3D positions")
    print(f"  segment_qWXYZ: {segment_orientations.shape}  # 23 x orientations")

    print("\nWearable IMU data (17 Xsens sensors):")
    print(f"  sensor_freeAcceleration: {sensor_accelerations.shape}  # 17 x 3D")
    print(f"  sensor_qWXYZ:            {sensor_orientations.shape}  # 17 x quaternion")

    head_segment = SEGMENT_NAMES.index("Head")
    head_sensor = SENSOR_NAMES.index("Head")
    print(f"\nExample frame {frame}, Head segment={head_segment}, Head sensor={head_sensor}:")
    print(f"  GT head position XYZ:       {segment_positions[frame, head_segment]}")
    print(f"  GT head orientation WXYZ:   {segment_orientations[frame, head_segment]}")
    print(f"  IMU head free acceleration: {sensor_accelerations[frame, head_sensor]}")
    print(f"  IMU head orientation WXYZ:  {sensor_orientations[frame, head_sensor]}")
    print(
        "  quaternion norms (GT, IMU): "
        f"{np.linalg.norm(segment_orientations[frame, head_segment]):.6f}, "
        f"{np.linalg.norm(sensor_orientations[frame, head_sensor]):.6f}"
    )


if __name__ == "__main__":
    main()
