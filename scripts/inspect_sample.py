#!/usr/bin/env python3
"""Inspect a manifest sample and synthesize sparse IMU proxy signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from itm.data.manifest import read_jsonl
from itm.data.synthetic_imu import (
    DEFAULT_CHILD_JOINT_INDICES,
    DEFAULT_PARENT_JOINT_INDICES,
    DEFAULT_SENSOR_JOINT_INDICES,
    synthesize_sparse_imu,
)


def _first_caption(text_path: str) -> str:
    path = Path(text_path)
    for line in path.read_text(errors="ignore").splitlines():
        if line.strip():
            return line.split("#", 1)[0]
    return ""


def _shape(value: object) -> tuple[int, ...]:
    if isinstance(value, np.ndarray):
        return value.shape
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    return (len(value),) + _shape(value[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    records = list(read_jsonl(args.manifest))
    if args.index < 0 or args.index >= len(records):
        raise IndexError(f"index {args.index} out of range for {len(records)} records")

    record = records[args.index]
    joints = np.load(record["joints_path"])
    joint_vec = np.load(record["joint_vec_path"])
    imu = synthesize_sparse_imu(
        joints.tolist(),
        DEFAULT_SENSOR_JOINT_INDICES,
        parent_joint_indices=DEFAULT_PARENT_JOINT_INDICES,
        child_joint_indices=DEFAULT_CHILD_JOINT_INDICES,
    )

    print(f"motion_id: {record['motion_id']}")
    print(f"split: {record['split']}")
    print(f"caption: {_first_caption(record['text_path'])}")
    print(f"joints_shape: {joints.shape}")
    print(f"joint_vec_shape: {joint_vec.shape}")
    print(f"sensor_joint_indices: {imu.sensor_joint_indices}")
    print(f"acceleration_shape: {_shape(imu.acceleration)}")
    print(f"orientation_shape: {_shape(imu.orientation_vectors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

