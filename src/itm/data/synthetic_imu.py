"""Synthetic sparse IMU extraction from joint trajectories.

This module uses the same simple proxy used by many early IMU-mocap baselines:
sensor acceleration is the second difference of selected joint positions, and
sensor orientation is represented by selected limb direction vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from itm.metrics.imu_consistency import limb_orientation_vectors, second_difference_acceleration


DEFAULT_SENSOR_JOINT_INDICES = [0, 7, 8, 15, 20, 21]
DEFAULT_PARENT_JOINT_INDICES = [0, 0, 0, 12, 18, 19]
DEFAULT_CHILD_JOINT_INDICES = [1, 4, 5, 15, 20, 21]


@dataclass(frozen=True)
class SyntheticIMU:
    """Sparse IMU proxy signals aligned to selected joints/limbs."""

    acceleration: list
    orientation_vectors: Optional[list]
    sensor_joint_indices: list[int]
    parent_joint_indices: Optional[list[int]]
    child_joint_indices: Optional[list[int]]


def _select_joints(joints: list, indices: list[int]) -> list:
    selected = []
    for frame in joints:
        selected.append([frame[index] for index in indices])
    return selected


def synthesize_sparse_imu(
    joints: list,
    sensor_joint_indices: list[int] | None = None,
    *,
    parent_joint_indices: Optional[list[int]] = None,
    child_joint_indices: Optional[list[int]] = None,
    dt: float = 1.0,
) -> SyntheticIMU:
    """Create sparse IMU proxy signals from joint positions.

    Args:
        joints: Joint positions shaped ``[T, J, 3]``.
        sensor_joint_indices: Joints used as acceleration sensor locations.
        parent_joint_indices: Parent joints for orientation vectors.
        child_joint_indices: Child joints for orientation vectors.
        dt: Frame interval for acceleration estimation.

    Returns:
        Synthetic IMU acceleration shaped ``[T - 2, S, 3]`` and optional
        orientation vectors shaped ``[T, S, 3]``.
    """

    if sensor_joint_indices is None:
        sensor_joint_indices = DEFAULT_SENSOR_JOINT_INDICES
    sensor_joints = _select_joints(joints, sensor_joint_indices)
    acceleration = second_difference_acceleration(sensor_joints, dt=dt)

    orientation_vectors = None
    if parent_joint_indices is not None or child_joint_indices is not None:
        if parent_joint_indices is None or child_joint_indices is None:
            raise ValueError("parent_joint_indices and child_joint_indices must be provided together")
        orientation_vectors = limb_orientation_vectors(joints, parent_joint_indices, child_joint_indices)

    return SyntheticIMU(
        acceleration=acceleration,
        orientation_vectors=orientation_vectors,
        sensor_joint_indices=[int(index) for index in sensor_joint_indices],
        parent_joint_indices=(
            [int(index) for index in parent_joint_indices]
            if parent_joint_indices is not None
            else None
        ),
        child_joint_indices=(
            [int(index) for index in child_joint_indices]
            if child_joint_indices is not None
            else None
        ),
    )
