"""Nymeria narration, Xsens motion, and real-IMU conversion helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


NYMERIA_JOINT_NAMES = (
    "pelvis", "spine1", "spine2", "spine3", "spine4", "neck", "head",
    "right_collar", "right_shoulder", "right_elbow", "right_wrist",
    "left_collar", "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle", "right_foot",
    "left_hip", "left_knee", "left_ankle", "left_foot",
)
HUMANML_JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot",
    "right_foot", "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
)

# Xsens MVN sensor indices used by Nymeria/Ego4o.
NYMERIA_SENSOR_TO_STANDARD_SLOT = {9: 0, 5: 1, 2: 4}
MOTION_TEXT_COLUMNS = (
    "Describe my body posture",
    "Describe my hands/arms motion",
    "Describe my legs/feet motion",
)


def read_motion_narrations(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def narration_text(row: dict[str, str]) -> str:
    return " ".join(row.get(key, "").strip() for key in MOTION_TEXT_COLUMNS if row.get(key, "").strip())


def device_seconds_to_timecode_us(device_seconds: float, audit: dict) -> int:
    """Map Head DEVICE_TIME seconds to common TIME_CODE microseconds.

    Nymeria clocks share nanosecond rate, so one audited VRS conversion pair
    supplies the per-recording offset without reopening the large VRS file.
    """

    offset_ns = int(audit["csv_start_timecode_ns"]) - round(
        float(audit["csv_start_device_s"]) * 1e9
    )
    return round((float(device_seconds) * 1e9 + offset_ns) / 1000)


def quaternion_wxyz_to_matrix(values: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(values, dtype=np.float64)
    xyzw = np.concatenate((quaternions[..., 1:], quaternions[..., :1]), axis=-1)
    if np.any(np.linalg.norm(xyzw, axis=-1) < 0.05):
        raise ValueError("Nymeria clip contains invalid sensor quaternions")
    return Rotation.from_quat(xyzw.reshape(-1, 4)).as_matrix().reshape(
        quaternions.shape[:-1] + (3, 3)
    ).astype(np.float32)


def humanml22_from_nymeria23(joints: np.ndarray) -> np.ndarray:
    values = np.asarray(joints, dtype=np.float32)
    source = {name: index for index, name in enumerate(NYMERIA_JOINT_NAMES)}
    return np.stack([values[..., source[name], :] for name in HUMANML_JOINT_NAMES], axis=-2)


def convert_clip(joints23: np.ndarray, acceleration17: np.ndarray, orientation17_wxyz: np.ndarray):
    """Convert Nymeria Z-up arrays to the aligned ITM Y-up six-slot format."""

    joints23 = _reshape_channels(joints23, 23, 3, "segment_tXYZ")
    acceleration17 = _reshape_channels(
        acceleration17, 17, 3, "sensor_freeAcceleration"
    )
    orientation17_wxyz = _reshape_channels(
        orientation17_wxyz, 17, 4, "sensor_qWXYZ"
    )
    joints = humanml22_from_nymeria23(joints23)
    coordinate = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
    joints = np.einsum("ij,tkj->tki", coordinate, joints)

    across = (joints[0, 2] - joints[0, 1]) + (joints[0, 17] - joints[0, 16])
    across /= max(np.linalg.norm(across), 1e-8)
    forward = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), across)
    forward /= max(np.linalg.norm(forward), 1e-8)
    yaw = np.arctan2(forward[0], forward[2])
    align = Rotation.from_euler("y", -yaw).as_matrix().astype(np.float32)
    root_xz = joints[0, 0] * np.array([1, 0, 1], dtype=np.float32)
    joints = np.einsum("ij,tkj->tki", align, joints - root_xz)
    joints[..., 1] -= joints[..., 1].min()

    acceleration = np.zeros((len(joints), 6, 3), dtype=np.float32)
    orientation = np.tile(np.eye(3, dtype=np.float32), (len(joints), 6, 1, 1))
    for source_slot, target_slot in NYMERIA_SENSOR_TO_STANDARD_SLOT.items():
        matrices = quaternion_wxyz_to_matrix(orientation17_wxyz[:, source_slot])
        acceleration[:, target_slot] = np.einsum(
            "ij,tj->ti", align @ coordinate, acceleration17[:, source_slot]
        )
        orientation[:, target_slot] = np.einsum(
            "ij,tjk->tik", align @ coordinate, matrices
        )
    return joints.astype(np.float32), acceleration, orientation


def _reshape_channels(values: np.ndarray, count: int, width: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[1:] == (count, width):
        return array
    if array.ndim == 2 and array.shape[1] == count * width:
        return array.reshape(len(array), count, width)
    raise ValueError(f"{name} must have shape (T, {count}, {width}) or flattened equivalent; got {array.shape}")


def nearest_time_slice(timestamps_us: np.ndarray, start_us: int, end_us: int) -> slice:
    timestamps = np.asarray(timestamps_us, dtype=np.int64)
    start = int(np.searchsorted(timestamps, start_us, side="left"))
    end = int(np.searchsorted(timestamps, end_us, side="right"))
    if end - start < 3:
        raise ValueError("Aligned narration clip contains fewer than three Xsens frames")
    return slice(start, end)
