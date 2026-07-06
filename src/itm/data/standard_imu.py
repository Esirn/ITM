"""Standard virtual IMU representation used by ITM and IMUPoser baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SENSOR_NAMES = (
    "left_wrist",
    "right_wrist",
    "left_thigh",
    "right_thigh",
    "head",
    "pelvis",
)
SENSOR_VERTEX_INDICES = (1961, 5424, 876, 4362, 411, 3021)
SENSOR_JOINT_INDICES = (18, 19, 1, 2, 15, 0)
MODEL_SENSOR_COUNT = 5


@dataclass(frozen=True)
class StandardIMU:
    acceleration: np.ndarray
    orientation: np.ndarray
    sensor_mask: np.ndarray
    fps: float


def synthesize_standard_imu(
    vertices: np.ndarray,
    global_rotations: np.ndarray,
    *,
    fps: float,
    sensor_slots: tuple[int, ...] | None = None,
    smooth_window: int = 5,
) -> StandardIMU:
    """Create six-slot virtual IMU from SMPL vertices and global rotations."""

    vertices = np.asarray(vertices, dtype=np.float32)
    rotations = np.asarray(global_rotations, dtype=np.float32)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"vertices must have shape (T, V, 3), got {vertices.shape}")
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError(f"global_rotations must have shape (T, J, 3, 3), got {rotations.shape}")
    if len(vertices) != len(rotations):
        raise ValueError("vertices and rotations must have equal frame counts")
    if vertices.shape[1] <= max(SENSOR_VERTEX_INDICES):
        raise ValueError("SMPL vertex array does not contain required sensor vertices")
    if rotations.shape[1] <= max(SENSOR_JOINT_INDICES):
        raise ValueError("SMPL rotations do not contain required sensor joints")
    if fps <= 0:
        raise ValueError("fps must be positive")

    positions = vertices[:, SENSOR_VERTEX_INDICES]
    acceleration = second_difference(positions, fps=fps)
    acceleration = moving_average(acceleration, smooth_window)
    orientation = rotations[:, SENSOR_JOINT_INDICES].copy()
    mask = sensor_mask(sensor_slots, sensor_count=len(SENSOR_NAMES))
    return StandardIMU(acceleration, orientation, mask, float(fps))


def second_difference(positions: np.ndarray, *, fps: float) -> np.ndarray:
    """Centered acceleration with zero-valued boundary frames."""

    values = np.asarray(positions, dtype=np.float32)
    result = np.zeros_like(values)
    if len(values) >= 3:
        result[1:-1] = (values[:-2] + values[2:] - 2.0 * values[1:-1]) * fps**2
    return result


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """NaN-free centered moving average preserving sequence length."""

    array = np.asarray(values, dtype=np.float32)
    if window <= 1 or len(array) == 0:
        return array.copy()
    if window % 2 == 0:
        raise ValueError("smooth_window must be odd")
    radius = window // 2
    padded = np.pad(array, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    return np.stack([padded[index : index + window].mean(0) for index in range(len(array))])


def sensor_mask(
    sensor_slots: tuple[int, ...] | None,
    *,
    sensor_count: int = MODEL_SENSOR_COUNT,
) -> np.ndarray:
    slots = tuple(range(sensor_count)) if sensor_slots is None else sensor_slots
    mask = np.zeros(sensor_count, dtype=bool)
    for slot in slots:
        if slot < 0 or slot >= sensor_count:
            raise IndexError(f"Sensor slot {slot} is outside [0, {sensor_count})")
        mask[slot] = True
    return mask


def imuposer_features(
    acceleration: np.ndarray,
    orientation: np.ndarray,
    mask: np.ndarray,
    *,
    acceleration_scale: float = 30.0,
) -> np.ndarray:
    """Mask and flatten five device slots to IMUPoser's 60D frame input."""

    acc = np.asarray(acceleration, dtype=np.float32)[:, :MODEL_SENSOR_COUNT].copy()
    ori = np.asarray(orientation, dtype=np.float32)[:, :MODEL_SENSOR_COUNT].copy()
    active = np.asarray(mask, dtype=bool)[:MODEL_SENSOR_COUNT]
    if acc.shape[1:] != (MODEL_SENSOR_COUNT, 3):
        raise ValueError(f"Expected acceleration (T, 5+, 3), got {acceleration.shape}")
    if ori.shape[1:] != (MODEL_SENSOR_COUNT, 3, 3):
        raise ValueError(f"Expected orientation (T, 5+, 3, 3), got {orientation.shape}")
    acc[:, ~active] = 0.0
    ori[:, ~active] = 0.0
    return np.concatenate((acc.reshape(len(acc), -1) / acceleration_scale, ori.reshape(len(ori), -1)), axis=1)


def humanml22_from_smpl24(joints: np.ndarray) -> np.ndarray:
    values = np.asarray(joints)
    if values.shape[-2:] != (24, 3):
        raise ValueError(f"Expected (..., 24, 3), got {values.shape}")
    return values[..., :22, :].copy()


def smpl24_from_humanml22(joints: np.ndarray) -> np.ndarray:
    """Pad HumanML joints to SMPL24; hand-tip joints copy their wrists."""

    values = np.asarray(joints)
    if values.shape[-2:] != (22, 3):
        raise ValueError(f"Expected (..., 22, 3), got {values.shape}")
    output = np.empty(values.shape[:-2] + (24, 3), dtype=values.dtype)
    output[..., :22, :] = values
    output[..., 22, :] = values[..., 20, :]
    output[..., 23, :] = values[..., 21, :]
    return output

