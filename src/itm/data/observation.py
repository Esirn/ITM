"""Versioned, explicit observation utilities; no learned IMU reconstruction.

The mesh operator assumes the caller supplies the very mesh/global rotations
used for rendering. Its analytic correctness does not certify a HumanML decoder.
Legacy cache values are never silently converted or overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from itm.data.standard_imu import SENSOR_JOINT_INDICES, SENSOR_VERTEX_INDICES


OPERATOR_VERSION = "smpl-vertex-observation-v1"
SENSOR_CONFIGS = {"head": (4,), "wrists": (0, 1)}


@dataclass(frozen=True)
class Observation:
    acceleration: np.ndarray
    orientation: np.ndarray
    acceleration_valid: np.ndarray
    orientation_valid: np.ndarray
    timestamps: np.ndarray
    metadata: dict


def uniform_timestamps(timestamps, count: int) -> tuple[np.ndarray, float]:
    times = np.asarray(timestamps, dtype=np.float64)
    if times.shape != (count,) or count < 3 or not np.isfinite(times).all():
        raise ValueError("at least three finite timestamps with matching length required")
    steps = np.diff(times)
    if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-5, atol=1e-9):
        raise ValueError("timestamps must be strictly increasing and uniform; resample explicitly")
    return times, 1.0 / float(steps[0])


def rotation_valid(rotations, tolerance=1e-4):
    rotations = np.asarray(rotations, dtype=np.float64)
    if rotations.shape[-2:] != (3, 3):
        raise ValueError("expected rotation matrices")
    finite = np.isfinite(rotations).all(axis=(-2, -1))
    safe = np.where(finite[..., None, None], rotations, np.eye(3))
    return (finite & (np.max(np.abs(safe @ safe.swapaxes(-1, -2) - np.eye(3)), axis=(-2, -1)) <= tolerance)
            & (np.abs(np.linalg.det(safe) - 1) <= tolerance))


def observe_sensor_positions(positions, orientation, timestamps, sensor_mask, *, smooth_window=5,
                             position_valid=None, orientation_valid=None) -> Observation:
    """Differentiate world positions in metres without bridging gaps or endpoints.

    Unlike legacy edge padding, invalid acceleration is NaN with an explicit mask.
    Interior values have the same centered-difference then box-smoothing definition.
    """
    positions = np.asarray(positions, dtype=np.float64)
    rotations = np.asarray(orientation, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (T, S, 3)")
    frames, sensors = positions.shape[:2]
    times, fps = uniform_timestamps(timestamps, frames)
    if rotations.shape != (frames, sensors, 3, 3):
        raise ValueError("orientation shape must match positions")
    active = np.asarray(sensor_mask)
    if active.dtype != np.bool_ or active.shape != (sensors,) or not active.any():
        raise ValueError("sensor_mask must be a nonempty boolean sensor vector")
    if not isinstance(smooth_window, int) or smooth_window < 1 or smooth_window % 2 != 1:
        raise ValueError("smooth_window must be a positive odd integer")
    valid = np.isfinite(positions).all(-1) & active[None]
    rot_valid = rotation_valid(rotations) & active[None]
    for mask, name in ((position_valid, "position_valid"), (orientation_valid, "orientation_valid")):
        if mask is not None and (np.asarray(mask).shape != (frames, sensors) or np.asarray(mask).dtype != np.bool_):
            raise ValueError(f"{name} must be a (T, S) boolean mask")
    if position_valid is not None:
        valid &= position_valid
    if orientation_valid is not None:
        rot_valid &= orientation_valid
    acceleration = np.full_like(positions, np.nan)
    raw = np.full_like(positions, np.nan)
    raw[1:-1] = (positions[:-2] + positions[2:] - 2 * positions[1:-1]) * fps**2
    stencil = np.zeros_like(valid)
    stencil[1:-1] = valid[:-2] & valid[1:-1] & valid[2:]
    stencil &= np.isfinite(raw).all(-1)
    acc_valid = np.zeros_like(valid)
    radius = smooth_window // 2
    for t in range(radius + 1, frames - radius - 1):
        good = stencil[t-radius:t+radius+1].all(0)
        acc_valid[t] = good
        acceleration[t, good] = raw[t-radius:t+radius+1, good].mean(0)
    rotations = np.where(rot_valid[..., None, None], rotations, np.nan)
    return Observation(acceleration, rotations, acc_valid, rot_valid, times, {
        "operator_version": OPERATOR_VERSION,
        "quantity": "world_linear_acceleration_and_global_orientation",
        "acceleration_unit": "m_per_s2", "gravity": "not_added",
        "smooth_window_frames": smooth_window, "smooth_window_seconds": smooth_window / fps,
        "boundary": "invalid_full_difference_and_smoothing_stencil", "fps": fps,
    })


def observe_mesh(vertices, global_rotations, timestamps, sensor_mask, **kwargs):
    vertices = np.asarray(vertices)
    rotations = np.asarray(global_rotations)
    if vertices.ndim != 3 or vertices.shape[1] <= max(SENSOR_VERTEX_INDICES):
        raise ValueError("required SMPL sensor vertices are missing")
    if rotations.ndim != 4 or rotations.shape[1] <= max(SENSOR_JOINT_INDICES):
        raise ValueError("required global SMPL segment rotations are missing")
    return observe_sensor_positions(vertices[:, SENSOR_VERTEX_INDICES], rotations[:, SENSOR_JOINT_INDICES],
                                    timestamps, sensor_mask, **kwargs)


def device_acceleration_to_world(values, body_to_world, *, quantity, gravity_world=None):
    """Explicit bridge: specific force needs g; linear acceleration must not add g."""
    values = np.asarray(values, dtype=np.float64)
    rotations = np.asarray(body_to_world, dtype=np.float64)
    if values.shape[-1:] != (3,) or rotations.shape != (*values.shape[:-1], 3, 3):
        raise ValueError("acceleration and rotation shapes disagree")
    if not np.isfinite(values).all() or not rotation_valid(rotations).all():
        raise ValueError("finite acceleration and proper body-to-world rotations required")
    output = (rotations @ values[..., None])[..., 0]
    if quantity == "body_specific_force":
        gravity = np.asarray(gravity_world, dtype=np.float64)
        if gravity.shape != (3,) or not np.isfinite(gravity).all():
            raise ValueError("specific force requires explicit finite gravity_world")
        return output + gravity
    if quantity != "body_linear_acceleration" or gravity_world is not None:
        raise ValueError("linear acceleration forbids adding gravity; unknown quantity rejected")
    return output


def load_standard_cache(record: dict, *, expected_fps: float | None = None) -> dict:
    """Read declared standard_imu_v1 only; shape alone never infers the schema.

    Legacy records lack frame validity and physical provenance. This reader checks
    stored values, exposes that limitation, and does not certify their synthesis.
    """
    if record.get("schema") != "standard_imu_v1":
        raise ValueError("explicit standard_imu_v1 manifest required; legacy schemas are separate")
    for key, value in {"quantity": "world_linear_acceleration_and_global_orientation",
                       "acceleration_unit": "m_per_s2", "gravity_convention": "not_added",
                       "rotation_convention": "segment_to_world_column_vectors"}.items():
        if key in record and record[key] != value:
            raise ValueError(f"incompatible standard cache physical metadata: {key}")
    with np.load(Path(record["imu_path"]), allow_pickle=False) as data:
        required = {"acceleration", "orientation", "fps", "motion_id", "sensor_mask"}
        if not required.issubset(data.files):
            raise ValueError("missing standard cache fields")
        arrays = {key: data[key].copy() for key in required}
    fps = float(arrays["fps"])
    frames = int(record["num_frames"])
    if frames < 3 or not np.isfinite(fps) or fps <= 0 or fps != float(record["fps"]):
        raise ValueError("invalid or mismatched FPS/frame count")
    if expected_fps is not None and fps != expected_fps:
        raise ValueError("cache FPS differs; explicit resampling required")
    if str(arrays["motion_id"].item()) != str(record["motion_id"]):
        raise ValueError("cache motion ID mismatch")
    if arrays["acceleration"].shape != (frames, 6, 3) or arrays["orientation"].shape != (frames, 6, 3, 3):
        raise ValueError("invalid standard cache shapes")
    active = arrays["sensor_mask"]
    if active.shape != (6,) or active.dtype != np.bool_ or not active.any():
        raise ValueError("invalid sensor mask")
    if not np.isfinite(arrays["acceleration"][:, active]).all() or not rotation_valid(arrays["orientation"][:, active]).all():
        raise ValueError("invalid active acceleration or orientation")
    arrays["timestamps"] = np.arange(frames, dtype=np.float64) / fps
    arrays["provenance_status"] = "declared_legacy_schema_not_synthesis_certified"
    return arrays
