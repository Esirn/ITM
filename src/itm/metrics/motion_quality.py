"""Pose accuracy and temporal stability metrics for IMU-only baselines."""

from __future__ import annotations

import numpy as np


def root_relative_mpjpe(prediction: np.ndarray, target: np.ndarray) -> float:
    pred, true = _joint_pair(prediction, target)
    pred = pred - pred[:, :1]
    true = true - true[:, :1]
    return float(np.linalg.norm(pred - true, axis=-1).mean())


def jerk_ratio(prediction: np.ndarray, target: np.ndarray, *, fps: float) -> float:
    pred, true = _joint_pair(prediction, target)
    pred_jerk = np.diff(pred, n=3, axis=0) * fps**3
    true_jerk = np.diff(true, n=3, axis=0) * fps**3
    numerator = np.linalg.norm(pred_jerk, axis=-1).mean()
    denominator = max(float(np.linalg.norm(true_jerk, axis=-1).mean()), 1e-8)
    return float(numerator / denominator)


def foot_skating(
    joints: np.ndarray,
    *,
    fps: float,
    foot_indices: tuple[int, int] = (10, 11),
    height_threshold: float = 0.05,
) -> float:
    values = np.asarray(joints, dtype=np.float32)
    feet = values[:, foot_indices]
    horizontal_speed = np.linalg.norm(np.diff(feet[..., (0, 2)], axis=0), axis=-1) * fps
    contact = feet[:-1, :, 1] < height_threshold
    return float(horizontal_speed[contact].mean()) if np.any(contact) else 0.0


def rotation_geodesic_error(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=np.float32)
    true = np.asarray(target, dtype=np.float32)
    relative = np.matmul(np.swapaxes(pred, -1, -2), true)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine).mean())


def _joint_pair(prediction, target):
    pred = np.asarray(prediction, dtype=np.float32)
    true = np.asarray(target, dtype=np.float32)
    if pred.shape != true.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"Expected matching (T, J, 3), got {pred.shape} and {true.shape}")
    return pred, true

