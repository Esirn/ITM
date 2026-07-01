"""HumanML3D representation recovery and skeleton metadata."""

from __future__ import annotations

import numpy as np


T2M_KINEMATIC_CHAIN = (
    (0, 2, 5, 8, 11),
    (0, 1, 4, 7, 10),
    (0, 3, 6, 9, 12, 15),
    (9, 14, 17, 19, 21),
    (9, 13, 16, 18, 20),
)


def recover_from_ric(features: np.ndarray, joints_num: int = 22) -> np.ndarray:
    """Recover global joints from raw HumanML root-invariant coordinates.

    This is a NumPy equivalent of the standard HumanML3D/MDM recovery routine.
    The input is unnormalized ``new_joint_vecs`` data with shape ``(..., T, 263)``.
    """

    data = np.asarray(features, dtype=np.float32)
    expected_minimum = 4 + (joints_num - 1) * 3
    if data.ndim < 2 or data.shape[-1] < expected_minimum:
        raise ValueError(
            f"Expected (..., T, D) with D >= {expected_minimum}, got {data.shape}"
        )

    rotation_velocity = data[..., 0]
    rotation_angle = np.zeros_like(rotation_velocity)
    rotation_angle[..., 1:] = rotation_velocity[..., :-1]
    rotation_angle = np.cumsum(rotation_angle, axis=-1)

    root_rotation = np.zeros(data.shape[:-1] + (4,), dtype=np.float32)
    root_rotation[..., 0] = np.cos(rotation_angle)
    root_rotation[..., 2] = np.sin(rotation_angle)

    root_position = np.zeros(data.shape[:-1] + (3,), dtype=np.float32)
    root_position[..., 1:, 0] = data[..., :-1, 1]
    root_position[..., 1:, 2] = data[..., :-1, 2]
    root_position = _quaternion_rotate(_quaternion_inverse(root_rotation), root_position)
    root_position = np.cumsum(root_position, axis=-2)
    root_position[..., 1] = data[..., 3]

    end = 4 + (joints_num - 1) * 3
    positions = data[..., 4:end].reshape(data.shape[:-1] + (joints_num - 1, 3))
    inverse_rotation = np.broadcast_to(
        _quaternion_inverse(root_rotation)[..., None, :], positions.shape[:-1] + (4,)
    )
    positions = _quaternion_rotate(inverse_rotation, positions)
    positions[..., 0] += root_position[..., 0, None]
    positions[..., 2] += root_position[..., 2, None]
    return np.concatenate((root_position[..., None, :], positions), axis=-2)


def _quaternion_inverse(quaternion: np.ndarray) -> np.ndarray:
    result = quaternion.copy()
    result[..., 1:] *= -1
    return result


def _quaternion_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q_vector = quaternion[..., 1:]
    uv = np.cross(q_vector, vector, axis=-1)
    uuv = np.cross(q_vector, uv, axis=-1)
    return vector + 2.0 * (quaternion[..., :1] * uv + uuv)
