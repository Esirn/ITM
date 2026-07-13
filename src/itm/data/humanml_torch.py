"""Torch utilities for HumanML3D motion representations."""

from __future__ import annotations

from itm.models.torch_frame_baseline import require_torch


def recover_from_ric_torch(features, joints_num: int = 22):
    """Recover global joints from HumanML3D root-invariant coordinates.

    Args:
        features: Tensor with shape ``(..., T, 263)`` in unnormalized HumanML3D
            representation.
        joints_num: Number of HumanML joints to recover.

    Returns:
        Tensor with shape ``(..., T, joints_num, 3)``.
    """

    torch = require_torch()
    data = features.float()
    expected_minimum = 4 + (joints_num - 1) * 3
    if data.ndim < 2 or data.shape[-1] < expected_minimum:
        raise ValueError(
            f"Expected (..., T, D) with D >= {expected_minimum}, got {tuple(data.shape)}"
        )

    rotation_velocity = data[..., 0]
    rotation_angle = torch.zeros_like(rotation_velocity)
    rotation_angle[..., 1:] = rotation_velocity[..., :-1]
    rotation_angle = torch.cumsum(rotation_angle, dim=-1)

    root_rotation = torch.zeros(
        data.shape[:-1] + (4,), dtype=data.dtype, device=data.device
    )
    root_rotation[..., 0] = torch.cos(rotation_angle)
    root_rotation[..., 2] = torch.sin(rotation_angle)

    root_position = torch.zeros(
        data.shape[:-1] + (3,), dtype=data.dtype, device=data.device
    )
    root_position[..., 1:, 0] = data[..., :-1, 1]
    root_position[..., 1:, 2] = data[..., :-1, 2]
    root_position = _quaternion_rotate(_quaternion_inverse(root_rotation), root_position)
    root_position = torch.cumsum(root_position, dim=-2)
    root_position[..., 1] = data[..., 3]

    end = 4 + (joints_num - 1) * 3
    positions = data[..., 4:end].reshape(data.shape[:-1] + (joints_num - 1, 3))
    inverse_rotation = _quaternion_inverse(root_rotation)[..., None, :].expand(
        positions.shape[:-1] + (4,)
    )
    positions = _quaternion_rotate(inverse_rotation, positions)
    positions[..., 0] = positions[..., 0] + root_position[..., 0, None]
    positions[..., 2] = positions[..., 2] + root_position[..., 2, None]
    return torch.cat((root_position[..., None, :], positions), dim=-2)


def _quaternion_inverse(quaternion):
    result = quaternion.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def _quaternion_rotate(quaternion, vector):
    torch = require_torch()
    q_vector = quaternion[..., 1:]
    uv = torch.cross(q_vector, vector, dim=-1)
    uuv = torch.cross(q_vector, uv, dim=-1)
    return vector + 2.0 * (quaternion[..., :1] * uv + uuv)
