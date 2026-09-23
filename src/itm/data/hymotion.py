"""HY-Motion rotation/FK helpers and joint-mounted IMU conventions."""

from __future__ import annotations

import numpy as np

from itm.data.standard_imu import moving_average, second_difference


BODY_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
SENSOR_JOINTS = (20, 21, 1, 2, 15, 0)
SENSOR_CONFIGS = {"head": (4,), "wrists": (0, 1)}


def rotation_6d_to_matrix(value):
    rotation = np.asarray(value, dtype=np.float32)
    columns = rotation.reshape(rotation.shape[:-1] + (3, 2))
    first, second = columns[..., 0], columns[..., 1]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def forward_kinematics(local_rotation, translation, template, parents=BODY_PARENTS):
    """Match WoodenMesh joint FK, including the supplied root translation."""
    local = np.asarray(local_rotation, dtype=np.float32)
    template = np.asarray(template, dtype=np.float32)
    translation = np.asarray(translation, dtype=np.float32)
    count = len(parents)
    if local.shape[-3:] != (count, 3, 3) or template.shape != (count, 3):
        raise ValueError("rotation/template dimensions do not match the skeleton")
    if translation.shape != local.shape[:-3] + (3,):
        raise ValueError("translation must match the rotation batch/frame dimensions")
    global_rotation = np.empty_like(local)
    positions = np.empty(local.shape[:-1], dtype=np.float32)
    global_rotation[..., 0, :, :] = local[..., 0, :, :]
    positions[..., 0, :] = template[0] + translation
    for joint in range(1, count):
        parent = int(parents[joint])
        if not 0 <= parent < joint:
            raise ValueError("parents must precede their children")
        global_rotation[..., joint, :, :] = global_rotation[..., parent, :, :] @ local[..., joint, :, :]
        offset = np.einsum("...ij,j->...i", global_rotation[..., parent, :, :], template[joint] - template[parent])
        positions[..., joint, :] = positions[..., parent, :] + offset
    return positions, global_rotation


def joint_imu_features(joints, global_rotation, fps=30.0):
    """Six slots of acceleration + row-flattened global rotation; no gravity."""
    acceleration = moving_average(second_difference(joints[:, SENSOR_JOINTS], fps=fps), 5)
    orientation = global_rotation[:, SENSOR_JOINTS]
    return np.concatenate((acceleration, orientation.reshape(len(joints), 6, 9)), axis=-1).astype(np.float32)


def joint_anchor_features(latent, imu):
    """Low-derivative control features: root-relative sensor positions + orientation."""
    latent = np.asarray(latent, dtype=np.float32)
    imu = np.asarray(imu, dtype=np.float32)
    if latent.ndim != 2 or latent.shape[1] != 201 or imu.shape != (len(latent), 6, 12):
        raise ValueError("expected latent [T,201] and IMU [T,6,12]")
    positions = latent[:, 135:].reshape(len(latent), 22, 3)[:, SENSOR_JOINTS]
    return np.concatenate((positions, imu[..., 3:]), axis=-1).astype(np.float32)


def world_joints_from_result(result, template, parents):
    """Reconstruct 52 world joints; the official result omits root translation."""
    rotations = rotation_6d_to_matrix(result["rot6d"])
    count = len(parents)
    if rotations.shape[-3] < count:
        fingers = np.broadcast_to(np.eye(3), rotations.shape[:-3] + (count - rotations.shape[-3], 3, 3))
        rotations = np.concatenate((rotations, fingers), axis=-3)
    return forward_kinematics(rotations, result["transl"], template, parents)[0]
