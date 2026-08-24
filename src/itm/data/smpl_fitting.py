"""Differentiable fitting of HumanML joints to a neutral SMPL body."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from itm.models.torch_frame_baseline import require_torch


@dataclass
class SMPLFitResult:
    joints: np.ndarray
    vertices: np.ndarray
    local_rotations: np.ndarray
    global_rotations: np.ndarray
    translation: np.ndarray
    scale: float
    final_loss: float


def load_neutral_smpl(model_path, *, batch_size, device):
    """Load legacy SMPL pickle through smplx without modifying the asset."""

    _enable_legacy_numpy_aliases()
    import smplx

    return smplx.SMPL(
        str(Path(model_path)),
        batch_size=batch_size,
        create_betas=False,
        create_global_orient=False,
        create_body_pose=False,
        create_transl=False,
    ).to(device)


def fit_humanml_joints_to_smpl(
    target_joints,
    model,
    *,
    iterations=200,
    learning_rate=0.05,
    pose_smoothness=0.02,
    translation_smoothness=0.05,
):
    """Fit one complete joint sequence while keeping neutral body shape fixed."""

    torch = require_torch()
    target = torch.as_tensor(target_joints, dtype=torch.float32, device=model.v_template.device)
    if target.ndim != 3 or target.shape[1:] != (22, 3):
        raise ValueError(f"target_joints must have shape (T, 22, 3), got {tuple(target.shape)}")
    frames = len(target)
    if frames < 3:
        raise ValueError("SMPL fitting requires at least three frames")
    global_orient = torch.zeros(frames, 3, device=target.device, requires_grad=True)
    body_pose = torch.zeros(frames, 69, device=target.device, requires_grad=True)
    translation = target[:, 0].detach().clone().requires_grad_(True)
    log_scale = torch.zeros((), device=target.device, requires_grad=True)
    betas = torch.zeros(frames, 10, device=target.device)
    optimizer = torch.optim.Adam(
        (global_orient, body_pose, translation, log_scale), lr=learning_rate
    )
    final_loss = None
    for _ in range(iterations):
        output = model(
            betas=betas, global_orient=global_orient, body_pose=body_pose
        )
        scale = log_scale.exp()
        joints = output.joints[:, :24] * scale + translation[:, None]
        relative = joints[:, :22] - joints[:, :1]
        target_relative = target - target[:, :1]
        joint_loss = (relative - target_relative).pow(2).mean()
        root_loss = (joints[:, 0] - target[:, 0]).pow(2).mean()
        smooth_pose = (body_pose[1:] - body_pose[:-1]).pow(2).mean()
        smooth_translation = (translation[1:] - translation[:-1]).pow(2).mean()
        pose_prior = body_pose.pow(2).mean()
        loss = (
            joint_loss
            + root_loss
            + pose_smoothness * smooth_pose
            + translation_smoothness * smooth_translation
            + 1e-4 * pose_prior
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = loss.detach()
    with torch.inference_mode():
        output = model(
            betas=betas, global_orient=global_orient, body_pose=body_pose
        )
        scale = log_scale.exp()
        joints = output.joints[:, :24] * scale + translation[:, None]
        vertices = output.vertices * scale + translation[:, None]
        axis_angle = torch.cat((global_orient[:, None], body_pose.view(frames, 23, 3)), dim=1)
        local = axis_angle_to_matrix(axis_angle)
        global_rotation = local_to_global_rotations(local, model.parents[:24])
    return SMPLFitResult(
        joints=joints.cpu().numpy().astype(np.float32),
        vertices=vertices.cpu().numpy().astype(np.float32),
        local_rotations=local.cpu().numpy().astype(np.float32),
        global_rotations=global_rotation.cpu().numpy().astype(np.float32),
        translation=translation.detach().cpu().numpy().astype(np.float32),
        scale=float(scale),
        final_loss=float(final_loss),
    )


def axis_angle_to_matrix(axis_angle):
    torch = require_torch()
    from smplx.lbs import batch_rodrigues

    shape = axis_angle.shape
    return batch_rodrigues(axis_angle.reshape(-1, 3)).reshape(shape[:-1] + (3, 3))


def local_to_global_rotations(local_rotations, parents):
    """Compose SMPL local rotations along the kinematic tree."""

    torch = require_torch()
    parents = torch.as_tensor(parents, device=local_rotations.device, dtype=torch.long)
    values = [local_rotations[:, 0]]
    for joint in range(1, local_rotations.shape[1]):
        values.append(values[int(parents[joint])] @ local_rotations[:, joint])
    return torch.stack(values, dim=1)


def _enable_legacy_numpy_aliases():
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
