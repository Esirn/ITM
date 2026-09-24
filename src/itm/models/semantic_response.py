"""Single-step semantic response experiment, explicitly using joint proxies.

This is an experimental training objective, not certified sensor consistency.
It consumes an already paired x_t/t and training-only compatible edit witnesses.
"""
from __future__ import annotations

from itm.models.mdm_imu_control import evaluating_preserving_modes
from itm.models.torch_frame_baseline import require_torch


def arm_spread_attribute(joints, frame_mask):
    """Mean root-relative horizontal wrist separation in metres; shape [B,1]."""
    torch = require_torch()
    delta = joints[:, :, 20] - joints[:, :, 21]
    spread = torch.linalg.vector_norm(delta[..., [0, 2]], dim=-1)
    valid = frame_mask.to(spread.dtype)
    return ((torch.where(frame_mask, spread, 0) * valid).sum(1)
            / valid.sum(1).clamp_min(1))[:, None]


def relative_response_loss(student0, student1, teacher0, teacher1, *,
                           compatibility_weight, mode="bounded_margin", margin=0.02,
                           teacher_min_delta=0.01):
    """Teacher confidence threshold is a declared train/validation hyperparameter.

    A bounded directional margin never asks the student to reproduce an arbitrary
    teacher instance's geometry. Exact delta magnitude matching is an ablation.
    Rejected teacher attributes remain in diagnostics; this is not test filtering.
    """
    torch = require_torch()
    if mode not in {"bounded_margin", "exact_delta"}:
        raise ValueError("Unknown response objective")
    if margin <= 0 or teacher_min_delta <= 0:
        raise ValueError("Positive margin and teacher confidence threshold required")
    delta = student1 - student0
    target = (teacher1 - teacher0).detach()
    accepted = torch.isfinite(target) & (target.abs() >= teacher_min_delta)
    weight = torch.as_tensor(compatibility_weight, device=delta.device, dtype=delta.dtype)
    if weight.shape != (len(delta),) or not torch.isfinite(weight).all() or (weight < 0).any() or (weight > 1).any():
        raise ValueError("Compatibility weights must have shape (B,) within [0,1]")
    safe_target = torch.where(accepted, target, 0)
    if mode == "exact_delta":
        errors = (delta - safe_target).square()
    else:
        bound = safe_target.abs().clamp(max=margin)
        errors = torch.relu(bound - safe_target.sign() * delta)
    weighted = torch.where(accepted, errors, 0) * weight[:, None]
    loss = weighted.sum() / accepted.sum().clamp_min(1)
    return loss, {"accepted_attributes": int(accepted.sum()), "total_attributes": accepted.numel(),
                  "teacher_delta": target.detach(), "student_delta": delta.detach(),
                  "reason_rejected": "nonfinite_or_below_declared_teacher_min_delta"}


def semantic_response_step(model, x_t, timesteps, y, *, caption0, caption1, witnesses,
                           decode_joints, target_proxy_joints, compatibility_weight,
                           mode="bounded_margin", margin=0.02, teacher_min_delta=0.01,
                           use_compatibility_weight=True):
    """Return response + two observed-joint proxy losses (caller supplies weights).

    Original paired diffusion training must continue separately. No full-body GT
    loss is applied to either edited branch. All four forwards use the same x_t/t;
    teacher bypasses every adapter and does not backpropagate. Student dropout is
    disabled for this paired comparison only, with exact module-mode restoration.
    """
    torch = require_torch()
    if model.seqTransEncoder.null_mode != "strict":
        raise ValueError("Semantic response requires strict B0 null semantics")
    if len(witnesses) != len(x_t) or len(caption0) != len(x_t) or len(caption1) != len(x_t):
        raise ValueError("One caption pair and witness per sample required")
    for witness in witnesses:
        if (witness.get("split") != "train" or not witness.get("compatibility_evidence")
                or witness.get("attribute_spec") != "arm_spread_m"):
            raise ValueError("Requires train-only compatibility evidence and arm_spread_m attribute")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not (name.startswith("imu_control_encoder.") or name.startswith("seqTransEncoder.adapters.")):
            raise ValueError("Teacher backbone must be frozen")
    base = {key: value for key, value in y.items()
            if key not in {"text_embed", "imu_control", "imu_condition_present", "imu_uncond", "uncond"}}
    if not y["sensor_mask"].bool().any(-1).all() or not y["imu_frame_mask"].bool().any(-1).all():
        raise ValueError("Edit branches require observed sensors and valid frames")
    base.update(imu_condition_present=torch.ones(len(x_t), device=x_t.device, dtype=torch.bool),
                imu_uncond=False, uncond=False)
    student, teacher = [], []
    with evaluating_preserving_modes(model):
        for captions in (caption0, caption1):
            condition = dict(base, text=list(captions))
            student.append(decode_joints(model(x_t, timesteps, y=condition)))
            teacher_condition = {key: value for key, value in condition.items()
                                 if not key.startswith("imu") and key != "sensor_mask"}
            with torch.no_grad():
                teacher.append(decode_joints(model(x_t, timesteps, y=teacher_condition)))
    valid = y["imu_frame_mask"].bool()
    attributes = [arm_spread_attribute(joints, valid) for joints in (*student, *teacher)]
    if not use_compatibility_weight:
        compatibility_weight = x_t.new_ones(len(x_t))
    response, diagnostics = relative_response_loss(*attributes, compatibility_weight=compatibility_weight,
                                                   mode=mode, margin=margin, teacher_min_delta=teacher_min_delta)
    joint_mask = x_t.new_zeros((len(x_t), 22))
    for slot, joint in ((0, 20), (1, 21), (4, 15)):
        joint_mask[:, joint] = y["sensor_mask"][:, slot]
    weight = valid[:, :, None].to(x_t.dtype) * joint_mask[:, None]
    proxy = sum((((joints - target_proxy_joints.detach()).square().sum(-1) * weight).sum()
                 / weight.sum().clamp_min(1)) for joints in student) / 2
    diagnostics.update(student_forwards=2, teacher_forwards=2, supervision="global_active_joint_proxy",
                       timestep=timesteps.detach(), teacher_confidence_threshold=teacher_min_delta)
    return {"relative_response": response, "edited_observation_joint_proxy": proxy}, diagnostics
