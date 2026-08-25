"""IMU-to-trajectory control adapter for a frozen MotionLab backbone."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from itm.models.torch_frame_baseline import require_torch


@dataclass(frozen=True)
class MotionLabIMUAdapterConfig:
    sensor_feature_dim: int = 12
    sensor_count: int = 6
    hidden_dim: int = 256
    encoder_layers: int = 2
    encoder_heads: int = 8
    dropout: float = 0.1
    output_dim: int = 66
    max_frames: int = 196
    sensor_fusion: str = "mean"
    sensor_encoder_layers: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def make_motionlab_imu_adapter(config: MotionLabIMUAdapterConfig):
    torch = require_torch()

    class MotionLabIMUAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if config.sensor_fusion not in {"mean", "concat", "attention"}:
                raise ValueError(f"unknown sensor_fusion: {config.sensor_fusion}")
            self.feature_projection = torch.nn.Linear(
                config.sensor_feature_dim, config.hidden_dim
            )
            self.sensor_embedding = torch.nn.Embedding(
                config.sensor_count, config.hidden_dim
            )
            self.sensor_fusion = (
                torch.nn.Linear(config.sensor_count * config.hidden_dim, config.hidden_dim)
                if config.sensor_fusion == "concat"
                else None
            )
            if config.sensor_fusion == "attention":
                sensor_layer = torch.nn.TransformerEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.encoder_heads,
                    dim_feedforward=config.hidden_dim * 2,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.sensor_encoder = torch.nn.TransformerEncoder(
                    sensor_layer, config.sensor_encoder_layers
                )
            else:
                self.sensor_encoder = None
            layer = torch.nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.encoder_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal_encoder = torch.nn.TransformerEncoder(
                layer, config.encoder_layers
            )
            self.output_projection = torch.nn.Linear(config.hidden_dim, config.output_dim)
            self.register_buffer(
                "position_encoding",
                _sinusoidal_encoding(torch, config.max_frames, config.hidden_dim),
                persistent=False,
            )

        def forward(self, imu, sensor_mask, frame_mask=None):
            if imu.ndim != 4:
                raise ValueError("imu must have shape [B,T,S,12]")
            batch, frames, sensors, features = imu.shape
            if sensors != config.sensor_count or features != config.sensor_feature_dim:
                raise ValueError(
                    f"expected S,D={config.sensor_count},{config.sensor_feature_dim}; "
                    f"got {sensors},{features}"
                )
            if frames > config.max_frames:
                raise ValueError(f"frame count {frames} exceeds {config.max_frames}")
            if sensor_mask.shape != (batch, sensors):
                raise ValueError("sensor_mask must have shape [B,S]")

            sensor_ids = torch.arange(sensors, device=imu.device)
            hidden = self.feature_projection(imu)
            hidden = hidden + self.sensor_embedding(sensor_ids)[None, None]
            active = sensor_mask[:, None, :, None].to(hidden.dtype)
            hidden = hidden * active
            if self.sensor_encoder is not None:
                flat = hidden.reshape(batch * frames, sensors, config.hidden_dim)
                padding = (~sensor_mask[:, None].expand(-1, frames, -1)).reshape(
                    batch * frames, sensors
                )
                flat = self.sensor_encoder(flat, src_key_padding_mask=padding)
                flat_active = (~padding)[:, :, None].to(flat.dtype)
                hidden = (flat * flat_active).sum(1) / flat_active.sum(1).clamp_min(1.0)
                hidden = hidden.reshape(batch, frames, config.hidden_dim)
            elif self.sensor_fusion is None:
                hidden = hidden.sum(2) / active.sum(2).clamp_min(1.0)
            else:
                hidden = self.sensor_fusion(hidden.flatten(2))
            hidden = hidden + self.position_encoding[:frames].to(hidden)[None]
            hidden = self.temporal_encoder(
                hidden,
                src_key_padding_mask=(~frame_mask if frame_mask is not None else None),
            )
            output = self.output_projection(hidden)
            if frame_mask is not None:
                output = output * frame_mask[:, :, None].to(output.dtype)
            return output

    return MotionLabIMUAdapter()


def active_joint_mask(sensor_mask, *, output_joints: int = 22):
    """Map standard IMU slots to the HumanML joints controlled by MotionLab."""
    torch = require_torch()
    if sensor_mask.ndim != 2 or sensor_mask.shape[1] != 6:
        raise ValueError("sensor_mask must have shape [B,6]")
    result = torch.zeros(
        sensor_mask.shape[0], output_joints, 3, dtype=torch.bool, device=sensor_mask.device
    )
    for slot, joint in ((0, 20), (1, 21), (4, 15)):
        result[:, joint, :] |= sensor_mask[:, slot, None].bool()
    return result


def masked_trajectory_losses(predicted, target, frame_mask, joint_mask):
    """Position and first-difference losses on active sensor joints only."""
    torch = require_torch()
    if predicted.shape != target.shape or predicted.shape[-1] != 66:
        raise ValueError("predicted and target must have equal shape [B,T,66]")
    batch, frames, _ = predicted.shape
    pred = predicted.view(batch, frames, 22, 3)
    truth = target.view(batch, frames, 22, 3)
    weight = frame_mask[:, :, None, None] & joint_mask[:, None]
    denominator = weight.sum().clamp_min(1).to(pred.dtype)
    trajectory = (((pred - truth) ** 2) * weight).sum() / denominator
    if frames < 2:
        return trajectory, pred.new_zeros(())
    velocity_weight = weight[:, 1:] & weight[:, :-1]
    velocity_denominator = velocity_weight.sum().clamp_min(1).to(pred.dtype)
    error = pred - truth
    velocity = (
        ((error[:, 1:] - error[:, :-1]) ** 2)
        * velocity_weight
    ).sum() / velocity_denominator
    return trajectory, velocity


def factorized_guidance(f00, f10, f01, f11, text_scale, imu_scale, joint_scale=1.0):
    """Combine independent text, IMU, and text-IMU interaction effects."""
    return (
        f00
        + text_scale * (f10 - f00)
        + imu_scale * (f01 - f00)
        + joint_scale * (f11 - f10 - f01 + f00)
    )


def paired_control_ranking_loss(paired_error, negative_error, margin=0.01):
    """Require paired IMU to explain its motion better than a negative IMU."""
    torch = require_torch()
    return torch.relu(paired_error - negative_error + margin)


def same_group_derangement(groups, device=None):
    """Return a within-group cyclic permutation and a mask for valid negatives."""
    torch = require_torch()
    permutation = torch.arange(len(groups), device=device)
    valid = torch.zeros(len(groups), dtype=torch.bool, device=device)
    for group in sorted(set(groups)):
        indices = [index for index, value in enumerate(groups) if value == group]
        if len(indices) < 2:
            continue
        source = torch.as_tensor(indices, device=device)
        permutation[source] = torch.roll(source, 1)
        valid[source] = True
    return permutation, valid


def _sinusoidal_encoding(torch, frames: int, dimension: int):
    position = torch.arange(frames, dtype=torch.float32)[:, None]
    divisor = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32)
        * (-math.log(10000.0) / dimension)
    )
    encoding = torch.zeros(frames, dimension)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor)
    return encoding
