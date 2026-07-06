"""IMUPoser-style bidirectional LSTM with flexible sensor masking."""

from __future__ import annotations

from dataclasses import dataclass

from itm.models.torch_frame_baseline import require_torch


@dataclass(frozen=True)
class FlexibleIMUPoserConfig:
    input_dim: int = 60
    hidden_dim: int = 512
    num_layers: int = 2
    output_joints: int = 24
    dropout: float = 0.2
    bidirectional: bool = True


def make_flexible_imu_poser(config: FlexibleIMUPoserConfig):
    torch = require_torch()

    class FlexibleIMUPoser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_projection = torch.nn.Linear(config.input_dim, config.hidden_dim)
            self.dropout = torch.nn.Dropout(config.dropout)
            self.lstm = torch.nn.LSTM(
                config.hidden_dim,
                config.hidden_dim,
                config.num_layers,
                batch_first=True,
                bidirectional=config.bidirectional,
            )
            width = config.hidden_dim * (2 if config.bidirectional else 1)
            self.output_projection = torch.nn.Linear(width, config.output_joints * 6)

        def forward(self, features, lengths):
            hidden = torch.relu(self.input_projection(self.dropout(features)))
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                hidden, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=features.shape[1]
            )
            return self.output_projection(output).reshape(
                features.shape[0], features.shape[1], config.output_joints, 6
            )

    return FlexibleIMUPoser()


def rotation_matrix_to_r6d(rotation):
    """Use the first two rotation-matrix columns, matching IMUPoser."""

    return rotation[..., :, :2].swapaxes(-1, -2).reshape(rotation.shape[:-2] + (6,))


def r6d_to_rotation_matrix(r6d):
    torch = require_torch()
    first = torch.nn.functional.normalize(r6d[..., :3], dim=-1)
    second_raw = r6d[..., 3:]
    second = torch.nn.functional.normalize(
        second_raw - (first * second_raw).sum(-1, keepdim=True) * first,
        dim=-1,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def masked_pose_loss(prediction, target, mask, *, velocity_weight=0.1, acceleration_weight=0.01):
    """Pose loss with MobilePoser-inspired temporal regularization."""

    valid = mask[..., None, None].expand_as(target)
    pose = ((prediction - target) ** 2)[valid].mean()
    velocity = _temporal_loss(prediction, target, mask, order=1)
    acceleration = _temporal_loss(prediction, target, mask, order=2)
    return pose + velocity_weight * velocity + acceleration_weight * acceleration


def _temporal_loss(prediction, target, mask, order):
    torch = require_torch()
    pred_delta = torch.diff(prediction, n=order, dim=1)
    target_delta = torch.diff(target, n=order, dim=1)
    valid = mask
    for _ in range(order):
        valid = valid[:, 1:] & valid[:, :-1]
    if not torch.any(valid):
        return prediction.new_zeros(())
    expanded = valid[..., None, None].expand_as(pred_delta)
    return ((pred_delta - target_delta) ** 2)[expanded].mean()
