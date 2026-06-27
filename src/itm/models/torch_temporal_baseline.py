"""Masked temporal Transformer baseline for IMU-conditioned motion reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from itm.baselines.linear_reconstruct import LinearBaselineConfig, build_frame_features
from itm.models.torch_frame_baseline import require_torch


@dataclass(frozen=True)
class TemporalBaselineConfig:
    """Architecture and input settings for the temporal baseline."""

    include_acceleration: bool = True
    include_orientation: bool = True
    include_text: bool = True
    sensor_slots: tuple[int, ...] | None = None
    model_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    feedforward_dim: int = 512
    dropout: float = 0.1


def build_sequence_item(
    sample: dict[str, Any],
    text_embeddings: Mapping[str, np.ndarray] | None,
    config: TemporalBaselineConfig,
) -> dict[str, Any]:
    """Build aligned sequence features, text condition, and motion target."""

    feature_config = LinearBaselineConfig(
        include_text=False,
        include_acceleration=config.include_acceleration,
        include_orientation=config.include_orientation,
    )
    selected_sample = _select_sensor_slots(sample, config.sensor_slots)
    sequence = build_frame_features(selected_sample, feature_config)
    target = np.asarray(sample["motion"], dtype=np.float32)
    if sequence.shape[0] != target.shape[0]:
        raise ValueError(f"Frame mismatch for {sample.get('motion_id')}")

    motion_id = str(sample["motion_id"])
    if config.include_text:
        if text_embeddings is None or motion_id not in text_embeddings:
            raise KeyError(f"Missing text embedding for motion_id={motion_id}")
        text = np.asarray(text_embeddings[motion_id], dtype=np.float32)
        if text.ndim != 1:
            raise ValueError(f"Text embedding for {motion_id} must be one-dimensional")
    else:
        text = np.zeros(0, dtype=np.float32)
    return {"motion_id": motion_id, "sequence": sequence, "text": text, "target": target}


def _select_sensor_slots(
    sample: dict[str, Any], sensor_slots: tuple[int, ...] | None
) -> dict[str, Any]:
    if sensor_slots is None:
        return sample
    if not sensor_slots:
        raise ValueError("sensor_slots must contain at least one sensor")
    if len(set(sensor_slots)) != len(sensor_slots) or min(sensor_slots) < 0:
        raise ValueError(f"Invalid sensor_slots: {sensor_slots}")
    selected = dict(sample)
    for key in ("imu_acceleration", "imu_orientation"):
        if key not in sample:
            continue
        values = np.asarray(sample[key])
        if max(sensor_slots) >= values.shape[1]:
            raise IndexError(
                f"sensor slot {max(sensor_slots)} exceeds {key} width {values.shape[1]}"
            )
        selected[key] = values[:, sensor_slots, :]
    for key in (
        "sensor_joint_indices",
        "parent_joint_indices",
        "child_joint_indices",
    ):
        if key in sample:
            selected[key] = np.asarray(sample[key])[list(sensor_slots)]
    return selected


def collate_sequence_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad temporal items and return a valid-frame mask."""

    if not items:
        raise ValueError("At least one sequence item is required")
    lengths = np.asarray([len(item["sequence"]) for item in items], dtype=np.int64)
    max_length = int(lengths.max())
    input_dim = items[0]["sequence"].shape[1]
    output_dim = items[0]["target"].shape[1]
    sequences = np.zeros((len(items), max_length, input_dim), dtype=np.float32)
    targets = np.zeros((len(items), max_length, output_dim), dtype=np.float32)
    mask = np.zeros((len(items), max_length), dtype=bool)
    for index, item in enumerate(items):
        length = lengths[index]
        sequences[index, :length] = item["sequence"]
        targets[index, :length] = item["target"]
        mask[index, :length] = True
    return {
        "motion_id": [item["motion_id"] for item in items],
        "sequence": sequences,
        "text": np.stack([item["text"] for item in items]),
        "target": targets,
        "mask": mask,
        "lengths": lengths,
    }


def sinusoidal_position_encoding(length: int, dim: int, device, dtype):
    """Create deterministic sinusoidal positions for arbitrary sequence length."""

    torch = require_torch()
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    even_dim = (dim + 1) // 2
    scale = torch.exp(
        torch.arange(even_dim, device=device, dtype=dtype)
        * (-math.log(10000.0) / max(dim, 1))
        * 2.0
    )
    encoding = torch.zeros((length, dim), device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(position * scale[: encoding[:, 0::2].shape[1]])
    encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
    return encoding


def make_temporal_model(
    input_dim: int,
    text_dim: int,
    output_dim: int,
    config: TemporalBaselineConfig,
):
    """Build a Transformer encoder with a broadcast text condition."""

    torch = require_torch()

    class TemporalMotionTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_projection = torch.nn.Linear(input_dim, config.model_dim)
            self.text_projection = (
                torch.nn.Linear(text_dim, config.model_dim, bias=False)
                if config.include_text
                else None
            )
            layer = torch.nn.TransformerEncoderLayer(
                d_model=config.model_dim,
                nhead=config.num_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(layer, config.num_layers)
            self.output_norm = torch.nn.LayerNorm(config.model_dim)
            self.output_projection = torch.nn.Linear(config.model_dim, output_dim)

        def forward(self, sequence, text, mask):
            hidden = self.input_projection(sequence)
            positions = sinusoidal_position_encoding(
                hidden.shape[1], hidden.shape[2], hidden.device, hidden.dtype
            )
            hidden = hidden + positions.unsqueeze(0)
            if self.text_projection is not None:
                hidden = hidden + self.text_projection(text).unsqueeze(1)
            hidden = self.encoder(hidden, src_key_padding_mask=~mask)
            return self.output_projection(self.output_norm(hidden))

    return TemporalMotionTransformer()


def masked_mse(prediction, target, mask):
    """Mean squared error over valid frames and all output channels."""

    valid = mask.unsqueeze(-1).expand_as(target)
    return ((prediction - target) ** 2)[valid].mean()
