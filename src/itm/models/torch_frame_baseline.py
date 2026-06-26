"""Torch per-frame baseline for text-and-IMU motion reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from itm.baselines.linear_reconstruct import LinearBaselineConfig, build_frame_features


@dataclass(frozen=True)
class TorchFrameBaselineConfig:
    """Training-independent architecture settings."""

    feature_config: LinearBaselineConfig
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0


def require_torch():
    """Import torch with a clear optional-dependency error."""

    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "PyTorch is required for neural baselines. Install torch in the itm "
            "environment before running this script."
        ) from error
    return torch


def build_frame_arrays(
    samples: list[dict[str, Any]],
    feature_config: LinearBaselineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert samples into frame-level feature and target arrays."""

    features = []
    targets = []
    for sample in samples:
        x = build_frame_features(sample, feature_config)
        y = np.asarray(sample["motion"], dtype=np.float32)
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Feature/target frame mismatch for {sample.get('motion_id')}: "
                f"{x.shape[0]} != {y.shape[0]}"
            )
        features.append(x)
        targets.append(y)
    if not features:
        raise ValueError("At least one sample is required")
    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def make_mlp(
    input_dim: int,
    output_dim: int,
    config: TorchFrameBaselineConfig,
):
    """Build a simple MLP for frame-level regression."""

    torch = require_torch()
    layers = []
    current_dim = input_dim
    for _ in range(config.num_layers):
        layers.append(torch.nn.Linear(current_dim, config.hidden_dim))
        layers.append(torch.nn.ReLU())
        if config.dropout > 0:
            layers.append(torch.nn.Dropout(config.dropout))
        current_dim = config.hidden_dim
    layers.append(torch.nn.Linear(current_dim, output_dim))
    return torch.nn.Sequential(*layers)


def evaluate_mlp(model, x, y, *, batch_size: int = 4096) -> dict[str, float]:
    """Evaluate MSE and MAE for tensors already placed on the target device."""

    torch = require_torch()
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    total_values = 0
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            prediction = model(x[start:end])
            residual = prediction - y[start:end]
            squared_error += float(torch.sum(residual * residual).item())
            absolute_error += float(torch.sum(torch.abs(residual)).item())
            total_values += int(residual.numel())
    return {
        "num_frames": float(x.shape[0]),
        "mse": squared_error / total_values,
        "mae": absolute_error / total_values,
    }
