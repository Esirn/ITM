"""Linear text-and-IMU baseline for HumanML3D joint-vector reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class LinearBaselineConfig:
    """Feature and solver settings for the linear baseline."""

    text_dim: int = 128
    ridge_alpha: float = 1e-2
    include_text: bool = True
    include_acceleration: bool = True
    include_orientation: bool = True


def text_hash_features(text: str, dim: int) -> np.ndarray:
    """Stable hashed bag-of-words text features."""

    features = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little", signed=False)
        index = value % dim
        sign = 1.0 if ((value >> 8) & 1) == 0 else -1.0
        features[index] += sign
    norm = np.linalg.norm(features)
    if norm > 0:
        features /= norm
    return features


def build_frame_features(
    sample: dict[str, Any],
    config: LinearBaselineConfig,
) -> np.ndarray:
    """Build per-frame features aligned with ``sample["motion"]``."""

    motion = np.asarray(sample["motion"], dtype=np.float32)
    num_frames = motion.shape[0]
    time = _time_features(num_frames)
    parts = [np.ones((num_frames, 1), dtype=np.float32), time]

    if config.include_text:
        text = np.tile(
            text_hash_features(str(sample.get("caption", "")), config.text_dim),
            (num_frames, 1),
        )
        parts.append(text)

    acceleration = None
    if config.include_acceleration:
        acceleration = _align_acceleration(
            np.asarray(sample["imu_acceleration"], dtype=np.float32),
            num_frames,
        )
        parts.append(acceleration)

    if config.include_orientation:
        orientation = sample.get("imu_orientation")
        if orientation is None:
            flat_dim = int(np.prod(np.asarray(sample["imu_acceleration"]).shape[1:]))
            orientation_flat = np.zeros((num_frames, flat_dim), dtype=np.float32)
        else:
            orientation_flat = _align_orientation(
                np.asarray(orientation, dtype=np.float32),
                num_frames,
            )
        parts.append(orientation_flat)

    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def fit_linear_baseline(
    samples: Iterable[dict[str, Any]],
    config: LinearBaselineConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit ridge regression from frame features to joint-vector frames."""

    xtx: np.ndarray | None = None
    xty: np.ndarray | None = None
    total_loss = 0.0
    total_frames = 0
    num_records = 0

    for sample in samples:
        x = build_frame_features(sample, config).astype(np.float64, copy=False)
        y = np.asarray(sample["motion"], dtype=np.float64)
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Feature/target frame mismatch for {sample.get('motion_id')}: "
                f"{x.shape[0]} != {y.shape[0]}"
            )
        xtx = x.T @ x if xtx is None else xtx + x.T @ x
        xty = x.T @ y if xty is None else xty + x.T @ y
        total_loss += float(np.sum(y * y))
        total_frames += int(y.shape[0])
        num_records += 1

    if xtx is None or xty is None:
        raise ValueError("fit_linear_baseline requires at least one sample")

    regularizer = np.eye(xtx.shape[0], dtype=np.float64) * float(config.ridge_alpha)
    regularizer[0, 0] = 0.0
    try:
        weights = np.linalg.solve(xtx + regularizer, xty)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(xtx + regularizer) @ xty

    stats = {
        "num_records": float(num_records),
        "num_frames": float(total_frames),
        "target_energy_mean": total_loss / max(total_frames, 1),
        "feature_dim": float(weights.shape[0]),
        "target_dim": float(weights.shape[1]),
    }
    return weights.astype(np.float32), stats


def predict_motion(
    sample: dict[str, Any],
    weights: np.ndarray,
    config: LinearBaselineConfig,
) -> np.ndarray:
    """Predict joint-vector frames for one sample."""

    return build_frame_features(sample, config) @ weights


def evaluate_linear_baseline(
    samples: Iterable[dict[str, Any]],
    weights: np.ndarray,
    config: LinearBaselineConfig,
) -> dict[str, float]:
    """Evaluate mean squared and absolute error over samples."""

    squared_error = 0.0
    absolute_error = 0.0
    total_values = 0
    total_frames = 0
    num_records = 0
    for sample in samples:
        target = np.asarray(sample["motion"], dtype=np.float32)
        prediction = predict_motion(sample, weights, config).astype(np.float32)
        residual = prediction - target
        squared_error += float(np.sum(residual * residual))
        absolute_error += float(np.sum(np.abs(residual)))
        total_values += int(residual.size)
        total_frames += int(target.shape[0])
        num_records += 1

    if total_values == 0:
        raise ValueError("evaluate_linear_baseline requires non-empty targets")
    return {
        "num_records": float(num_records),
        "num_frames": float(total_frames),
        "mse": squared_error / total_values,
        "mae": absolute_error / total_values,
    }


def save_linear_baseline(
    output_path: str | Path,
    weights: np.ndarray,
    config: LinearBaselineConfig,
    *,
    train_stats: dict[str, float] | None = None,
    eval_stats: dict[str, float] | None = None,
) -> None:
    """Save weights and metadata to a compressed npz file."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": asdict(config),
        "train_stats": train_stats or {},
        "eval_stats": eval_stats or {},
    }
    np.savez_compressed(
        output,
        weights=np.asarray(weights, dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )


def load_linear_baseline(
    model_path: str | Path,
) -> tuple[np.ndarray, LinearBaselineConfig, dict[str, Any]]:
    """Load a saved linear baseline."""

    with np.load(model_path) as data:
        weights = data["weights"].astype(np.float32, copy=False)
        metadata = json.loads(str(data["metadata"]))
    config = LinearBaselineConfig(**metadata["config"])
    return weights, config, metadata


def _time_features(num_frames: int) -> np.ndarray:
    if num_frames <= 1:
        progress = np.zeros((num_frames, 1), dtype=np.float32)
    else:
        progress = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)[:, None]
    return np.concatenate(
        [
            progress,
            np.sin(2.0 * np.pi * progress).astype(np.float32),
            np.cos(2.0 * np.pi * progress).astype(np.float32),
        ],
        axis=1,
    )


def _align_acceleration(acceleration: np.ndarray, num_frames: int) -> np.ndarray:
    flat_dim = int(np.prod(acceleration.shape[1:]))
    aligned = np.zeros((num_frames, flat_dim), dtype=np.float32)
    if num_frames <= 0 or acceleration.shape[0] == 0:
        return aligned
    usable = min(acceleration.shape[0], max(num_frames - 2, 0))
    if usable > 0:
        aligned[1 : 1 + usable] = acceleration[:usable].reshape(usable, flat_dim)
    return aligned


def _align_orientation(orientation: np.ndarray, num_frames: int) -> np.ndarray:
    flat_dim = int(np.prod(orientation.shape[1:]))
    aligned = np.zeros((num_frames, flat_dim), dtype=np.float32)
    usable = min(orientation.shape[0], num_frames)
    if usable > 0:
        aligned[:usable] = orientation[:usable].reshape(usable, flat_dim)
    return aligned
