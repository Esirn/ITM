"""Utilities for MDM IMU-control counterfactual experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from itm.data.dataset import read_caption_records
from itm.data.manifest import read_jsonl


FPS = 20.0
SENSOR_SLOTS = {"head": (4,), "wrists": (0, 1)}
SENSOR_JOINTS = {"head": (15,), "wrists": (20, 21)}
HEAD_JOINT = 15
WRIST_JOINTS = (20, 21)
SHOULDER_JOINTS = (16, 17)
FOOT_JOINTS = (10, 11)


@dataclass(frozen=True)
class SampleRecord:
    """Catalog entry with the first HumanML3D caption."""

    motion_id: str
    caption: str
    captions: tuple[str, ...]
    frames: int


def sample_catalog(manifest_path: str | Path, imu_manifest_path: str | Path) -> list[SampleRecord]:
    """Return samples present in both the motion and standard-IMU manifests."""

    available = {str(row["motion_id"]): row for row in read_jsonl(imu_manifest_path)}
    rows: list[SampleRecord] = []
    for row in read_jsonl(manifest_path):
        motion_id = str(row["motion_id"])
        if motion_id not in available:
            continue
        captions = read_caption_records(row["text_path"])
        if not captions:
            continue
        rows.append(
            SampleRecord(
                motion_id=motion_id,
                caption=captions[0].caption,
                captions=tuple(value.caption for value in captions),
                frames=int(available[motion_id]["num_frames"]),
            )
        )
    return rows


def serialize_browser_result(
    *,
    run_id: str,
    request: dict[str, Any],
    result_path: str | Path,
    samples: dict[str, SampleRecord],
) -> dict[str, Any]:
    """Create a frontend-friendly JSON payload from a sampler ``results.npz`` file."""

    with np.load(result_path) as data:
        motion = data["motion"].astype(np.float32)
        gt = data["gt"].astype(np.float32)
        acceleration = data["acceleration"].astype(np.float32)
        orientation = data["orientation"].astype(np.float32)
        sensor_mask = data["sensor_mask"].astype(np.float32)
        metadata = json.loads(str(data["metadata"]))

    panels: list[dict[str, Any]] = []
    first_gt_by_motion: dict[str, int] = {}
    for index, case in enumerate(metadata["cases"]):
        first_gt_by_motion.setdefault(str(case["motion_id"]), index)
    for motion_id, index in first_gt_by_motion.items():
        sample = samples.get(motion_id)
        panels.append(
            {
                "label": f"Ground truth {motion_id}",
                "kind": "ground_truth",
                "motion_id": motion_id,
                "caption": sample.caption if sample is not None else "",
                "motion": gt[index].tolist(),
            }
        )
    for index, case in enumerate(metadata["cases"]):
        panels.append(
            {
                "label": case["label"],
                "kind": "generated",
                "text_source": case.get("text_source"),
                "imu_source": case.get("imu_source"),
                "motion_id": case["motion_id"],
                "caption": case["text"],
                "sensor_config": case["sensor_config"],
                "text_scale": case.get("text_scale", metadata["text_scale"]),
                "imu_scale": case.get("imu_scale", metadata["imu_scale"]),
                "motion": motion[index].tolist(),
            }
        )

    imu: dict[str, Any] = {}
    for index, case in enumerate(metadata["cases"]):
        key = str(case.get("imu_source") or case["motion_id"])
        if key in imu:
            continue
        slots = np.flatnonzero(sensor_mask[index] > 0.5)
        slot = int(slots[0]) if len(slots) else 0
        imu[key] = {
            "motion_id": case["motion_id"],
            "sensor_config": case["sensor_config"],
            "slot": slot,
            "acceleration": acceleration[index, :, slot].tolist(),
            "orientation": orientation[index, :, slot, :, 0].tolist(),
        }

    return {
        "run_id": run_id,
        "request": request,
        "samples": {key: record.__dict__ for key, record in samples.items()},
        "fps": int(FPS),
        "frame_count": int(motion.shape[1]),
        "panels": panels,
        "imu": imu,
    }


def compute_run_metrics(result_path: str | Path) -> dict[str, Any]:
    """Compute lightweight proxy metrics for one generated sampler output."""

    with np.load(result_path) as data:
        motion = data["motion"].astype(np.float32)
        gt = data["gt"].astype(np.float32)
        acceleration = data["acceleration"].astype(np.float32)
        sensor_mask = data["sensor_mask"].astype(np.float32)
        metadata = json.loads(str(data["metadata"]))

    case_metrics = []
    for index, case in enumerate(metadata["cases"]):
        generated = motion[index]
        target = gt[index, :, : generated.shape[1]]
        sensor_config = case["sensor_config"]
        case_metrics.append(
            {
                "index": index,
                "label": case["label"],
                "motion_id": case["motion_id"],
                "text_source": case.get("text_source"),
                "imu_source": case.get("imu_source"),
                "sensor_config": sensor_config,
                "text_scale": float(case.get("text_scale", metadata["text_scale"])),
                "imu_scale": float(case.get("imu_scale", metadata["imu_scale"])),
                "root_trajectory_error_m": root_trajectory_error(generated, target),
                "head_trajectory_error_m": joint_trajectory_error(generated, target, (HEAD_JOINT,)),
                "wrist_trajectory_error_m": joint_trajectory_error(generated, target, WRIST_JOINTS),
                "active_sensor_trajectory_error_m": joint_trajectory_error(
                    generated,
                    target,
                    SENSOR_JOINTS[sensor_config],
                ),
                "root_relative_motion_error_m": root_relative_motion_error(generated, target),
                "jerk_ratio": jerk_ratio(generated, target, fps=FPS),
                "active_sensor_acceleration_error_mps2": active_sensor_acceleration_error(
                    generated,
                    acceleration[index],
                    sensor_mask[index],
                    sensor_config=sensor_config,
                    fps=FPS,
                ),
                "active_sensor_acceleration_ratio": active_sensor_acceleration_ratio(
                    generated,
                    acceleration[index],
                    sensor_mask[index],
                    sensor_config=sensor_config,
                    fps=FPS,
                ),
                "arm_swing_proxy_m": arm_swing_proxy(generated),
                "root_travel_m": root_travel(generated),
                "step_frequency_hz": step_frequency_proxy(generated, fps=FPS),
            }
        )

    return {
        "metadata": metadata,
        "cases": case_metrics,
        "aggregate": aggregate_metrics(case_metrics),
        "pairwise_root_relative_distance_m": pairwise_root_relative_distance(motion),
    }


def aggregate_metrics(cases: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate numeric case metrics by label and by sensor configuration."""

    rows = list(cases)
    return {
        "all": _mean_numeric(rows),
        "by_sensor_config": {
            key: _mean_numeric([row for row in rows if row["sensor_config"] == key])
            for key in sorted({row["sensor_config"] for row in rows})
        },
        "by_label": {
            key: _mean_numeric([row for row in rows if row["label"] == key])
            for key in sorted({row["label"] for row in rows})
        },
    }


def write_summary_markdown(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Write a compact human-readable summary for one run."""

    metadata = metrics["metadata"]
    lines = [
        f"# {metadata.get('experiment_name', 'MDM Control Experiment')}",
        "",
        f"- checkpoint: `{metadata['control_checkpoint']}`",
        f"- seed: `{metadata['seed']}`",
        f"- device: `{metadata['device']}`",
        f"- frame count: `{metadata['frame_count']}`",
        f"- pairwise root-relative distance: `{metrics['pairwise_root_relative_distance_m']:.4f} m`",
        "",
        "## Aggregate",
        "",
        _metric_table([{"group": "all", **metrics["aggregate"]["all"]}]),
        "",
        "## By Sensor",
        "",
        _metric_table(
            [
                {"group": group, **values}
                for group, values in metrics["aggregate"]["by_sensor_config"].items()
            ]
        ),
        "",
        "## Cases",
        "",
        _metric_table(metrics["cases"], include_label=True),
        "",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def root_trajectory_error(prediction: np.ndarray, target: np.ndarray) -> float:
    pred, true = _align(prediction, target)
    pred_root = pred[:, 0] - pred[:1, 0]
    true_root = true[:, 0] - true[:1, 0]
    return float(np.linalg.norm(pred_root - true_root, axis=-1).mean())


def joint_trajectory_error(prediction: np.ndarray, target: np.ndarray, joints: tuple[int, ...]) -> float:
    pred, true = _align(prediction, target)
    pred_rel = pred[:, joints] - pred[:, :1]
    true_rel = true[:, joints] - true[:, :1]
    return float(np.linalg.norm(pred_rel - true_rel, axis=-1).mean())


def root_relative_motion_error(prediction: np.ndarray, target: np.ndarray) -> float:
    pred, true = _align(prediction, target)
    pred_rel = pred - pred[:, :1]
    true_rel = true - true[:, :1]
    return float(np.linalg.norm(pred_rel - true_rel, axis=-1).mean())


def jerk_ratio(prediction: np.ndarray, target: np.ndarray, *, fps: float) -> float:
    pred, true = _align(prediction, target)
    if len(pred) < 4:
        return 0.0
    pred_jerk = np.diff(pred, n=3, axis=0) * fps**3
    true_jerk = np.diff(true, n=3, axis=0) * fps**3
    numerator = float(np.linalg.norm(pred_jerk, axis=-1).mean())
    denominator = max(float(np.linalg.norm(true_jerk, axis=-1).mean()), 1e-8)
    return numerator / denominator


def active_sensor_acceleration_error(
    prediction: np.ndarray,
    target_acceleration: np.ndarray,
    sensor_mask: np.ndarray,
    *,
    sensor_config: str,
    fps: float,
) -> float:
    """Mean L2 error between generated active-joint and target IMU acceleration.

    This is a joint-derived acceleration proxy. It is useful for paper
    diagnostics, but it is not a full virtual-IMU orientation consistency
    metric because the generated HumanML3D motion is not decoded to SMPL sensor
    frames here.
    """

    pred_acc, target_acc = _active_sensor_accelerations(
        prediction,
        target_acceleration,
        sensor_mask,
        sensor_config=sensor_config,
        fps=fps,
    )
    if pred_acc.size == 0:
        return 0.0
    return float(np.linalg.norm(pred_acc - target_acc, axis=-1).mean())


def active_sensor_acceleration_ratio(
    prediction: np.ndarray,
    target_acceleration: np.ndarray,
    sensor_mask: np.ndarray,
    *,
    sensor_config: str,
    fps: float,
) -> float:
    """Generated/target active-sensor acceleration magnitude ratio."""

    pred_acc, target_acc = _active_sensor_accelerations(
        prediction,
        target_acceleration,
        sensor_mask,
        sensor_config=sensor_config,
        fps=fps,
    )
    if pred_acc.size == 0:
        return 0.0
    numerator = float(np.linalg.norm(pred_acc, axis=-1).mean())
    denominator = max(float(np.linalg.norm(target_acc, axis=-1).mean()), 1e-8)
    return numerator / denominator


def arm_swing_proxy(joints: np.ndarray) -> float:
    values = np.asarray(joints, dtype=np.float32)
    left = values[:, WRIST_JOINTS[0]] - values[:, SHOULDER_JOINTS[0]]
    right = values[:, WRIST_JOINTS[1]] - values[:, SHOULDER_JOINTS[1]]
    relative = np.stack((left, right), axis=1)
    centered = relative - relative.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(centered, axis=-1).mean())


def root_travel(joints: np.ndarray) -> float:
    values = np.asarray(joints, dtype=np.float32)
    root = values[:, 0]
    return float(np.linalg.norm(root[-1, (0, 2)] - root[0, (0, 2)]))


def step_frequency_proxy(joints: np.ndarray, *, fps: float) -> float:
    values = np.asarray(joints, dtype=np.float32)
    if len(values) < 8:
        return 0.0
    feet = values[:, FOOT_JOINTS][:, :, (0, 2)]
    speed = np.linalg.norm(np.diff(feet, axis=0), axis=-1).mean(axis=1)
    speed = speed - speed.mean()
    if float(np.abs(speed).sum()) < 1e-8:
        return 0.0
    freqs = np.fft.rfftfreq(len(speed), d=1.0 / fps)
    spectrum = np.abs(np.fft.rfft(speed))
    valid = (freqs >= 0.3) & (freqs <= 4.0)
    if not np.any(valid):
        return 0.0
    valid_indices = np.flatnonzero(valid)
    return float(freqs[valid_indices[int(np.argmax(spectrum[valid]))]])


def pairwise_root_relative_distance(motions: np.ndarray) -> float:
    values = np.asarray(motions, dtype=np.float32)
    if len(values) < 2:
        return 0.0
    distances = []
    rel = values - values[:, :, :1]
    for left in range(len(rel)):
        for right in range(left + 1, len(rel)):
            distances.append(np.linalg.norm(rel[left] - rel[right], axis=-1).mean())
    return float(np.mean(distances))


def _align(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float32)
    true = np.asarray(target, dtype=np.float32)
    frames = min(len(pred), len(true))
    joints = min(pred.shape[1], true.shape[1])
    return pred[:frames, :joints], true[:frames, :joints]


def _active_sensor_accelerations(
    prediction: np.ndarray,
    target_acceleration: np.ndarray,
    sensor_mask: np.ndarray,
    *,
    sensor_config: str,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float32)
    target_acc = np.asarray(target_acceleration, dtype=np.float32)
    mask = np.asarray(sensor_mask, dtype=np.float32)
    if len(pred) < 3:
        return (
            np.zeros((0, 0, 3), dtype=np.float32),
            np.zeros((0, 0, 3), dtype=np.float32),
        )
    active_slots = [slot for slot in SENSOR_SLOTS[sensor_config] if slot < len(mask) and mask[slot] > 0.5]
    if not active_slots:
        active_slots = list(SENSOR_SLOTS[sensor_config])
    active_joints = SENSOR_JOINTS[sensor_config][: len(active_slots)]
    pred_acc = np.diff(pred[:, active_joints], n=2, axis=0) * fps**2
    target = target_acc[:, active_slots]
    if len(target) == len(pred_acc) + 2:
        target = target[1:-1]
    frames = min(len(pred_acc), len(target))
    sensors = min(pred_acc.shape[1], target.shape[1])
    return pred_acc[:frames, :sensors], target[:frames, :sensors]


def _mean_numeric(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "root_trajectory_error_m",
        "head_trajectory_error_m",
        "wrist_trajectory_error_m",
        "active_sensor_trajectory_error_m",
        "root_relative_motion_error_m",
        "jerk_ratio",
        "active_sensor_acceleration_error_mps2",
        "active_sensor_acceleration_ratio",
        "arm_swing_proxy_m",
        "root_travel_m",
        "step_frequency_hz",
    ]
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def _metric_table(rows: list[dict[str, Any]], *, include_label: bool = False) -> str:
    if not rows:
        return "_No rows._"
    fields = [
        "root_trajectory_error_m",
        "head_trajectory_error_m",
        "wrist_trajectory_error_m",
        "active_sensor_trajectory_error_m",
        "jerk_ratio",
        "active_sensor_acceleration_error_mps2",
        "active_sensor_acceleration_ratio",
        "arm_swing_proxy_m",
        "root_travel_m",
        "step_frequency_hz",
    ]
    leading = ["label", "sensor_config"] if include_label else ["group"]
    header = leading + fields
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        cells = []
        for key in header:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
