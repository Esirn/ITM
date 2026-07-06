#!/usr/bin/env python
"""Export four-way qualitative comparisons from temporal baseline checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from itm.data.dataset import TextIMUMotionDataset
from itm.data.humanml import recover_from_ric
from itm.data.text_embeddings import load_text_embedding_cache
from itm.models.torch_frame_baseline import require_torch
from itm.models.torch_temporal_baseline import (
    TemporalBaselineConfig,
    build_sequence_item,
    collate_sequence_items,
    make_temporal_model,
)
from itm.visualization.skeleton import save_motion_comparison


SENSOR_NAMES = ("pelvis", "left ankle", "right ankle", "head", "left wrist", "right wrist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-checkpoint", required=True)
    parser.add_argument("--imu-checkpoint", required=True)
    parser.add_argument("--conditioned-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--imu-cache-manifest", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--output-dir", default="outputs/visualizations/temporal")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-each", type=int, default=2)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--format", choices=("gif", "mp4"), default="gif")
    parser.add_argument(
        "--imu-display-slot",
        type=int,
        help="Zero-based cache sensor slot to plot; defaults to the first conditioned sensor.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch = require_torch()
    if args.device == "cuda:1":
        raise ValueError("Refusing to use cuda:1 because GPU 1 is reserved in this workspace.")
    device = torch.device(args.device)
    dataset = TextIMUMotionDataset(
        args.manifest,
        imu_cache_manifest_path=args.imu_cache_manifest,
        load_joints=True,
        load_joint_vec=True,
        load_imu=True,
        synthesize_imu_if_missing=False,
    )
    count = len(dataset) if args.max_records is None else min(len(dataset), args.max_records)
    samples = [dataset[index] for index in range(count)]
    embeddings = load_text_embedding_cache(args.text_cache).embeddings
    conditioned_config = _load_config(args.conditioned_checkpoint)
    conditioned_slots = (
        tuple(conditioned_config.sensor_slots)
        if conditioned_config.sensor_slots is not None
        else tuple(range(samples[0]["imu_acceleration"].shape[1]))
    )
    display_slot = args.imu_display_slot
    if display_slot is None:
        display_slot = conditioned_slots[0]
    if display_slot not in conditioned_slots:
        raise ValueError(
            f"Displayed IMU slot {display_slot} is not used by the conditioned model: "
            f"{conditioned_slots}"
        )
    sensor_joint_index = int(samples[0]["sensor_joint_indices"][display_slot])
    sensor_name = _sensor_name(display_slot)
    conditioned_names = ", ".join(_sensor_name(slot) for slot in conditioned_slots)
    if len(conditioned_slots) == 1:
        imu_label = f"{sensor_name.title()} IMU"
        imu_title = imu_label
        comparison_labels = [
            "Ground truth",
            "Text only",
            f"{imu_label} only",
            f"Text + {imu_label}",
        ]
    else:
        imu_title = f"Model IMUs: {conditioned_names} | displayed: {sensor_name}"
        comparison_labels = ["Ground truth", "Text only", "IMU only", "Text + IMU"]
    text_predictions = _predict(args.text_checkpoint, samples, embeddings, device)
    imu_predictions = _predict(args.imu_checkpoint, samples, None, device)
    conditioned_predictions = _predict(
        args.conditioned_checkpoint, samples, embeddings, device
    )

    imu_errors = np.asarray(
        [np.mean((prediction - sample["motion"]) ** 2) for prediction, sample in zip(imu_predictions, samples)]
    )
    text_errors = np.asarray(
        [
            np.mean((prediction - sample["motion"]) ** 2)
            for prediction, sample in zip(text_predictions, samples)
        ]
    )
    conditioned_errors = np.asarray(
        [np.mean((prediction - sample["motion"]) ** 2) for prediction, sample in zip(conditioned_predictions, samples)]
    )
    improvement = imu_errors - conditioned_errors
    selected = _select_examples(improvement, args.num_each, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for category, index in selected:
        sample = samples[index]
        motion_id = str(sample["motion_id"])
        stem = f"{category}_{motion_id}"
        gt_joints = recover_from_ric(sample["motion"])
        text_joints = recover_from_ric(text_predictions[index])
        imu_joints = recover_from_ric(imu_predictions[index])
        conditioned_joints = recover_from_ric(conditioned_predictions[index])
        imu_acceleration = _align_acceleration_for_plot(
            sample["imu_acceleration"][:, display_slot], len(sample["motion"])
        )
        imu_orientation = np.asarray(
            sample["imu_orientation"][:, display_slot], dtype=np.float32
        )
        np.savez_compressed(
            output_dir / f"{stem}.npz",
            ground_truth=sample["motion"],
            text_prediction=text_predictions[index],
            imu_prediction=imu_predictions[index],
            conditioned_prediction=conditioned_predictions[index],
            ground_truth_joints=gt_joints,
            text_joints=text_joints,
            imu_joints=imu_joints,
            conditioned_joints=conditioned_joints,
            displayed_imu_acceleration=imu_acceleration,
            displayed_imu_orientation=imu_orientation,
            displayed_imu_slot=np.asarray(display_slot),
            displayed_imu_joint=np.asarray(sensor_joint_index),
            caption=np.asarray(sample["caption"]),
        )
        save_motion_comparison(
            output_dir / f"{stem}.{args.format}",
            [gt_joints, text_joints, imu_joints, conditioned_joints],
            comparison_labels,
            str(sample["caption"]),
            fps=args.fps,
            imu_acceleration=imu_acceleration,
            imu_orientation=imu_orientation,
            imu_title=imu_title,
        )
        records.append(
            {
                "category": category,
                "motion_id": motion_id,
                "caption": sample["caption"],
                "conditioned_imu_slots": list(conditioned_slots),
                "conditioned_imu_names": [
                    _sensor_name(slot) for slot in conditioned_slots
                ],
                "displayed_imu_slot": display_slot,
                "displayed_imu_name": sensor_name,
                "displayed_imu_joint": sensor_joint_index,
                "text_mse": float(text_errors[index]),
                "imu_mse": float(imu_errors[index]),
                "conditioned_mse": float(conditioned_errors[index]),
                "mse_improvement": float(improvement[index]),
                "animation": f"{stem}.{args.format}",
                "prediction": f"{stem}.npz",
            }
        )
        print(f"rendered {stem}: delta_mse={improvement[index]:+.6f}")
    (output_dir / "index.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"index: {output_dir / 'index.json'}")
    return 0


def _predict(checkpoint_path, samples, embeddings, device):
    torch = require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = TemporalBaselineConfig(**checkpoint["model_config"])
    if config.include_text and embeddings is None:
        raise ValueError(f"Text cache required by checkpoint: {checkpoint_path}")
    items = [build_sequence_item(sample, embeddings, config) for sample in samples]
    batch = collate_sequence_items(items)
    model = make_temporal_model(
        checkpoint["input_dim"], checkpoint["text_dim"], checkpoint["output_dim"], config
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.inference_mode():
        prediction = model(
            torch.from_numpy(batch["sequence"]).to(device),
            torch.from_numpy(batch["text"]).to(device),
            torch.from_numpy(batch["mask"]).to(device),
        ).cpu().numpy()
    return [prediction[index, :length] for index, length in enumerate(batch["lengths"])]


def _load_config(checkpoint_path):
    torch = require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return TemporalBaselineConfig(**checkpoint["model_config"])


def _align_acceleration_for_plot(acceleration, num_frames):
    acceleration = np.asarray(acceleration, dtype=np.float32)
    aligned = np.full((num_frames, 3), np.nan, dtype=np.float32)
    usable = min(len(acceleration), max(num_frames - 2, 0))
    if usable:
        aligned[1 : 1 + usable] = acceleration[:usable]
    return aligned


def _sensor_name(slot):
    return SENSOR_NAMES[slot] if 0 <= slot < len(SENSOR_NAMES) else f"sensor {slot}"


def _select_examples(improvement: np.ndarray, num_each: int, seed: int):
    if num_each < 1:
        raise ValueError("--num-each must be positive")
    count = len(improvement)
    num_each = min(num_each, count)
    best = np.argsort(improvement)[-num_each:][::-1]
    worst = np.argsort(improvement)[:num_each]
    reserved = set(best) | set(worst)
    candidates = np.asarray([index for index in range(count) if index not in reserved])
    rng = np.random.default_rng(seed)
    random_indices = rng.choice(candidates, size=min(num_each, len(candidates)), replace=False)
    return (
        [("text_helps", int(index)) for index in best]
        + [("text_hurts", int(index)) for index in worst]
        + [("random", int(index)) for index in random_indices]
    )


if __name__ == "__main__":
    raise SystemExit(main())
