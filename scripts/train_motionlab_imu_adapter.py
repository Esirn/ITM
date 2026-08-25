#!/usr/bin/env python3
"""Train sparse standard IMU to MotionLab trajectory-hint adapter."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from itm.data.dataset import TextIMUMotionDataset
from itm.data.manifest import read_jsonl
from itm.data.standard_imu import resample_standard_imu, sensor_mask
from itm.models.motionlab_imu_adapter import (
    MotionLabIMUAdapterConfig,
    active_joint_mask,
    make_motionlab_imu_adapter,
    masked_trajectory_losses,
)
from itm.models.torch_frame_baseline import require_torch


SENSOR_CONFIGS = {"head": (4,), "wrists": (0, 1)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", default="outputs/manifests_full/train.jsonl")
    parser.add_argument(
        "--train-imu-manifest", default="outputs/manifests_full/train_standard_imu.jsonl"
    )
    parser.add_argument("--val-manifest", default="outputs/manifests_full/val.jsonl")
    parser.add_argument(
        "--val-imu-manifest", default="outputs/manifests_full/val_standard_imu.jsonl"
    )
    parser.add_argument(
        "--motionlab-root", default="/home/a200/mount/a40/relatedworks/MotionLab"
    )
    parser.add_argument("--sensor-configs", default="head,wrists")
    parser.add_argument("--output", default="outputs/motionlab/imu_adapter/adapter.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--velocity-loss-weight", type=float, default=0.2)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


class PairedIMUTrajectoryDataset:
    def __init__(
        self,
        manifest: Path,
        imu_manifest: Path,
        configs: dict[str, tuple[int, ...]],
        motion_mean: np.ndarray,
        motion_std: np.ndarray,
        normalization: dict | None = None,
        limit: int | None = None,
    ):
        motion_data = TextIMUMotionDataset(
            manifest, load_joints=True, load_joint_vec=False, load_imu=False
        )
        motions = {}
        invalid = []
        for index in range(len(motion_data)):
            sample = motion_data[index]
            joints = np.asarray(sample["joints"])
            if joints.ndim != 3 or joints.shape[1:] != (22, 3):
                invalid.append((str(sample["motion_id"]), tuple(joints.shape)))
                continue
            motions[str(sample["motion_id"])] = sample
        if invalid:
            print(f"skipping_invalid_joint_records={invalid}")
        imu = {str(record["motion_id"]): record for record in read_jsonl(imu_manifest)}
        ids = sorted(set(motions) & set(imu))
        if limit is not None:
            ids = ids[:limit]
        self.items = [
            (motions[motion_id], imu[motion_id], name, slots)
            for motion_id in ids
            for name, slots in configs.items()
        ]
        if not self.items:
            raise ValueError("No overlapping motion and standard IMU records")
        self.motion_mean = motion_mean.reshape(1, 22, 3)
        self.motion_std = motion_std.reshape(1, 22, 3)
        self.normalization = normalization
        self.invalid_records = invalid

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        sample, record, config_name, slots = self.items[index]
        with np.load(record["imu_path"]) as data:
            acceleration, orientation = resample_standard_imu(
                data["acceleration"],
                data["orientation"],
                source_fps=float(data["fps"]),
                target_fps=20.0,
            )
        if self.normalization is not None:
            mean = np.asarray(self.normalization["acceleration_mean"], dtype=np.float32)
            std = np.asarray(self.normalization["acceleration_std"], dtype=np.float32)
            acceleration = (acceleration - mean[None]) / std[None]
        imu = np.concatenate(
            [acceleration, orientation.reshape(len(orientation), 6, 9)], axis=-1
        ).astype(np.float32)
        joints = np.asarray(sample["joints"], dtype=np.float32)
        length = min(len(joints), len(imu), 196)
        target = (joints[:length] - self.motion_mean) / self.motion_std
        return {
            "imu": imu[:length],
            "target": target.reshape(length, 66).astype(np.float32),
            "sensor_mask": sensor_mask(slots, sensor_count=6),
            "sensor_config": config_name,
            "motion_id": sample["motion_id"],
        }

    def compute_normalization(self) -> dict:
        total = np.zeros((6, 3), dtype=np.float64)
        squared = np.zeros((6, 3), dtype=np.float64)
        count = 0
        seen = set()
        for _, record, _, _ in self.items:
            path = Path(record["imu_path"])
            if path in seen:
                continue
            seen.add(path)
            with np.load(path) as data:
                acceleration = np.asarray(data["acceleration"], dtype=np.float64)
            total += acceleration.sum(0)
            squared += np.square(acceleration).sum(0)
            count += len(acceleration)
        mean = total / count
        variance = np.maximum(squared / count - mean**2, 1e-8)
        return {
            "acceleration_mean": mean.astype(np.float32).tolist(),
            "acceleration_std": np.sqrt(variance).astype(np.float32).tolist(),
            "orientation": "rotation_matrix_unscaled",
        }


def collate(items):
    torch = require_torch()
    lengths = torch.tensor([len(item["imu"]) for item in items], dtype=torch.long)
    frames = int(lengths.max())
    imu = torch.zeros(len(items), frames, 6, 12)
    target = torch.zeros(len(items), frames, 66)
    frame_mask = torch.arange(frames)[None] < lengths[:, None]
    for index, item in enumerate(items):
        length = lengths[index]
        imu[index, :length] = torch.from_numpy(item["imu"])
        target[index, :length] = torch.from_numpy(item["target"])
    return {
        "imu": imu,
        "target": target,
        "sensor_mask": torch.from_numpy(np.stack([item["sensor_mask"] for item in items])),
        "frame_mask": frame_mask,
        "lengths": lengths,
    }


def evaluate(model, loader, device, velocity_weight):
    torch = require_torch()
    model.eval()
    totals = {"loss": 0.0, "trajectory_loss": 0.0, "velocity_loss": 0.0}
    count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            prediction = model(batch["imu"], batch["sensor_mask"], batch["frame_mask"])
            trajectory, velocity = masked_trajectory_losses(
                prediction,
                batch["target"],
                batch["frame_mask"],
                active_joint_mask(batch["sensor_mask"]),
            )
            loss = trajectory + velocity_weight * velocity
            size = len(batch["imu"])
            totals["loss"] += float(loss) * size
            totals["trajectory_loss"] += float(trajectory) * size
            totals["velocity_loss"] += float(velocity) * size
            count += size
    model.train()
    return {key: value / count for key, value in totals.items()}


def main() -> int:
    args = parse_args()
    torch = require_torch()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    configs = _parse_configs(args.sensor_configs)
    motionlab_root = Path(args.motionlab_root).resolve()
    mean = np.load(motionlab_root / "datasets/all/mean_motion.npy").astype(np.float32)
    std = np.load(motionlab_root / "datasets/all/std_motion.npy").astype(np.float32)
    train = PairedIMUTrajectoryDataset(
        Path(args.train_manifest),
        Path(args.train_imu_manifest),
        configs,
        mean,
        std,
        limit=args.max_records,
    )
    normalization = train.compute_normalization()
    train.normalization = normalization
    val = PairedIMUTrajectoryDataset(
        Path(args.val_manifest),
        Path(args.val_imu_manifest),
        configs,
        mean,
        std,
        normalization=normalization,
        limit=args.max_records,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        generator=generator,
    )
    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )
    config = MotionLabIMUAdapterConfig()
    model = make_motionlab_imu_adapter(config)
    device = torch.device(args.device)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    best_val = float("inf")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        totals = {"loss": 0.0, "trajectory_loss": 0.0, "velocity_loss": 0.0}
        count = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            prediction = model(batch["imu"], batch["sensor_mask"], batch["frame_mask"])
            trajectory, velocity = masked_trajectory_losses(
                prediction,
                batch["target"],
                batch["frame_mask"],
                active_joint_mask(batch["sensor_mask"]),
            )
            loss = trajectory + args.velocity_loss_weight * velocity
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = len(batch["imu"])
            totals["loss"] += float(loss) * size
            totals["trajectory_loss"] += float(trajectory) * size
            totals["velocity_loss"] += float(velocity) * size
            count += size
        train_metrics = {key: value / count for key, value in totals.items()}
        val_metrics = evaluate(model, val_loader, device, args.velocity_loss_weight)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record))
        _save(output.with_name(f"{output.stem}_epoch{epoch:03d}{output.suffix}"), model, optimizer, config, normalization, configs, history, args)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            _save(output.with_name(f"{output.stem}_best{output.suffix}"), model, optimizer, config, normalization, configs, history, args)
    _save(output, model, optimizer, config, normalization, configs, history, args)
    print(f"checkpoint: {output}")
    return 0


def _parse_configs(value: str) -> dict[str, tuple[int, ...]]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(SENSOR_CONFIGS))
    if unknown:
        raise ValueError(f"unknown sensor configs: {unknown}")
    return {name: SENSOR_CONFIGS[name] for name in names}


def _save(path, model, optimizer, config, normalization, sensor_configs, history, args):
    torch = require_torch()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
            "imu_normalization": normalization,
            "sensor_configs": {key: list(value) for key, value in sensor_configs.items()},
            "history": history,
            "training_args": vars(args),
            "target": "MotionLab normalized 22-joint trajectory hints",
        },
        path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
