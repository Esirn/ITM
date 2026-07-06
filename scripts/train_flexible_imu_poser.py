#!/usr/bin/env python
"""Train an IMUPoser-style SMPL rotation model on standard IMU caches."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np

from itm.data.manifest import read_jsonl
from itm.data.standard_imu import imuposer_features, sensor_mask
from itm.models.flexible_imu_poser import (
    FlexibleIMUPoserConfig,
    make_flexible_imu_poser,
    masked_pose_loss,
    rotation_matrix_to_r6d,
)
from itm.models.torch_frame_baseline import require_torch


SENSOR_CONFIGS = {
    "head": (4,),
    "wrists": (0, 1),
    "left_wrist": (0,),
    "right_wrist": (1,),
    "head_wrists": (0, 1, 4),
    "all_five": (0, 1, 2, 3, 4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache-manifest", required=True)
    parser.add_argument("--eval-cache-manifest", required=True)
    parser.add_argument("--output", default="outputs/imu_poser/flexible_imu_poser.pt")
    parser.add_argument("--sensor-configs", default="head,wrists")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--velocity-weight", type=float, default=0.1)
    parser.add_argument("--acceleration-weight", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-eval-records", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch = require_torch()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    configs = _parse_sensor_configs(args.sensor_configs)
    train_records = list(read_jsonl(args.train_cache_manifest))[: args.max_train_records]
    eval_records = list(read_jsonl(args.eval_cache_manifest))[: args.max_eval_records]
    if not train_records or not eval_records:
        raise ValueError("Train and eval cache manifests must be non-empty")

    train_dataset = _CacheDataset(train_records, configs)
    eval_dataset = _CacheDataset(eval_records, configs)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate
    )
    config = FlexibleIMUPoserConfig(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    device = torch.device(args.device)
    model = make_flexible_imu_poser(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_mse = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_stats = _run_epoch(model, train_loader, device, args, optimizer)
        eval_stats = _run_epoch(model, eval_loader, device, args, None)
        history.append({"epoch": epoch, "train": train_stats, "eval": eval_stats})
        if eval_stats["pose_mse"] < best_mse:
            best_mse = eval_stats["pose_mse"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": asdict(config),
                    "sensor_configs": {name: list(slots) for name, slots in configs.items()},
                    "epoch": epoch,
                    "eval_stats": eval_stats,
                },
                output,
            )
        print(
            f"epoch={epoch} train_mse={train_stats['pose_mse']:.6f} "
            f"eval_mse={eval_stats['pose_mse']:.6f} best={best_mse:.6f}"
        )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "model_config": asdict(config),
                "sensor_configs": {name: list(slots) for name, slots in configs.items()},
                "train_cache_manifest": args.train_cache_manifest,
                "eval_cache_manifest": args.eval_cache_manifest,
                "best_pose_mse": best_mse,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"model: {output}")
    return 0


class _CacheDataset:
    def __init__(self, records, configs):
        self.items = [(record, name, slots) for record in records for name, slots in configs.items()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        record, config_name, slots = self.items[index]
        with np.load(record["imu_path"]) as data:
            mask = sensor_mask(slots, sensor_count=6)
            features = imuposer_features(data["acceleration"], data["orientation"], mask)
            target = rotation_matrix_to_r6d(data["local_rotation"])
        return {
            "motion_id": record["motion_id"],
            "sensor_config": config_name,
            "features": features.astype(np.float32),
            "target": target.astype(np.float32),
        }


def _parse_sensor_configs(value):
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in SENSOR_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown sensor configs: {unknown}; choose from {sorted(SENSOR_CONFIGS)}")
    if not names:
        raise ValueError("At least one sensor config is required")
    return {name: SENSOR_CONFIGS[name] for name in names}


def _collate(items):
    lengths = np.asarray([len(item["features"]) for item in items], dtype=np.int64)
    max_length = int(lengths.max())
    features = np.zeros((len(items), max_length, 60), dtype=np.float32)
    targets = np.zeros((len(items), max_length, 24, 6), dtype=np.float32)
    mask = np.zeros((len(items), max_length), dtype=bool)
    for index, item in enumerate(items):
        length = lengths[index]
        features[index, :length] = item["features"]
        targets[index, :length] = item["target"]
        mask[index, :length] = True
    return {"features": features, "target": targets, "mask": mask, "lengths": lengths}


def _run_epoch(model, loader, device, args, optimizer):
    torch = require_torch()
    model.train(optimizer is not None)
    squared_error = 0.0
    total_values = 0
    for batch in loader:
        features = torch.from_numpy(batch["features"]).to(device)
        target = torch.from_numpy(batch["target"]).to(device)
        mask = torch.from_numpy(batch["mask"]).to(device)
        lengths = torch.from_numpy(batch["lengths"])
        with torch.set_grad_enabled(optimizer is not None):
            prediction = model(features, lengths)
            loss = masked_pose_loss(
                prediction,
                target,
                mask,
                velocity_weight=args.velocity_weight,
                acceleration_weight=args.acceleration_weight,
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        valid = mask[..., None, None].expand_as(target)
        residual = (prediction - target)[valid]
        squared_error += float(torch.sum(residual * residual).item())
        total_values += int(residual.numel())
    return {"pose_mse": squared_error / total_values, "num_values": total_values}


if __name__ == "__main__":
    raise SystemExit(main())
