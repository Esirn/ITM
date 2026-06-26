#!/usr/bin/env python
"""Train a small Torch MLP from text/IMU frame features to joint vectors."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from itertools import islice
from pathlib import Path

import numpy as np

from itm.baselines.linear_reconstruct import LinearBaselineConfig
from itm.data.dataset import TextIMUMotionDataset
from itm.models.torch_frame_baseline import (
    TorchFrameBaselineConfig,
    build_frame_arrays,
    evaluate_mlp,
    make_mlp,
    require_torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--imu-cache-manifest")
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--eval-imu-cache-manifest")
    parser.add_argument("--output", default="outputs/neural/frame_mlp.pt")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--eval-max-records", type=int)
    parser.add_argument("--text-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-text", action="store_true")
    parser.add_argument("--no-acceleration", action="store_true")
    parser.add_argument("--no-orientation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch = require_torch()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feature_config = LinearBaselineConfig(
        text_dim=args.text_dim,
        include_text=not args.no_text,
        include_acceleration=not args.no_acceleration,
        include_orientation=not args.no_orientation,
    )
    model_config = TorchFrameBaselineConfig(
        feature_config=feature_config,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    train_samples = _load_samples(
        args.manifest,
        args.imu_cache_manifest,
        args.max_records,
    )
    eval_samples = _load_samples(
        args.eval_manifest,
        args.eval_imu_cache_manifest,
        args.eval_max_records,
    )
    train_x_np, train_y_np = build_frame_arrays(train_samples, feature_config)
    eval_x_np, eval_y_np = build_frame_arrays(eval_samples, feature_config)

    device = _resolve_device(torch, args.device)
    train_x = torch.from_numpy(train_x_np).to(device)
    train_y = torch.from_numpy(train_y_np).to(device)
    eval_x = torch.from_numpy(eval_x_np).to(device)
    eval_y = torch.from_numpy(eval_y_np).to(device)

    model = make_mlp(train_x.shape[1], train_y.shape[1], model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(train_x.shape[0], device=device)
        total_loss = 0.0
        total_frames = 0
        for start in range(0, train_x.shape[0], args.batch_size):
            indices = permutation[start : start + args.batch_size]
            prediction = model(train_x[indices])
            loss = loss_fn(prediction, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(indices.numel())
            total_frames += int(indices.numel())
        if epoch == 1 or epoch == args.epochs or epoch % max(args.epochs // 5, 1) == 0:
            print(f"epoch={epoch} train_frame_mse={total_loss / total_frames:.6f}")

    train_stats = evaluate_mlp(model, train_x, train_y, batch_size=args.batch_size)
    eval_stats = evaluate_mlp(model, eval_x, eval_y, batch_size=args.batch_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model_config),
            "input_dim": train_x.shape[1],
            "output_dim": train_y.shape[1],
            "train_stats": train_stats,
            "eval_stats": eval_stats,
        },
        output,
    )
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "model_config": asdict(model_config),
                "device": str(device),
                "train_stats": train_stats,
                "eval_stats": eval_stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"device: {device}")
    print(f"train_mse: {train_stats['mse']:.6f}")
    print(f"train_mae: {train_stats['mae']:.6f}")
    print(f"eval_mse: {eval_stats['mse']:.6f}")
    print(f"eval_mae: {eval_stats['mae']:.6f}")
    print(f"model: {output}")
    print(f"metadata: {metadata_path}")
    return 0


def _load_samples(
    manifest_path: str,
    imu_cache_manifest_path: str | None,
    limit: int | None,
) -> list[dict]:
    dataset = TextIMUMotionDataset(
        manifest_path,
        imu_cache_manifest_path=imu_cache_manifest_path,
        load_joints=False,
        load_joint_vec=True,
        load_imu=True,
    )
    return list(islice((dataset[index] for index in range(len(dataset))), limit))


def _resolve_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cuda:1":
        raise ValueError("Refusing to use cuda:1 because GPU 1 is reserved in this workspace.")
    return torch.device(requested)


if __name__ == "__main__":
    raise SystemExit(main())
