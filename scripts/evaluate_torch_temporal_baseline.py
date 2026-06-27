#!/usr/bin/env python
"""Evaluate a saved temporal Transformer checkpoint on a held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from itm.data.dataset import TextIMUMotionDataset
from itm.data.text_embeddings import load_text_embedding_cache
from itm.models.torch_frame_baseline import require_torch
from itm.models.torch_temporal_baseline import (
    TemporalBaselineConfig,
    build_sequence_item,
    collate_sequence_items,
    make_temporal_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--imu-cache-manifest", required=True)
    parser.add_argument("--text-cache")
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch = require_torch()
    if args.device == "cuda:1":
        raise ValueError("Refusing to use cuda:1 because GPU 1 is reserved in this workspace.")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = TemporalBaselineConfig(**checkpoint["model_config"])
    if config.include_text and not args.text_cache:
        raise ValueError("--text-cache is required for a text-conditioned checkpoint")
    embeddings = (
        load_text_embedding_cache(args.text_cache).embeddings if config.include_text else None
    )
    dataset = TextIMUMotionDataset(
        args.manifest,
        imu_cache_manifest_path=args.imu_cache_manifest,
        load_joints=False,
        load_joint_vec=True,
        load_imu=True,
        synthesize_imu_if_missing=False,
    )
    items = [
        build_sequence_item(dataset[index], embeddings, config)
        for index in range(len(dataset))
    ]
    model = make_temporal_model(
        checkpoint["input_dim"],
        checkpoint["text_dim"],
        checkpoint["output_dim"],
        config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    squared_error = 0.0
    absolute_error = 0.0
    total_values = 0
    total_frames = 0
    with torch.inference_mode():
        for start in range(0, len(items), args.batch_size):
            batch = collate_sequence_items(items[start : start + args.batch_size])
            sequence = torch.from_numpy(batch["sequence"]).to(device)
            text = torch.from_numpy(batch["text"]).to(device)
            target = torch.from_numpy(batch["target"]).to(device)
            mask = torch.from_numpy(batch["mask"]).to(device)
            prediction = model(sequence, text, mask)
            valid = mask.unsqueeze(-1).expand_as(target)
            residual = (prediction - target)[valid]
            squared_error += float(torch.sum(residual * residual).item())
            absolute_error += float(torch.sum(torch.abs(residual)).item())
            total_values += int(residual.numel())
            total_frames += int(mask.sum().item())
    stats = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "num_records": len(items),
        "num_frames": total_frames,
        "mse": squared_error / total_values,
        "mae": absolute_error / total_values,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
