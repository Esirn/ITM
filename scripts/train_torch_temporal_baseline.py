#!/usr/bin/env python
"""Train a masked temporal Transformer from text and sparse IMU to motion."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path
import random

import numpy as np

from itm.data.dataset import TextIMUMotionDataset
from itm.data.text_embeddings import load_text_embedding_cache
from itm.models.torch_frame_baseline import require_torch
from itm.models.torch_temporal_baseline import (
    TemporalBaselineConfig,
    build_sequence_item,
    collate_sequence_items,
    make_temporal_model,
    masked_mse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--imu-cache-manifest", required=True)
    parser.add_argument("--text-cache")
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--eval-imu-cache-manifest", required=True)
    parser.add_argument("--eval-text-cache")
    parser.add_argument("--output", default="outputs/neural/temporal_transformer.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--eval-max-records", type=int)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-text", action="store_true")
    parser.add_argument("--no-acceleration", action="store_true")
    parser.add_argument("--no-orientation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch = require_torch()
    if args.device == "cuda:1":
        raise ValueError("Refusing to use cuda:1 because GPU 1 is reserved in this workspace.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = TemporalBaselineConfig(
        include_acceleration=not args.no_acceleration,
        include_orientation=not args.no_orientation,
        include_text=not args.no_text,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    )
    if config.include_text and (not args.text_cache or not args.eval_text_cache):
        raise ValueError("--text-cache and --eval-text-cache are required unless --no-text")
    train_embeddings = (
        load_text_embedding_cache(args.text_cache).embeddings if config.include_text else None
    )
    eval_embeddings = (
        load_text_embedding_cache(args.eval_text_cache).embeddings if config.include_text else None
    )
    train_items = _load_items(
        args.manifest, args.imu_cache_manifest, train_embeddings, config, args.max_records
    )
    eval_items = _load_items(
        args.eval_manifest,
        args.eval_imu_cache_manifest,
        eval_embeddings,
        config,
        args.eval_max_records,
    )
    input_dim = train_items[0]["sequence"].shape[1]
    text_dim = train_items[0]["text"].shape[0]
    output_dim = train_items[0]["target"].shape[1]
    device = torch.device(args.device)
    model = make_temporal_model(input_dim, text_dim, output_dim, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_eval_mse = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_stats = _run_epoch(
            model, train_items, args.batch_size, device, optimizer=optimizer, shuffle=True
        )
        eval_stats = _run_epoch(
            model, eval_items, args.batch_size, device, optimizer=None, shuffle=False
        )
        history.append({"epoch": epoch, "train": train_stats, "eval": eval_stats})
        if eval_stats["mse"] < best_eval_mse:
            best_eval_mse = eval_stats["mse"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": asdict(config),
                    "input_dim": input_dim,
                    "text_dim": text_dim,
                    "output_dim": output_dim,
                    "epoch": epoch,
                    "eval_stats": eval_stats,
                },
                output,
            )
        print(
            f"epoch={epoch} train_mse={train_stats['mse']:.6f} "
            f"eval_mse={eval_stats['mse']:.6f} best={best_eval_mse:.6f}"
        )

    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "model_config": asdict(config),
                "device": str(device),
                "train_manifest": args.manifest,
                "eval_manifest": args.eval_manifest,
                "text_cache": args.text_cache,
                "eval_text_cache": args.eval_text_cache,
                "input_dim": input_dim,
                "text_dim": text_dim,
                "output_dim": output_dim,
                "best_epoch": best_epoch,
                "best_eval_mse": best_eval_mse,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"best_epoch: {best_epoch}")
    print(f"best_eval_mse: {best_eval_mse:.6f}")
    print(f"model: {output}")
    return 0


def _load_items(manifest, imu_manifest, embeddings, config, limit):
    dataset = TextIMUMotionDataset(
        manifest,
        imu_cache_manifest_path=imu_manifest,
        load_joints=False,
        load_joint_vec=True,
        load_imu=True,
        synthesize_imu_if_missing=False,
    )
    samples = islice((dataset[index] for index in range(len(dataset))), limit)
    items = [build_sequence_item(sample, embeddings, config) for sample in samples]
    if not items:
        raise ValueError("The selected dataset is empty")
    return items


def _run_epoch(model, items, batch_size, device, *, optimizer, shuffle):
    torch = require_torch()
    order = list(range(len(items)))
    if shuffle:
        random.shuffle(order)
    model.train(optimizer is not None)
    squared_error = 0.0
    absolute_error = 0.0
    total_values = 0
    for start in range(0, len(order), batch_size):
        batch = collate_sequence_items([items[index] for index in order[start : start + batch_size]])
        sequence = torch.from_numpy(batch["sequence"]).to(device)
        text = torch.from_numpy(batch["text"]).to(device)
        target = torch.from_numpy(batch["target"]).to(device)
        mask = torch.from_numpy(batch["mask"]).to(device)
        with torch.set_grad_enabled(optimizer is not None):
            prediction = model(sequence, text, mask)
            loss = masked_mse(prediction, target, mask)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        valid = mask.unsqueeze(-1).expand_as(target)
        residual = (prediction - target)[valid]
        squared_error += float(torch.sum(residual * residual).item())
        absolute_error += float(torch.sum(torch.abs(residual)).item())
        total_values += int(residual.numel())
    return {
        "mse": squared_error / total_values,
        "mae": absolute_error / total_values,
        "num_values": total_values,
    }


if __name__ == "__main__":
    raise SystemExit(main())
