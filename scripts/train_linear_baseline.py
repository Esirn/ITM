#!/usr/bin/env python
"""Train a CPU ridge-regression baseline from text and synthetic IMU to motion."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from itm.baselines.linear_reconstruct import (
    LinearBaselineConfig,
    evaluate_linear_baseline,
    fit_linear_baseline,
    save_linear_baseline,
)
from itm.data.dataset import TextIMUMotionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Training manifest JSONL.")
    parser.add_argument(
        "--imu-cache-manifest",
        help="Optional training synthetic IMU cache manifest JSONL.",
    )
    parser.add_argument(
        "--eval-manifest",
        help="Optional held-out evaluation manifest JSONL.",
    )
    parser.add_argument(
        "--eval-imu-cache-manifest",
        help="Optional held-out synthetic IMU cache manifest JSONL.",
    )
    parser.add_argument(
        "--output",
        default="outputs/baselines/linear_text_imu.npz",
        help="Output model npz path.",
    )
    parser.add_argument("--text-dim", type=int, default=128)
    parser.add_argument("--ridge-alpha", type=float, default=1e-2)
    parser.add_argument("--max-records", type=int, help="Limit records for smoke runs.")
    parser.add_argument(
        "--eval-max-records",
        type=int,
        help="Limit held-out records for smoke runs.",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Drop hashed caption features.",
    )
    parser.add_argument(
        "--no-acceleration",
        action="store_true",
        help="Drop IMU acceleration features.",
    )
    parser.add_argument(
        "--no-orientation",
        action="store_true",
        help="Use acceleration but not orientation features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LinearBaselineConfig(
        text_dim=args.text_dim,
        ridge_alpha=args.ridge_alpha,
        include_text=not args.no_text,
        include_acceleration=not args.no_acceleration,
        include_orientation=not args.no_orientation,
    )
    train_dataset = TextIMUMotionDataset(
        args.manifest,
        imu_cache_manifest_path=args.imu_cache_manifest,
        load_joints=False,
        load_joint_vec=True,
        load_imu=True,
    )
    train_samples = _load_samples(train_dataset, args.max_records)
    weights, train_stats = fit_linear_baseline(train_samples, config)
    train_eval_stats = evaluate_linear_baseline(train_samples, weights, config)

    heldout_eval_stats = None
    if args.eval_manifest:
        eval_dataset = TextIMUMotionDataset(
            args.eval_manifest,
            imu_cache_manifest_path=args.eval_imu_cache_manifest,
            load_joints=False,
            load_joint_vec=True,
            load_imu=True,
        )
        eval_samples = _load_samples(eval_dataset, args.eval_max_records)
        heldout_eval_stats = evaluate_linear_baseline(eval_samples, weights, config)

    save_linear_baseline(
        args.output,
        weights,
        config,
        train_stats=train_stats,
        eval_stats={
            "train": train_eval_stats,
            "heldout": heldout_eval_stats or {},
        },
    )

    print(f"manifest: {Path(args.manifest)}")
    print(f"train_records: {int(train_eval_stats['num_records'])}")
    print(f"train_frames: {int(train_eval_stats['num_frames'])}")
    print(f"feature_dim: {weights.shape[0]}")
    print(f"target_dim: {weights.shape[1]}")
    print(f"include_text: {config.include_text}")
    print(f"include_acceleration: {config.include_acceleration}")
    print(f"include_orientation: {config.include_orientation}")
    print(f"train_mse: {train_eval_stats['mse']:.6f}")
    print(f"train_mae: {train_eval_stats['mae']:.6f}")
    if heldout_eval_stats is not None:
        print(f"eval_manifest: {Path(args.eval_manifest)}")
        print(f"eval_records: {int(heldout_eval_stats['num_records'])}")
        print(f"eval_frames: {int(heldout_eval_stats['num_frames'])}")
        print(f"eval_mse: {heldout_eval_stats['mse']:.6f}")
        print(f"eval_mae: {heldout_eval_stats['mae']:.6f}")
    print(f"model: {Path(args.output)}")


def _load_samples(dataset: TextIMUMotionDataset, limit: int | None) -> list[dict]:
    return list(islice((dataset[index] for index in range(len(dataset))), limit))


if __name__ == "__main__":
    main()
