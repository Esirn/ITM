#!/usr/bin/env python
"""Run CPU linear baseline ablations and write a compact results table."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
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
    parser.add_argument("--imu-cache-manifest", help="Training IMU cache manifest.")
    parser.add_argument("--eval-manifest", required=True, help="Evaluation manifest JSONL.")
    parser.add_argument("--eval-imu-cache-manifest", help="Evaluation IMU cache manifest.")
    parser.add_argument("--output-dir", default="outputs/baselines/linear_ablation")
    parser.add_argument("--text-dim", type=int, default=128)
    parser.add_argument("--ridge-alpha", type=float, default=1e-2)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--eval-max-records", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = TextIMUMotionDataset(
        args.manifest,
        imu_cache_manifest_path=args.imu_cache_manifest,
        load_joints=False,
        load_joint_vec=True,
        load_imu=True,
    )
    eval_dataset = TextIMUMotionDataset(
        args.eval_manifest,
        imu_cache_manifest_path=args.eval_imu_cache_manifest,
        load_joints=False,
        load_joint_vec=True,
        load_imu=True,
    )
    train_samples = _load_samples(train_dataset, args.max_records)
    eval_samples = _load_samples(eval_dataset, args.eval_max_records)

    variants = {
        "full": LinearBaselineConfig(
            text_dim=args.text_dim,
            ridge_alpha=args.ridge_alpha,
            include_text=True,
            include_acceleration=True,
            include_orientation=True,
        ),
        "text_only": LinearBaselineConfig(
            text_dim=args.text_dim,
            ridge_alpha=args.ridge_alpha,
            include_text=True,
            include_acceleration=False,
            include_orientation=False,
        ),
        "imu_only": LinearBaselineConfig(
            text_dim=args.text_dim,
            ridge_alpha=args.ridge_alpha,
            include_text=False,
            include_acceleration=True,
            include_orientation=True,
        ),
        "acceleration_only": LinearBaselineConfig(
            text_dim=args.text_dim,
            ridge_alpha=args.ridge_alpha,
            include_text=False,
            include_acceleration=True,
            include_orientation=False,
        ),
        "orientation_only": LinearBaselineConfig(
            text_dim=args.text_dim,
            ridge_alpha=args.ridge_alpha,
            include_text=False,
            include_acceleration=False,
            include_orientation=True,
        ),
        "time_only": LinearBaselineConfig(
            text_dim=args.text_dim,
            ridge_alpha=args.ridge_alpha,
            include_text=False,
            include_acceleration=False,
            include_orientation=False,
        ),
    }

    rows = []
    for name, config in variants.items():
        weights, train_stats = fit_linear_baseline(train_samples, config)
        train_eval = evaluate_linear_baseline(train_samples, weights, config)
        heldout_eval = evaluate_linear_baseline(eval_samples, weights, config)
        model_path = output_dir / f"{name}.npz"
        save_linear_baseline(
            model_path,
            weights,
            config,
            train_stats=train_stats,
            eval_stats={"train": train_eval, "heldout": heldout_eval},
        )
        row = {
            "variant": name,
            "feature_dim": weights.shape[0],
            "target_dim": weights.shape[1],
            "train_records": int(train_eval["num_records"]),
            "train_frames": int(train_eval["num_frames"]),
            "train_mse": train_eval["mse"],
            "train_mae": train_eval["mae"],
            "eval_records": int(heldout_eval["num_records"]),
            "eval_frames": int(heldout_eval["num_frames"]),
            "eval_mse": heldout_eval["mse"],
            "eval_mae": heldout_eval["mae"],
            "model_path": str(model_path),
            **asdict(config),
        }
        rows.append(row)

    rows.sort(key=lambda item: item["eval_mse"])
    _write_csv(output_dir / "summary.csv", rows)
    _write_jsonl(output_dir / "summary.jsonl", rows)
    print(f"wrote {len(rows)} ablation rows to {output_dir / 'summary.csv'}")
    for row in rows:
        print(
            f"{row['variant']}: eval_mse={row['eval_mse']:.6f}, "
            f"eval_mae={row['eval_mae']:.6f}, feature_dim={row['feature_dim']}"
        )


def _load_samples(dataset: TextIMUMotionDataset, limit: int | None) -> list[dict]:
    return list(islice((dataset[index] for index in range(len(dataset))), limit))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


if __name__ == "__main__":
    main()
