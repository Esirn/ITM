#!/usr/bin/env python
"""Export qualitative comparisons from two temporal baseline checkpoints."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    imu_predictions = _predict(args.imu_checkpoint, samples, None, device)
    conditioned_predictions = _predict(
        args.conditioned_checkpoint, samples, embeddings, device
    )

    imu_errors = np.asarray(
        [np.mean((prediction - sample["motion"]) ** 2) for prediction, sample in zip(imu_predictions, samples)]
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
        imu_joints = recover_from_ric(imu_predictions[index])
        conditioned_joints = recover_from_ric(conditioned_predictions[index])
        np.savez_compressed(
            output_dir / f"{stem}.npz",
            ground_truth=sample["motion"],
            imu_prediction=imu_predictions[index],
            conditioned_prediction=conditioned_predictions[index],
            ground_truth_joints=gt_joints,
            imu_joints=imu_joints,
            conditioned_joints=conditioned_joints,
            caption=np.asarray(sample["caption"]),
        )
        save_motion_comparison(
            output_dir / f"{stem}.{args.format}",
            [gt_joints, imu_joints, conditioned_joints],
            ["Ground truth", "IMU only", "Text + IMU"],
            str(sample["caption"]),
            fps=args.fps,
        )
        records.append(
            {
                "category": category,
                "motion_id": motion_id,
                "caption": sample["caption"],
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
