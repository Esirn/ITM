#!/usr/bin/env python3
"""Batch MotionLab generation on the fixed IMU-mapped HumanML test subset."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from itm.data.manifest import read_jsonl
from itm.data.standard_imu import resample_standard_imu, sensor_mask
from itm.models.motionlab_imu_adapter import MotionLabIMUAdapterConfig, make_motionlab_imu_adapter

from sample_motionlab_text import (
    MinimalHumanMLDataModule,
    disable_unrelated_initializers,
    load_config,
    sample_batch,
    sample_text_control_tokens,
)


SENSOR_CONFIGS = {"head": (4,), "wrists": (0, 1)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motionlab-root", type=Path, required=True)
    parser.add_argument("--motionlab-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--imu-manifest", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--sensor-config", choices=tuple(SENSOR_CONFIGS))
    parser.add_argument("--num-samples", type=int, default=672)
    parser.add_argument("--replication-times", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--guidance-mode", choices=("legacy", "factorized"), default="factorized")
    parser.add_argument("--text-scale", type=float, default=1.75)
    parser.add_argument("--imu-scale", type=float, default=1.0)
    parser.add_argument("--joint-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-index", type=Path, required=True)
    return parser.parse_args()


def load_records(manifest, imu_manifest, limit):
    motion = {str(item["motion_id"]): item for item in read_jsonl(manifest)}
    imu = {str(item["motion_id"]): item for item in read_jsonl(imu_manifest)}
    records = []
    for motion_id in sorted(set(motion) & set(imu)):
        row = motion[motion_id]
        vector = np.load(row["joint_vec_path"], mmap_mode="r")
        if vector.ndim != 2 or vector.shape[1] != 263 or not 40 <= len(vector) <= 196:
            continue
        caption = ""
        for line in Path(row["text_path"]).read_text(errors="ignore").splitlines():
            if line.strip():
                caption = line.split("#", 1)[0].strip()
                break
        if not caption:
            continue
        records.append({
            "motion_id": motion_id,
            "caption": caption,
            "length": len(vector),
            "imu_path": imu[motion_id]["imu_path"],
        })
        if len(records) >= limit:
            break
    return records


def encode_control(torch, adapter, records, normalization, slots, device):
    sequences = []
    mean = np.asarray(normalization["acceleration_mean"], dtype=np.float32)
    std = np.asarray(normalization["acceleration_std"], dtype=np.float32)
    for record in records:
        with np.load(record["imu_path"]) as data:
            acceleration, orientation = resample_standard_imu(
                data["acceleration"], data["orientation"],
                source_fps=float(data["fps"]), target_fps=20.0,
            )
        acceleration = (acceleration - mean[None]) / std[None]
        sequences.append(np.concatenate(
            [acceleration, orientation.reshape(len(orientation), 6, 9)], axis=-1
        ).astype(np.float32))
    frames = max(record["length"] for record in records)
    imu = np.zeros((len(records), frames, 6, 12), dtype=np.float32)
    mask = np.zeros((len(records), frames), dtype=bool)
    for index, (record, sequence) in enumerate(zip(records, sequences)):
        length = min(record["length"], len(sequence), frames)
        imu[index, :length] = sequence[:length]
        mask[index, :length] = True
    sensors = np.stack([sensor_mask(slots, sensor_count=6)] * len(records))
    with torch.inference_mode():
        tokens = adapter(
            torch.from_numpy(imu).to(device),
            torch.from_numpy(sensors).to(device),
            torch.from_numpy(mask).to(device),
        )
    return tokens.cpu().numpy(), mask


def main():
    args = parse_args()
    if bool(args.adapter_checkpoint) != bool(args.sensor_config):
        raise ValueError("adapter checkpoint and sensor config must be provided together")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    manifest = args.manifest.resolve()
    imu_manifest = args.imu_manifest.resolve()
    adapter_checkpoint = args.adapter_checkpoint.resolve() if args.adapter_checkpoint else None
    output_index = args.output_index.resolve()
    motionlab_checkpoint = args.motionlab_checkpoint.resolve()
    records = load_records(manifest, imu_manifest, args.num_samples)
    usable = (len(records) // args.batch_size) * args.batch_size
    records = records[:usable]
    if not records:
        raise ValueError("no complete evaluation batch")

    motionlab_root = args.motionlab_root.resolve()
    os.chdir(motionlab_root)
    sys.path.insert(0, str(motionlab_root))
    import torch
    from rfmotion.models.get_model import get_model

    cfg = load_config(motionlab_root)
    disable_unrelated_initializers()
    datamodule = MinimalHumanMLDataModule(motionlab_root)
    model = get_model(cfg, datamodule)
    model.load_state_dict(torch.load(motionlab_checkpoint, map_location="cpu")["state_dict"])
    model.to(args.device).eval()

    adapter = None
    adapter_state = None
    if adapter_checkpoint:
        adapter_state = torch.load(adapter_checkpoint, map_location="cpu")
        config = MotionLabIMUAdapterConfig(**adapter_state["adapter_config"])
        adapter = make_motionlab_imu_adapter(config)
        adapter.load_state_dict(adapter_state["adapter"])
        adapter.to(args.device).eval()

    output_index.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for replication in range(args.replication_times):
        seed = args.seed + replication
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        feature_batches, joint_batches = [], []
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            texts = [item["caption"] for item in batch]
            lengths = [item["length"] for item in batch]
            if adapter is None:
                features, joints = sample_batch(model, datamodule, texts, lengths, args.device)
            else:
                tokens, mask = encode_control(
                    torch, adapter, batch, adapter_state["imu_normalization"],
                    SENSOR_CONFIGS[args.sensor_config], args.device,
                )
                features, joints = sample_text_control_tokens(
                    model, datamodule, texts, lengths, args.device, tokens, mask,
                    guidance_mode=args.guidance_mode,
                    text_scale=args.text_scale,
                    imu_scale=args.imu_scale,
                    joint_scale=args.joint_scale,
                )
            padded_features = np.zeros((len(batch), 196, 263), dtype=np.float32)
            padded_joints = np.zeros((len(batch), 196, 22, 3), dtype=np.float32)
            padded_features[:, :features.shape[1]] = features
            padded_joints[:, :joints.shape[1]] = joints
            feature_batches.append(padded_features)
            joint_batches.append(padded_joints)
            print(f"replication={replication} samples={start + len(batch)}/{len(records)}", flush=True)
        path = output_index.with_name(f"{output_index.stem}_rep{replication:03d}.npz")
        np.savez_compressed(
            path,
            features=np.concatenate(feature_batches),
            joints=np.concatenate(joint_batches),
            lengths=np.asarray([item["length"] for item in records]),
            motion_ids=np.asarray([item["motion_id"] for item in records]),
        )
        files.append(str(path))
    payload = {
        "backbone": "MotionLab MotionFlow",
        "mode": "text_only" if adapter is None else "text_imu",
        "adapter_checkpoint": str(adapter_checkpoint) if adapter_checkpoint else None,
        "sensor_config": args.sensor_config,
        "guidance_mode": args.guidance_mode,
        "seed": args.seed,
        "num_samples": len(records),
        "replications": args.replication_times,
        "files": files,
        "motion_ids": [item["motion_id"] for item in records],
        "captions": [item["caption"] for item in records],
        "lengths": [item["length"] for item in records],
    }
    output_index.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
