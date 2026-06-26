#!/usr/bin/env python
"""Inspect a padded text-motion-IMU batch."""

from __future__ import annotations

import argparse
from pathlib import Path

from itm.data.dataset import TextIMUMotionDataset, collate_text_imu_motion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Input manifest JSONL.")
    parser.add_argument(
        "--imu-cache-manifest",
        help="Optional synthetic IMU cache manifest JSONL.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--no-joints",
        action="store_true",
        help="Skip loading full 22-joint arrays.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = TextIMUMotionDataset(
        args.manifest,
        imu_cache_manifest_path=args.imu_cache_manifest,
        load_joints=not args.no_joints,
    )
    end_index = min(len(dataset), args.start_index + args.batch_size)
    samples = [dataset[index] for index in range(args.start_index, end_index)]
    batch = collate_text_imu_motion(samples)

    print(f"manifest: {Path(args.manifest)}")
    print(f"num_samples: {len(dataset)}")
    print(f"batch_motion_ids: {batch['motion_id']}")
    print(f"captions: {batch['caption']}")
    for key in [
        "motion",
        "joints",
        "imu_acceleration",
        "imu_orientation",
    ]:
        if key in batch:
            print(f"{key}_shape: {batch[key].shape}")
            print(f"{key}_length: {batch[f'{key}_length'].tolist()}")
    if "sensor_joint_indices" in batch:
        print(f"sensor_joint_indices: {batch['sensor_joint_indices'].tolist()}")


if __name__ == "__main__":
    main()
