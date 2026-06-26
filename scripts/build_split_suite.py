#!/usr/bin/env python
"""Build manifests and synthetic IMU caches for multiple splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from itm.config import dataset_paths, load_toml
from itm.data.imu_cache import build_imu_cache, write_imu_cache_manifest
from itm.data.manifest import build_manifest_entries, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paths.toml"))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Default per-split limit used when split-specific limits are omitted.",
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--manifest-dir", type=Path, default=Path("outputs/manifests"))
    parser.add_argument("--imu-output-dir", type=Path, default=Path("outputs/synthetic_imu"))
    parser.add_argument("--no-orientation", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument(
        "--strict-invalid",
        action="store_true",
        help="fail on invalid joint arrays instead of filtering them out",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = dataset_paths(load_toml(args.config))
    for split in args.splits:
        limit = _split_limit(args, split)
        split_file = Path(paths["motion_splits"]) / f"{split}.txt"
        manifest_path = args.manifest_dir / f"{split}.jsonl"
        cache_manifest_path = args.manifest_dir / f"{split}_imu_cache.jsonl"

        manifest_entries = build_manifest_entries(
            split=split,
            split_file=split_file,
            texts_dir=paths["humanml3d_texts"],
            joints_dir=paths["humanml3d_joints"],
            joint_vecs_dir=paths["humanml3d_joint_vecs"],
            limit=limit,
        )
        manifest_count = write_jsonl(manifest_entries, manifest_path)
        cache_entries = build_imu_cache(
            manifest_path,
            args.imu_output_dir,
            include_orientation=not args.no_orientation,
            overwrite=args.overwrite_cache,
            skip_invalid=not args.strict_invalid,
        )
        if len(cache_entries) != len(manifest_entries):
            valid_ids = {entry.motion_id for entry in cache_entries}
            manifest_entries = [
                entry for entry in manifest_entries if entry.motion_id in valid_ids
            ]
            manifest_count = write_jsonl(manifest_entries, manifest_path)
        cache_count = write_imu_cache_manifest(cache_entries, cache_manifest_path)
        print(
            f"{split}: wrote {manifest_count} manifest entries to {manifest_path}; "
            f"{cache_count} IMU entries to {cache_manifest_path}"
        )
    return 0


def _split_limit(args: argparse.Namespace, split: str) -> int | None:
    value = getattr(args, f"{split}_limit")
    return value if value is not None else args.limit


if __name__ == "__main__":
    raise SystemExit(main())
