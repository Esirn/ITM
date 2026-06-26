#!/usr/bin/env python3
"""Build cached synthetic sparse-IMU artifacts from a motion manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from itm.data.imu_cache import build_imu_cache, write_imu_cache_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic_imu"))
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=Path("outputs/manifests/imu_cache.jsonl"),
    )
    parser.add_argument(
        "--no-orientation",
        action="store_true",
        help="write acceleration-only IMU artifacts",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute existing cache files",
    )
    args = parser.parse_args()

    entries = build_imu_cache(
        args.manifest,
        args.output_dir,
        include_orientation=not args.no_orientation,
        overwrite=args.overwrite,
    )
    count = write_imu_cache_manifest(entries, args.cache_manifest)
    print(f"wrote {count} IMU cache entries to {args.cache_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
