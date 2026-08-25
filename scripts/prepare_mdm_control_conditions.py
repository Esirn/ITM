#!/usr/bin/env python3
"""Prepare paired, zero, and shuffled MDM control specs from a fixed subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    subset = json.loads(args.subset_spec.read_text(encoding="utf-8"))
    count = len(subset["motion_ids"])
    if count < 2:
        raise ValueError("subset must contain at least two motions")
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(count)
    if np.any(permutation == np.arange(count)):
        permutation = np.roll(np.arange(count), 1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(args.subset_spec.resolve()),
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "motion_ids": subset["motion_ids"],
        "shuffled_motion_ids": [subset["motion_ids"][int(i)] for i in permutation],
        "specs": {},
    }
    for config in ("head", "wrists"):
        for mode in ("text", "paired", "zero", "shuffled"):
            cases = []
            for index, motion_id in enumerate(subset["motion_ids"]):
                imu_motion_id = (
                    subset["motion_ids"][int(permutation[index])]
                    if mode == "shuffled"
                    else motion_id
                )
                cases.append({
                    "motion_id": motion_id,
                    "target_motion_id": motion_id,
                    "imu_motion_id": imu_motion_id,
                    "imu_mode": "zero" if mode == "zero" else "paired" if mode == "text" else mode,
                    "sensor_config": config,
                    "text": subset["captions"][index],
                    "length": int(subset["lengths"][index]),
                    "imu_scale": 0.0 if mode == "text" else 1.0,
                    "joint_scale": 0.0 if mode == "text" else 1.0,
                    "label": f"{motion_id}_{config}_{mode}",
                })
            paths = []
            for start in range(0, count, args.chunk_size):
                path = args.output_dir / f"{config}_{mode}_{start:03d}.json"
                path.write_text(
                    json.dumps(cases[start : start + args.chunk_size], indent=2),
                    encoding="utf-8",
                )
                paths.append(str(path.resolve()))
            manifest["specs"][f"{config}_{mode}"] = paths
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"count": count, "spec_groups": len(manifest["specs"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
