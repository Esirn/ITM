#!/usr/bin/env python3
"""Prepare fixed-subset specs for factorized joint-guidance sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_scales(value: str) -> list[float]:
    scales = [float(item) for item in value.split(",") if item.strip()]
    if not scales or any(scale < 0 for scale in scales):
        raise ValueError("joint scales must be a non-empty list of non-negative values")
    if len(set(scales)) != len(scales):
        raise ValueError("joint scales must be unique")
    return scales


def scale_key(scale: float) -> str:
    return f"joint_{scale:g}".replace(".", "p")


def build_cases(subset: dict, sensor_config: str, joint_scale: float) -> list[dict]:
    return [
        {
            "motion_id": motion_id,
            "target_motion_id": motion_id,
            "imu_motion_id": motion_id,
            "imu_mode": "paired",
            "sensor_config": sensor_config,
            "text": subset["captions"][index],
            "length": int(subset["lengths"][index]),
            "text_scale": 2.5,
            "imu_scale": 1.0,
            "joint_scale": joint_scale,
            "label": f"{motion_id}_{sensor_config}_joint_{joint_scale:g}",
        }
        for index, motion_id in enumerate(subset["motion_ids"])
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--joint-scales", default="0,0.25,0.5,1,1.5")
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    subset = json.loads(args.subset_spec.read_text(encoding="utf-8"))
    scales = parse_scales(args.joint_scales)
    count = len(subset["motion_ids"])
    if not (count == len(subset["captions"]) == len(subset["lengths"])):
        raise ValueError("motion_ids, captions, and lengths must have equal sizes")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(args.subset_spec.resolve()),
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "motion_ids": subset["motion_ids"],
        "joint_scales": scales,
        "fixed_scales": {"text_scale": 2.5, "imu_scale": 1.0},
        "specs": {},
    }
    for config in ("head", "wrists"):
        for scale in scales:
            group = f"{config}_{scale_key(scale)}"
            cases = build_cases(subset, config, scale)
            paths = []
            for start in range(0, count, args.chunk_size):
                path = args.output_dir / f"{group}_{start:03d}.json"
                path.write_text(
                    json.dumps(cases[start : start + args.chunk_size], indent=2),
                    encoding="utf-8",
                )
                paths.append(str(path.resolve()))
            manifest["specs"][group] = paths
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"samples": count, "groups": len(manifest["specs"]), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
