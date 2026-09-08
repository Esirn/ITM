#!/usr/bin/env python3
"""Prepare a length-stratified multi-seed MDM control experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def choose_stratified(subset: dict, count_per_bin: int, seed: int) -> list[int]:
    bins = {"short": [], "medium": [], "long": []}
    for index, length in enumerate(subset["lengths"]):
        key = "short" if length <= 80 else "medium" if length <= 140 else "long"
        bins[key].append(index)
    rng = np.random.default_rng(seed)
    selected = []
    for key in ("short", "medium", "long"):
        if len(bins[key]) < count_per_bin:
            raise ValueError(f"{key} has only {len(bins[key])} samples")
        selected.extend(sorted(rng.choice(bins[key], count_per_bin, replace=False).tolist()))
    return selected


def derangement(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(count)
    while np.any(permutation == np.arange(count)):
        permutation = rng.permutation(count)
    return permutation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="1234,2345,3456,4567,5678")
    parser.add_argument("--count-per-length-bin", type=int, default=10)
    parser.add_argument("--selection-seed", type=int, default=20260825)
    parser.add_argument("--chunk-size", type=int, default=10)
    args = parser.parse_args()

    source = json.loads(args.subset_spec.read_text(encoding="utf-8"))
    indices = choose_stratified(source, args.count_per_length_bin, args.selection_seed)
    subset = {key: [source[key][index] for index in indices] for key in ("motion_ids", "captions", "lengths")}
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds):
        raise ValueError("diffusion seeds must be unique")
    shuffled = derangement(len(indices), args.selection_seed + 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(args.subset_spec.resolve()),
        "selection_seed": args.selection_seed,
        "diffusion_seeds": seeds,
        "chunk_size": args.chunk_size,
        "motion_ids": subset["motion_ids"],
        "captions": subset["captions"],
        "lengths": subset["lengths"],
        "shuffled_motion_ids": [subset["motion_ids"][int(index)] for index in shuffled],
        "group_seeds": {},
        "specs": {},
    }
    for diffusion_seed in seeds:
        for config in ("head", "wrists"):
            for mode in ("text", "paired", "shuffled"):
                group = f"seed{diffusion_seed}_{config}_{mode}"
                manifest["group_seeds"][group] = diffusion_seed
                cases = []
                for index, motion_id in enumerate(subset["motion_ids"]):
                    imu_id = subset["motion_ids"][int(shuffled[index])] if mode == "shuffled" else motion_id
                    cases.append({
                        "motion_id": motion_id,
                        "target_motion_id": motion_id,
                        "imu_motion_id": imu_id,
                        "imu_mode": "paired" if mode == "text" else mode,
                        "sensor_config": config,
                        "text": subset["captions"][index],
                        "length": int(subset["lengths"][index]),
                        "imu_scale": 0.0 if mode == "text" else 1.0,
                        "joint_scale": 0.0 if mode == "text" else 1.0,
                        "label": f"{motion_id}_{config}_{mode}_seed{diffusion_seed}",
                    })
                paths = []
                for start in range(0, len(cases), args.chunk_size):
                    path = args.output_dir / f"{group}_{start:03d}.json"
                    path.write_text(json.dumps(cases[start:start + args.chunk_size], indent=2), encoding="utf-8")
                    paths.append(str(path.resolve()))
                manifest["specs"][group] = paths
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"motions": len(indices), "seeds": seeds, "groups": len(manifest["specs"]), "chunks": sum(map(len, manifest["specs"].values())), "manifest": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
