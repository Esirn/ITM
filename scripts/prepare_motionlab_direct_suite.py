#!/usr/bin/env python3
"""Prepare fixed test IDs, captions, lengths, and GT paths for MotionLab sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from itm.data.manifest import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--imu-manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    motion = {str(item["motion_id"]): item for item in read_jsonl(args.manifest)}
    imu_ids = {str(item["motion_id"]) for item in read_jsonl(args.imu_manifest)}
    candidates = []
    for motion_id in sorted(set(motion) & imu_ids):
        record = motion[motion_id]
        joints = np.load(record["joints_path"], mmap_mode="r")
        if joints.ndim != 3 or joints.shape[1:] != (22, 3) or len(joints) < 40:
            continue
        captions = [
            line.split("#", 1)[0].strip()
            for line in Path(record["text_path"]).read_text(errors="ignore").splitlines()
            if line.strip()
        ]
        if captions:
            candidates.append((motion_id, record, captions[0], min(len(joints), 196)))
    rng = np.random.default_rng(args.seed)
    selected_indices = np.sort(rng.choice(len(candidates), size=args.count, replace=False))
    selected = [candidates[int(index)] for index in selected_indices]
    request = {
        "text": [item[2] for item in selected],
        "lengths": [item[3] for item in selected],
        "seed": args.seed,
    }
    spec = {
        "motion_ids": [item[0] for item in selected],
        "captions": request["text"],
        "lengths": request["lengths"],
        "ground_truth_paths": [str(Path(item[1]["joints_path"]).resolve()) for item in selected],
        "seed": args.seed,
        "selection": "seeded random sample from aligned test split",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    (args.output_dir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(selected), "motion_ids": spec["motion_ids"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
