#!/usr/bin/env python3
"""Split a six-prompt text-attribute spec without separating motion groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--motions-per-chunk", type=int, default=5)
    args = parser.parse_args()
    cases = json.loads(args.source_spec.read_text(encoding="utf-8"))
    groups = []
    by_motion = {}
    for case in cases:
        motion_id = str(case["motion_id"])
        if motion_id not in by_motion:
            groups.append(motion_id)
            by_motion[motion_id] = []
        enriched = dict(case)
        enriched["target_motion_id"] = motion_id
        enriched["imu_motion_id"] = motion_id
        enriched["imu_mode"] = "paired"
        enriched["joint_scale"] = 1.0
        by_motion[motion_id].append(enriched)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, len(groups), args.motions_per_chunk):
        selected = groups[start : start + args.motions_per_chunk]
        chunk = [case for motion_id in selected for case in by_motion[motion_id]]
        path = args.output_dir / f"stage2b_attributes_{start:03d}.json"
        path.write_text(json.dumps(chunk, indent=2), encoding="utf-8")
        paths.append(str(path.resolve()))
    manifest = {
        "source": str(args.source_spec.resolve()),
        "motion_ids": groups,
        "specs": {"stage2b_attributes": paths},
        "seed": 1234,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"motions": len(groups), "chunks": len(paths)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
