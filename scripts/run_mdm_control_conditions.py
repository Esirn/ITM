#!/usr/bin/env python3
"""Run resumable MDM paired/zero/shuffled control-condition chunks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--mdm-args", type=Path, required=True)
    parser.add_argument("--standard-imu-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", default="")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--text-scale", type=float, default=2.5)
    parser.add_argument("--imu-scale", type=float, default=1.0)
    parser.add_argument("--joint-scale", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = set(filter(None, args.groups.split(",")))
    groups = {
        key: value for key, value in manifest["specs"].items()
        if not selected or key in selected
    }
    unknown = selected - set(manifest["specs"])
    if unknown:
        raise ValueError(f"unknown groups: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0
    for group, specs in groups.items():
        group_seed = int(manifest.get("group_seeds", {}).get(group, args.seed))
        for spec_value in specs:
            spec = Path(spec_value)
            output = args.output_dir / f"{spec.stem}.npz"
            if output.exists() and not args.overwrite:
                skipped += 1
                continue
            command = [
                sys.executable,
                str(ROOT / "scripts" / "sample_mdm_imu_control.py"),
                "--control-checkpoint", str(args.control_checkpoint.resolve()),
                "--mdm-args", str(args.mdm_args.resolve()),
                "--standard-imu-manifest", str(args.standard_imu_manifest.resolve()),
                "--spec", str(spec.resolve()),
                "--output", str(output.resolve()),
                "--seed", str(group_seed),
                "--text-scale", str(args.text_scale),
                "--imu-scale", str(args.imu_scale),
                "--joint-scale", str(args.joint_scale),
                "--guidance-mode", "factorized",
                "--branch-execution", "batched",
                "--device", args.device,
            ]
            log = output.with_suffix(".log")
            with log.open("w", encoding="utf-8") as handle:
                subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
            completed += 1
            print(f"completed {group}: {output.name}", flush=True)
    summary = {
        "manifest": str(args.manifest.resolve()),
        "groups": list(groups),
        "completed_chunks": completed,
        "skipped_chunks": skipped,
        "seed": args.seed,
        "group_seeds": {group: int(manifest.get("group_seeds", {}).get(group, args.seed)) for group in groups},
        "guidance_mode": "factorized",
        "device": args.device,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
