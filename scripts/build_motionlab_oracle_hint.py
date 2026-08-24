#!/usr/bin/env python3
"""Build a sparse trajectory-hint NPZ from joint motion for MotionLab smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--joints", default="15", help="comma-separated HumanML joint IDs")
    args = parser.parse_args()

    source = np.load(args.input)
    joints = np.asarray(source["joints"], dtype=np.float32)
    active = [int(value) for value in args.joints.split(",") if value.strip()]
    if joints.ndim != 4 or joints.shape[-2:] != (22, 3):
        raise ValueError("input joints must have shape [B,T,22,3]")
    if not active or min(active) < 0 or max(active) >= 22:
        raise ValueError("joint IDs must be in [0, 21]")
    mask = np.zeros_like(joints, dtype=bool)
    mask[:, :, active, :] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, joints=joints, mask=mask)
    print(f"active joints: {active}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
