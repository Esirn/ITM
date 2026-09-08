#!/usr/bin/env python3
"""Render the 22 body joints produced by the minimal HY-Motion sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from itm.visualization.skeleton import save_motion_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = np.load(args.results)
    metadata = json.loads(result["metadata"].item())
    joints = np.asarray(result["keypoints3d"], dtype=np.float32)
    if joints.ndim != 4 or joints.shape[0] != 1 or joints.shape[2] < 22:
        raise ValueError(f"Expected [1,T,J>=22,3] keypoints, got {joints.shape}")
    save_motion_comparison(
        args.output,
        [joints[0, :, :22]],
        [metadata["model"]],
        metadata["text"],
        fps=int(metadata["fps"]),
    )
    print(f"frames: {joints.shape[1]}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
