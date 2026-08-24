#!/usr/bin/env python3
"""Render joints produced by the renderer-free MotionLab sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from itm.visualization.skeleton import save_motion_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    result = np.load(args.results)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    joints = result["joints"]
    lengths = result["lengths"]
    motions = [joints[index, : int(length)] for index, length in enumerate(lengths)]
    labels = [f"MotionLab sample {index + 1}" for index in range(len(motions))]
    caption = " | ".join(metadata["texts"])
    save_motion_comparison(args.output, motions, labels, caption, fps=args.fps)
    print(f"samples: {len(motions)}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
