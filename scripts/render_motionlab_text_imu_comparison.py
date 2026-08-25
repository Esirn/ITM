#!/usr/bin/env python3
"""Render GT and fixed-seed MotionLab Text/Text+IMU outputs side by side."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from itm.visualization.skeleton import save_motion_comparison


def load_motion(path: Path) -> np.ndarray:
    payload = np.load(path)
    return np.asarray(payload["joints"][0, : int(payload["lengths"][0])], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--text-only", type=Path, required=True)
    parser.add_argument("--text-head", type=Path, required=True)
    parser.add_argument("--text-wrists", type=Path, required=True)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generated = [
        load_motion(args.text_only),
        load_motion(args.text_head),
        load_motion(args.text_wrists),
    ]
    frames = min(len(item) for item in generated)
    ground_truth = np.load(args.ground_truth).astype(np.float32)[:frames]
    save_motion_comparison(
        args.output,
        [ground_truth, *generated],
        ["Ground truth", "Text only", "Text + head IMU", "Text + wrists IMU"],
        args.caption,
        fps=20,
    )
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
