#!/usr/bin/env python
"""Render official MDM results.npy with ITM's maintained visualizer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from itm.visualization.skeleton import save_motion_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    payload = np.load(args.results, allow_pickle=True).item()
    motions = np.asarray(payload["motion"], dtype=np.float32)
    count = min(len(motions), args.max_samples)
    joints = [motions[index, :, :, : int(payload["lengths"][index])].transpose(2, 0, 1) for index in range(count)]
    labels = [f"MDM sample {index + 1}" for index in range(count)]
    texts = [str(value) for value in payload["text"][:count]]
    caption = texts[0] if len(set(texts)) == 1 else " | ".join(texts)
    save_motion_comparison(args.output, joints, labels, caption, fps=args.fps)
    print(f"samples: {count}")
    print(f"output: {Path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
