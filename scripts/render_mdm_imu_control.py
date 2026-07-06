#!/usr/bin/env python
"""Render a controlled MDM NPZ with per-case target IMU traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from itm.visualization.skeleton import save_control_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.results) as data:
        motions = data["motion"]
        acceleration = data["acceleration"]
        orientation = data["orientation"]
        masks = data["sensor_mask"]
        metadata = json.loads(str(data["metadata"]))
    cases = metadata["cases"]
    slots = [int(np.flatnonzero(mask)[0]) for mask in masks]
    save_control_comparison(
        args.output,
        list(motions),
        [case["label"] for case in cases],
        [case["text"] or "[text disabled]" for case in cases],
        [acceleration[index, :, slot] for index, slot in enumerate(slots)],
        [orientation[index, :, slot, :, 0] for index, slot in enumerate(slots)],
        [f"{case['sensor_config']} | slot {slot}" for case, slot in zip(cases, slots)],
    )
    print(f"visualization: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
