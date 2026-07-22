#!/usr/bin/env python
"""Render readable per-clip Nymeria GT/MDM/ITM pilot comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from itm.visualization.skeleton import save_control_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.results) as data:
        motion = data["motion"]
        gt = data["gt"]
        acceleration = data["acceleration"]
        orientation = data["orientation"]
        masks = data["sensor_mask"]
        metadata = json.loads(str(data["metadata"]))
    grouped: dict[str, list[int]] = {}
    for index, case in enumerate(metadata["cases"]):
        grouped.setdefault(case["motion_id"], []).append(index)
    index_rows = []
    for motion_id, indices in grouped.items():
        first = indices[0]
        cases = [metadata["cases"][index] for index in indices]
        motions = [gt[first], *(motion[index] for index in indices)]
        labels = ["Nymeria Ground truth", *(case["label"].split(" | ")[-1] for case in cases)]
        texts = [cases[0]["text"], *(case["text"] for case in cases)]
        trace_acc, trace_ori, trace_labels = [], [], []
        for index in [first, *indices]:
            slot = int(np.flatnonzero(masks[index])[0])
            trace_acc.append(acceleration[index, :, slot])
            trace_ori.append(orientation[index, :, slot, :, 0])
            trace_labels.append("target IMU | " + metadata["cases"][index]["sensor_config"])
        output = args.output_dir / f"{motion_id}.gif"
        save_control_comparison(
            output, motions, labels, texts, trace_acc, trace_ori, trace_labels
        )
        index_rows.append({"motion_id": motion_id, "caption": cases[0]["text"], "visualization": str(output.resolve())})
        print(f"visualization: {output}")
    (args.output_dir / "index.json").write_text(
        json.dumps(index_rows, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
