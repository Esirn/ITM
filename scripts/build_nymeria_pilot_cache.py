#!/usr/bin/env python
"""Build a small exactly time-aligned Nymeria cache for zero-shot ITM tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from itm.data.nymeria import (
    convert_clip,
    device_seconds_to_timecode_us,
    narration_text,
    nearest_time_slice,
    read_motion_narrations,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--alignment-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-clips", type=int, default=8)
    parser.add_argument("--min-seconds", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=9.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audits = {row["sequence_id"]: row for row in json.loads(args.alignment_audit.read_text()) if row.get("status") == "valid"}
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    candidates = []
    for sequence_id, audit in audits.items():
        sequence = args.root / sequence_id
        csv_path = sequence / "narration" / "motion_narration.csv"
        xdata_path = sequence / "body" / "xdata.npz"
        if not csv_path.is_file() or not xdata_path.is_file():
            continue
        for row_index, row in enumerate(read_motion_narrations(csv_path)):
            duration = float(row["end_time"]) - float(row["start_time"])
            text = narration_text(row)
            if args.min_seconds <= duration <= args.max_seconds and text:
                candidates.append((sequence_id, row_index, row, audit, xdata_path, duration, text))
    # Deterministic spread over recordings and durations for reproducibility.
    candidates.sort(key=lambda item: (item[0], -item[5], item[1]))
    selected = []
    used_sequences = set()
    for item in candidates:
        if item[0] not in used_sequences:
            selected.append(item)
            used_sequences.add(item[0])
        if len(selected) == args.max_clips:
            break
    failures = []
    for sequence_id, row_index, row, audit, xdata_path, duration, text in selected:
        try:
            with np.load(xdata_path) as data:
                start_us = device_seconds_to_timecode_us(float(row["start_time"]), audit)
                end_us = device_seconds_to_timecode_us(float(row["end_time"]), audit)
                clip_slice = nearest_time_slice(data["timestamps_us"], start_us, end_us)
                joints, acceleration, orientation = convert_clip(
                    data["segment_tXYZ"][clip_slice],
                    data["sensor_freeAcceleration"][clip_slice],
                    data["sensor_qWXYZ"][clip_slice],
                )
                fps = float(np.asarray(data["frameRate"]).reshape(-1)[0])
                timestamps = data["timestamps_us"][clip_slice].astype(np.int64)
        except (OSError, ValueError) as error:
            failures.append({"sequence_id": sequence_id, "row": row_index, "error": str(error)})
            continue
        motion_id = f"nymeria_{sequence_id}_{row_index:04d}"
        cache_path = args.output / f"{motion_id}.npz"
        np.savez_compressed(
            cache_path, acceleration=acceleration, orientation=orientation,
            joints=joints, fps=np.float32(fps), timestamps_us=timestamps,
        )
        records.append({
            "motion_id": motion_id, "sequence_id": sequence_id, "narration_row": row_index,
            "caption": text, "duration_s": duration, "fps": fps,
            "imu_path": str(cache_path.resolve()), "source_xdata": str(xdata_path),
            "time_alignment": "Head DEVICE_TIME -> VRS TIME_CODE -> Xsens timestamps_us",
            "available_sensor_configs": ["head", "wrists"],
        })
    manifest = args.output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    cases = []
    for row in records:
        common = {
            "motion_id": row["motion_id"],
            "text": row["caption"],
            "text_source": row["motion_id"],
            "imu_source": row["motion_id"],
        }
        cases.extend(
            [
                {**common, "label": f"{row['motion_id']} | MDM Text-only", "sensor_config": "head", "imu_scale": 0.0},
                {**common, "label": f"{row['motion_id']} | ITM head", "sensor_config": "head", "imu_scale": 1.0},
                {**common, "label": f"{row['motion_id']} | ITM wrists", "sensor_config": "wrists", "imu_scale": 1.0},
            ]
        )
    (args.output / "pilot_spec.json").write_text(
        json.dumps(cases, indent=2), encoding="utf-8"
    )
    (args.output / "README.md").write_text(
        "# Nymeria zero-shot pilot cache\n\n"
        "This cache uses real Xsens IMUs and is evaluated without Nymeria fine-tuning. "
        "It is a cross-dataset/domain-shift diagnostic, not a formal in-domain score.\n",
        encoding="utf-8",
    )
    (args.output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"clips={len(records)} failures={len(failures)} manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
