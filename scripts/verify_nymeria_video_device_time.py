#!/usr/bin/env python3
"""Check whether narration CSV times fall in the head RGB DEVICE_TIME range."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess


def read_narration_range(sequence: Path) -> tuple[float, float, int]:
    rows: list[dict[str, str]] = []
    for path in sorted((sequence / "narration").glob("*.csv")):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("missing_narration")
    starts = [float(row["start_time"]) for row in rows if row.get("start_time")]
    ends = [float(row["end_time"]) for row in rows if row.get("end_time")]
    if not starts or not ends:
        raise RuntimeError("missing_narration_times")
    return min(starts), max(ends), len(rows)


def read_video_device_range(video: Path) -> tuple[float, float, int]:
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags=description",
            "-of",
            "default=nw=1:nk=1",
            str(video),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    timestamps_ns = [int(value) for value in re.findall(r"\d+", process.stdout)]
    if not timestamps_ns:
        raise RuntimeError("missing_video_device_timestamps")
    return timestamps_ns[0] / 1e9, timestamps_ns[-1] / 1e9, len(timestamps_ns)


def audit_sequence(sequence: Path) -> dict[str, object]:
    result: dict[str, object] = {"sequence_id": sequence.name}
    try:
        csv_start, csv_end, row_count = read_narration_range(sequence)
        video_start, video_end, video_frames = read_video_device_range(
            sequence / "video_main_rgb.mp4"
        )
        result.update(
            {
                "csv_start_device_s": csv_start,
                "csv_end_device_s": csv_end,
                "narration_rows": row_count,
                "video_start_device_s": video_start,
                "video_end_device_s": video_end,
                "video_frames": video_frames,
                "start_delta_to_video_s": csv_start - video_start,
                "end_delta_to_video_s": csv_end - video_end,
                "status": "inside"
                if video_start <= csv_start <= csv_end <= video_end
                else "outside",
            }
        )
    except Exception as error:
        result["status"] = str(error)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = [audit_sequence(path) for path in sorted(args.root.iterdir()) if path.is_dir()]
    if args.output:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    inside = [r for r in results if r["status"] == "inside"]
    outside = [r for r in results if r["status"] == "outside"]
    other = [r for r in results if r["status"] not in {"inside", "outside"}]
    print(
        f"SUMMARY total={len(results)} inside={len(inside)} "
        f"outside={len(outside)} other={len(other)}"
    )
    print("NON_INSIDE")
    for row in outside + other:
        print(
            row["sequence_id"],
            row["status"],
            row.get("start_delta_to_video_s"),
            row.get("end_delta_to_video_s"),
        )


if __name__ == "__main__":
    main()
