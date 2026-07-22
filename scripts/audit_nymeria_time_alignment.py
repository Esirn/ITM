#!/usr/bin/env python3
"""Audit narration-to-body timing across downloaded Nymeria sequences."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import subprocess

import imageio_ffmpeg
import numpy as np
from projectaria_tools.core import data_provider


def narration_range(sequence: Path) -> tuple[float, float]:
    rows: list[dict[str, str]] = []
    for path in sorted((sequence / "narration").glob("*.csv")):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("no narration rows")
    return (
        min(float(row["start_time"]) for row in rows),
        max(float(row["end_time"]) for row in rows),
    )


def video_device_time_range(sequence: Path) -> tuple[float, float]:
    video = sequence / "video_main_rgb.mp4"
    process = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-i",
            str(video),
            "-f",
            "ffmetadata",
            "-",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    match = re.search(r"description\s*[:=]\s*\[([^\]]+)\]", process.stdout + process.stderr)
    if match is None:
        raise RuntimeError("video has no embedded device timestamps")
    timestamps_ns = [int(value) for value in re.findall(r"\d+", match.group(1))]
    return timestamps_ns[0] / 1e9, timestamps_ns[-1] / 1e9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--start-tolerance", type=float, default=1.0)
    parser.add_argument(
        "--clock-source", choices=("vrs", "video"), default="vrs"
    )
    args = parser.parse_args()

    results = []
    for sequence in sorted(args.root.iterdir()):
        xdata_path = sequence / "body" / "xdata.npz"
        vrs_path = sequence / "recording_head" / "data" / "motion.vrs"
        if not vrs_path.is_file():
            vrs_path = sequence / "recording_head" / "data" / "data.vrs"
        required_clock = (
            vrs_path if args.clock_source == "vrs" else sequence / "video_main_rgb.mp4"
        )
        if not sequence.is_dir() or not xdata_path.is_file() or not required_clock.is_file():
            continue

        try:
            xdata = np.load(xdata_path, allow_pickle=False)
            timecode_start_ns = int(xdata["timestamps_us"][0]) * 1000
            timecode_end_ns = int(xdata["timestamps_us"][-1]) * 1000
            if args.clock_source == "vrs":
                provider = data_provider.create_vrs_data_provider(str(vrs_path))
                if provider is None:
                    raise RuntimeError("VRS provider creation failed")
                device_start = (
                    provider.convert_from_timecode_to_device_time_ns(timecode_start_ns)
                    / 1e9
                )
                device_end = (
                    provider.convert_from_timecode_to_device_time_ns(timecode_end_ns)
                    / 1e9
                )
            else:
                device_start, device_end = video_device_time_range(sequence)
            csv_start, csv_end = narration_range(sequence)
            start_delta = csv_start - device_start
            end_delta = csv_end - device_end
            if args.clock_source == "vrs":
                status = "direct" if abs(start_delta) <= args.start_tolerance else "offset"
            else:
                status = "inside" if device_start <= csv_start <= csv_end <= device_end else "outside"
            results.append(
                (sequence.name, status, start_delta, end_delta, csv_end - csv_start)
            )
        except Exception as error:  # Keep auditing after a corrupt sequence.
            results.append((sequence.name, "error", float("nan"), float("nan"), 0.0))
            print(f"ERROR {sequence.name}: {error}")

    direct = [row for row in results if row[1] in {"direct", "inside"}]
    offset = [row for row in results if row[1] in {"offset", "outside"}]
    errors = [row for row in results if row[1] == "error"]
    print("\nsequence,status,start_delta_s,end_delta_s,narration_span_s")
    for row in results:
        print(f"{row[0]},{row[1]},{row[2]:.6f},{row[3]:.6f},{row[4]:.3f}")
    print(
        f"\nSUMMARY total={len(results)} direct={len(direct)} "
        f"offset={len(offset)} error={len(errors)}"
    )
    if direct:
        starts = np.asarray([row[2] for row in direct])
        print(
            f"DIRECT start delta mean={starts.mean():.6f}s "
            f"std={starts.std():.6f}s range=[{starts.min():.6f}, {starts.max():.6f}]s"
        )


if __name__ == "__main__":
    main()
