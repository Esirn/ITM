#!/usr/bin/env python3
"""Verify Nymeria CSV/XSens/VRS timing on a local dataset copy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from projectaria_tools.core import data_provider


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


def find_head_vrs(sequence: Path) -> tuple[Path, str]:
    motion = sequence / "recording_head" / "data" / "motion.vrs"
    if motion.is_file():
        return motion, "motion"
    data = sequence / "recording_head" / "data" / "data.vrs"
    if data.is_file():
        return data, "data"
    raise RuntimeError("missing_head_vrs")


def xsens_stats(xdata_path: Path) -> dict[str, int | float | bool]:
    xdata = np.load(xdata_path, allow_pickle=False)
    timestamps_us = xdata["timestamps_us"].astype(np.int64)
    dt = np.diff(timestamps_us)
    nominal = 1_000_000.0 / float(np.asarray(xdata.get("frameRate", 240)).reshape(-1)[0])
    invalid = np.abs(dt - nominal) > 1000
    monotonic = bool(np.all(dt > 0))
    return {
        "start_timecode_ns": int(timestamps_us[0]) * 1000,
        "end_timecode_ns": int(timestamps_us[-1]) * 1000,
        "frames": int(timestamps_us.shape[0]),
        "frame_rate": float(np.asarray(xdata.get("frameRate", 240)).reshape(-1)[0]),
        "monotonic": monotonic,
        "invalid_dt_count": int(invalid.sum()),
    }


def audit_sequence(sequence: Path, start_tolerance_s: float) -> dict[str, object]:
    result: dict[str, object] = {"sequence_id": sequence.name}
    try:
        xdata_path = sequence / "body" / "xdata.npz"
        if not xdata_path.is_file():
            raise RuntimeError("missing_xdata")
        result["xdata_path"] = str(xdata_path)
        result.update({f"xsens_{k}": v for k, v in xsens_stats(xdata_path).items()})

        csv_start_s, csv_end_s, row_count = read_narration_range(sequence)
        result["csv_start_device_s"] = csv_start_s
        result["csv_end_device_s"] = csv_end_s
        result["narration_rows"] = row_count

        vrs_path, vrs_type = find_head_vrs(sequence)
        result["vrs_path"] = str(vrs_path)
        result["vrs_type"] = vrs_type
        provider = data_provider.create_vrs_data_provider(str(vrs_path))
        if provider is None:
            raise RuntimeError("vrs_provider_creation_failed")
        result["vrs_readable"] = True

        csv_start_timecode_ns = provider.convert_from_device_time_to_timecode_ns(
            round(csv_start_s * 1e9)
        )
        csv_end_timecode_ns = provider.convert_from_device_time_to_timecode_ns(
            round(csv_end_s * 1e9)
        )
        result["csv_start_timecode_ns"] = int(csv_start_timecode_ns)
        result["csv_end_timecode_ns"] = int(csv_end_timecode_ns)

        xsens_start = int(result["xsens_start_timecode_ns"])
        xsens_end = int(result["xsens_end_timecode_ns"])
        start_delta_s = (int(csv_start_timecode_ns) - xsens_start) / 1e9
        end_delta_s = (int(csv_end_timecode_ns) - xsens_end) / 1e9
        result["start_delta_to_xsens_s"] = start_delta_s
        result["end_delta_to_xsens_s"] = end_delta_s
        result["csv_inside_xsens_timecode_range"] = (
            xsens_start <= int(csv_start_timecode_ns) <= int(csv_end_timecode_ns) <= xsens_end
        )
        result["xsens_start_as_head_device_s"] = (
            provider.convert_from_timecode_to_device_time_ns(xsens_start) / 1e9
        )
        result["status"] = (
            "valid"
            if abs(start_delta_s) <= start_tolerance_s
            and bool(result["csv_inside_xsens_timecode_range"])
            else "timing_offset"
        )
    except Exception as error:
        result["status"] = str(error)
        result.setdefault("vrs_readable", False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--start-tolerance-s", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = [
        audit_sequence(path, args.start_tolerance_s)
        for path in sorted(args.root.iterdir())
        if path.is_dir()
    ]
    if args.output:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    valid = [r for r in results if r["status"] == "valid"]
    offsets = [r for r in results if r["status"] == "timing_offset"]
    other = [r for r in results if r["status"] not in {"valid", "timing_offset"}]
    print(
        f"SUMMARY total={len(results)} valid={len(valid)} "
        f"timing_offset={len(offsets)} other={len(other)}"
    )
    if valid:
        starts = np.array([r["start_delta_to_xsens_s"] for r in valid], dtype=float)
        ends = np.array([r["end_delta_to_xsens_s"] for r in valid], dtype=float)
        print(
            "VALID start_delta_s "
            f"mean={starts.mean():.6f} std={starts.std():.6f} "
            f"range=[{starts.min():.6f}, {starts.max():.6f}]"
        )
        print(
            "VALID end_delta_s "
            f"mean={ends.mean():.6f} std={ends.std():.6f} "
            f"range=[{ends.min():.6f}, {ends.max():.6f}]"
        )
    print("NON_VALID")
    for row in offsets + other:
        print(
            row["sequence_id"],
            row["status"],
            row.get("start_delta_to_xsens_s"),
            row.get("end_delta_to_xsens_s"),
        )


if __name__ == "__main__":
    main()
