#!/usr/bin/env python3
"""Print VRS streams and demonstrate DEVICE_TIME <-> TIME_CODE conversion."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions


def narration_start(sequence: Path) -> float | None:
    starts: list[float] = []
    for path in sorted((sequence / "narration").glob("*.csv")):
        with path.open(newline="") as fp:
            for row in csv.DictReader(fp):
                if row.get("start_time"):
                    starts.append(float(row["start_time"]))
    return min(starts) if starts else None


def print_range(provider, label: str, domain: TimeDomain) -> None:
    try:
        first = provider.get_first_time_ns_all_streams(domain)
        last = provider.get_last_time_ns_all_streams(domain)
        print(f"  {label:<11} {first:>16} .. {last:<16} ns  ({first/1e9:.6f} .. {last/1e9:.6f} s)")
    except Exception as error:
        print(f"  {label:<11} unavailable ({type(error).__name__}: {error})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", type=Path, help="Nymeria sequence directory")
    parser.add_argument(
        "--vrs",
        type=Path,
        help="VRS path. Defaults to recording_head/data/motion.vrs, then data.vrs.",
    )
    args = parser.parse_args()

    sequence = args.sequence
    vrs_path = args.vrs
    if vrs_path is None:
        motion = sequence / "recording_head" / "data" / "motion.vrs"
        data = sequence / "recording_head" / "data" / "data.vrs"
        vrs_path = motion if motion.is_file() else data

    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise RuntimeError(f"could not open VRS: {vrs_path}")

    print(f"VRS: {vrs_path}")
    print(f"time_sync_mode: {provider.get_time_sync_mode()}")

    print("\nStreams:")
    for stream_id in provider.get_all_streams():
        print(
            f"  {stream_id}: {provider.get_label_from_stream_id(stream_id)}, "
            f"active={provider.check_stream_is_active(stream_id)}"
        )

    print("\nAll-stream time ranges:")
    print_range(provider, "DEVICE", TimeDomain.DEVICE_TIME)
    print_range(provider, "TIME_CODE", TimeDomain.TIME_CODE)
    print_range(provider, "RECORD", TimeDomain.RECORD_TIME)
    print_range(provider, "HOST", TimeDomain.HOST_TIME)

    print("\nPer-stream DEVICE_TIME and TIME_CODE ranges:")
    for stream_id in provider.get_all_streams():
        if not provider.check_stream_is_active(stream_id):
            continue
        label = provider.get_label_from_stream_id(stream_id)
        print(f"  {stream_id} {label}")
        for domain_label, domain in (
            ("DEVICE", TimeDomain.DEVICE_TIME),
            ("TIME_CODE", TimeDomain.TIME_CODE),
        ):
            first = provider.get_first_time_ns(stream_id, domain)
            last = provider.get_last_time_ns(stream_id, domain)
            print(
                f"    {domain_label:<9} {first:>16} .. {last:<16} ns  "
                f"({first/1e9:.6f} .. {last/1e9:.6f} s)"
            )

    print("\nConversion proof:")
    csv_start_s = narration_start(sequence)
    if csv_start_s is not None:
        device_ns = round(csv_start_s * 1e9)
        timecode_ns = provider.convert_from_device_time_to_timecode_ns(device_ns)
        roundtrip_device_ns = provider.convert_from_timecode_to_device_time_ns(timecode_ns)
        print(f"  CSV start_time          {csv_start_s:.9f} s")
        print(f"  as DEVICE_TIME          {device_ns} ns")
        print(f"  converted TIME_CODE     {timecode_ns} ns ({timecode_ns/1e9:.9f} s)")
        print(f"  roundtrip DEVICE_TIME   {roundtrip_device_ns} ns ({roundtrip_device_ns/1e9:.9f} s)")
    else:
        print("  no narration start_time found")

    xdata_path = sequence / "body" / "xdata.npz"
    if xdata_path.is_file() and csv_start_s is not None:
        xdata = np.load(xdata_path, allow_pickle=False)
        xsens_ns = xdata["timestamps_us"].astype(np.int64) * 1000
        print(f"  Xsens first TIME_CODE   {int(xsens_ns[0])} ns ({xsens_ns[0]/1e9:.9f} s)")
        print(f"  delta CSV-Xsens start   {(timecode_ns - int(xsens_ns[0]))/1e9:.9f} s")

    print("\nOne record example:")
    for label in ("camera-rgb", "imu-left", "imu-right"):
        try:
            stream_id = provider.get_stream_id_from_label(label)
        except Exception:
            continue
        device_t = provider.get_first_time_ns(stream_id, TimeDomain.DEVICE_TIME)
        timecode_t = provider.convert_from_device_time_to_timecode_ns(device_t)
        print(f"  {label}: first DEVICE={device_t}, converted TIME_CODE={timecode_t}")
        if label.startswith("imu"):
            imu = provider.get_imu_data_by_time_ns(
                stream_id, device_t, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST
            )
            print(f"    capture_timestamp_ns={imu.capture_timestamp_ns}")
            print(f"    accel_msec2={list(imu.accel_msec2)}")
            print(f"    gyro_radsec={list(imu.gyro_radsec)}")
            break


if __name__ == "__main__":
    main()
