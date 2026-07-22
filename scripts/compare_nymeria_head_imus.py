#!/usr/bin/env python3
"""Compare the Aria headset IMUs with the Xsens head sensor in Nymeria."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions


XSENS_HEAD_SENSOR = 2
XSENS_HEAD_SEGMENT = 6
GRAVITY = 9.80665


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--sample-hz", type=float, default=20.0)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    args = parser.parse_args()

    npz_path = args.sequence / "body" / "xdata.npz"
    vrs_path = args.sequence / "recording_head" / "data" / "motion.vrs"
    xsens = np.load(npz_path, allow_pickle=False)
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise RuntimeError(f"Could not open {vrs_path}")

    n = int(xsens["frameCount"][0])
    xsens_time_ns = xsens["timestamps_us"].astype(np.int64) * 1000
    xsens_acc = xsens["sensor_freeAcceleration"].reshape(n, 17, 3)
    xsens_ori = xsens["sensor_qWXYZ"].reshape(n, 17, 4)
    xsens_gyro = xsens["segment_angularVelocity"].reshape(n, 23, 3)

    stream_info = []
    for label in ("imu-right", "imu-left"):
        stream = provider.get_stream_id_from_label(label)
        config = provider.get_imu_configuration(stream)
        stream_info.append((label, stream, config))

    start_ns = max(
        int(xsens_time_ns[0]),
        *(provider.get_first_time_ns(s, TimeDomain.TIME_CODE) for _, s, _ in stream_info),
    )
    end_ns = min(
        int(xsens_time_ns[-1]),
        *(provider.get_last_time_ns(s, TimeDomain.TIME_CODE) for _, s, _ in stream_info),
    )
    if args.max_seconds > 0:
        end_ns = min(end_ns, start_ns + int(args.max_seconds * 1e9))
    step_ns = int(1e9 / args.sample_hz)
    query_ns = np.arange(start_ns, end_ns, step_ns, dtype=np.int64)
    xsens_idx = np.searchsorted(xsens_time_ns, query_ns)
    xsens_idx = np.clip(xsens_idx, 0, n - 1)

    print(f"Sequence: {args.sequence.name}")
    print("\nXsens suit head sensor in body/xdata.npz:")
    print(f"  nominal rate: {int(xsens['frameRate'][0])} Hz")
    print("  acceleration: sensor_freeAcceleration (gravity removed)")
    print("  orientation:  sensor_qWXYZ (fused global orientation)")
    print("  physical unit: Xsens sensor mounted on the head as part of the suit")

    aria_samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    print("\nAria headset sensors in recording_head/data/motion.vrs:")
    for label, stream, config in stream_info:
        accelerations = []
        gyroscopes = []
        for timestamp_ns in query_ns:
            sample = provider.get_imu_data_by_time_ns(
                stream,
                int(timestamp_ns),
                TimeDomain.TIME_CODE,
                TimeQueryOptions.CLOSEST,
            )
            accelerations.append(np.asarray(sample.accel_msec2, dtype=np.float64))
            gyroscopes.append(np.asarray(sample.gyro_radsec, dtype=np.float64))
        acc = np.stack(accelerations)
        gyro = np.stack(gyroscopes)
        aria_samples[label] = acc, gyro
        print(
            f"  {label}: {config.sensor_model}, {config.nominal_rate_hz:g} Hz, "
            "raw acceleration (includes gravity) + angular velocity"
        )

    x_acc = xsens_acc[xsens_idx, XSENS_HEAD_SENSOR]
    x_ori = xsens_ori[xsens_idx, XSENS_HEAD_SENSOR]
    x_gyro = xsens_gyro[xsens_idx, XSENS_HEAD_SEGMENT]
    print(
        f"\nSynchronized comparison: {len(query_ns)} samples at "
        f"{args.sample_hz:g} Hz, timecode {start_ns / 1e9:.3f}--{end_ns / 1e9:.3f} s"
    )
    print(f"  first Xsens free acceleration: {x_acc[0]}")
    print(f"  first Xsens orientation WXYZ:  {x_ori[0]}")
    print(f"  Xsens free-acc norm mean/std:  {np.linalg.norm(x_acc, axis=1).mean():.3f} / {np.linalg.norm(x_acc, axis=1).std():.3f}")

    x_acc_norm = np.linalg.norm(x_acc, axis=1)
    x_gyro_norm = np.linalg.norm(x_gyro, axis=1)
    for label, (acc, gyro) in aria_samples.items():
        acc_norm = np.linalg.norm(acc, axis=1)
        gyro_norm = np.linalg.norm(gyro, axis=1)
        print(f"\n  {label} first raw acceleration: {acc[0]}")
        print(f"  {label} first gyroscope:        {gyro[0]}")
        print(f"  {label} raw-acc norm mean/std:  {acc_norm.mean():.3f} / {acc_norm.std():.3f}")
        print(
            f"  corr(|Xsens free acc|, ||Aria acc||-g absolute): "
            f"{correlation(x_acc_norm, np.abs(acc_norm - GRAVITY)):.3f}"
        )
        print(
            f"  corr(|Xsens head angular velocity|, |Aria gyro|): "
            f"{correlation(x_gyro_norm, gyro_norm):.3f}"
        )

    print("\nConclusion: these streams observe the same head motion but are not duplicates.")
    print("  VRS has two high-rate raw headset IMUs and no fused orientation stream.")
    print("  xdata.npz has one Xsens head IMU with free acceleration and fused orientation.")


if __name__ == "__main__":
    main()
