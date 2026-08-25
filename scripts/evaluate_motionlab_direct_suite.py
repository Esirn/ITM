#!/usr/bin/env python3
"""Summarize fixed-seed MotionLab Text-only versus direct Text+IMU outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ACTIVE_JOINTS = {"head": (15,), "wrists": (20, 21)}


def root_relative(motion: np.ndarray) -> np.ndarray:
    return motion - motion[:, :1]


def active_error(generated, target, joints):
    generated = root_relative(generated)[:, joints]
    target = root_relative(target)[:, joints]
    return float(np.linalg.norm(generated - target, axis=-1).mean())


def motion_difference(first, second):
    difference = root_relative(first) - root_relative(second)
    return float(np.sqrt(np.mean(difference**2)))


def jerk_ratio(generated, target):
    if len(generated) < 4:
        return 0.0
    def jerk(motion):
        values = root_relative(motion)
        third = values[3:] - 3 * values[2:-1] + 3 * values[1:-2] - values[:-3]
        return float(np.linalg.norm(third, axis=-1).mean())
    return jerk(generated) / max(jerk(target), 1e-8)


def root_travel(motion):
    if len(motion) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(motion[:, 0], axis=0), axis=-1).sum())


def step_frequency(motion, fps=20.0):
    """Estimate cadence from alternating ankle-height minima."""
    if len(motion) < 5:
        return 0.0
    signal = motion[:, 10, 1] - motion[:, 11, 1]
    centered = signal - signal.mean()
    crossings = np.count_nonzero(centered[1:] * centered[:-1] < 0)
    duration = (len(motion) - 1) / fps
    return float(crossings / max(2.0 * duration, 1e-8))


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--text-only", type=Path, required=True)
    parser.add_argument("--text-head", type=Path, required=True)
    parser.add_argument("--text-wrists", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    outputs = {
        "text_only": np.load(args.text_only)["joints"],
        "head": np.load(args.text_head)["joints"],
        "wrists": np.load(args.text_wrists)["joints"],
    }
    records = []
    for index, motion_id in enumerate(spec["motion_ids"]):
        length = int(spec["lengths"][index])
        target = np.load(spec["ground_truth_paths"][index]).astype(np.float32)[:length]
        text = outputs["text_only"][index, :length]
        record = {
            "motion_id": motion_id,
            "caption": spec["captions"][index],
            "length": length,
            "configs": {},
        }
        for name in ("head", "wrists"):
            controlled = outputs[name][index, :length]
            text_error = active_error(text, target, ACTIVE_JOINTS[name])
            control_error = active_error(controlled, target, ACTIVE_JOINTS[name])
            record["configs"][name] = {
                "text_only_active_error_m": text_error,
                "text_imu_active_error_m": control_error,
                "active_error_change_m": control_error - text_error,
                "improved": control_error < text_error,
                "motion_difference_rms_m": motion_difference(text, controlled),
                "text_only_jerk_ratio": jerk_ratio(text, target),
                "text_imu_jerk_ratio": jerk_ratio(controlled, target),
                "text_only_root_travel_m": root_travel(text),
                "text_imu_root_travel_m": root_travel(controlled),
                "text_only_step_frequency_hz": step_frequency(text),
                "text_imu_step_frequency_hz": step_frequency(controlled),
            }
        records.append(record)
    summary = {}
    for name in ("head", "wrists"):
        items = [record["configs"][name] for record in records]
        summary[name] = {
            "samples": len(items),
            "improved_samples": sum(item["improved"] for item in items),
            "improvement_rate": sum(item["improved"] for item in items) / len(items),
            "text_only_active_error_m": summarize(
                [item["text_only_active_error_m"] for item in items]
            ),
            "text_imu_active_error_m": summarize(
                [item["text_imu_active_error_m"] for item in items]
            ),
            "active_error_change_m": summarize(
                [item["active_error_change_m"] for item in items]
            ),
            "motion_difference_rms_m": summarize(
                [item["motion_difference_rms_m"] for item in items]
            ),
            "text_only_jerk_ratio": summarize(
                [item["text_only_jerk_ratio"] for item in items]
            ),
            "text_imu_jerk_ratio": summarize(
                [item["text_imu_jerk_ratio"] for item in items]
            ),
            "text_only_root_travel_m": summarize(
                [item["text_only_root_travel_m"] for item in items]
            ),
            "text_imu_root_travel_m": summarize(
                [item["text_imu_root_travel_m"] for item in items]
            ),
            "text_only_step_frequency_hz": summarize(
                [item["text_only_step_frequency_hz"] for item in items]
            ),
            "text_imu_step_frequency_hz": summarize(
                [item["text_imu_step_frequency_hz"] for item in items]
            ),
        }
    result = {
        "protocol": {
            "spec": str(args.spec.resolve()),
            "seed": spec["seed"],
            "coordinate_protocol": "per-frame root-relative HumanML joints",
        },
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    markdown = [
        "# MotionLab Direct IMU Test-20",
        "",
        "All variants use the same text, length, seed, and initial diffusion noise.",
        "Active-joint errors are root-relative and are diagnostic rather than real IMU errors.",
        "",
        "| Config | Text-only error | Text+IMU error | Change | Improved | Motion difference | Jerk Text | Jerk Text+IMU |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("head", "wrists"):
        item = summary[name]
        markdown.append(
            f"| {name} | {item['text_only_active_error_m']['mean']:.4f} m | "
            f"{item['text_imu_active_error_m']['mean']:.4f} m | "
            f"{item['active_error_change_m']['mean']:+.4f} m | "
            f"{item['improved_samples']}/{item['samples']} | "
            f"{item['motion_difference_rms_m']['mean']:.4f} m | "
            f"{item['text_only_jerk_ratio']['mean']:.3f} | "
            f"{item['text_imu_jerk_ratio']['mean']:.3f} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
