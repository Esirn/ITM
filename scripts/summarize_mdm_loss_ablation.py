#!/usr/bin/env python
"""Create paper-ready tables for the strict MDM-control loss ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VARIANTS = ("diffusion", "trajectory", "trajectory_velocity", "full_regularized")
EXPERIMENTS = ("same_text_different_imu", "same_head_imu_different_text", "matrix")
METRICS = (
    "active_sensor_trajectory_error_m",
    "active_sensor_acceleration_error_mps2",
    "jerk_ratio",
    "arm_swing_proxy_m",
    "root_travel_m",
    "step_frequency_hz",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/mdm_control/loss_ablation")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    summaries = {}
    for variant in VARIANTS:
        path = root / f"experiments_{variant}/RESULTS_SUMMARY.json"
        summaries[variant] = json.loads(path.read_text(encoding="utf-8"))
    baseline = summaries["diffusion"]
    rows = []
    for experiment in EXPERIMENTS:
        for variant in VARIANTS:
            values = summaries[variant]["by_experiment"][experiment]
            reference = baseline["by_experiment"][experiment]
            row = {"experiment": experiment, "variant": variant}
            for metric in METRICS:
                row[metric] = values[metric]
                row[f"{metric}_change_percent"] = _percent_change(
                    reference[metric], values[metric]
                )
            rows.append(row)
    result = {
        "root": str(root.resolve()),
        "protocol": {
            "resume": "stage1_full_pilot_v2.pt",
            "seed": 1234,
            "epochs": 8,
            "guidance": "legacy",
            "run_count_per_variant": summaries["diffusion"]["run_count"],
            "case_count_per_variant": summaries["diffusion"]["case_count"],
        },
        "rows": rows,
    }
    output_json = Path(args.output_json) if args.output_json else root / "LOSS_ABLATION_SUMMARY.json"
    output_md = Path(args.output_md) if args.output_md else root / "LOSS_ABLATION_SUMMARY.md"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    output_md.write_text(_markdown(result), encoding="utf-8")
    print(f"summary json: {output_json}")
    print(f"summary md: {output_md}")
    return 0


def _percent_change(reference, value):
    return None if abs(reference) < 1e-12 else (value - reference) / reference * 100.0


def _markdown(result):
    labels = {
        "same_text_different_imu": "Same text, different IMU",
        "same_head_imu_different_text": "Same head IMU, different text",
        "matrix": "Text/IMU matrix",
    }
    lines = [
        "# Strict Loss Ablation",
        "",
        "All variants resume the same Stage-1 checkpoint and use the same seed, data, epochs, sensor configurations, and legacy guidance.",
        "",
    ]
    for experiment in EXPERIMENTS:
        lines.extend(
            [
                f"## {labels[experiment]}",
                "",
                "| Variant | Active trajectory (m) | Acc. proxy (m/s^2) | Jerk ratio | Arm swing (m) | Root travel (m) | Step freq. (Hz) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["rows"]:
            if row["experiment"] != experiment:
                continue
            lines.append(
                "| {variant} | {active_sensor_trajectory_error_m:.4f} | "
                "{active_sensor_acceleration_error_mps2:.4f} | {jerk_ratio:.4f} | "
                "{arm_swing_proxy_m:.4f} | {root_travel_m:.4f} | {step_frequency_hz:.4f} |".format(**row)
            )
        lines.append("")
    lines.extend(
        [
            "## Relative To Diffusion-only",
            "",
            "Negative values mean a reduction. Whether a reduction is desirable depends on the metric; lower trajectory error, acceleration error, and jerk are preferred, while arm swing/root travel/step frequency describe behavior rather than universal quality.",
            "",
            "| Experiment | Variant | Trajectory | Acc. proxy | Jerk | Arm swing |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["rows"]:
        if row["variant"] == "diffusion":
            continue
        lines.append(
            "| {experiment} | {variant} | {active_sensor_trajectory_error_m_change_percent:+.1f}% | "
            "{active_sensor_acceleration_error_mps2_change_percent:+.1f}% | "
            "{jerk_ratio_change_percent:+.1f}% | {arm_swing_proxy_m_change_percent:+.1f}% |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
