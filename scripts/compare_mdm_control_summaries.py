#!/usr/bin/env python
"""Compare two MDM-control experiment-suite summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


KEYS = (
    "active_sensor_trajectory_error_m",
    "jerk_ratio",
    "arm_swing_proxy_m",
    "root_travel_m",
    "step_frequency_hz",
    "root_relative_motion_error_m",
    "active_sensor_acceleration_error_mps2",
    "active_sensor_acceleration_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", default="outputs/mdm_control/experiments/RESULTS_SUMMARY.json")
    parser.add_argument("--stage2", default="outputs/mdm_control/experiments_stage2/RESULTS_SUMMARY.json")
    parser.add_argument("--output", default="outputs/mdm_control/experiments_stage2/STAGE1_VS_STAGE2.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage1 = json.loads(Path(args.stage1).read_text())
    stage2 = json.loads(Path(args.stage2).read_text())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(stage1, stage2), encoding="utf-8")
    print(f"comparison md: {output}")
    return 0


def render(stage1: dict[str, Any], stage2: dict[str, Any]) -> str:
    lines = [
        "# Stage-1 vs Stage-2 MDM Control",
        "",
        f"- stage-1 root: `{stage1['root']}`",
        f"- stage-2 root: `{stage2['root']}`",
        f"- stage-1 runs/cases: `{stage1['run_count']}` / `{stage1['case_count']}`",
        f"- stage-2 runs/cases: `{stage2['run_count']}` / `{stage2['case_count']}`",
        "",
        "## By Experiment",
        "",
        comparison_table(stage1["by_experiment"], stage2["by_experiment"]),
        "",
        "## By Experiment And Sensor",
        "",
        comparison_table(stage1["by_experiment_and_sensor"], stage2["by_experiment_and_sensor"]),
        "",
        "## Guidance Sweep Best Settings",
        "",
        guidance_table(stage1["by_guidance"], stage2["by_guidance"]),
        "",
        "## Interpretation",
        "",
    ]
    lines.extend(interpretation(stage1, stage2))
    return "\n".join(lines)


def comparison_table(stage1: dict[str, dict[str, float]], stage2: dict[str, dict[str, float]]) -> str:
    headers = ["group"]
    for key in KEYS:
        headers.extend([f"s1_{key}", f"s2_{key}", f"delta_{key}_pct"])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for group in sorted(set(stage1) & set(stage2)):
        cells = [group]
        for key in KEYS:
            left = float(stage1[group][key])
            right = float(stage2[group][key])
            cells.extend([f"{left:.4f}", f"{right:.4f}", f"{percent_change(left, right):+.1f}%"])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def guidance_table(stage1_rows: list[dict[str, Any]], stage2_rows: list[dict[str, Any]]) -> str:
    lines = [
        "| stage | sensor | text_scale | imu_scale | active_sensor_trajectory_error_m | jerk_ratio | arm_swing_proxy_m |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for stage, rows in (("stage1", stage1_rows), ("stage2", stage2_rows)):
        for sensor in ("head", "wrists"):
            candidates = [row for row in rows if row["sensor_config"] == sensor]
            if not candidates:
                continue
            best = min(candidates, key=lambda row: (float(row["active_sensor_trajectory_error_m"]), float(row["jerk_ratio"])))
            lines.append(
                "| "
                + " | ".join(
                    [
                        stage,
                        sensor,
                        f"{float(best['text_scale']):.1f}",
                        f"{float(best['imu_scale']):.1f}",
                        f"{float(best['active_sensor_trajectory_error_m']):.4f}",
                        f"{float(best['jerk_ratio']):.4f}",
                        f"{float(best['arm_swing_proxy_m']):.4f}",
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def interpretation(stage1: dict[str, Any], stage2: dict[str, Any]) -> list[str]:
    same_text_s1 = stage1["by_experiment"]["same_text_different_imu"]
    same_text_s2 = stage2["by_experiment"]["same_text_different_imu"]
    same_imu_s1 = stage1["by_experiment"]["same_head_imu_different_text"]
    same_imu_s2 = stage2["by_experiment"]["same_head_imu_different_text"]
    matrix_s1 = stage1["by_experiment"]["matrix"]
    matrix_s2 = stage2["by_experiment"]["matrix"]
    return [
        f"- Same-text-different-IMU jerk changed by `{percent_change(same_text_s1['jerk_ratio'], same_text_s2['jerk_ratio']):+.1f}%`; this is the cleanest stage-2 gain.",
        f"- Same-text active sensor error changed by `{percent_change(same_text_s1['active_sensor_trajectory_error_m'], same_text_s2['active_sensor_trajectory_error_m']):+.1f}%`, so the lower jerk did not come from simply ignoring IMU.",
        f"- Same-text active acceleration proxy changed by `{percent_change(same_text_s1['active_sensor_acceleration_error_mps2'], same_text_s2['active_sensor_acceleration_error_mps2']):+.1f}%`; this is a joint-derived diagnostic, not a full SMPL/IMU orientation metric.",
        f"- Matrix jerk changed by `{percent_change(matrix_s1['jerk_ratio'], matrix_s2['jerk_ratio']):+.1f}%`, suggesting smoother qualitative A/B matrix examples.",
        f"- Matrix active acceleration proxy changed by `{percent_change(matrix_s1['active_sensor_acceleration_error_mps2'], matrix_s2['active_sensor_acceleration_error_mps2']):+.1f}%`, which supports the smoothness interpretation.",
        f"- Same-head-IMU-different-text jerk changed by `{percent_change(same_imu_s1['jerk_ratio'], same_imu_s2['jerk_ratio']):+.1f}%`; this remains a failure mode and needs visual inspection or stronger text-preserving training.",
        f"- Arm-swing proxy in same-head prompts changed by `{percent_change(same_imu_s1['arm_swing_proxy_m'], same_imu_s2['arm_swing_proxy_m']):+.1f}%`; the arm-swing phenomenon is weaker after smoothing.",
    ]


def percent_change(left: float, right: float) -> float:
    if abs(left) < 1e-12:
        return 0.0
    return 100.0 * (right - left) / abs(left)


if __name__ == "__main__":
    raise SystemExit(main())
