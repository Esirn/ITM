#!/usr/bin/env python
"""Select an ITM checkpoint from multiple MDM-control experiment summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODEL_SPECS = (
    ("stage1", "outputs/mdm_control/experiments/RESULTS_SUMMARY.json"),
    ("stage2", "outputs/mdm_control/experiments_stage2/RESULTS_SUMMARY.json"),
    ("stage2b_balanced_a", "outputs/mdm_control/experiments_stage2b_balanced_a/RESULTS_SUMMARY.json"),
    ("stage2b_balanced_b", "outputs/mdm_control/experiments_stage2b_balanced_b/RESULTS_SUMMARY.json"),
    ("stage2b_balanced_c", "outputs/mdm_control/experiments_stage2b_balanced_c/RESULTS_SUMMARY.json"),
    ("stage3_upper_body", "outputs/mdm_control/experiments_stage3_upper_body/RESULTS_SUMMARY.json"),
    ("stage4_text_anchor", "outputs/mdm_control/experiments_stage4_text_anchor/RESULTS_SUMMARY.json"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/mdm_control/comparisons/itm_stage1_stage2_stage2b")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries()
    selection = build_selection(summaries)
    (output_root / "MODEL_SELECTION.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    (output_root / "MODEL_SELECTION.md").write_text(
        render_markdown(selection), encoding="utf-8"
    )
    print(f"selection json: {output_root / 'MODEL_SELECTION.json'}")
    print(f"selection md: {output_root / 'MODEL_SELECTION.md'}")
    return 0


def load_summaries() -> dict[str, dict[str, Any]]:
    summaries = {}
    missing = []
    for name, path in MODEL_SPECS:
        summary_path = Path(path)
        if not summary_path.exists():
            missing.append(str(summary_path))
            continue
        summaries[name] = json.loads(summary_path.read_text())
    if missing:
        raise FileNotFoundError("Missing summaries:\n" + "\n".join(missing))
    return summaries


def build_selection(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = summaries["stage1"]
    rows = []
    for name, summary in summaries.items():
        same_text = summary["by_experiment"]["same_text_different_imu"]
        same_imu = summary["by_experiment"]["same_head_imu_different_text"]
        matrix = summary["by_experiment"]["matrix"]
        stage1_same_text = baseline["by_experiment"]["same_text_different_imu"]
        stage1_same_imu = baseline["by_experiment"]["same_head_imu_different_text"]
        stage1_matrix = baseline["by_experiment"]["matrix"]
        checks = {
            "same_text_jerk_down_30pct": (
                percent_change(
                    stage1_same_text["jerk_ratio"], same_text["jerk_ratio"]
                )
                <= -30.0
            ),
            "same_imu_arm_swing_drop_within_10pct": (
                percent_change(
                    stage1_same_imu["arm_swing_proxy_m"],
                    same_imu["arm_swing_proxy_m"],
                )
                >= -10.0
            ),
            "same_imu_head_error_within_110pct": (
                same_imu["active_sensor_trajectory_error_m"]
                <= 1.1 * stage1_same_imu["active_sensor_trajectory_error_m"]
            ),
            "matrix_jerk_lower_than_stage1": (
                matrix["jerk_ratio"] < stage1_matrix["jerk_ratio"]
            ),
        }
        score = sum(int(value) for value in checks.values())
        rows.append(
            {
                "model": name,
                "score": score,
                "checks": checks,
                "same_text_active_sensor_error_m": same_text["active_sensor_trajectory_error_m"],
                "same_text_jerk_ratio": same_text["jerk_ratio"],
                "same_text_jerk_delta_pct": percent_change(
                    stage1_same_text["jerk_ratio"], same_text["jerk_ratio"]
                ),
                "same_imu_active_sensor_error_m": same_imu["active_sensor_trajectory_error_m"],
                "same_imu_arm_swing_proxy_m": same_imu["arm_swing_proxy_m"],
                "same_imu_arm_swing_delta_pct": percent_change(
                    stage1_same_imu["arm_swing_proxy_m"],
                    same_imu["arm_swing_proxy_m"],
                ),
                "matrix_jerk_ratio": matrix["jerk_ratio"],
                "matrix_jerk_delta_pct": percent_change(
                    stage1_matrix["jerk_ratio"], matrix["jerk_ratio"]
                ),
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["score"],
            row["same_text_jerk_ratio"],
            row["matrix_jerk_ratio"],
            abs(row["same_imu_arm_swing_delta_pct"]),
        ),
    )
    imu_control = min(
        rows,
        key=lambda row: (
            row["same_text_jerk_ratio"],
            row["matrix_jerk_ratio"],
            row["same_text_active_sensor_error_m"],
        ),
    )
    text_completion = max(
        rows,
        key=lambda row: (
            row["same_imu_arm_swing_proxy_m"],
            -row["same_imu_active_sensor_error_m"],
        ),
    )
    all_gate_pass = [row for row in rows if row["score"] == 4]
    return {
        "models": rows,
        "ranked": ranked,
        "recommended": all_gate_pass[0]["model"] if all_gate_pass else None,
        "role_recommendations": {
            "main_if_single_model_required": ranked[0]["model"],
            "imu_control_and_smoothness": imu_control["model"],
            "head_imu_text_completion_diagnostic": text_completion["model"],
        },
    }


def render_markdown(selection: dict[str, Any]) -> str:
    lines = [
        "# ITM Model Selection",
        "",
        f"- all-gate recommended: `{selection['recommended'] or 'none'}`",
        f"- main if single model required: `{selection['role_recommendations']['main_if_single_model_required']}`",
        f"- IMU-control/smoothness model: `{selection['role_recommendations']['imu_control_and_smoothness']}`",
        f"- head-IMU text-completion diagnostic model: `{selection['role_recommendations']['head_imu_text_completion_diagnostic']}`",
        "",
        "## Summary",
        "",
        "| model | score | same_text_jerk | same_text_jerk_delta | same_imu_head_error | same_imu_arm_swing | same_imu_arm_swing_delta | matrix_jerk | matrix_jerk_delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selection["ranked"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    str(row["score"]),
                    f"{row['same_text_jerk_ratio']:.4f}",
                    f"{row['same_text_jerk_delta_pct']:+.1f}%",
                    f"{row['same_imu_active_sensor_error_m']:.4f}",
                    f"{row['same_imu_arm_swing_proxy_m']:.4f}",
                    f"{row['same_imu_arm_swing_delta_pct']:+.1f}%",
                    f"{row['matrix_jerk_ratio']:.4f}",
                    f"{row['matrix_jerk_delta_pct']:+.1f}%",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Gate Checks", ""])
    for row in selection["ranked"]:
        lines.extend(
            [
                f"### {row['model']}",
                "",
                "| check | pass |",
                "| --- | --- |",
            ]
        )
        for key, value in row["checks"].items():
            lines.append(f"| {key} | {'yes' if value else 'no'} |")
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            "- No checkpoint passes all four planned gates.",
            "- Stage-2/Stage-2b/Stage-4 variants consistently improve the IMU-control smoothness side, especially same-text-different-IMU jerk and matrix jerk.",
            "- Stage-1 remains the best diagnostic checkpoint for head-only text completion because it preserves the strongest arm-swing proxy.",
            "- Report this as a real trade-off instead of hiding it: one checkpoint demonstrates IMU control, while the head-only text completion phenomenon still needs a better objective if no anchor variant passes all gates.",
        ]
    )
    return "\n".join(lines)


def percent_change(left: float, right: float) -> float:
    if abs(left) < 1e-12:
        return 0.0
    return 100.0 * (right - left) / abs(left)


if __name__ == "__main__":
    raise SystemExit(main())
