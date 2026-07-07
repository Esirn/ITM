#!/usr/bin/env python
"""Summarize completed MDM-control experiment-suite metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


NUMERIC_KEYS = (
    "root_trajectory_error_m",
    "head_trajectory_error_m",
    "wrist_trajectory_error_m",
    "active_sensor_trajectory_error_m",
    "root_relative_motion_error_m",
    "jerk_ratio",
    "arm_swing_proxy_m",
    "root_travel_m",
    "step_frequency_hz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/mdm_control/experiments")
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    rows = load_rows(root)
    if not rows:
        raise RuntimeError(f"No metrics.json files found under {root}")
    summary = {
        "root": str(root),
        "run_count": len({row["run"] for row in rows}),
        "case_count": len(rows),
        "by_experiment": summarize_by(rows, "experiment"),
        "by_experiment_and_sensor": summarize_by(rows, "experiment_sensor"),
        "by_guidance": summarize_guidance(rows),
        "matrix_candidates": matrix_candidates(rows),
        "same_imu_by_prompt": summarize_same_imu_prompts(rows),
    }
    output_json = Path(args.output_json) if args.output_json else root / "RESULTS_SUMMARY.json"
    output_md = Path(args.output_md) if args.output_md else root / "RESULTS_SUMMARY.md"
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(f"summary json: {output_json}")
    print(f"summary md: {output_md}")
    return 0


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("**/metrics.json")):
        metrics = json.loads(path.read_text())
        run = path.parent.relative_to(root).as_posix()
        experiment = _experiment_from_run(run)
        for case in metrics["cases"]:
            row = dict(case)
            row["run"] = run
            row["experiment"] = experiment
            row["experiment_sensor"] = f"{experiment}:{case['sensor_config']}"
            rows.append(row)
    return rows


def summarize_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return {name: mean_numeric(values) for name, values in sorted(grouped.items())}


def summarize_guidance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        if row["experiment"] != "guidance_sweep":
            continue
        key = (row["sensor_config"], float(row["text_scale"]), float(row["imu_scale"]))
        grouped.setdefault(key, []).append(row)
    result = []
    for (sensor, text_scale, imu_scale), values in sorted(grouped.items()):
        result.append(
            {
                "sensor_config": sensor,
                "text_scale": text_scale,
                "imu_scale": imu_scale,
                **mean_numeric(values),
            }
        )
    return result


def matrix_candidates(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    matrix_rows = [row for row in rows if row["experiment"] == "matrix"]
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in matrix_rows:
        by_run.setdefault(row["run"], []).append(row)
    run_scores = []
    for run, values in by_run.items():
        score = mean_numeric(values)
        run_scores.append({"run": run, **score})
    return {
        "lowest_jerk": sorted(run_scores, key=lambda row: row["jerk_ratio"])[:5],
        "highest_pair_signal_proxy": sorted(
            run_scores,
            key=lambda row: row["root_relative_motion_error_m"],
            reverse=True,
        )[:5],
        "lowest_active_sensor_error": sorted(
            run_scores,
            key=lambda row: row["active_sensor_trajectory_error_m"],
        )[:5],
    }


def summarize_same_imu_prompts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["experiment"] == "same_head_imu_different_text":
            grouped.setdefault(row["label"], []).append(row)
    result = [{"prompt": prompt, **mean_numeric(values)} for prompt, values in sorted(grouped.items())]
    return sorted(result, key=lambda row: row["arm_swing_proxy_m"], reverse=True)


def mean_numeric(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {key: mean(float(row[key]) for row in rows) for key in NUMERIC_KEYS}


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# MDM Control Experiment Results",
        "",
        f"- root: `{summary['root']}`",
        f"- runs with metrics: `{summary['run_count']}`",
        f"- generated cases: `{summary['case_count']}`",
        "",
        "## By Experiment",
        "",
        table_from_mapping(summary["by_experiment"]),
        "",
        "## By Experiment And Sensor",
        "",
        table_from_mapping(summary["by_experiment_and_sensor"]),
        "",
        "## Guidance Sweep",
        "",
        table_from_rows(summary["by_guidance"], ("sensor_config", "text_scale", "imu_scale")),
        "",
        "## Same Head IMU, Different Text Prompts",
        "",
        table_from_rows(summary["same_imu_by_prompt"], ("prompt",)),
        "",
        "## Matrix Candidates",
        "",
    ]
    for title, rows in summary["matrix_candidates"].items():
        lines.extend([f"### {title}", "", table_from_rows(rows, ("run",)), ""])
    return "\n".join(lines)


def table_from_mapping(values: dict[str, dict[str, float]]) -> str:
    return table_from_rows([{"group": key, **value} for key, value in values.items()], ("group",))


def table_from_rows(rows: list[dict[str, Any]], leading: tuple[str, ...]) -> str:
    if not rows:
        return "_No rows._"
    keys = leading + NUMERIC_KEYS
    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
    ]
    for row in rows:
        cells = []
        for key in keys:
            value = row.get(key, "")
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _experiment_from_run(run: str) -> str:
    if run.startswith("matrix_test_"):
        return "matrix"
    if run.startswith("same_text_different_imu/"):
        return "same_text_different_imu"
    if run.startswith("same_head_imu_different_text/"):
        return "same_head_imu_different_text"
    if run.startswith("guidance_sweep/"):
        return "guidance_sweep"
    return run.split("/", 1)[0]


if __name__ == "__main__":
    raise SystemExit(main())
