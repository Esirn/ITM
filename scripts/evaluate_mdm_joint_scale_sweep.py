#!/usr/bin/env python3
"""Evaluate a factorized joint-guidance sweep on a fixed motion subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_mdm_control_conditions import load_group
from evaluate_motionlab_direct_suite import (
    ACTIVE_JOINTS,
    active_error,
    jerk_ratio,
    motion_difference,
    summarize,
)
from prepare_mdm_joint_scale_sweep import scale_key


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, samples=20_000):
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def sign_flip_pvalue(values: np.ndarray, rng: np.random.Generator, samples=20_000):
    observed = abs(float(values.mean()))
    signs = rng.choice((-1.0, 1.0), size=(samples, len(values)))
    permuted = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (samples + 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--text-reference-dir", type=Path, required=True)
    parser.add_argument("--text-reference-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    reference_manifest = json.loads(args.text_reference_manifest.read_text(encoding="utf-8"))
    rows = []
    records = []
    for config in ("head", "wrists"):
        text, targets, _ = load_group(
            reference_manifest["specs"][f"{config}_text"], args.text_reference_dir
        )
        for scale in manifest["joint_scales"]:
            group = f"{config}_{scale_key(float(scale))}"
            generated, generated_targets, _ = load_group(
                manifest["specs"][group], args.results_dir
            )
            if len(generated) != len(text):
                raise ValueError(f"sample count mismatch for {group}")
            scale_records = []
            for index, motion_id in enumerate(manifest["motion_ids"]):
                target = generated_targets[index]
                if target.shape != targets[index].shape:
                    raise ValueError(f"target shape mismatch for {motion_id}")
                text_error = active_error(text[index], target, ACTIVE_JOINTS[config])
                control_error = active_error(generated[index], target, ACTIVE_JOINTS[config])
                item = {
                    "motion_id": motion_id,
                    "sensor_config": config,
                    "joint_scale": float(scale),
                    "text_active_error_m": text_error,
                    "control_active_error_m": control_error,
                    "improvement_m": text_error - control_error,
                    "improves_text": control_error < text_error,
                    "text_jerk_ratio": jerk_ratio(text[index], target),
                    "control_jerk_ratio": jerk_ratio(generated[index], target),
                    "motion_difference_m": motion_difference(text[index], generated[index]),
                }
                scale_records.append(item)
                records.append(item)
            rows.append(summarize_scale(config, float(scale), scale_records))

    rng = np.random.default_rng(20260825)
    comparisons = compare_to_default(records, rng)
    payload = {
        "protocol": {
            "manifest": str(args.manifest.resolve()),
            "text_reference_manifest": str(args.text_reference_manifest.resolve()),
            "guidance_mode": "factorized",
            "text_scale": manifest["fixed_scales"]["text_scale"],
            "imu_scale": manifest["fixed_scales"]["imu_scale"],
            "seed": manifest["seed"],
        },
        "summary": rows,
        "comparisons_to_joint_scale_1": comparisons,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


def summarize_scale(config: str, scale: float, records: list[dict]) -> dict:
    improvements = np.asarray([row["improvement_m"] for row in records])
    rng = np.random.default_rng(20260825 + (0 if config == "head" else 1000) + int(scale * 100))
    return {
        "sensor_config": config,
        "joint_scale": scale,
        "samples": len(records),
        "control_active_error_m": summarize([row["control_active_error_m"] for row in records]),
        "improvement_m": summarize([row["improvement_m"] for row in records]),
        "improvement_ci95_m": bootstrap_mean_ci(improvements, rng),
        "improvement_sign_flip_pvalue": sign_flip_pvalue(improvements, rng),
        "improves_text": sum(row["improves_text"] for row in records),
        "control_jerk_ratio": summarize([row["control_jerk_ratio"] for row in records]),
        "motion_difference_m": summarize([row["motion_difference_m"] for row in records]),
    }


def compare_to_default(records: list[dict], rng: np.random.Generator) -> list[dict]:
    results = []
    for config in ("head", "wrists"):
        by_scale = {}
        for record in records:
            if record["sensor_config"] == config:
                by_scale.setdefault(record["joint_scale"], {})[record["motion_id"]] = record
        if 1.0 not in by_scale:
            raise ValueError(f"joint_scale=1 reference is missing for {config}")
        reference = by_scale[1.0]
        for scale, current in sorted(by_scale.items()):
            if scale == 1.0:
                continue
            ids = sorted(reference)
            error_difference = np.asarray([
                current[motion_id]["control_active_error_m"]
                - reference[motion_id]["control_active_error_m"]
                for motion_id in ids
            ])
            jerk_difference = np.asarray([
                current[motion_id]["control_jerk_ratio"]
                - reference[motion_id]["control_jerk_ratio"]
                for motion_id in ids
            ])
            results.append({
                "sensor_config": config,
                "joint_scale": scale,
                "active_error_minus_default_mean_m": float(error_difference.mean()),
                "active_error_minus_default_ci95_m": bootstrap_mean_ci(error_difference, rng),
                "active_error_sign_flip_pvalue": sign_flip_pvalue(error_difference, rng),
                "jerk_minus_default_mean": float(jerk_difference.mean()),
                "jerk_minus_default_ci95": bootstrap_mean_ci(jerk_difference, rng),
                "jerk_sign_flip_pvalue": sign_flip_pvalue(jerk_difference, rng),
            })
    return results


def markdown(payload: dict) -> str:
    lines = [
        "# MDM Factorized Joint-Scale Sweep", "",
        "Fixed protocol: 100 test motions, text scale 2.5, IMU scale 1.0, paired IMU, shared seed 1234.", "",
        "| Sensor | Joint scale | Active error (m) | Improvement vs Text (m), 95% CI | Improved | Jerk ratio | Motion difference (m) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['sensor_config']} | {row['joint_scale']:g} | "
            f"{row['control_active_error_m']['mean']:.4f} | {row['improvement_m']['mean']:+.4f} "
            f"[{row['improvement_ci95_m'][0]:+.4f}, {row['improvement_ci95_m'][1]:+.4f}] | "
            f"{row['improves_text']}/{row['samples']} | {row['control_jerk_ratio']['mean']:.3f} | "
            f"{row['motion_difference_m']['mean']:.4f} |"
        )
    lines.extend([
        "", "## Paired comparison to joint scale 1", "",
        "Negative differences favor the alternative scale. Intervals use 20,000 paired bootstrap resamples.", "",
        "| Sensor | Joint scale | Active error change (m), 95% CI | p | Jerk change, 95% CI | p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in payload["comparisons_to_joint_scale_1"]:
        error_ci = row["active_error_minus_default_ci95_m"]
        jerk_ci = row["jerk_minus_default_ci95"]
        lines.append(
            f"| {row['sensor_config']} | {row['joint_scale']:g} | "
            f"{row['active_error_minus_default_mean_m']:+.4f} [{error_ci[0]:+.4f}, {error_ci[1]:+.4f}] | "
            f"{row['active_error_sign_flip_pvalue']:.4f} | "
            f"{row['jerk_minus_default_mean']:+.4f} [{jerk_ci[0]:+.4f}, {jerk_ci[1]:+.4f}] | "
            f"{row['jerk_sign_flip_pvalue']:.4f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
