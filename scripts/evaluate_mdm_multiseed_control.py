#!/usr/bin/env python3
"""Evaluate MDM control stability across diffusion seeds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_mdm_control_conditions import load_group
from evaluate_motionlab_direct_suite import ACTIVE_JOINTS, active_error, jerk_ratio


def bootstrap_motion_ci(values: np.ndarray, seed=20260825, samples=20_000):
    """Bootstrap motion-level means after averaging repeated seeds."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = []
    for seed in manifest["diffusion_seeds"]:
        for config in ("head", "wrists"):
            groups = {
                mode: load_group(manifest["specs"][f"seed{seed}_{config}_{mode}"], args.results_dir)
                for mode in ("text", "paired", "shuffled")
            }
            for index, motion_id in enumerate(manifest["motion_ids"]):
                target = groups["paired"][1][index]
                text = groups["text"][0][index]
                paired = groups["paired"][0][index]
                shuffled = groups["shuffled"][0][index]
                text_error = active_error(text, target, ACTIVE_JOINTS[config])
                paired_error = active_error(paired, target, ACTIVE_JOINTS[config])
                shuffled_error = active_error(shuffled, target, ACTIVE_JOINTS[config])
                records.append({
                    "motion_id": motion_id,
                    "length": manifest["lengths"][index],
                    "seed": seed,
                    "sensor_config": config,
                    "paired_minus_text_error_m": paired_error - text_error,
                    "paired_minus_shuffled_error_m": paired_error - shuffled_error,
                    "paired_minus_text_jerk": jerk_ratio(paired, target) - jerk_ratio(text, target),
                })
    summary = [summarize_config(records, config, manifest["diffusion_seeds"]) for config in ("head", "wrists")]
    length_strata = [summarize_length(records, config, name, low, high) for config in ("head", "wrists") for name, low, high in (("short_1_80", 1, 80), ("medium_81_140", 81, 140), ("long_141_196", 141, 196))]
    payload = {"protocol": {"manifest": str(args.manifest.resolve()), "motions": len(manifest["motion_ids"]), "seeds": manifest["diffusion_seeds"], "bootstrap_unit": "motion after averaging seeds"}, "summary": summary, "length_strata": length_strata, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")
    print(markdown(payload))
    return 0


def summarize_config(records: list[dict], config: str, seeds: list[int]) -> dict:
    selected = [row for row in records if row["sensor_config"] == config]
    per_seed = []
    for seed in seeds:
        rows = [row for row in selected if row["seed"] == seed]
        per_seed.append({
            "seed": seed,
            "paired_minus_text_error_m": float(np.mean([row["paired_minus_text_error_m"] for row in rows])),
            "paired_minus_shuffled_error_m": float(np.mean([row["paired_minus_shuffled_error_m"] for row in rows])),
            "paired_minus_text_jerk": float(np.mean([row["paired_minus_text_jerk"] for row in rows])),
            "paired_beats_text": sum(row["paired_minus_text_error_m"] < 0 for row in rows),
            "paired_beats_shuffled": sum(row["paired_minus_shuffled_error_m"] < 0 for row in rows),
        })
    by_motion = defaultdict(list)
    for row in selected:
        by_motion[row["motion_id"]].append(row)
    motion_text = np.asarray([np.mean([row["paired_minus_text_error_m"] for row in rows]) for rows in by_motion.values()])
    motion_shuffled = np.asarray([np.mean([row["paired_minus_shuffled_error_m"] for row in rows]) for rows in by_motion.values()])
    motion_jerk = np.asarray([np.mean([row["paired_minus_text_jerk"] for row in rows]) for rows in by_motion.values()])
    majority = (len(seeds) // 2) + 1
    return {
        "sensor_config": config,
        "per_seed": per_seed,
        "motion_averaged_paired_minus_text_error_m": float(motion_text.mean()),
        "motion_averaged_paired_minus_text_ci95_m": bootstrap_motion_ci(motion_text),
        "motion_averaged_paired_minus_shuffled_error_m": float(motion_shuffled.mean()),
        "motion_averaged_paired_minus_shuffled_ci95_m": bootstrap_motion_ci(motion_shuffled, seed=20260826),
        "motion_averaged_jerk_change": float(motion_jerk.mean()),
        "motion_averaged_jerk_ci95": bootstrap_motion_ci(motion_jerk, seed=20260827),
        "motions_beating_text_majority_seeds": sum(sum(row["paired_minus_text_error_m"] < 0 for row in rows) >= majority for rows in by_motion.values()),
        "motions_beating_shuffled_majority_seeds": sum(sum(row["paired_minus_shuffled_error_m"] < 0 for row in rows) >= majority for rows in by_motion.values()),
    }


def summarize_length(records: list[dict], config: str, name: str, low: int, high: int) -> dict:
    selected = [row for row in records if row["sensor_config"] == config and low <= row["length"] <= high]
    by_motion = defaultdict(list)
    for row in selected:
        by_motion[row["motion_id"]].append(row)
    text = np.asarray([np.mean([row["paired_minus_text_error_m"] for row in rows]) for rows in by_motion.values()])
    shuffled = np.asarray([np.mean([row["paired_minus_shuffled_error_m"] for row in rows]) for rows in by_motion.values()])
    jerk = np.asarray([np.mean([row["paired_minus_text_jerk"] for row in rows]) for rows in by_motion.values()])
    return {
        "sensor_config": config,
        "length_bin": name,
        "motions": len(by_motion),
        "paired_minus_text_error_m": float(text.mean()),
        "paired_minus_text_ci95_m": bootstrap_motion_ci(text, seed=20260830 + low),
        "paired_minus_shuffled_error_m": float(shuffled.mean()),
        "paired_minus_shuffled_ci95_m": bootstrap_motion_ci(shuffled, seed=20260831 + low),
        "jerk_change": float(jerk.mean()),
        "jerk_ci95": bootstrap_motion_ci(jerk, seed=20260832 + low),
    }


def markdown(payload: dict) -> str:
    lines = ["# MDM Multi-Seed Control Stability", "", "Negative differences favor paired IMU. Confidence intervals bootstrap motions after averaging the five repeated seeds.", ""]
    for item in payload["summary"]:
        lines.extend([
            f"## {item['sensor_config']}", "",
            "| Seed | Paired-Text error (m) | Better than Text | Paired-Shuffled error (m) | Better than Shuffled | Jerk change |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in item["per_seed"]:
            lines.append(f"| {row['seed']} | {row['paired_minus_text_error_m']:+.4f} | {row['paired_beats_text']}/30 | {row['paired_minus_shuffled_error_m']:+.4f} | {row['paired_beats_shuffled']}/30 | {row['paired_minus_text_jerk']:+.3f} |")
        text_ci = item["motion_averaged_paired_minus_text_ci95_m"]
        shuffle_ci = item["motion_averaged_paired_minus_shuffled_ci95_m"]
        jerk_ci = item["motion_averaged_jerk_ci95"]
        lines.extend([
            "", f"- Motion-averaged paired-Text: {item['motion_averaged_paired_minus_text_error_m']:+.4f} m [{text_ci[0]:+.4f}, {text_ci[1]:+.4f}].",
            f"- Motion-averaged paired-shuffled: {item['motion_averaged_paired_minus_shuffled_error_m']:+.4f} m [{shuffle_ci[0]:+.4f}, {shuffle_ci[1]:+.4f}].",
            f"- Motion-averaged jerk change: {item['motion_averaged_jerk_change']:+.3f} [{jerk_ci[0]:+.3f}, {jerk_ci[1]:+.3f}].",
            f"- Consistent on >=3/5 seeds: {item['motions_beating_text_majority_seeds']}/30 vs Text, {item['motions_beating_shuffled_majority_seeds']}/30 vs shuffled.", "",
        ])
    lines.extend([
        "## Length-stratified motion averages", "",
        "Each bin contains 10 motions and each motion is averaged over five seeds.", "",
        "| Sensor | Length | Paired-Text error (m), 95% CI | Paired-Shuffled error (m), 95% CI | Jerk change, 95% CI |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for row in payload["length_strata"]:
        text_ci = row["paired_minus_text_ci95_m"]
        shuffled_ci = row["paired_minus_shuffled_ci95_m"]
        jerk_ci = row["jerk_ci95"]
        lines.append(
            f"| {row['sensor_config']} | {row['length_bin']} | {row['paired_minus_text_error_m']:+.4f} "
            f"[{text_ci[0]:+.4f}, {text_ci[1]:+.4f}] | {row['paired_minus_shuffled_error_m']:+.4f} "
            f"[{shuffled_ci[0]:+.4f}, {shuffled_ci[1]:+.4f}] | {row['jerk_change']:+.3f} "
            f"[{jerk_ci[0]:+.3f}, {jerk_ci[1]:+.3f}] |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
