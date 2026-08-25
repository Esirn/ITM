#!/usr/bin/env python3
"""Evaluate chunked MDM Text/paired/zero/shuffled control conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_motionlab_direct_suite import (
    ACTIVE_JOINTS,
    active_error,
    jerk_ratio,
    motion_difference,
    root_travel,
    step_frequency,
    summarize,
)


def load_group(paths, output_dir):
    sequences = []
    targets = []
    cases = []
    for spec_value in paths:
        stem = Path(spec_value).stem
        path = output_dir / f"{stem}.npz"
        with np.load(path) as data:
            motion = data["motion"]
            gt = data["gt"]
            metadata = json.loads(str(data["metadata"].item()))
        for index, length in enumerate(metadata["effective_lengths"]):
            sequences.append(motion[index, : int(length)].astype(np.float32))
            targets.append(gt[index, : int(length)].astype(np.float32))
            cases.append(metadata["cases"][index])
    return sequences, targets, cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    groups = {
        key: load_group(paths, args.results_dir)
        for key, paths in manifest["specs"].items()
    }
    records = []
    text_equivalence = []
    for index, motion_id in enumerate(manifest["motion_ids"]):
        record = {"motion_id": motion_id, "configs": {}}
        head_text = groups["head_text"][0][index]
        wrists_text = groups["wrists_text"][0][index]
        text_equivalence.append(float(np.max(np.abs(head_text - wrists_text))))
        for config in ("head", "wrists"):
            text_motion = groups[f"{config}_text"][0][index]
            paired = groups[f"{config}_paired"][0][index]
            zero = groups[f"{config}_zero"][0][index]
            shuffled = groups[f"{config}_shuffled"][0][index]
            target = groups[f"{config}_paired"][1][index]
            cases = groups[f"{config}_paired"][2]
            errors = {
                "text": active_error(text_motion, target, ACTIVE_JOINTS[config]),
                "paired": active_error(paired, target, ACTIVE_JOINTS[config]),
                "zero": active_error(zero, target, ACTIVE_JOINTS[config]),
                "shuffled": active_error(shuffled, target, ACTIVE_JOINTS[config]),
            }
            record["caption"] = cases[index]["text"]
            record["configs"][config] = {
                "active_error_m": errors,
                "paired_improves_text": errors["paired"] < errors["text"],
                "paired_beats_zero": errors["paired"] < errors["zero"],
                "paired_beats_shuffled": errors["paired"] < errors["shuffled"],
                "paired_motion_difference_m": motion_difference(text_motion, paired),
                "text_jerk_ratio": jerk_ratio(text_motion, target),
                "paired_jerk_ratio": jerk_ratio(paired, target),
                "text_root_travel_m": root_travel(text_motion),
                "paired_root_travel_m": root_travel(paired),
                "text_step_frequency_hz": step_frequency(text_motion),
                "paired_step_frequency_hz": step_frequency(paired),
            }
        records.append(record)

    summary = {}
    for config in ("head", "wrists"):
        items = [record["configs"][config] for record in records]
        summary[config] = {
            "samples": len(items),
            "paired_improves_text": sum(item["paired_improves_text"] for item in items),
            "paired_beats_zero": sum(item["paired_beats_zero"] for item in items),
            "paired_beats_shuffled": sum(item["paired_beats_shuffled"] for item in items),
        }
        for mode in ("text", "paired", "zero", "shuffled"):
            summary[config][f"{mode}_active_error_m"] = summarize(
                [item["active_error_m"][mode] for item in items]
            )
        for key in (
            "paired_motion_difference_m", "text_jerk_ratio", "paired_jerk_ratio",
            "text_root_travel_m", "paired_root_travel_m",
            "text_step_frequency_hz", "paired_step_frequency_hz",
        ):
            summary[config][key] = summarize([item[key] for item in items])

    payload = {
        "protocol": {
            "manifest": str(args.manifest.resolve()),
            "seed": manifest["seed"],
            "guidance_mode": "factorized",
            "text_only_head_wrists_max_abs_difference": max(text_equivalence),
        },
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# MDM Stage-2b Control Conditions", "",
        "| Config | Text error | Paired error | Zero error | Shuffled error | Paired<Text | Paired<Zero | Paired<Shuffled | Jerk Text | Jerk Paired |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for config, item in summary.items():
        lines.append(
            f"| {config} | {item['text_active_error_m']['mean']:.4f} | "
            f"{item['paired_active_error_m']['mean']:.4f} | {item['zero_active_error_m']['mean']:.4f} | "
            f"{item['shuffled_active_error_m']['mean']:.4f} | {item['paired_improves_text']}/100 | "
            f"{item['paired_beats_zero']}/100 | {item['paired_beats_shuffled']}/100 | "
            f"{item['text_jerk_ratio']['mean']:.3f} | {item['paired_jerk_ratio']['mean']:.3f} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
