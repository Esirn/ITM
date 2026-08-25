#!/usr/bin/env python3
"""Compare Text-only and paired/zero/shuffled MotionLab IMU controls."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--text-only", type=Path, required=True)
    for config in ("head", "wrists"):
        for mode in ("paired", "zero", "shuffled"):
            parser.add_argument(f"--{config}-{mode}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    text = np.load(args.text_only)["joints"]
    arrays = {
        config: {
            mode: np.load(getattr(args, f"{config}_{mode}"))["joints"]
            for mode in ("paired", "zero", "shuffled")
        }
        for config in ("head", "wrists")
    }
    records = []
    for index, motion_id in enumerate(spec["motion_ids"]):
        length = int(spec["lengths"][index])
        target = np.load(spec["ground_truth_paths"][index]).astype(np.float32)[:length]
        text_motion = text[index, :length]
        record = {"motion_id": motion_id, "caption": spec["captions"][index], "configs": {}}
        for config in ("head", "wrists"):
            values = {
                mode: arrays[config][mode][index, :length]
                for mode in ("paired", "zero", "shuffled")
            }
            errors = {
                "text": active_error(text_motion, target, ACTIVE_JOINTS[config]),
                **{
                    mode: active_error(motion, target, ACTIVE_JOINTS[config])
                    for mode, motion in values.items()
                },
            }
            record["configs"][config] = {
                "active_error_m": errors,
                "paired_improves_text": errors["paired"] < errors["text"],
                "paired_beats_zero": errors["paired"] < errors["zero"],
                "paired_beats_shuffled": errors["paired"] < errors["shuffled"],
                "paired_motion_difference_m": motion_difference(text_motion, values["paired"]),
                "text_jerk_ratio": jerk_ratio(text_motion, target),
                "paired_jerk_ratio": jerk_ratio(values["paired"], target),
                "text_root_travel_m": root_travel(text_motion),
                "paired_root_travel_m": root_travel(values["paired"]),
                "text_step_frequency_hz": step_frequency(text_motion),
                "paired_step_frequency_hz": step_frequency(values["paired"]),
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
        "protocol": {"spec": str(args.spec.resolve()), "seed": spec["seed"]},
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# MotionLab Control Conditions", "",
        "| Config | Text error | Paired error | Zero error | Shuffled error | Paired<Text | Paired<Zero | Paired<Shuffled | Jerk Text | Jerk Paired |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for config, item in summary.items():
        lines.append(
            f"| {config} | {item['text_active_error_m']['mean']:.4f} | "
            f"{item['paired_active_error_m']['mean']:.4f} | {item['zero_active_error_m']['mean']:.4f} | "
            f"{item['shuffled_active_error_m']['mean']:.4f} | {item['paired_improves_text']}/{item['samples']} | "
            f"{item['paired_beats_zero']}/{item['samples']} | {item['paired_beats_shuffled']}/{item['samples']} | "
            f"{item['text_jerk_ratio']['mean']:.3f} | {item['paired_jerk_ratio']['mean']:.3f} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
