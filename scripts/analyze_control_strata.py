#!/usr/bin/env python3
"""Stratify fixed-subset control results by caption semantics and length."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


SEMANTIC_PATTERNS = {
    "locomotion": re.compile(
        r"\b(walk|walks|walking|run|runs|running|jog|jogs|jogging|step|steps|stepping|march|marches|marching)\b"
    ),
    "turning": re.compile(r"\b(turn|turns|turning|rotate|rotates|rotating|spin|spins|spinning|circle|circles|circling)\b"),
    "upper_body": re.compile(
        r"\b(arm|arms|hand|hands|wrist|wrists|shoulder|shoulders|elbow|elbows|wave|waves|waving|clap|claps|clapping|punch|punches|punching|throw|throws|throwing)\b"
    ),
    "lower_body": re.compile(
        r"\b(leg|legs|foot|feet|knee|knees|kick|kicks|kicking|squat|squats|squatting|jump|jumps|jumping|hop|hops|hopping)\b"
    ),
    "sit_stand": re.compile(
        r"\b(sit|sits|sitting|sat|stand|stands|standing|stood|chair|rise|rises|rising)\b"
    ),
}


def semantic_tags(caption: str) -> list[str]:
    text = caption.lower()
    tags = [name for name, pattern in SEMANTIC_PATTERNS.items() if pattern.search(text)]
    return tags or ["other"]


def length_bin(frames: int) -> str:
    if frames <= 80:
        return "short_1_80"
    if frames <= 140:
        return "medium_81_140"
    return "long_141_196"


def summarize(rows: list[dict]) -> dict:
    def mean(key):
        return float(np.mean([row[key] for row in rows]))

    return {
        "samples": len(rows),
        "paired_minus_text_error_m": mean("paired_minus_text_error_m"),
        "paired_minus_shuffled_error_m": mean("paired_minus_shuffled_error_m"),
        "paired_minus_text_jerk": mean("paired_minus_text_jerk"),
        "paired_beats_text": sum(row["paired_minus_text_error_m"] < 0 for row in rows),
        "paired_beats_shuffled": sum(row["paired_minus_shuffled_error_m"] < 0 for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-spec", type=Path, required=True)
    parser.add_argument("--mdm", type=Path, required=True)
    parser.add_argument("--motionlab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    subset = json.loads(args.subset_spec.read_text(encoding="utf-8"))
    metadata = {
        motion_id: {
            "caption": subset["captions"][index],
            "length": int(subset["lengths"][index]),
        }
        for index, motion_id in enumerate(subset["motion_ids"])
    }
    model_sources = {
        "MDM Stage-2b": json.loads(args.mdm.read_text(encoding="utf-8")),
        "MotionLab Direct-V2": json.loads(args.motionlab.read_text(encoding="utf-8")),
    }
    detailed = []
    for model, source in model_sources.items():
        if [row["motion_id"] for row in source["records"]] != subset["motion_ids"]:
            raise ValueError(f"motion ID order mismatch for {model}")
        for record in source["records"]:
            item = metadata[record["motion_id"]]
            for sensor in ("head", "wrists"):
                config = record["configs"][sensor]
                detailed.append({
                    "model": model,
                    "motion_id": record["motion_id"],
                    "sensor_config": sensor,
                    "caption": item["caption"],
                    "length": item["length"],
                    "semantic_tags": semantic_tags(item["caption"]),
                    "length_bin": length_bin(item["length"]),
                    "paired_minus_text_error_m": config["active_error_m"]["paired"] - config["active_error_m"]["text"],
                    "paired_minus_shuffled_error_m": config["active_error_m"]["paired"] - config["active_error_m"]["shuffled"],
                    "paired_minus_text_jerk": config["paired_jerk_ratio"] - config["text_jerk_ratio"],
                })

    strata = []
    for model in model_sources:
        for sensor in ("head", "wrists"):
            selected = [row for row in detailed if row["model"] == model and row["sensor_config"] == sensor]
            semantic_groups = defaultdict(list)
            length_groups = defaultdict(list)
            for row in selected:
                for tag in row["semantic_tags"]:
                    semantic_groups[tag].append(row)
                length_groups[row["length_bin"]].append(row)
            for kind, groups in (("semantic", semantic_groups), ("length", length_groups)):
                for name, rows in sorted(groups.items()):
                    strata.append({"model": model, "sensor_config": sensor, "stratum_type": kind, "stratum": name, **summarize(rows)})

    payload = {
        "protocol": {
            "subset_spec": str(args.subset_spec.resolve()),
            "samples": len(subset["motion_ids"]),
            "semantic_assignment": "fixed keyword-based multi-label tags; unmatched captions are other",
            "length_bins": {"short_1_80": "1-80", "medium_81_140": "81-140", "long_141_196": "141-196"},
            "interpretation": "exploratory descriptive analysis; no category-wise significance claims",
        },
        "strata": strata,
        "records": detailed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8")
    print(render_markdown(payload))
    return 0


def render_markdown(payload: dict) -> str:
    lines = [
        "# Control Performance by Motion Stratum", "",
        "Exploratory descriptive analysis on the fixed 100-motion test subset. Semantic groups use fixed keyword-based multi-label rules and may overlap. Negative error changes favor paired IMU.", "",
    ]
    for kind in ("semantic", "length"):
        lines.extend([
            f"## {kind.title()} strata", "",
            "| Model | Sensor | Stratum | N | Paired-Text error (m) | Better than Text | Paired-Shuffled error (m) | Better than Shuffled | Jerk change |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in payload["strata"]:
            if row["stratum_type"] != kind:
                continue
            lines.append(
                f"| {row['model']} | {row['sensor_config']} | {row['stratum']} | {row['samples']} | "
                f"{row['paired_minus_text_error_m']:+.4f} | {row['paired_beats_text']}/{row['samples']} | "
                f"{row['paired_minus_shuffled_error_m']:+.4f} | {row['paired_beats_shuffled']}/{row['samples']} | "
                f"{row['paired_minus_text_jerk']:+.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
