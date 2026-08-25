#!/usr/bin/env python3
"""Measure paired text-attribute responses under a fixed head IMU."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from itm.experiments.mdm_control_suite import arm_swing_proxy


BASE = "a person walks"
SWING = "a person walks while swinging arms"
SLOW = "a person walks slowly"
QUICK = "a person walks quickly"
TURN = "a person turns while walking"
LARGE = "a person walks with large steps"


def path_speed(joints: np.ndarray, fps: float = 20.0) -> float:
    root = joints[:, 0, (0, 2)]
    return float(np.linalg.norm(np.diff(root, axis=0), axis=-1).sum() * fps / max(len(root) - 1, 1))


def turning_amount(joints: np.ndarray) -> float:
    # HumanML left/right hips define a stable body-right direction in the ground plane.
    right = joints[:, 2, (0, 2)] - joints[:, 1, (0, 2)]
    yaw = np.unwrap(np.arctan2(right[:, 1], right[:, 0]))
    return float(np.abs(np.diff(yaw)).sum())


def stride_span(joints: np.ndarray) -> float:
    root = joints[:, 0, (0, 2)]
    right = joints[:, 2, (0, 2)] - joints[:, 1, (0, 2)]
    right /= np.linalg.norm(right, axis=-1, keepdims=True).clip(1e-6)
    forward = np.stack((-right[:, 1], right[:, 0]), axis=-1)
    spans = []
    for foot in (10, 11):
        relative = joints[:, foot, (0, 2)] - root
        projection = np.sum(relative * forward, axis=-1)
        spans.append(np.quantile(projection, 0.95) - np.quantile(projection, 0.05))
    return float(np.mean(spans))


def head_error(joints: np.ndarray, target: np.ndarray) -> float:
    prediction = joints[:, 15] - joints[:, 0]
    truth = target[:, 15] - target[:, 0]
    return float(np.linalg.norm(prediction - truth, axis=-1).mean())


def bootstrap(values: np.ndarray, rng, samples=20_000):
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(values, rng):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "ci95": bootstrap(values, rng),
        "desired_count": int((values > 0).sum()),
        "samples": len(values),
    }


def evaluate(paths: list[Path], seed: int):
    motions = []
    targets = []
    cases = []
    checkpoint = None
    generation_seed = None
    lengths = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            motion_batch = data["motion"].astype(np.float32)
            target_batch = data["gt"].astype(np.float32)
            metadata = json.loads(str(data["metadata"].item()))
        checkpoint = checkpoint or metadata["control_checkpoint"]
        generation_seed = generation_seed if generation_seed is not None else metadata["seed"]
        effective = metadata.get("effective_lengths", [motion_batch.shape[1]] * len(motion_batch))
        for index, length in enumerate(effective):
            motions.append(motion_batch[index, : int(length)])
            targets.append(target_batch[index, : int(length)])
            lengths.append(int(length))
        cases.extend(metadata["cases"])
    grouped = {}
    for index, case in enumerate(cases):
        grouped.setdefault(str(case["motion_id"]), {})[case["text"]] = index
    rng = np.random.default_rng(seed)
    records = []
    for motion_id, prompts in grouped.items():
        missing = {BASE, SWING, SLOW, QUICK, TURN, LARGE} - set(prompts)
        if missing:
            raise ValueError(f"{motion_id} lacks prompts: {sorted(missing)}")
        values = {
            prompt: {
                "speed_mps": path_speed(motions[index]),
                "turning_rad": turning_amount(motions[index]),
                "arm_swing_m": arm_swing_proxy(motions[index]),
                "stride_span_m": stride_span(motions[index]),
                "head_error_m": head_error(motions[index], targets[index]),
            }
            for prompt, index in prompts.items()
        }
        records.append({
            "motion_id": motion_id,
            "values": values,
            "contrasts": {
                "quick_minus_slow_speed_mps": values[QUICK]["speed_mps"] - values[SLOW]["speed_mps"],
                "turn_minus_base_turning_rad": values[TURN]["turning_rad"] - values[BASE]["turning_rad"],
                "swing_minus_base_arm_swing_m": values[SWING]["arm_swing_m"] - values[BASE]["arm_swing_m"],
                "large_minus_base_stride_span_m": values[LARGE]["stride_span_m"] - values[BASE]["stride_span_m"],
            },
        })
    summary = {
        key: summarize([record["contrasts"][key] for record in records], rng)
        for key in records[0]["contrasts"]
    }
    edited_prompts = (SWING, SLOW, QUICK, TURN, LARGE)
    head_differences = [
        record["values"][prompt]["head_error_m"] - record["values"][BASE]["head_error_m"]
        for record in records for prompt in edited_prompts
    ]
    summary["edited_minus_base_head_error_m"] = {
        "mean": float(np.mean(head_differences)),
        "ci95": bootstrap(np.asarray(head_differences), rng),
        "samples": len(head_differences),
    }
    return {
        "sources": [str(path.resolve()) for path in paths],
        "checkpoint": checkpoint,
        "seed": generation_seed,
        "samples": len(records),
        "frame_lengths": lengths,
        "summary": summary,
        "records": records,
    }


def render(payloads):
    lines = [
        "# Text Attribute Control under Fixed Head IMU", "",
        "Positive values are the desired direction for the four semantic contrasts.", "",
        "| Model | Quick-Slow speed | Turn-Base yaw | Swing-Base arms | Large-Base stride | Head error change |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    keys = (
        "quick_minus_slow_speed_mps", "turn_minus_base_turning_rad",
        "swing_minus_base_arm_swing_m", "large_minus_base_stride_span_m",
    )
    for name, payload in payloads.items():
        cells = []
        for key in keys:
            item = payload["summary"][key]
            cells.append(f"{item['mean']:+.4f} ({item['desired_count']}/{item['samples']})")
        head = payload["summary"]["edited_minus_base_head_error_m"]["mean"]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {head:+.4f} m |")
    variable_length = any(len(set(payload["frame_lengths"])) > 1 for payload in payloads.values())
    note = (
        "Each motion retains its own effective length; comparisons are paired within the same motion, head IMU, seed, and initial diffusion noise."
        if variable_length else
        "The historical suites were generated before variable-length batching was fixed; all cases were truncated to the shortest sequence in their batch. These results are short-clip diagnostics, not final long-horizon attribute measurements."
    )
    lines.extend(["", note, ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="NAME=results.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    payloads = {}
    for value in args.input:
        name, separator, path = value.partition("=")
        if not separator:
            raise ValueError("--input must use NAME=PATH")
        paths = [Path(value) for value in sorted(glob.glob(path))]
        if not paths:
            raise ValueError(f"input pattern matched no files: {path}")
        payloads[name] = evaluate(paths, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payloads, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(render(payloads), encoding="utf-8")
    print(render(payloads))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
