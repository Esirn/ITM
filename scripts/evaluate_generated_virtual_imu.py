#!/usr/bin/env python
"""Fit generated HumanML joints to SMPL and evaluate virtual IMU consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from itm.data.smpl_fitting import fit_humanml_joints_to_smpl, load_neutral_smpl
from itm.data.standard_imu import SENSOR_JOINT_INDICES, synthesize_standard_imu
from itm.metrics.virtual_imu import rotation_geodesic_error, vector_l2_error
from itm.models.torch_frame_baseline import require_torch


DEFAULT_SMPL = "/home/a200/mount/a40/datasets/SMPL/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--motion-key",
        choices=("motion", "gt"),
        default="motion",
        help="Fit generated motion or run a ground-truth round-trip validation.",
    )
    parser.add_argument("--smpl-model", default=DEFAULT_SMPL)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--validation-summary")
    parser.add_argument("--max-roundtrip-trajectory-m", type=float, default=0.05)
    parser.add_argument("--max-roundtrip-acceleration-mps2", type=float, default=0.5)
    parser.add_argument("--max-roundtrip-orientation-rad", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    torch = require_torch()
    device = torch.device(args.device)
    with np.load(args.results, allow_pickle=False) as data:
        motion = data[args.motion_key].astype(np.float32)[..., :22, :]
        target_acceleration = data["acceleration"].astype(np.float32)
        target_orientation = data["orientation"].astype(np.float32)
        sensor_masks = data["sensor_mask"].astype(bool)
        metadata = json.loads(str(data["metadata"].item()))
    lengths = metadata.get("effective_lengths", [motion.shape[1]] * len(motion))
    count = len(motion) if args.max_cases is None else min(len(motion), args.max_cases)
    model = load_neutral_smpl(args.smpl_model, batch_size=max(lengths[:count]), device=device)
    cases = []
    for index in range(count):
        frames = int(lengths[index])
        try:
            fit = fit_humanml_joints_to_smpl(
                motion[index, :frames], model, iterations=args.iterations
            )
            generated = synthesize_standard_imu(
                fit.vertices, fit.global_rotations, fps=20.0, smooth_window=5
            )
            active = sensor_masks[index]
            trajectory_error = vector_l2_error(
                fit.joints[:, SENSOR_JOINT_INDICES],
                np.asarray(data_or_none(args.results, "gt", index, frames))[:, SENSOR_JOINT_INDICES],
                active,
            )
            cases.append(
                {
                    "case_index": index,
                    "status": "ok",
                    "fit_loss": fit.final_loss,
                    "scale": fit.scale,
                    "active_sensor_trajectory_error_m": trajectory_error,
                    "virtual_acceleration_error_mps2": vector_l2_error(
                        generated.acceleration,
                        target_acceleration[index, :frames],
                        active,
                    ),
                    "virtual_orientation_error_rad": rotation_geodesic_error(
                        generated.orientation,
                        target_orientation[index, :frames],
                        active,
                    ),
                }
            )
        except Exception as error:
            cases.append({"case_index": index, "status": "failed", "error": str(error)})
        print(f"[{index + 1}/{count}] {cases[-1]['status']}")
    successful = [case for case in cases if case["status"] == "ok"]
    means = {
        key: float(np.mean([case[key] for case in successful]))
        for key in (
            "active_sensor_trajectory_error_m",
            "virtual_acceleration_error_mps2",
            "virtual_orientation_error_rad",
        )
    } if successful else {}
    validation = _validation_status(args, means)
    summary = {
        "results": str(Path(args.results).resolve()),
        "motion_key": args.motion_key,
        "smpl_model": str(Path(args.smpl_model).resolve()),
        "device": str(device),
        "iterations": args.iterations,
        "successful_cases": len(successful),
        "failed_cases": len(cases) - len(successful),
        "metrics_mean": means,
        "validation": validation,
        "eligible_for_paper_claims": bool(validation.get("passed", False)),
        "cases": cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary: {output}")
    return 0 if successful else 2


def data_or_none(path, key, index, frames):
    with np.load(path, allow_pickle=False) as data:
        if key not in data:
            raise KeyError(f"{path} does not contain {key}")
        return data[key][index, :frames]


def _validation_status(args, means):
    thresholds = {
        "active_sensor_trajectory_error_m": args.max_roundtrip_trajectory_m,
        "virtual_acceleration_error_mps2": args.max_roundtrip_acceleration_mps2,
        "virtual_orientation_error_rad": args.max_roundtrip_orientation_rad,
    }
    if args.motion_key == "gt":
        checks = {key: means.get(key, float("inf")) <= value for key, value in thresholds.items()}
        return {"kind": "ground_truth_roundtrip", "thresholds": thresholds, "checks": checks, "passed": all(checks.values())}
    if not args.validation_summary:
        return {
            "kind": "generated_motion",
            "passed": False,
            "reason": "No passing ground-truth round-trip summary was supplied.",
        }
    validation = json.loads(Path(args.validation_summary).read_text(encoding="utf-8"))
    passed = bool(validation.get("validation", {}).get("passed", False))
    return {
        "kind": "generated_motion",
        "roundtrip_summary": str(Path(args.validation_summary).resolve()),
        "passed": passed,
        "reason": None if passed else "Ground-truth round-trip validation did not pass.",
    }


if __name__ == "__main__":
    raise SystemExit(main())
