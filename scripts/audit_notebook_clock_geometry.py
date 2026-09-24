#!/usr/bin/env python
"""Compare two predeclared clocks on common supported frames of 13 fixed SSM clips.

GT yaw is an oracle diagnostic, never an admissible input canonicalization.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_caption_intervals import artifact, write_json
from audit_humanml_imu_alignment import oracle_yaw_residual
from itm.data.manifest import read_jsonl
from itm.data.physical_time import resample_positions_at_times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    clock = json.loads(args.clock_report.read_text())
    if len(clock["results"]) != 13:
        raise ValueError("Only the predeclared 13 clips are allowed")
    support_path = Path("docs/itm_rework/protocol/caption_audit_20260924_v1/time_support.json")
    support = {r["motion_id"]: r for r in json.loads(support_path.read_text())["failures"]}
    motions = {r["motion_id"]: r for r in read_jsonl("outputs/manifests_full/train.jsonl")}
    caches = {r["motion_id"]: r for r in read_jsonl("outputs/manifests_full/train_standard_imu.jsonl")}
    rows = []
    for plan in clock["results"]:
        mid = plan["motion_id"]
        source = support[mid]
        if artifact(source["cache_artifact"]["path"])["sha256"] != source["cache_artifact"]["sha256"]:
            raise ValueError("Cache changed since source audit")
        target_path = Path(motions[mid]["joints_path"])
        target = np.load(target_path, allow_pickle=False)[:, :22].astype(np.float64)
        if len(target) != plan["predicted_feature_frames"]:
            raise ValueError("Saved joint representation length differs from predicted feature length")
        with np.load(caches[mid]["imu_path"], allow_pickle=False) as cache:
            joints, fps = cache["joints"][:, :22].astype(np.float64), float(cache["fps"])
        cache_times = np.arange(len(joints), dtype=np.float64) / fps
        nominal = np.arange(len(target), dtype=np.float64) / 20.
        source_origin = source["source_declared_crop"][0] / source["source_fps"]
        replayed = np.asarray(plan["source_times_seconds"][:-1]) - source_origin
        common = ((nominal >= 0.) & (nominal <= cache_times[-1])
                  & (replayed >= 0.) & (replayed <= cache_times[-1]))
        indices = np.flatnonzero(common)
        if len(indices) < 2:
            raise ValueError("Insufficient common support")
        metrics = {}
        for name, times in (("nominal_20hz", nominal), ("notebook_source_clock", replayed)):
            positions = resample_positions_at_times(joints, cache_times, times[common])
            metrics[name] = oracle_yaw_residual(positions, target[common])
        rows.append({"motion_id": mid, "split": "train", "feature_frames": len(target),
            "common_frame_count": len(indices), "common_frame_fraction": len(indices) / len(target),
            "common_frame_indices": indices.tolist(), "cache_source_origin_seconds": source_origin,
            "nominal_source_interval_seconds": [float(nominal[common][0]), float(nominal[common][-1])],
            "notebook_source_interval_seconds": [float(replayed[common][0]), float(replayed[common][-1])],
            "metrics": metrics, "artifacts": [artifact(target_path), source["cache_artifact"]]})
    metric = "oracle_constant_yaw_mean_m"
    report = {"kind": "fixed_clock_geometry_diagnostic", "rows": rows,
        "notebook_clock_lower_oracle_residual_count": sum(r["metrics"]["notebook_source_clock"][metric]
            < r["metrics"]["nominal_20hz"][metric] for r in rows),
        "per_clip_equal_mean_oracle_residual_m": {clock_name: float(np.mean([r["metrics"][clock_name][metric] for r in rows]))
            for clock_name in ("nominal_20hz", "notebook_source_clock")},
        "budget": {"clips": 13, "clock_hypotheses": 2, "body_model_forwards": 0,
                   "model_generation": 0, "gpu_jobs": 0},
        "artifacts": [artifact(args.clock_report), artifact(support_path), artifact(Path(__file__)),
                      artifact(Path("scripts/audit_humanml_imu_alignment.py")), artifact(Path("src/itm/data/physical_time.py"))],
        "limitations": ["Oracle yaw uses full target joints and must not enter deployment input preprocessing.",
                        "Comparison uses identical common frames, not the full clips; missing support is reported.",
                        "One constant yaw is separately fitted per clock, without searching time shifts/scales.",
                        "Body retargeting and source FPS calibration remain unresolved; no physical IMU closure certification."]}
    write_json(args.output, report)
    print(json.dumps({"cases": len(rows), "improved": report["notebook_clock_lower_oracle_residual_count"],
                      "mean_m": report["per_clip_equal_mean_oracle_residual_m"]}))


if __name__ == "__main__":
    main()
