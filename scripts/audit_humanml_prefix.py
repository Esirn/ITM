#!/usr/bin/env python
"""Predeclare ten train-only prefix probes, then compare three clocks on common frames."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

import numpy as np

from audit_caption_intervals import artifact, write_json
from audit_caption_time_support import pose_frames
from audit_humanml_imu_alignment import oracle_yaw_residual
from itm.data.humanml_clock import PREFIX_TRIM_SECONDS, notebook_source_indices
from itm.data.manifest import read_jsonl
from itm.data.physical_time import resample_positions_at_times

VERSION = "humanml-prefix-fixed-probes-v1"
CANDIDATES = Path("docs/itm_rework/protocol/caption_audit_20260924_v1/full_clip_candidates.jsonl")
ORIGINS = Path("docs/itm_rework/protocol/source_audit_20260924_v1/source_intervals.jsonl")


def comparison_clocks(*, path, fps, source_frames, start, end, feature_frames, cache_times):
    """The same target frames must support all three independently defined clocks."""
    clocks = {"nominal_20hz": np.arange(feature_frames) / 20.}
    origin_seconds = round(start * fps / 20.) / fps
    metadata = {}
    for name, source_path in (("stride_only", "untrimmed_diagnostic"), ("stride_and_prefix", path)):
        plan = notebook_source_indices(source_path=source_path, source_fps=fps, source_frames=source_frames,
                                       start_frame=start, end_frame=end)
        if plan["predicted_feature_frames"] != feature_frames:
            raise ValueError(f"{name}: predicted feature length differs; no silent crop")
        clocks[name] = plan["source_times_seconds"][:-1] - origin_seconds
        metadata[name] = {key: plan[key] for key in (
            "integer_stride", "dataset_prefix_trim_nominal_seconds", "notebook_end_would_clamp",
            "physical_sample_rate_hz", "nominal_to_physical_duration_ratio")}
    common = np.ones(feature_frames, dtype=bool)
    for times in clocks.values():
        common &= (times >= cache_times[0]) & (times <= cache_times[-1])
    return clocks, common, metadata


def make_plan(path):
    if path.exists():
        raise FileExistsError(path)
    candidates = sorted(read_jsonl(CANDIDATES), key=lambda row: row["motion_id"])
    cases = []
    for dataset, seconds in PREFIX_TRIM_SECONDS.items():
        selected, sources = [], set()
        for row in candidates:
            if (row["split"] != "train" or dataset not in row["source_group_id"].split("/")
                    or row["frame_slice"][1] <= seconds * 20 + 20 or row["source_group_id"] in sources):
                continue
            sources.add(row["source_group_id"])
            selected.append({**row, "dataset": dataset})
            if len(selected) == 2:
                break
        if len(selected) != 2:
            raise ValueError(f"Insufficient fixed candidates for {dataset}")
        cases.extend(selected)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"version": VERSION, "read_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "selection": "First two distinct train-exclusive sources per dataset, sorted motion ID; complete full-clip caption, <=196 frames and > prefix_nominal_frames+20; no geometry inspected.",
        "cases": cases, "inputs": [artifact(CANDIDATES), artifact(ORIGINS)],
        "budget": {"cases": 10, "clock_hypotheses": ["nominal_20hz", "stride_only", "stride_and_prefix"],
                   "minimum_common_frames": 20, "source_headers": 10, "body_model_forwards": 0,
                   "model_generation": 0, "training_updates": 0, "gpu_jobs": 0},
        "code": [artifact(Path(__file__)), artifact(Path("src/itm/data/humanml_clock.py"))]})
    print(json.dumps({"plan": str(path), "ids": [r["motion_id"] for r in cases]}))


def run_plan(plan_path, output):
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(plan_path.read_text())
    if plan["version"] != VERSION or len(plan["cases"]) != 10 or any(r["split"] != "train" for r in plan["cases"]):
        raise ValueError("Fixed ten-case train plan required")
    for a in plan["inputs"] + plan["code"]:
        if artifact(a["path"])["sha256"] != a["sha256"]:
            raise ValueError(f"Plan input or code changed: {a['path']}")
    origins = {r["motion_id"]: r for r in read_jsonl(ORIGINS)}
    motions = {r["motion_id"]: r for r in read_jsonl("outputs/manifests_full/train.jsonl")}
    caches = {r["motion_id"]: r for r in read_jsonl("outputs/manifests_full/train_standard_imu.jsonl")}
    rows = []
    for selected in plan["cases"]:
        mid = selected["motion_id"]
        source, motion, cache = origins[mid], motions[mid], caches[mid]
        if artifact(selected["text_artifact"]["path"])["sha256"] != selected["text_artifact"]["sha256"]:
            raise ValueError("Selected caption artifact changed")
        path = Path(source["resolved_source_path"])
        count = pose_frames(path)
        with np.load(path, allow_pickle=False) as data:
            fps = float(data["mocap_framerate"])
        target = np.load(motion["joints_path"], allow_pickle=False)[:, :22].astype(np.float64)
        with np.load(cache["imu_path"], allow_pickle=False) as data:
            source_joints = data["joints"][:, :22].astype(np.float64)
            cache_fps = float(data["fps"])
        if len(target) != selected["frame_slice"][1] or len(source_joints) != cache["num_frames"] or cache_fps != cache["fps"]:
            raise ValueError("Plan/manifest array lengths or FPS changed")
        row = {"motion_id": mid, "dataset": selected["dataset"], "source_group_id": source["source_group_id"],
               "feature_frames": len(target), "source_fps": fps, "source_frames": count,
               "artifacts": [artifact(p) for p in (path, motion["joints_path"], cache["imu_path"])], "metrics": {}}
        times = np.arange(len(source_joints)) / cache_fps
        try:
            clocks, common, metadata = comparison_clocks(path=source["source_group_id"], fps=fps,
                source_frames=count, start=source["start_frame_20fps"], end=source["end_frame_20fps"],
                feature_frames=len(target), cache_times=times)
        except ValueError as exc:
            row.update(status="not_estimable", reason=str(exc))
            rows.append(row)
            continue
        indices = np.flatnonzero(common)
        row.update(common_frame_count=len(indices), common_frame_fraction=len(indices) / len(target),
                   common_frame_indices=indices.tolist(), clock_metadata=metadata)
        if len(indices) < plan["budget"]["minimum_common_frames"]:
            row.update(status="not_estimable", reason="fewer_than_20_common_frames_no_replacement")
        else:
            row["status"] = "estimated_on_common_frames"
            row["clock_intervals_seconds"] = {}
            for name, clock_times in clocks.items():
                row["clock_intervals_seconds"][name] = [float(clock_times[common][0]), float(clock_times[common][-1])]
                positions = resample_positions_at_times(source_joints, times, clock_times[common])
                row["metrics"][name] = oracle_yaw_residual(positions, target[common])
        rows.append(row)
        print(json.dumps({"motion_id": mid, "status": row["status"], "common_frames": len(indices)}), flush=True)
    estimated = [r for r in rows if r["status"] == "estimated_on_common_frames"]
    key = "oracle_constant_yaw_mean_m"
    report = {"version": VERSION, "plan_artifact": artifact(plan_path), "rows": rows,
        "status_counts": dict(Counter(r["status"] for r in rows)), "planned_cases": 10,
        "prefix_lower_than_stride_only_count": sum(r["metrics"]["stride_and_prefix"][key] < r["metrics"]["stride_only"][key] for r in estimated),
        "equal_case_mean_oracle_residual_m": {name: float(np.mean([r["metrics"][name][key] for r in estimated])) if estimated else None
            for name in plan["budget"]["clock_hypotheses"]},
        "budget": plan["budget"],
        "dependencies": [artifact(p) for p in (Path("scripts/audit_humanml_imu_alignment.py"), Path("src/itm/data/physical_time.py"))],
        "limitations": ["Only ten predeclared train clips; not prevalence estimation over 2177 records.",
                        "Oracle GT yaw and common supported subset only; not a deployable transform or full-clip certification.",
                        "No threshold tuning, time-shift search, replacement of failed cases, runtime mapping or cache changes."]}
    write_json(output, report)
    print(json.dumps({"status_counts": report["status_counts"], "prefix_lower": report["prefix_lower_than_stride_only_count"],
                      "mean_m": report["equal_case_mean_oracle_residual_m"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("plan"); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run"); p.add_argument("--plan", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "plan":
        make_plan(args.output)
    else:
        run_plan(args.plan, args.output)
