#!/usr/bin/env python
"""Check all audited prefixes independently of caption status, then at most 32 source headers."""
import argparse
from pathlib import Path
import zipfile

import numpy as np

from audit_caption_intervals import artifact, write_json
from itm.data.manifest import read_jsonl


def pose_frames(path):
    with zipfile.ZipFile(path) as archive, archive.open("poses.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, _ = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported NPY header {version}")
    return shape[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caption-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    clips = list(read_jsonl(args.caption_audit / "clips.jsonl"))
    if len(clips) > 7500 or any(r["split"] not in ("train", "val") for r in clips):
        raise ValueError("Bounded train/val audit required")
    caches = {r["motion_id"]: r for split in ("train", "val") for r in read_jsonl(f"outputs/manifests_full/{split}_standard_imu.jsonl")}
    origins = {r["motion_id"]: r for r in read_jsonl("docs/itm_rework/protocol/source_audit_20260924_v1/source_intervals.jsonl")}
    failures = []
    for row in clips:
        cache = caches[row["motion_id"]]
        last = (cache["num_frames"] - 1) / cache["fps"]
        requested_last = (row["current_prefix_frames"] - 1) / 20
        if requested_last > last:
            failures.append({"motion_id": row["motion_id"], "split": row["split"],
                "source_group_id": row["source_group_id"], "caption_first_status": row["first_caption_status"],
                "requested_prefix_last_seconds": requested_last, "cache_last_seconds": last,
                "feature_frames": row["motion_frames"], "cache_frames": cache["num_frames"]})
    if len(failures) > 32:
        raise ValueError("Source-header follow-up exceeds 32-file budget")
    for row in failures:
        mid = row["motion_id"]
        origin, cache = origins[mid], caches[mid]
        source = Path(origin["resolved_source_path"])
        count = pose_frames(source)
        with np.load(source, allow_pickle=False) as data:
            fps = float(data["mocap_framerate"])
        start = round(origin["start_frame_20fps"] * fps / 20)
        stop = count if origin["end_frame_20fps"] < 0 else round(origin["end_frame_20fps"] * fps / 20)
        available = max(0, min(stop, count) - min(start, count))
        with np.load(cache["imu_path"], allow_pickle=False) as data:
            actual_frames, actual_fps = len(data["acceleration"]), float(data["fps"])
        if actual_frames != cache["num_frames"] or actual_fps != cache["fps"]:
            raise ValueError("Cache differs from audited manifest")
        predicted = int(np.floor((available - 1) / fps * actual_fps)) + 1 if available else 0
        row.update(source_pose_frames=count, source_fps=fps, source_declared_crop=[start, stop],
                   crop_exceeds_source=stop > count, available_source_crop_frames=available,
                   predicted_resampled_cache_frames=predicted, actual_cache_frames=actual_frames,
                   predicted_cache_frames_match=predicted == actual_frames,
                   source_artifact=artifact(source), cache_artifact=artifact(cache["imu_path"]))
    write_json(args.output, {"kind": "caption_independent_prefix_support_audit", "clips": len(clips),
        "failures": failures, "failure_count": len(failures),
        "inputs": [artifact(args.caption_audit / "clips.jsonl")], "code": artifact(Path(__file__)),
        "budget": {"maximum_source_headers": 32, "actual_source_headers": len(failures), "model_forwards": 0, "gpu_jobs": 0},
        "limitations": ["Current source FPS metadata is checked, not independently calibrated.",
                        "No changed FPS, synthetic cache, time stretch or HumanML reconstruction.",
                        "Cause of the HumanML/source duration discrepancy is not certified."]})
    print(f"{len(clips)} prefixes checked; {len(failures)} failures with source headers audited")


if __name__ == "__main__":
    main()
