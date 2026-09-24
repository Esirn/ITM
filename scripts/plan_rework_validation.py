#!/usr/bin/env python3
"""Construct a deterministic val-only candidate panel and gated generation plan.

This command never launches GPU work. The plan cannot be promoted until the
method owner supplies B0 correctness, synchronization and an explicit run budget.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from audit_itm_rework import artifact, dump
from itm.data.dataset import read_caption_records
from itm.data.manifest import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-intervals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    intervals = list(read_jsonl(args.source_intervals))
    by_id = {row["motion_id"]: row for row in intervals}
    splits = defaultdict(set)
    for row in intervals: splits[row["source_group_id"]].add(row["split"])
    motions = {r["motion_id"]: r for r in read_jsonl("outputs/manifests_full/val.jsonl")}
    bins = {"short": [], "medium": [], "long": []}
    excluded = []
    for imu in sorted(read_jsonl("outputs/manifests_full/val_standard_imu.jsonl"), key=lambda r: r["motion_id"]):
        mid = imu["motion_id"]
        origin = by_id[mid]
        if splits[origin["source_group_id"]] != {"val"}:
            excluded.append({"motion_id": mid, "reason": "source_shared_with_train_or_test"}); continue
        record = motions[mid]
        motion = np.load(record["joint_vec_path"], mmap_mode="r", allow_pickle=False)
        length = len(motion)
        captions = read_caption_records(record["text_path"])
        caption = next((c for c in captions if c.start == 0 and c.end == 0 and c.caption), None)
        if not 40 <= length <= 196 or caption is None:
            excluded.append({"motion_id": mid, "reason": "length_or_no_full_clip_caption"}); continue
        name = "short" if length <= 80 else "medium" if length <= 140 else "long"
        bins[name].append({"case_id": f"val-{mid}", "motion_id": mid, "source_group_id": origin["source_group_id"],
            "split_role": "val", "caption": caption.caption, "caption_interval_seconds": [0., (length-1)/20],
            "length_frames_20fps": length, "length_bin": name, "imu_path": imu["imu_path"],
            "motion_path": record["joint_vec_path"], "source_interval_20fps": [origin["start_frame_20fps"], origin["end_frame_20fps"]],
            "sensor_configs": {"head": [4], "wrists": [0,1]}, "synchronization_status": "pending",
            "asymmetric_wrists_review": "pending", "compatibility_edit_status": "not_constructed"})
    selected, used = [], set()
    for name, rows in bins.items():
        chosen = []
        for row in rows:
            if row["source_group_id"] in used: continue
            chosen.append(row); used.add(row["source_group_id"])
            if len(chosen) == 2: break
        selected.extend(chosen)
    with (args.output / "panel.jsonl").open("w") as f:
        for row in selected: f.write(json.dumps(row) + "\n")
    conditions = ["text_only", "imu_only", "paired", "same_class_near_length_shuffled", "bypass", "numeric_zero", "physical_static"]
    dump(args.output / "dry_run.json", {"status": "blocked_for_generation", "panel": artifact(args.output / "panel.jsonl"),
        "panel_status": "candidate_ids_fixed_not_protocol_frozen", "selected_cases": len(selected),
        "eligible_per_length_bin": {k: len(v) for k,v in bins.items()}, "excluded": excluded,
        "stages": [
            {"gate": "G1", "action": "CPU schema/analytic observation/RNG tests", "gpu_jobs": 0},
            {"gate": "G2", "action": "B0 null/mask/sampler correctness supplied by method owner", "status": "awaiting_method_handoff"},
            {"gate": "G3", "action": "full-generation P then compatible C on candidate val panel", "status": "blocked"},
            {"gate": "G4", "action": "sealed evaluation of final checkpoint", "status": "blocked"}],
        "proposed_P_plan": {"conditions": conditions, "sensor_configs": ["head", "wrists"], "K": 1,
            "sampling_seeds": [20260924], "training_seed": None, "outputs_per_model": len(selected)*2*len(conditions),
            "sampler": None, "steps": None, "forward_calls": None, "gpu": None, "memory_limit_gib": None,
            "authorized_gpu_budget": 0},
        "blockers": ["no fixed B0 checkpoint/model card from owner 01", "input synchronization/canonicalization not certified",
            "no independent semantic/quality thresholds fitted on validation", "no verified compatibility witnesses",
            "same-class near-length shuffle partners and asymmetric wrist coverage need review",
            "sampler, guidance, runtime and memory budget not frozen", "independent test pool and pretrained exposure unresolved"]})
    print(json.dumps({"panel_cases": len(selected), "proposed_P_outputs_per_model": len(selected)*14, "gpu_jobs_started": 0}))


if __name__ == "__main__":
    main()
