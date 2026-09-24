#!/usr/bin/env python
"""Bounded train/val caption/header audit; no test captions or model execution."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from itm.data.caption_time import VERSION, caption_interval, parse_timed_caption, plan_caption_crop
from itm.data.manifest import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def artifact(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    budget = {"maximum_clips": 7500, "splits": ["train", "val"], "caption_reads_per_clip": 1,
              "feature_access": "npy_header_mmap_shape_only", "imu_support": "previously_audited_manifest",
              "model_forwards": 0, "training_updates": 0, "gpu_jobs": 0}
    write_json(out / "budget.json", budget)
    origins_path = ROOT / "docs/itm_rework/protocol/source_audit_20260924_v1/source_intervals.jsonl"
    origins = list(read_jsonl(origins_path))
    by_id = {r["motion_id"]: r for r in origins}
    group_splits = defaultdict(set)
    for r in origins:
        group_splits[r["source_group_id"]].add(r["split"])
    panel_path = ROOT / "docs/itm_rework/protocol/validation_20260924_v1/panel.jsonl"
    panel = {r["motion_id"]: r for r in read_jsonl(panel_path)}
    inputs = []
    source_artifacts = [artifact(origins_path), artifact(panel_path)]
    for split in budget["splits"]:
        motion_path = ROOT / f"outputs/manifests_full/{split}.jsonl"
        imu_path = ROOT / f"outputs/manifests_full/{split}_standard_imu.jsonl"
        records = {r["motion_id"]: r for r in read_jsonl(motion_path)}
        for imu in read_jsonl(imu_path):
            if imu["motion_id"] not in records:
                raise ValueError(f"Missing paired motion {imu['motion_id']}")
            inputs.append((records[imu["motion_id"]], imu))
        source_artifacts.extend([artifact(motion_path), artifact(imu_path)])
    if len(inputs) > budget["maximum_clips"]:
        raise ValueError("Audit exceeds fixed clip budget")
    counts = defaultdict(Counter)
    details, candidates, selected = [], [], []
    for record, imu in inputs:
        mid, split = record["motion_id"], record["split"]
        if split != imu["split"] or split not in budget["splits"]:
            raise ValueError("Split mismatch")
        motion = np.load(record["joint_vec_path"], mmap_mode="r", allow_pickle=False)
        if motion.ndim != 2 or motion.shape[1] != 263:
            raise ValueError(f"Unexpected motion shape: {mid}")
        frames = len(motion)
        del motion
        text_path = Path(record["text_path"])
        text_bytes = text_path.read_bytes()
        lines = [line for line in text_bytes.decode().splitlines() if line.strip()]
        annotations, full_candidates = [], []
        source_times = np.arange(imu["num_frames"], dtype=np.float64) / imu["fps"]
        for index, line in enumerate(lines):
            annotation = {"caption_index": index}
            try:
                caption = parse_timed_caption(line)
                interval = caption_interval(caption, frames)
                annotation.update(interval)
                annotation["caption"] = caption.text
                try:
                    plan_caption_crop(caption, frames, source_times)
                    annotation["crop_status"] = "complete_interval_supported"
                    if interval["annotation_kind"] == "full_clip_sentinel":
                        full_candidates.append(annotation)
                except ValueError as exc:
                    annotation["crop_status"] = "blocked"
                    annotation["reason"] = str(exc)
            except ValueError as exc:
                annotation.update(crop_status="invalid_tags_or_interval", reason=str(exc), raw_line=line)
            annotations.append(annotation)
        first = annotations[0] if annotations else {}
        if "frame_slice" not in first:
            status = "invalid_or_missing_first_caption"
        elif first["frame_slice"] != [0, min(frames, 196)]:
            status = ("full_caption_model_truncated" if first["annotation_kind"] == "full_clip_sentinel"
                      else "tagged_caption_mismatched_to_current_prefix")
        elif first["crop_status"] != "complete_interval_supported":
            status = "first_interval_matches_but_imu_support_blocked"
        else:
            status = "first_caption_interval_matches_current_prefix"
        counts[split][status] += 1
        counts[split]["clips"] += 1
        counts[split]["annotations"] += len(annotations)
        counts[split]["invalid_annotations"] += sum(a["crop_status"] == "invalid_tags_or_interval" for a in annotations)
        origin = by_id[mid]
        exclusive = group_splits[origin["source_group_id"]] == {split}
        row = {"motion_id": mid, "split": split, "source_group_id": origin["source_group_id"],
               "source_exclusive_to_split": exclusive, "motion_frames": frames,
               "current_prefix_frames": min(frames, 196), "first_caption_status": status,
               "text_artifact": {"path": str(text_path), "sha256": hashlib.sha256(text_bytes).hexdigest()},
               "motion_path": record["joint_vec_path"], "annotations": annotations}
        details.append(row)
        if full_candidates and exclusive:
            candidate = full_candidates[0]
            candidates.append({"motion_id": mid, "split": split, "source_group_id": origin["source_group_id"],
                               "caption_index": candidate["caption_index"], "caption": candidate["caption"],
                               "frame_slice": candidate["frame_slice"], "text_artifact": row["text_artifact"],
                               "status": "full_clip_interval_candidate_only", "training_admissible": False,
                               "world_alignment_certified": False, "semantic_alignment_certified": False})
            counts[split]["source_exclusive_complete_full_caption_candidates"] += 1
        if mid in panel or (split == "train" and mid in {"000004", "000006", "000008", "000010"}):
            if mid in panel:
                row["fixed_panel_caption_matches"] = [a["caption_index"] for a in annotations
                                                      if a.get("caption") == panel[mid]["caption"]]
                if not row["fixed_panel_caption_matches"]:
                    raise ValueError(f"Fixed panel caption disappeared: {mid}")
            selected.append(row)
    for name, rows in (("clips.jsonl", details), ("full_clip_candidates.jsonl", candidates), ("fixed_cases.jsonl", selected)):
        with (out / name).open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    upstream = Path("/home/a200/0relatedworks/motion-diffusion-model/data_loaders/humanml/data/dataset.py")
    summary = {"version": VERSION, "read_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "counts": dict(counts), "budget": budget, "input_artifacts": source_artifacts,
               "code_artifacts": [artifact(p) for p in (Path(__file__), ROOT / "src/itm/data/caption_time.py",
                   ROOT / "src/itm/data/dataset.py", ROOT / "scripts/train_mdm_imu_control.py", upstream)],
               "outputs": [artifact(out / name) for name in ("clips.jsonl", "full_clip_candidates.jsonl", "fixed_cases.jsonl")],
               "limitations": ["Header/tag/source-support audit, not semantic or physical alignment certification.",
                               "Candidate metadata is not a training manifest or a compatible edit witness.",
                               "Source exclusivity does not certify backbone pretraining independence.",
                               "No test caption reads, no model execution, no changed method loader or contract."]}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary["counts"]))


if __name__ == "__main__":
    main()
