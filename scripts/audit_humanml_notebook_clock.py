#!/usr/bin/env python
"""Bounded notebook-clock replay for the 13 observed source-support failures."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import subprocess

import numpy as np

from audit_caption_intervals import artifact, write_json
from itm.data.humanml_clock import PREFIX_TRIM_SECONDS, VERSION, notebook_source_indices
from itm.data.manifest import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    notebook_root = Path("/home/a200/mount/a40/relatedworks/mdm/HumanML3D")
    raw_path = notebook_root / "raw_pose_processing.ipynb"
    representation_path = notebook_root / "motion_representation.ipynb"
    notebook = json.loads(raw_path.read_text())
    raw_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    stride_cell = next(cell for cell in raw_cells if "down_sample = int(fps / ex_fps)" in cell)
    trim_cell = next(cell for cell in raw_cells if "data = data[start_frame:end_frame]" in cell)
    # Check the code we are replaying, but never execute the notebook/body models.
    assert any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ex_fps" for t in n.targets)
               and isinstance(n.value, ast.Constant) and n.value.value == 20 for n in ast.walk(ast.parse(stride_cell)))
    assert "bdata['poses'][::down_sample,...]" in stride_cell
    for dataset, seconds in PREFIX_TRIM_SECONDS.items():
        line = "data = data[int(0.5*fps):]" if seconds == .5 else f"data = data[{int(seconds)}*fps:]"
        assert f"if '{dataset}' in source_path:" in trim_cell and line in trim_cell
    (out / "raw_notebook_clock_cells.txt").write_text(stride_cell + "\n\n" + trim_cell + "\n")
    support_path = Path("docs/itm_rework/protocol/caption_audit_20260924_v1/time_support.json")
    support = json.loads(support_path.read_text())
    if support["failure_count"] != 13 or len(support["failures"]) != 13:
        raise ValueError("Fixed 13-case replay required")
    origins = {r["motion_id"]: r for r in read_jsonl("docs/itm_rework/protocol/source_audit_20260924_v1/source_intervals.jsonl")}
    results = []
    for row in support["failures"]:
        origin = origins[row["motion_id"]]
        source_artifact = row["source_artifact"]
        if artifact(source_artifact["path"])["sha256"] != source_artifact["sha256"]:
            raise ValueError("Source changed since support audit")
        plan = notebook_source_indices(source_path=origin["source_group_id"], source_fps=row["source_fps"],
            source_frames=row["source_pose_frames"], start_frame=origin["start_frame_20fps"], end_frame=origin["end_frame_20fps"])
        values = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in plan.items()}
        results.append({"motion_id": row["motion_id"], "source_artifact": source_artifact,
            "saved_feature_frames": row["feature_frames"],
            "feature_frame_count_matches": row["feature_frames"] == plan["predicted_feature_frames"],
            "bridge_assumed_source_start_seconds": origin["start_frame_20fps"] / 20., **values})
    # A bounded metadata census of known notebook dataset prefix corrections.
    clips_path = Path("docs/itm_rework/protocol/caption_audit_20260924_v1/clips.jsonl")
    clips = list(read_jsonl(clips_path))
    if len(clips) > 7500 or any(r["split"] not in ("train", "val") for r in clips):
        raise ValueError("Only bounded train/val metadata allowed")
    prefix_candidates, counts = [], Counter()
    for row in clips:
        for dataset, seconds in PREFIX_TRIM_SECONDS.items():
            if dataset in row["source_group_id"].split("/"):
                prefix_candidates.append({"motion_id": row["motion_id"], "split": row["split"],
                    "source_group_id": row["source_group_id"], "notebook_prefix_trim_nominal_seconds": seconds,
                    "status": "code_implied_offset_requires_artifact_alignment_audit",
                    "physical_seconds_certified": False, "training_admissible": False})
                counts[f"{row['split']}/{dataset}"] += 1
    (out / "prefix_candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prefix_candidates))
    report = {"version": VERSION, "read_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "results": results, "feature_frame_count_matches": sum(r["feature_frame_count_matches"] for r in results),
        "prefix_candidate_counts": dict(counts), "prefix_candidates_total": len(prefix_candidates),
        "artifacts": [artifact(p) for p in (raw_path, representation_path, support_path, clips_path,
            out / "raw_notebook_clock_cells.txt", out / "prefix_candidates.jsonl")],
        "code": [artifact(p) for p in (Path(__file__), Path("src/itm/data/humanml_clock.py"),
            Path("src/itm/data/amass_bridge.py"))],
        "budget": {"source_hash_checks": 13, "metadata_clips": len(clips), "body_model_forwards": 0,
                   "training_updates": 0, "generation_samples": 0, "gpu_jobs": 0},
        "limitations": ["Notebook formula plus frame counts support a clock hypothesis; historical raw processed joints are unavailable.",
                        "No joint-value or independently calibrated source-clock reproduction in this audit.",
                        "Prefix candidates are inferred from dataset names/code, not individually verified temporal misalignments.",
                        "Do not relabel existing IMU FPS, stretch signals, or claim deployed alignment from this diagnostic."]}
    write_json(out / "report.json", report)
    print(json.dumps({"fixed_cases": 13, "feature_frame_count_matches": report["feature_frame_count_matches"],
                      "prefix_candidates_total": len(prefix_candidates), "prefix_candidate_counts": dict(counts)}))


if __name__ == "__main__":
    main()
