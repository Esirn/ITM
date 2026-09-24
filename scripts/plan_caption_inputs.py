#!/usr/bin/env python
"""Plan the fixed six val cases and four existing train diagnostics; never run a model."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_caption_intervals import artifact, write_json
from itm.data.caption_time import parse_timed_caption, plan_caption_crop
from itm.data.manifest import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    panel_path = Path("docs/itm_rework/protocol/validation_20260924_v1/panel.jsonl")
    panel = list(read_jsonl(panel_path))
    if len(panel) != 6:
        raise ValueError("Expected the unchanged six-case panel")
    selected = [("train", mid, None) for mid in ("000004", "000006", "000008", "000010")]
    selected += [("val", r["motion_id"], r["caption"]) for r in panel]
    motions, caches = {}, {}
    for split in ("train", "val"):
        motions.update({r["motion_id"]: r for r in read_jsonl(f"outputs/manifests_full/{split}.jsonl")})
        caches.update({r["motion_id"]: r for r in read_jsonl(f"outputs/manifests_full/{split}_standard_imu.jsonl")})
    plans = []
    for split, mid, selected_text in selected:
        motion, cache = motions[mid], caches[mid]
        if motion["split"] != split or cache["split"] != split:
            raise ValueError("Split mismatch")
        lines = [line for line in Path(motion["text_path"]).read_text().splitlines() if line.strip()]
        indices = [0] if selected_text is None else [i for i, line in enumerate(lines) if line.split("#")[0].strip() == selected_text]
        if len(indices) != 1:
            raise ValueError("Caption selection must be unambiguous")
        index = indices[0]
        row = {"motion_id": mid, "split": split, "caption_index": index,
               "caption_selection": "existing_first_caption" if selected_text is None else "fixed_panel_exact_text",
               "sensor_configs": ["head", "wrists"], "training_admissible": False,
               "artifacts": [artifact(motion["text_path"]), artifact(motion["joint_vec_path"]), artifact(cache["imu_path"])]}
        caption = parse_timed_caption(lines[index])
        row["caption"] = caption.text
        with np.load(cache["imu_path"], allow_pickle=False) as actual:
            frames, fps = len(actual["acceleration"]), float(actual["fps"])
        if frames != cache["num_frames"] or fps != cache["fps"]:
            raise ValueError("Cached time support differs from manifest")
        try:
            plan = plan_caption_crop(caption, len(np.load(motion["joint_vec_path"], mmap_mode="r")), np.arange(frames) / fps)
            row.update({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in plan.items()})
            row["status"] = "time_interval_plan_passed_not_alignment_certified"
        except ValueError as exc:
            row.update(status="blocked", reason=str(exc))
        plans.append(row)
    output = args.output_dir / "cases.jsonl"
    output.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in plans))
    write_json(args.output_dir / "summary.json", {"cases": len(plans),
        "planned": sum(r["status"] != "blocked" for r in plans), "blocked": sum(r["status"] == "blocked" for r in plans),
        "panel": artifact(panel_path), "cases_artifact": artifact(output),
        "code": [artifact(Path(__file__)), artifact(Path("src/itm/data/caption_time.py"))],
        "budget": {"maximum_cases": 10, "model_forwards": 0, "training_updates": 0, "gpu_jobs": 0},
        "limitations": ["No method loader change; integration remains with owner 01.",
                        "Time support checks do not certify semantics, coordinates or a compatible witness.",
                        "No HumanML feature recanonicalization; cropped representation alignment remains open."]})
    print(json.dumps({"cases": len(plans), "planned": sum(r["status"] != "blocked" for r in plans)}))


if __name__ == "__main__":
    main()
