#!/usr/bin/env python3
"""Bounded CPU G0 audit; writes a new directory, never changes legacy assets."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from itm.data.manifest import read_jsonl
from itm.data.observation import load_standard_cache


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_group(path):
    """Group by sequence across historical mount roots, not by cropped motion ID."""
    value = Path(path).as_posix()
    if "/AMASS/" not in value:
        raise ValueError(f"unrecognized source root: {path}")
    return "AMASS/" + value.split("/AMASS/", 1)[1]


def artifact(path, *, hash_file=True):
    path = Path(path)
    return {"path": str(path), "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path) if hash_file and path.is_file() else None}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-samples-per-split", type=int, default=8)
    args = parser.parse_args()
    if args.cache_samples_per_split < 1:
        parser.error("cache sample budget must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assets, groups, split_report, motion_map = [], defaultdict(list), {}, {}
    for split in ("train", "val", "test"):
        standard = Path(f"outputs/manifests_full/{split}_standard_imu.jsonl")
        manifest = Path(f"outputs/manifests_full/{split}.jsonl")
        assets.extend([artifact(standard), artifact(manifest)])
        rows = list(read_jsonl(standard))
        ids = [str(row["motion_id"]) for row in rows]
        report = {"records": len(rows), "unique_ids": len(set(ids)),
                  "schema_counts": dict(Counter(r.get("schema") for r in rows)),
                  "fps_counts": dict(Counter(str(r.get("fps")) for r in rows)),
                  "missing_cache_ids": [], "missing_source_ids": [], "sampled_cache_checks": []}
        # Entire manifest path checks; numerical validation is explicitly a sample.
        selected = set(np.linspace(0, len(rows)-1, min(args.cache_samples_per_split, len(rows)), dtype=int))
        for index, row in enumerate(rows):
            group = source_group(row["source_path"])
            item = {"motion_id": row["motion_id"], "split": split, "source_group_id": group,
                    "source_path": row["source_path"], "interval_status": "index_csv_unavailable",
                    "mirror_status": "not_recorded_in_standard_manifest"}
            groups[group].append(item)
            motion_map[row["motion_id"]] = item
            if not Path(row["imu_path"]).is_file():
                report["missing_cache_ids"].append(row["motion_id"])
            if not Path(row["source_path"]).is_file():
                report["missing_source_ids"].append(row["motion_id"])
            if index in selected:
                check = {"motion_id": row["motion_id"], **artifact(row["imu_path"])}
                try:
                    load_standard_cache(row)
                    check["status"] = "passed_schema_values_only"
                except (ValueError, OSError, KeyError) as exc:
                    check.update(status="failed", reason=str(exc))
                report["sampled_cache_checks"].append(check)
        split_report[split] = report
    overlaps = [{"source_group_id": key, "members": rows} for key, rows in sorted(groups.items())
                if len({row["split"] for row in rows}) > 1]
    dump(args.output / "data_audit.json", {"read_commit": head, "splits": split_report,
         "unique_source_sequences": len(groups), "cross_split_source_groups": len(overlaps),
         "cross_split_groups": overlaps,
         "limitations": ["sequence sharing does not prove temporal overlap",
                         "caption and mirror intervals need original index",
                         "cache checks do not certify original synthesis or HumanML alignment"]})
    with (args.output / "source_groups.jsonl").open("w") as f:
        for row in sorted(motion_map.values(), key=lambda x: (x["split"], x["motion_id"])):
            f.write(json.dumps(row) + "\n")

    # Match the existing evaluator's actual ordered filter, including its tail.
    test_rows = list(read_jsonl("outputs/manifests_full/test.jsonl"))
    mapped_ids = {row["motion_id"] for row in motion_map.values() if row["split"] == "test"}
    filtered, eligible = [], []
    for row in test_rows:
        if row["motion_id"] not in mapped_ids:
            continue
        item = {"motion_id": row["motion_id"]}
        try:
            length = len(np.load(row["joint_vec_path"], mmap_mode="r", allow_pickle=False))
            item["length"] = length
            captions = [line.split("#") for line in Path(row["text_path"]).read_text().splitlines()]
            caption = next((parts[0] for parts in captions if len(parts) >= 2), "")
            item["reason"] = "length_outside_40_196" if not 40 <= length <= 196 else "empty_caption" if not caption else "eligible"
        except (OSError, ValueError) as exc:
            item.update(reason="unavailable", detail=str(exc))
        if item["reason"] == "eligible":
            eligible.append(item)
        else:
            filtered.append(item)
    used = len(eligible) // 32 * 32
    dump(args.output / "quality_subset.json", {"mapped_count": len(mapped_ids), "eligible_count": len(eligible),
        "batch_size": 32, "evaluated_count_if_drop_last": used, "filtered": filtered,
        "evaluated": eligible[:used], "tail_dropped": eligible[used:],
        "status": "reconstructed_from_current_files_not_proof_of_historical_batch_order"})

    # Conservatively register any identified historical test usage as development
    # diagnostics. Absence from this scan NEVER certifies an untouched test set.
    exposure = defaultdict(set)
    def collect(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"motion_id", "target_motion_id", "imu_motion_id", "motion_ids", "shuffled_motion_ids"}:
                    for mid in child if isinstance(child, list) else [child]:
                        if isinstance(mid, str) and mid in motion_map:
                            exposure[mid].add(str(path))
                elif isinstance(child, (dict, list)):
                    collect(child, path)
        elif isinstance(value, list):
            for child in value:
                collect(child, path)
    scanned = 0
    for base in ("outputs/mdm_control", "outputs/motionlab", "outputs/nymeria_pilot"):
        for path in sorted(Path(base).rglob("*.json")):
            if path.stat().st_size > 20_000_000:
                continue
            try:
                collect(json.loads(path.read_text()), path)
                scanned += 1
            except (ValueError, OSError):
                continue
    exposed_groups = {motion_map[mid]["source_group_id"] for mid in exposure}
    with (args.output / "development_exposure.jsonl").open("w") as f:
        for mid, row in sorted(motion_map.items()):
            direct = sorted(exposure[mid])
            role = "historical_development_diagnostic" if direct else "same_source_as_exposed" if row["source_group_id"] in exposed_groups else "exposure_unknown"
            f.write(json.dumps({**row, "exposure_role": role, "decision_use": "requires_run_specific_review",
                "evidence_paths": direct, "audit_date": "2026-09-24", "sealed_test_eligible": False}) + "\n")
    dump(args.output / "exposure_summary.json", {"json_files_scanned": scanned, "direct_motion_count": sum(bool(v) for v in exposure.values()),
         "exposed_source_groups": len(exposed_groups), "no_unexposed_test_claim": True})

    evidence_paths = [
        "outputs/mdm_control/stage1_full_pilot_v2.pt", "outputs/mdm_control/stage2b_balanced_b.pt",
        "outputs/mdm_control/stage4_text_anchor.pt",
        "outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/model000475000.pt",
        "outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/args.json",
        "/home/a200/0proj/datasets/all/Mean.npy", "/home/a200/0proj/datasets/all/Std.npy",
        "/home/a200/mount/a40/relatedworks/mdm/HumanML3D/index.csv",
        "outputs/mdm_control/virtual_imu_roundtrip/gt_5x200.json",
        "outputs/mdm_control/virtual_imu_roundtrip/gt_wrists_5x200.json",
        "outputs/hymotion/prescreen_pad360_eval128_20260909/summary.json",
        "outputs/hymotion/output_velocity_x4_fresh64_128_192_audited_20260923/summary.json",
    ]
    for base in ("outputs/mdm_control/control_test100", "outputs/mdm_control/control_multiseed30",
                 "outputs/mdm_control/text_attribute_formal", "outputs/mdm_control/joint_scale_test100",
                 "outputs/motionlab/control_test100", "outputs/nymeria_pilot/stage2b_zero_shot"):
        for path in sorted(Path(base).rglob("*")):
            if path.is_file() and path.suffix in {".npz", ".json"}:
                evidence_paths.append(str(path))
    assets.extend(artifact(p) for p in sorted(set(evidence_paths)))
    with (args.output / "assets.jsonl").open("w") as f:
        for row in assets:
            f.write(json.dumps(row) + "\n")
    dump(args.output / "run.json", {"read_commit": head, "command": __import__("sys").argv,
         "assets": len(assets), "cache_sample_budget_per_split": args.cache_samples_per_split,
         "gpu_jobs": 0, "read_only_inputs": True})
    print(json.dumps({"output": str(args.output), "assets": len(assets), "source_groups": len(groups),
                     "cross_split_source_groups": len(overlaps), "quality_eligible": len(eligible)}))


if __name__ == "__main__":
    main()
