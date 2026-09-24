#!/usr/bin/env python3
"""Resolve historical AMASS roots and audit source intervals without rebuilding caches."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from audit_itm_rework import artifact, dump, source_group
from itm.data.amass_bridge import load_humanml_amass_index
from itm.data.manifest import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--amass-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mapping = load_humanml_amass_index(args.index, args.amass_root)
    groups = defaultdict(list)
    for split in ("train", "val", "test"):
        for row in read_jsonl(f"outputs/manifests_full/{split}_standard_imu.jsonl"):
            record = mapping.get(row["motion_id"])
            item = {"motion_id": row["motion_id"], "split": split,
                    "source_group_id": source_group(row["source_path"])}
            if record is None:
                item["status"] = "index_missing"
            else:
                item.update(start_frame_20fps=record.start_frame_20fps, end_frame_20fps=record.end_frame_20fps,
                    resolved_source_path=str(record.source_path), source_exists=record.source_path.is_file(),
                    index_source_matches_manifest=source_group(record.source_path) == item["source_group_id"],
                    status="index_resolved")
            groups[item["source_group_id"]].append(item)
    pairs = []
    for group, members in sorted(groups.items()):
        for i, a in enumerate(members):
            for b in members[:i]:
                if a["split"] == b["split"] or "start_frame_20fps" not in a or "start_frame_20fps" not in b:
                    continue
                end_a, end_b = a["end_frame_20fps"], b["end_frame_20fps"]
                if end_a < 0 or end_b < 0:
                    status = "unknown_open_interval"
                else:
                    overlap = min(end_a, end_b) - max(a["start_frame_20fps"], b["start_frame_20fps"])
                    status = "overlap" if overlap > 0 else "disjoint_or_touching"
                pairs.append({"source_group_id": group, "a": a["motion_id"], "b": b["motion_id"],
                              "splits": [a["split"], b["split"]], "status": status})
    rows = [r for group in sorted(groups) for r in groups[group]]
    with (args.output / "source_intervals.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    dump(args.output / "source_overlap.json", {"index": artifact(args.index), "amass_root": str(args.amass_root),
        "mapped_records": len(rows), "resolved_sources_present": sum(r.get("source_exists", False) for r in rows),
        "source_mismatches": [r for r in rows if r.get("index_source_matches_manifest") is False],
        "cross_split_interval_pairs": pairs,
        "confirmed_overlapping_pairs": sum(p["status"] == "overlap" for p in pairs),
        "unknown_interval_pairs": sum(p["status"] == "unknown_open_interval" for p in pairs),
        "mirror_coverage": "standard manifests contain original AMASS IDs only; official mirrored HumanML pool not certified"})
    print(json.dumps({"output": str(args.output), "records": len(rows), "pairs": len(pairs),
                      "overlapping_pairs": sum(p["status"] == "overlap" for p in pairs)}))


if __name__ == "__main__":
    main()
