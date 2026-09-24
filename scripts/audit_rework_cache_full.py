#!/usr/bin/env python3
"""Reuse full cache validation with explicit AMASS-root migration sidecars."""
import argparse
import json
from pathlib import Path

from audit_itm_rework import artifact, dump
from audit_standard_imu_cache import audit_split
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
    report = {"index": artifact(args.index), "amass_root": str(args.amass_root), "splits": {},
              "scope": "all legacy cache arrays; root remap only; no cache values changed",
              "not_certified": ["HumanML synchronization", "source-disjoint split", "output IMU roundtrip"]}
    for split in ("train", "val", "test"):
        original = Path(f"outputs/manifests_full/{split}_standard_imu.jsonl")
        rows = list(read_jsonl(original))
        expected = {r["motion_id"] for r in read_jsonl(f"outputs/manifests_full/{split}.jsonl") if r["motion_id"] in mapping}
        derived = args.output / f"{split}_relocated_manifest.jsonl"
        with derived.open("w") as f:
            for row in rows:
                source = mapping[row["motion_id"]].source_path
                if row["split"] != split:
                    raise ValueError("wrong split label")
                f.write(json.dumps({**row, "source_path": str(source)}) + "\n")
        result = audit_split(derived, expected, fps=30.)
        result["original_manifest"] = artifact(original)
        report["splits"][split] = result
    dump(args.output / "summary.json", report)
    print(json.dumps({"output": str(args.output / "summary.json"),
                      "validated_records": sum(r["records"] for r in report["splits"].values())}))


if __name__ == "__main__":
    main()
