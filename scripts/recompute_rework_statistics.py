#!/usr/bin/env python3
"""Audit saved MDM output identities, then add source-group CIs to CPU re-evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_itm_rework import artifact, dump
from itm.data.manifest import read_jsonl
from itm.metrics.rework_protocol import grouped_bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--recomputed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    groups = {r["motion_id"]: r["source_group_id"] for r in read_jsonl(args.audit / "source_groups.jsonl")}
    assets = {r["path"]: r for r in read_jsonl(args.audit / "assets.jsonl")}
    checkpoint = assets["outputs/mdm_control/stage2b_balanced_b.pt"]["sha256"]
    imu_manifest = {r["motion_id"]: r for r in read_jsonl("outputs/manifests_full/test_standard_imu.jsonl")}
    checks = []
    for name in ("control_test100", "control_multiseed30", "text_attribute_formal"):
        base = Path("outputs/mdm_control") / name
        manifest = json.loads((base / "specs/manifest.json").read_text())
        results_dir = base / "results" if name == "text_attribute_formal" else base
        for group, specs in manifest["specs"].items():
            for spec in specs:
                spec_path = base / "specs" / Path(spec).name
                declared = json.loads(spec_path.read_text())
                path = results_dir / (Path(spec).stem + ".npz")
                with np.load(path, allow_pickle=False) as data:
                    meta = json.loads(str(data["metadata"].item()))
                    arrays = {key: data[key] for key in ("motion", "gt", "acceleration", "orientation")}
                    cases = meta["cases"]
                    lengths = meta["effective_lengths"]
                    if len(cases) != len(declared) or len(lengths) != len(cases):
                        raise ValueError(f"count mismatch: {path}")
                    if meta.get("control_checkpoint_sha256") != checkpoint:
                        raise ValueError(f"checkpoint hash mismatch or absent: {path}")
                    if meta.get("mdm_checkpoint_sha256") != assets["outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/model000475000.pt"]["sha256"]:
                        raise ValueError(f"backbone hash mismatch: {path}")
                    stretched = []
                    for index, (case, expected) in enumerate(zip(cases, declared)):
                        for key in ("motion_id", "target_motion_id", "imu_motion_id", "imu_mode", "sensor_config", "text"):
                            if case.get(key) != expected.get(key):
                                raise ValueError(f"case identity mismatch: {path}: {index}: {key}")
                        length = int(lengths[index])
                        # Actual sampler targets cache joints, not HumanML263.
                        # Derive its documented frame count independently of NPZ.
                        declared_length = expected.get("length")
                        if declared_length is None:
                            target_cache = imu_manifest[expected["target_motion_id"]]
                            declared_length = min(round(target_cache["num_frames"] * 20 / target_cache["fps"]), 196)
                        if length != int(declared_length):
                            raise ValueError(f"length mismatch: {path}: {index}")
                        for key in ("motion", "gt", "acceleration", "orientation"):
                            if length > arrays[key].shape[1] or not np.isfinite(arrays[key][index, :length]).all():
                                raise ValueError(f"invalid array: {path}: {key}")
                        input_cache = imu_manifest[expected["imu_motion_id"]]
                        physical_count = round(input_cache["num_frames"] * 20 / input_cache["fps"])
                        if physical_count != length:
                            stretched.append({"case_index": index, "physical_frames_20fps": physical_count,
                                              "fitted_frames": length})
                    checks.append({"path": str(path), "cases": len(cases), "status": "passed",
                        "checkpoint_hash": checkpoint, "stored_seed": meta["seed"],
                        "legacy_time_fit_cases": stretched,
                        "random_stream_certification": "initial_seed_recorded_full_stream_not_archived"})
    dump(args.output / "output_identity_checks.json", {"checks": checks, "sample_count": sum(r["cases"] for r in checks),
          "limitations": ["finite outputs and matching hashes do not establish input synchronization",
                           "no proof of shared subsequent sampler noise or historical code commit"]})
    results, per_sample = {}, []
    for name in ("control_test100", "control_multiseed30"):
        payload = json.loads((args.recomputed / f"{name}.json").read_text())
        for config in ("head", "wrists"):
            rows = []
            for record in payload["records"]:
                if name == "control_multiseed30":
                    if record["sensor_config"] != config: continue
                    item = dict(record)
                else:
                    values = record["configs"][config]
                    errors = values["active_error_m"]
                    item = {"motion_id": record["motion_id"], "sample_seed": 1234,
                        "paired_minus_text_error_m": errors["paired"]-errors["text"],
                        "paired_minus_shuffled_error_m": errors["paired"]-errors["shuffled"],
                        "paired_minus_text_jerk": values["paired_jerk_ratio"]-values["text_jerk_ratio"]}
                item.update(case_id=item["motion_id"], source_group_id=groups[item["motion_id"]],
                            run_id=name, sensor_config=config, observation_space="joint_proxy",
                            exposure_role="historical_development_diagnostic")
                rows.append(item)
            per_sample.extend(rows)
            results[f"{name}/{config}"] = {key: grouped_bootstrap(rows, key) for key in
                ("paired_minus_text_error_m", "paired_minus_shuffled_error_m", "paired_minus_text_jerk")}
    attributes = json.loads((args.recomputed / "text_attributes.json").read_text())["stage2b"]
    rows = []
    for record in attributes["records"]:
        vals = record["values"]
        base_error = vals["a person walks"]["head_error_m"]
        deltas = [v["head_error_m"]-base_error for k,v in vals.items() if k != "a person walks"]
        rows.append({"case_id": record["motion_id"], "source_group_id": groups[record["motion_id"]],
                     **record["contrasts"], "edited_minus_base_head_error_m": float(np.mean(deltas))})
    results["text_attributes/exploratory_not_compatible_C"] = {
        key: grouped_bootstrap(rows, key) for key in (*attributes["records"][0]["contrasts"], "edited_minus_base_head_error_m")}
    dump(args.output / "grouped_statistics.json", results)
    with (args.output / "paired_differences.jsonl").open("w") as f:
        for row in per_sample: f.write(json.dumps(row) + "\n")
    roundtrip = {}
    for name in ("gt_5x200", "gt_wrists_5x200"):
        path = Path("outputs/mdm_control/virtual_imu_roundtrip") / f"{name}.json"
        payload = json.loads(path.read_text())
        result = {"source": artifact(path), "fit_failure_rate": payload["failed_cases"] / len(payload["cases"]),
                  "certified": False, "metrics": {}}
        for key, threshold in payload["validation"]["thresholds"].items():
            values = np.array([row[key] for row in payload["cases"] if row["status"] == "ok"])
            result["metrics"][key] = {"mean": float(values.mean()), "median": float(np.median(values)),
                "p95": float(np.quantile(values, .95)), "threshold": threshold,
                "exceedance_rate": float(np.mean(values > threshold))}
        roundtrip[name] = result
    dump(args.output / "roundtrip_distributions.json", roundtrip)
    print(json.dumps({"output": str(args.output), "checked_npz": len(checks),
                      "checked_samples": sum(r["cases"] for r in checks)}))


if __name__ == "__main__":
    main()
