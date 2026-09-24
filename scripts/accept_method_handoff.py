#!/usr/bin/env python3
"""Owner-02 independent acceptance of saved method diagnostics, without sampling."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from audit_itm_rework import artifact, dump, sha256, source_group
from itm.data.humanml_torch import recover_from_ric_torch
from itm.data.manifest import read_jsonl
from itm.data.random_plan import sampling_noise, VERSION


def verify_random_plan(plan):
    draws = plan["noise_draws"]
    expected_steps = [-1, *range(plan["steps"]-1, 0, -1)]
    if plan["version"] != VERSION or [d["step"] for d in draws] != expected_steps:
        raise ValueError("incorrect addressed DDPM draw coverage")
    if plan["sampler"] != "ddpm" or len(plan["timestep_map"]) != plan["steps"]:
        raise ValueError("incorrect sampler or timestep map")
    addresses = {(d["backbone_id"],d["case_id"],d["sample_seed"],tuple(d["shape"])) for d in draws}
    if len(addresses) != 1:
        raise ValueError("one plan must keep its backbone/case/seed/shape address")
    payload = json.dumps(draws, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(payload).hexdigest() != plan["random_plan_id"]:
        raise ValueError("random plan digest mismatch")
    for draw in draws:
        _, actual = sampling_noise(tuple(draw["shape"]), backbone_id=draw["backbone_id"],
            case_id=draw["case_id"], sample_seed=draw["sample_seed"], step=draw["step"])
        if actual != draw:
            raise ValueError("addressed noise metadata/hash mismatch")
    return {"random_plan_id": plan["random_plan_id"], "draws_recomputed": len(draws), "status": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-dir", type=Path, default=Path("docs/itm_rework/method"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(2)
    snapshot = json.loads((args.method_dir / "code_snapshot.json").read_text())
    checked = []
    # Contract is versioned separately after acceptance. The old snapshot's
    # dependency hash is recorded, not treated as a mutable permanent contract.
    for path, digest in {**snapshot["owned"], **{k:v for k,v in snapshot["external_owner02_dependencies"].items() if k.endswith(".py")}}.items():
        if sha256(path) != digest:
            raise ValueError(f"handoff source changed: {path}")
        checked.append(artifact(path))
    rows = list(read_jsonl(args.method_dir / "smoke_records.jsonl"))
    if len(rows) != 2 or [r["case_index"] for r in rows] != [0,1]:
        raise ValueError("expected the complete two-case diagnostic handoff")
    schema = json.loads(Path("docs/itm_rework/protocol/result.schema.json").read_text())
    sources = {r["motion_id"]:r for r in read_jsonl("outputs/manifests_full/train_standard_imu.jsonl")}
    npz_path = Path(rows[0]["artifact_path"])
    with np.load(npz_path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    meta = json.loads(str(data["metadata"].item()))
    mean = torch.from_numpy(np.load("/home/a200/0proj/datasets/all/Mean.npy"))
    std = torch.from_numpy(np.load("/home/a200/0proj/datasets/all/Std.npy"))
    raw = torch.from_numpy(data["normalized_humanml263"]).squeeze(2).permute(0,2,1) * std + mean
    decoded = recover_from_ric_torch(raw).numpy()
    np.testing.assert_allclose(decoded, data["motion"], atol=1e-6, rtol=1e-5)
    joint_delta = float(np.max(np.abs(decoded-data["motion"])))
    random_checks, normalized = [], []
    for index, row in enumerate(rows):
        for path_key, hash_key in (("artifact_path","artifact_sha256"),("checkpoint_path","checkpoint_sha256")):
            if sha256(row[path_key]) != row[hash_key]:
                raise ValueError(f"changed {path_key}")
        if Path(row["artifact_path"]) != npz_path or meta["cases"][index] != row["case"]:
            raise ValueError("case/output identity mismatch")
        if json.loads(Path(row["metadata_path"]).read_text()) != meta:
            raise ValueError("NPZ metadata differs from sidecar")
        length = meta["effective_lengths"][index]
        times = data["timestamps_seconds"][:length]
        np.testing.assert_allclose(times, np.arange(length)/20, atol=1e-12, rtol=0)
        for key in ("motion", "normalized_humanml263", "gt", "acceleration", "orientation"):
            if not np.isfinite(data[key][index]).all():
                raise ValueError(f"nonfinite {key}")
        expected_mask = np.arange(data["motion"].shape[1]) < length
        np.testing.assert_array_equal(data["frame_mask"][index], expected_mask)
        plan = meta["random_plans"][index]
        if row["random_plan_id"] != plan["random_plan_id"]:
            raise ValueError("handoff random plan mismatch")
        address = plan["noise_draws"][0]
        if address["sample_seed"] != row["sampling_seed"] or address["backbone_id"] != meta["mdm_checkpoint_sha256"] or address["case_id"] != row["case"]["random_plan_case_id"]:
            raise ValueError("plan address differs from handoff seed/backbone/case")
        random_checks.append(verify_random_plan(plan))
        normalized.append({"artifact_schema":"owner02-method-diagnostic-v1", "case_id":f"train-{row['case']['motion_id']}-{row['case']['imu_mode']}",
            "source_group_id":source_group(sources[row["case"]["motion_id"]]["source_path"]), "split_role":"train_diagnostic",
            "protocol":None, "research_evaluation_eligible":False, "condition":row["case"]["imu_mode"],
            "sensor_config":row["case"]["sensor_config"], "output":artifact(npz_path), "case_index":index,
            "checkpoint_hash":row["checkpoint_sha256"], "train_seed":row["training_seed"],
            "sample_seed":row["sampling_seed"], "random_plan_id":row["random_plan_id"],
            "frames":length, "physical_seconds":[float(times[0]),float(times[-1])],
            "decode_max_abs_difference":joint_delta, "finite":True,
            "joint_success":None,"semantic_metrics":None,"observation_metrics":None,"quality_metrics":None,
            "exclusion_reason":"single-update train diagnostic; synchronization, compatibility and independent thresholds uncertified"})
    if random_checks[0]["random_plan_id"] != random_checks[1]["random_plan_id"]:
        raise ValueError("paired/null do not share their random plan")
    audit_json = args.method_dir / "checkpoint_audit_20260924_v1.json"
    audit = json.loads(audit_json.read_text())
    audit_npz = audit_json.with_suffix(".npz")
    comparisons = {}
    with np.load(audit_npz, allow_pickle=False) as data:
        for kind in ("prediction", "sample"):
            reference = data[f"original_{kind}"]
            for mode in ("strict", "legacy"):
                other = data[f"{mode}_null_{kind}"]
                value = float(np.max(np.abs(reference-other)))
                comparisons[f"{mode}_{kind}_max_abs"] = value
                if mode == "strict": np.testing.assert_allclose(reference, other, atol=1e-6, rtol=1e-5)
    random_checks.append(verify_random_plan(audit["random_plan"][0]))
    with (args.output / "accepted_diagnostics.jsonl").open("w") as f:
        for row in normalized: f.write(json.dumps(row)+"\n")
    report = {"read_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "kind":"independent_saved_artifact_acceptance_not_new_generation", "source_hash_checks":checked,
        "smoke_checkpoint":artifact(rows[0]["checkpoint_path"]),"smoke_output":artifact(npz_path),
        "decode_max_abs_difference":joint_delta, "random_plan_checks":random_checks,
        "saved_null_comparisons":comparisons,"saved_audit_artifacts":[artifact(audit_json),artifact(audit_npz)],
        "formal_schema_admission":{"accepted":False, "missing_fields":sorted(set(schema["required"])-set(rows[0])),
          "reason":"method handoff is explicitly diagnostic; do not invent metrics or change train to val"},
        "budget":{"device":"cpu","threads":2,"new_model_forwards":0,"new_generated_clips":0,"noise_arrays_recomputed":sum(r["draws_recomputed"] for r in random_checks)},
        "limitations":["no new null full-generation run; saved predictions independently compared",
                       "no motion quality or semantic evaluation inferred from finite values",
                       "B0 trained once, not validation-qualified"]}
    dump(args.output / "acceptance.json", report)
    print(json.dumps({"output":str(args.output), "decode_max_abs_difference":joint_delta,
                      "noise_draws_checked":report["budget"]["noise_arrays_recomputed"],"formal_admission":False}))


if __name__ == "__main__":
    main()
