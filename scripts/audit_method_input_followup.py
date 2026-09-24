#!/usr/bin/env python
"""Owner-02 bounded, input-only acceptance of the method seconds-v2 follow-up."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from train_mdm_imu_control import _PairedDataset, _collate

ROOT = Path(__file__).resolve().parents[1]
TRAIN_IDS = ["000004", "000008", "000010"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    snapshot_path = ROOT / "docs/itm_rework/method/code_snapshot_v2.json"
    snapshot = json.loads(snapshot_path.read_text())
    verified = {}
    for group in ("owned", "owner02_dependencies"):
        for path, expected in snapshot[group].items():
            actual = sha(ROOT / path)
            if actual != expected:
                raise ValueError(f"Snapshot mismatch: {path}")
            verified[path] = actual
    replay_path = out / "input_replay.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_mdm_input_handoff.py"), "--output", str(replay_path.resolve())],
        cwd=ROOT, text=True, capture_output=True, timeout=120,
    )
    (out / "input_replay.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    saved_path = ROOT / "docs/itm_rework/method/input_handoff_20260924_v2.json"
    saved = json.loads(saved_path.read_text())
    replay = json.loads(replay_path.read_text())
    if replay != saved:
        raise ValueError("Input-only replay differs from the saved method report")

    # Select by explicit IDs; taking the first three common IDs selects 000006.
    source_manifest = ROOT / "outputs/manifests_full/train.jsonl"
    source_rows = {r["motion_id"]: r for r in rows(source_manifest)}
    selected_path = out / "explicit_train_manifest.jsonl"
    selected_path.write_text("".join(json.dumps(source_rows[mid]) + "\n" for mid in TRAIN_IDS))
    checkpoint = ROOT / "outputs/itm_rework/method/b0_cpu_smoke_20260924_v1/model.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if sha(checkpoint) != saved["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the recorded smoke checkpoint")
    dataset = _PairedDataset(
        selected_path, ROOT / "outputs/manifests_full/train_standard_imu.jsonl",
        np.load("/home/a200/0proj/datasets/all/Mean.npy"),
        np.load("/home/a200/0proj/datasets/all/Std.npy"),
        {"head": (4,), "wrists": (0, 1)}, None, ROOT,
    )
    dataset.set_normalization(state["imu_normalization"])
    items = [dataset[i] for i in range(len(dataset))]
    motion, y = _collate(items)
    if not torch.isfinite(motion).all() or not torch.isfinite(y["imu"]).all():
        raise ValueError("Nonfinite training input")
    if sorted({item["motion_id"] for item in items}) != TRAIN_IDS:
        raise ValueError("Unexpected explicit panel")
    for item in items:
        np.testing.assert_array_equal(item["timestamps_seconds"], np.arange(len(item["motion"])) / 20.)
    report = {
        "kind": "owner02_input_followup_acceptance",
        "read_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "contract_version": "itm-rework-v0.2-draft",
        "pipeline_version": "itm-method-seconds-v2",
        "status": "input_checks_passed_with_documented_panel_discrepancy",
        "saved_input_report_reproduced_exactly": True,
        "method_report_actual_train_ids": sorted({r["motion_id"] for r in saved["train"]}),
        "followup_document_claimed_train_ids": TRAIN_IDS,
        "explicit_train": [dict(motion_id=i["motion_id"], sensor_config=i["sensor_config"],
                                metadata=i["input_metadata"], finite=True) for i in items],
        "verified_code_and_contract_hashes": verified,
        "artifacts": {str(p.relative_to(ROOT)): sha(p) for p in (
            snapshot_path, saved_path, checkpoint, source_manifest,
            ROOT / "docs/itm_rework/method/FOLLOWUP_V02.md",
            Path(__file__).resolve(),
        )},
        "budget": {"threads": 1, "input_replay_timeout_seconds": 120,
                   "explicit_train_cases": 3, "sensor_configs": 2,
                   "model_forwards": 0, "training_updates": 0, "generated_samples": 0, "gpu_jobs": 0},
        "limitations": ["No world-frame, skeleton or caption alignment certification.",
                        "No witness promotion, threshold change, new checkpoint or research outcome.",
                        "199-frame candidate still requires a separately reviewed 196-frame witness.",
                        "The corrected explicit panel is a new owner-02 check, not a correction to the old artifact."],
    }
    (out / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "explicit_train_inputs": len(items), "model_forwards": 0}))


if __name__ == "__main__":
    main()
