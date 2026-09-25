#!/usr/bin/env python
"""Owner-02 acceptance of caption-seconds-v3, without training or generation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def artifact(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "sha256": h.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    snapshot_path = ROOT / "docs/itm_rework/method/code_snapshot_caption_v3.json"
    snapshot = json.loads(snapshot_path.read_text())
    verified = []
    for group in ("owned", "read_dependencies"):
        for path, expected in snapshot[group].items():
            actual = artifact(ROOT / path)
            if actual["sha256"] != expected:
                raise ValueError(f"Snapshot mismatch: {path}")
            verified.append(actual)
    card_path = ROOT / "docs/itm_rework/method/b0_smoke_model_card.json"
    card = json.loads(card_path.read_text())
    checkpoint = artifact(card["checkpoint"])
    if checkpoint["sha256"] != card["checkpoint_sha256"]:
        raise ValueError("Smoke checkpoint hash mismatch")
    state = torch.load(card["checkpoint"], map_location="cpu", weights_only=True)
    if state["imu_normalization"] != card["imu_normalization"]:
        raise ValueError("Input-check model card normalization differs from checkpoint")
    replay_path = out / "input_replay.json"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/check_mdm_caption_handoff.py"),
                             "--output", str(replay_path.resolve())], cwd=ROOT,
                            timeout=120, text=True, capture_output=True)
    (out / "input_replay.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    replay = json.loads(replay_path.read_text())
    original_path = ROOT / "docs/itm_rework/method/caption_input_20260925_v1.json"
    original = json.loads(original_path.read_text())
    if replay != original:
        raise ValueError("Input replay differs from method artifact")
    passed = [r for r in replay["cases"] if r["status"] == "input_passed_not_alignment_certified"]
    blocked = [r for r in replay["cases"] if r["status"] == "blocked_as_planned"]
    if len(passed) != 18 or len(blocked) != 2 or len(replay["ssm_rejections"]) != 13:
        raise ValueError("Unexpected fixed-panel outcome")
    for row in passed:
        metadata = row["metadata"]
        if (metadata["training_admissible"] is not False
                or metadata["alignment_status"] != "uncertified_diagnostic"
                or metadata["notebook_clock_correction"] != "none"):
            raise ValueError("Unexpected admission or runtime clock mapping change")
        if row["motion_id"] == "000133" and metadata["caption_index"] != 1:
            raise ValueError("Fixed val caption selection changed")
    if {r["motion_id"] for r in blocked} != {"000004"}:
        raise ValueError("Unexpected blocked source")
    if not replay["training_admission"].startswith("rejected:"):
        raise ValueError("Diagnostic plan was admitted for training")
    report = {"kind": "owner02_caption_input_acceptance", "pipeline": replay["pipeline"],
        "method_commit": "1a7e887ae4c258cdd7325ca5853c2ff3001009d0",
        "read_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "status": "input_interface_accepted_not_physical_alignment_certified",
        "saved_method_report_reproduced_exactly": True, "verified_snapshot_files": verified,
        "model_card_normalization_matches_checkpoint": True,
        "checkpoint_training_contract": state["contract_version"],
        "counts": {"passed_inputs": len(passed), "passed_train_inputs": sum(r["split"] == "train" for r in passed),
                   "passed_val_inputs": sum(r["split"] == "val" for r in passed),
                   "blocked_inputs": len(blocked), "ssm_rejected": len(replay["ssm_rejections"])},
        "artifacts": [artifact(p) for p in (snapshot_path, card_path, original_path,
            ROOT / "docs/itm_rework/method/CAPTION_HANDOFF_20260925.md", Path(__file__), replay_path)] + [checkpoint],
        "budget": {"cpu_threads": 1, "replay_timeout_seconds": 120, "model_forwards": 0,
                   "normalization_recomputed": False, "training_updates": 0, "generated_samples": 0, "gpu_jobs": 0},
        "limitations": ["Nonzero source origins are covered by synthetic regression fixtures, not these fixed real plans.",
                        "No new checkpoint or promotion of old v0.1 training to the new input pipeline.",
                        "World/skeleton/caption semantics, SSM clocks, dataset prefixes and nonzero-crop root frames remain uncertified.",
                        "No compatible witness, threshold freeze, formal P/C/D result or training permission."]}
    (out / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], **report["counts"]}))


if __name__ == "__main__":
    main()
