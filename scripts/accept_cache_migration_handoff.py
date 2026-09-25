#!/usr/bin/env python
"""Bounded owner-02 acceptance of method cache isolation and resume provenance."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from accept_caption_handoff import artifact
from itm.models.mdm_training_provenance import validate_training_resume

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    snapshot_path = ROOT / "docs/itm_rework/method/code_snapshot_cache_migration_v1.json"
    snapshot = json.loads(snapshot_path.read_text())
    verified = []
    for expected in snapshot["owned_files"] + snapshot["dependencies"]:
        actual = artifact(ROOT / expected["path"])
        if actual["sha256"] != expected["sha256"]:
            raise ValueError(f"Snapshot mismatch: {expected['path']}")
        verified.append(actual)
    replays = {}
    for name, script, saved in (
        ("cache_migration", "check_mdm_cache_migration.py", "cache_migration_20260925_v1.json"),
        ("caption", "check_mdm_caption_handoff.py", "caption_input_migration_20260925_v1.json"),
    ):
        replay_path = out / f"{name}_replay.json"
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--output", str(replay_path.resolve())],
                                cwd=ROOT, timeout=120, text=True, capture_output=True)
        (out / f"{name}_replay.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        replay = json.loads(replay_path.read_text())
        saved_path = ROOT / "docs/itm_rework/method" / saved
        if replay != json.loads(saved_path.read_text()):
            raise ValueError(f"Independent {name} replay differs from saved method report")
        replays[name] = {"saved": artifact(saved_path), "replay": artifact(replay_path), "exact_match": True}
    migration = json.loads((out / "cache_migration_replay.json").read_text())
    if len(migration["cases"]) != 5 or any(r[key]["status"] != "rejected" for r in migration["cases"]
                                         for key in ("original_manifest", "relabelled_manifest")):
        raise ValueError("Expected five original and five relabelled manifest rejections")
    caption = json.loads((out / "caption_replay.json").read_text())
    original_caption = ROOT / "docs/itm_rework/method/caption_input_20260925_v1.json"
    if caption != json.loads(original_caption.read_text()):
        raise ValueError("Caption inputs changed relative to the original accepted v3 report")
    card_path = ROOT / "docs/itm_rework/method/b0_smoke_model_card.json"
    card = json.loads(card_path.read_text())
    checkpoint = artifact(card["checkpoint"])
    if checkpoint["sha256"] != card["checkpoint_sha256"]:
        raise ValueError("Old smoke checkpoint changed")
    state = torch.load(card["checkpoint"], map_location="cpu", weights_only=True)
    if state["imu_normalization"] != card["imu_normalization"]:
        raise ValueError("Normalization differs from the card used for input-only replay")
    if "training_input_provenance" in state:
        raise ValueError("Expected old checkpoint without new provenance")
    try:
        validate_training_resume(state, {}, state["imu_normalization"])
    except ValueError as exc:
        legacy_rejection = str(exc)
        if "lacks training input provenance" not in legacy_rejection:
            raise
    else:
        raise AssertionError("Unbound legacy checkpoint was admitted for resume")
    report = {
        "kind": "owner02_cache_migration_acceptance", "method_commit": "b4f70fcd61a07c5539bdbc5c473a5c8fe91a6256",
        "read_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "status": "cache_identity_and_resume_guards_accepted_not_training_admission",
        "verified_snapshot_files": verified, "replays": replays,
        "counts": {"diagnostic_files": 5, "original_manifest_rejections": 5, "relabelled_manifest_rejections": 5,
                   "caption_passed_inputs": sum(r["status"] == "input_passed_not_alignment_certified" for r in caption["cases"]),
                   "caption_blocked_inputs": sum(r["status"] == "blocked_as_planned" for r in caption["cases"]),
                   "ssm_rejections": len(caption["ssm_rejections"])},
        "real_legacy_checkpoint_resume_rejection": legacy_rejection,
        "old_checkpoint_normalization_matches_model_card": True,
        "artifacts": [artifact(p) for p in (snapshot_path, original_caption, card_path,
            ROOT / "docs/itm_rework/method/CACHE_MIGRATION_HANDOFF_20260925.md", Path(__file__),
            ROOT / "scripts/accept_caption_handoff.py")] + [checkpoint],
        "budget": {"cpu_threads": 1, "replay_processes": 2, "timeout_seconds_per_replay": 120,
                   "real_training_statistics_recomputed": False, "model_forwards": 0,
                   "training_updates": 0, "generated_samples": 0, "gpu_jobs": 0},
        "limitations": ["Embedded-schema guard is method-side; public observation reader remains unchanged.",
                        "Matching provenance permits only a fresh-optimizer warm start, not exact optimizer/RNG resume.",
                        "Old checkpoint CPU loading and input replay checked; no new full sampling run performed.",
                        "Guard binds the listed paired inputs; it is not certification of all training semantics or physical alignment.",
                        "No new B0, compatible witness, source-clock migration or dataset training admission."]}
    (out / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], **report["counts"]}))


if __name__ == "__main__":
    main()
