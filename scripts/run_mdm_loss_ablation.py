#!/usr/bin/env python
"""Plan and optionally execute the strict MDM-control loss ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "diffusion": (0.0, 0.0, 0.0),
    "trajectory": (1.0, 0.0, 0.0),
    "trajectory_velocity": (1.0, 0.1, 0.0),
    "full_regularized": (1.0, 0.1, 0.003),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/manifests_full/train.jsonl")
    parser.add_argument(
        "--standard-imu-manifest",
        default="outputs/manifests_full/train_standard_imu.jsonl",
    )
    parser.add_argument(
        "--mdm-checkpoint",
        default="outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/model000475000.pt",
    )
    parser.add_argument(
        "--mdm-args",
        default="outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/args.json",
    )
    parser.add_argument("--mdm-root", default="outputs/mdm/runtime_root")
    parser.add_argument("--resume", default="outputs/mdm_control/stage1_full_pilot_v2.pt")
    parser.add_argument("--output-root", default="outputs/mdm_control/loss_ablation")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--execute-train", action="store_true")
    parser.add_argument("--execute-suite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    return parser.parse_args()


def main():
    args = parse_args()
    selected = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = sorted(set(selected) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {', '.join(unknown)}")
    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    plan = {"seed": args.seed, "variants": []}
    for name in selected:
        trajectory, velocity, jerk = VARIANTS[name]
        checkpoint = output_root / f"{name}.pt"
        experiment_root = output_root / f"experiments_{name}"
        train = _train_command(args, checkpoint, trajectory, velocity, jerk)
        suite = _suite_command(args, checkpoint, experiment_root)
        item = {
            "name": name,
            "loss_weights": {
                "trajectory": trajectory,
                "velocity": velocity,
                "jerk": jerk,
            },
            "checkpoint": str(checkpoint),
            "experiment_root": str(experiment_root),
            "train_command": train,
            "suite_command": suite,
            "train_status": "planned",
            "suite_status": "planned",
        }
        if args.execute_train and not (args.skip_existing and checkpoint.exists()):
            _run(train, output_root / f"{name}_train.log")
            item["train_status"] = "executed"
        elif checkpoint.exists():
            item["train_status"] = "existing"
        if args.execute_suite:
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing checkpoint for {name}: {checkpoint}")
            _run(suite, output_root / f"{name}_suite.log")
            item["suite_status"] = "executed"
        plan["variants"].append(item)
        (output_root / "ablation_plan.json").write_text(
            json.dumps(plan, indent=2), encoding="utf-8"
        )
    print(f"ablation plan: {output_root / 'ablation_plan.json'}")
    return 0


def _train_command(args, checkpoint, trajectory, velocity, jerk):
    command = [
        sys.executable,
        str(ROOT / "scripts/train_mdm_imu_control.py"),
        "--manifest", str((ROOT / args.manifest).resolve()),
        "--standard-imu-manifest", str((ROOT / args.standard_imu_manifest).resolve()),
        "--mdm-checkpoint", str((ROOT / args.mdm_checkpoint).resolve()),
        "--mdm-args", str((ROOT / args.mdm_args).resolve()),
        "--mdm-root", str((ROOT / args.mdm_root).resolve()),
        "--resume", str((ROOT / args.resume).resolve()),
        "--output", str(checkpoint),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--trajectory-loss-weight", str(trajectory),
        "--velocity-loss-weight", str(velocity),
        "--jerk-loss-weight", str(jerk),
        "--sensor-configs", "head,wrists",
        "--seed", str(args.seed),
        "--device", args.device,
    ]
    if args.max_records is not None:
        command.extend(("--max-records", str(args.max_records)))
    return command


def _suite_command(args, checkpoint, experiment_root):
    command = [
        sys.executable,
        str(ROOT / "scripts/run_mdm_control_experiment_suite.py"),
        "--control-checkpoint", str(checkpoint),
        "--output-root", str(experiment_root),
        "--device", args.device,
        "--seed", str(args.seed),
        "--execute",
    ]
    if args.skip_existing:
        command.append("--skip-existing")
    return command


def _run(command, log_path):
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    log_path.write_text(process.stdout + "\n" + process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"Command failed; inspect {log_path}")


if __name__ == "__main__":
    raise SystemExit(main())
