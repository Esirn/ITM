#!/usr/bin/env python
"""Build and optionally execute the next ITM MDM-control experiment suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from itm.experiments.mdm_control_suite import (
    SampleRecord,
    compute_run_metrics,
    sample_catalog,
    serialize_browser_result,
    write_summary_markdown,
)


DEFAULT_CHECKPOINT = ROOT / "outputs/mdm_control/stage1_full_pilot_v2.pt"
DEFAULT_MDM_ASSET_DIR = ROOT / "outputs/mdm/checkpoints_extracted/humanml_trans_enc_512"
DEFAULT_MDM_ARGS = DEFAULT_MDM_ASSET_DIR / "args.json"
DEFAULT_MDM_CHECKPOINT = DEFAULT_MDM_ASSET_DIR / "model000475000.pt"
DEFAULT_MDM_ROOT = ROOT / "outputs/mdm/runtime_root"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/mdm_control/experiments"
PROMPTS = [
    "a person walks",
    "a person walks while swinging arms",
    "a person walks slowly",
    "a person walks quickly",
    "a person turns while walking",
    "a person walks with large steps",
]
GUIDANCE_TEXT_SCALES = (1.0, 2.5, 4.0)
GUIDANCE_IMU_SCALES = (0.5, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--control-checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--mdm-args", default=str(DEFAULT_MDM_ARGS))
    parser.add_argument("--mdm-checkpoint", default=str(DEFAULT_MDM_CHECKPOINT))
    parser.add_argument("--mdm-root", default=str(DEFAULT_MDM_ROOT))
    parser.add_argument("--sample-script", default=str(ROOT / "scripts/sample_mdm_imu_control.py"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--standard-imu-manifest", default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--text-scale", type=float, default=2.5)
    parser.add_argument("--imu-scale", type=float, default=1.0)
    parser.add_argument("--joint-scale", type=float, default=1.0)
    parser.add_argument(
        "--guidance-mode", choices=("legacy", "factorized"), default="legacy"
    )
    parser.add_argument(
        "--branch-execution", choices=("sequential", "batched"), default="sequential"
    )
    parser.add_argument("--matrix-count", type=int, default=20)
    parser.add_argument("--same-text-count", type=int, default=30)
    parser.add_argument("--same-imu-count", type=int, default=30)
    parser.add_argument("--sweep-count", type=int, default=10)
    parser.add_argument(
        "--suite",
        choices=("all", "matrix", "same_text", "same_imu", "sweep"),
        default="all",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the expensive sampler. Without this flag only specs and requests are written.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest) if args.manifest else ROOT / f"outputs/manifests_full/{args.split}.jsonl"
    imu_manifest = (
        Path(args.standard_imu_manifest)
        if args.standard_imu_manifest
        else ROOT / f"outputs/manifests_full/{args.split}_standard_imu.jsonl"
    )
    catalog = sample_catalog(manifest, imu_manifest)
    if not catalog:
        raise RuntimeError(f"No samples found for split={args.split}")
    rng = random.Random(args.seed)
    runs = build_runs(args, catalog, rng)
    if args.max_runs is not None:
        runs = runs[: args.max_runs]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    completed = []
    for run in runs:
        run_dir = output_root / run["name"]
        wrote = write_run(args, run, run_dir, imu_manifest)
        completed.append(wrote)
        print(f"{wrote['status']}: {run_dir}")

    suite_summary = {
        "split": args.split,
        "seed": args.seed,
        "checkpoint": str(Path(args.control_checkpoint).resolve()),
        "mdm_checkpoint": str(Path(args.mdm_checkpoint).resolve()),
        "mdm_args": str(Path(args.mdm_args).resolve()),
        "mdm_root": str(Path(args.mdm_root).resolve()),
        "execute": args.execute,
        "guidance_mode": args.guidance_mode,
        "branch_execution": args.branch_execution,
        "joint_scale": args.joint_scale,
        "runs": completed,
    }
    (output_root / "suite_index.json").write_text(json.dumps(suite_summary, indent=2), encoding="utf-8")
    write_suite_summary(output_root / "SUMMARY.md", suite_summary)
    print(f"suite index: {output_root / 'suite_index.json'}")
    return 0


def build_runs(args: argparse.Namespace, catalog: list[SampleRecord], rng: random.Random) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if args.suite in ("all", "matrix"):
        pairs = random_pairs(catalog, args.matrix_count, rng)
        for sensor in ("head", "wrists"):
            for index, (sample_a, sample_b) in enumerate(pairs):
                runs.append(
                    matrix_run(
                        name=f"matrix_test_{sensor}/pair_{index:03d}_{sample_a.motion_id}_{sample_b.motion_id}",
                        split=args.split,
                        sample_a=sample_a,
                        sample_b=sample_b,
                        sensor_config=sensor,
                        seed=args.seed,
                        text_scale=args.text_scale,
                        imu_scale=args.imu_scale,
                    )
                )
    if args.suite in ("all", "same_text"):
        samples = walking_samples(catalog, args.same_text_count, rng)
        runs.append(
            same_text_run(
                name="same_text_different_imu/test_walk_head_wrists",
                split=args.split,
                samples=samples,
                seed=args.seed,
                text_scale=args.text_scale,
                imu_scale=args.imu_scale,
            )
        )
    if args.suite in ("all", "same_imu"):
        samples = walking_samples(catalog, args.same_imu_count, rng)
        runs.append(
            same_imu_run(
                name="same_head_imu_different_text/test_prompts",
                split=args.split,
                samples=samples,
                seed=args.seed,
                text_scale=args.text_scale,
                imu_scale=args.imu_scale,
            )
        )
    if args.suite in ("all", "sweep"):
        samples = walking_samples(catalog, args.sweep_count, rng)
        for sensor in ("head", "wrists"):
            for text_scale in GUIDANCE_TEXT_SCALES:
                for imu_scale in GUIDANCE_IMU_SCALES:
                    runs.append(
                        guidance_sweep_run(
                            name=(
                                f"guidance_sweep/test_{sensor}_"
                                f"text{text_scale:g}_imu{imu_scale:g}"
                            ),
                            split=args.split,
                            samples=samples,
                            sensor_config=sensor,
                            seed=args.seed,
                            text_scale=text_scale,
                            imu_scale=imu_scale,
                        )
                    )
    return runs


def write_run(
    args: argparse.Namespace,
    run: dict[str, Any],
    run_dir: Path,
    imu_manifest: Path,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "experiment": run["experiment"],
        "split": args.split,
        "seed": run["seed"],
        "text_scale": run["text_scale"],
        "imu_scale": run["imu_scale"],
        "joint_scale": args.joint_scale,
        "guidance_mode": args.guidance_mode,
        "branch_execution": args.branch_execution,
        "device": args.device,
        "checkpoint": str(Path(args.control_checkpoint).resolve()),
        "mdm_checkpoint": str(Path(args.mdm_checkpoint).resolve()),
        "mdm_args": str(Path(args.mdm_args).resolve()),
        "mdm_root": str(Path(args.mdm_root).resolve()),
        **run["request"],
    }
    (run_dir / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    (run_dir / "spec.json").write_text(json.dumps(run["spec"], indent=2), encoding="utf-8")
    if args.skip_existing and (run_dir / "results.npz").exists():
        status = "skipped-existing"
    elif args.execute:
        command = [
            sys.executable,
            str(Path(args.sample_script).resolve()),
            "--control-checkpoint",
            str(Path(args.control_checkpoint).resolve()),
        "--mdm-args",
        str(Path(args.mdm_args).resolve()),
        "--mdm-checkpoint",
        str(Path(args.mdm_checkpoint).resolve()),
        "--mdm-root",
        str(Path(args.mdm_root).resolve()),
            "--standard-imu-manifest",
            str(imu_manifest.resolve()),
            "--spec",
            str((run_dir / "spec.json").resolve()),
            "--output",
            str((run_dir / "results.npz").resolve()),
            "--seed",
            str(run["seed"]),
            "--text-scale",
            str(run["text_scale"]),
            "--imu-scale",
            str(run["imu_scale"]),
            "--joint-scale",
            str(args.joint_scale),
            "--guidance-mode",
            args.guidance_mode,
            "--branch-execution",
            args.branch_execution,
            "--device",
            args.device,
        ]
        process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        (run_dir / "generation.log").write_text(process.stdout + "\n" + process.stderr, encoding="utf-8")
        if process.returncode:
            raise RuntimeError(f"Generation failed for {run_dir}; inspect generation.log")
        status = "executed"
    else:
        status = "planned"

    if (run_dir / "results.npz").exists():
        metrics = compute_run_metrics(run_dir / "results.npz")
        metrics["metadata"]["experiment_name"] = run["name"]
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        write_summary_markdown(metrics, run_dir / "summary.md")
        result = serialize_browser_result(
            run_id=run["name"].replace("/", "__"),
            request=request,
            result_path=run_dir / "results.npz",
            samples=run["samples"],
        )
        (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return {
        "name": run["name"],
        "directory": str(run_dir),
        "status": status,
        "case_count": len(run["spec"]),
    }


def random_pairs(
    catalog: list[SampleRecord],
    count: int,
    rng: random.Random,
) -> list[tuple[SampleRecord, SampleRecord]]:
    if len(catalog) < 2:
        raise ValueError("At least two samples are required for matrix experiments")
    pairs = []
    for _ in range(count):
        sample_a, sample_b = rng.sample(catalog, 2)
        pairs.append((sample_a, sample_b))
    return pairs


def walking_samples(catalog: list[SampleRecord], count: int, rng: random.Random) -> list[SampleRecord]:
    keywords = ("walk", "walking", "stroll", "step", "limp", "turn")
    scored = [
        sample
        for sample in catalog
        if any(keyword in " ".join(sample.captions).lower() for keyword in keywords)
    ]
    pool = scored if len(scored) >= count else catalog
    if count > len(pool):
        count = len(pool)
    return rng.sample(pool, count)


def matrix_run(
    *,
    name: str,
    split: str,
    sample_a: SampleRecord,
    sample_b: SampleRecord,
    sensor_config: str,
    seed: int,
    text_scale: float,
    imu_scale: float,
) -> dict[str, Any]:
    common = {"sensor_config": sensor_config, "split": split}
    spec = [
        {**common, "motion_id": sample_a.motion_id, "text": "", "label": "None + IMU A", "text_scale": 0.0, "joint_scale": 0.0, "text_source": None, "imu_source": "A"},
        {**common, "motion_id": sample_b.motion_id, "text": "", "label": "None + IMU B", "text_scale": 0.0, "joint_scale": 0.0, "text_source": None, "imu_source": "B"},
        {**common, "motion_id": sample_a.motion_id, "text": sample_a.caption, "label": "Text A + None", "imu_scale": 0.0, "joint_scale": 0.0, "text_source": "A", "imu_source": None},
        {**common, "motion_id": sample_a.motion_id, "text": sample_a.caption, "label": "Text A + IMU A", "text_source": "A", "imu_source": "A"},
        {**common, "motion_id": sample_b.motion_id, "text": sample_a.caption, "label": "Text A + IMU B", "text_source": "A", "imu_source": "B"},
        {**common, "motion_id": sample_a.motion_id, "text": sample_b.caption, "label": "Text B + None", "imu_scale": 0.0, "joint_scale": 0.0, "text_source": "B", "imu_source": None},
        {**common, "motion_id": sample_a.motion_id, "text": sample_b.caption, "label": "Text B + IMU A", "text_source": "B", "imu_source": "A"},
        {**common, "motion_id": sample_b.motion_id, "text": sample_b.caption, "label": "Text B + IMU B", "text_source": "B", "imu_source": "B"},
    ]
    return {
        "name": name,
        "experiment": "matrix",
        "seed": seed,
        "text_scale": text_scale,
        "imu_scale": imu_scale,
        "request": {
            "sensor_config": sensor_config,
            "sample_a": sample_a.motion_id,
            "sample_b": sample_b.motion_id,
        },
        "samples": {"A": sample_a, "B": sample_b},
        "spec": spec,
    }


def same_text_run(
    *,
    name: str,
    split: str,
    samples: list[SampleRecord],
    seed: int,
    text_scale: float,
    imu_scale: float,
) -> dict[str, Any]:
    spec = []
    for sample in samples:
        for label, sensor, case_text_scale, case_imu_scale in (
            ("MDM Text-only", "head", text_scale, 0.0),
            ("ITM Text + head IMU", "head", text_scale, imu_scale),
            ("ITM Text + wrist IMUs", "wrists", text_scale, imu_scale),
        ):
            spec.append(
                {
                    "motion_id": sample.motion_id,
                    "text": "a person walks",
                    "label": label,
                    "sensor_config": sensor,
                    "split": split,
                    "text_scale": case_text_scale,
                    "imu_scale": case_imu_scale,
                    "joint_scale": 0.0 if case_imu_scale == 0.0 else 1.0,
                    "text_source": "fixed_walk",
                    "imu_source": sample.motion_id if case_imu_scale else None,
                }
            )
    return {
        "name": name,
        "experiment": "same_text_different_imu",
        "seed": seed,
        "text_scale": text_scale,
        "imu_scale": imu_scale,
        "request": {"fixed_text": "a person walks", "sample_count": len(samples)},
        "samples": {sample.motion_id: sample for sample in samples},
        "spec": spec,
    }


def same_imu_run(
    *,
    name: str,
    split: str,
    samples: list[SampleRecord],
    seed: int,
    text_scale: float,
    imu_scale: float,
) -> dict[str, Any]:
    spec = []
    for sample in samples:
        for prompt in PROMPTS:
            spec.append(
                {
                    "motion_id": sample.motion_id,
                    "text": prompt,
                    "label": prompt,
                    "sensor_config": "head",
                    "split": split,
                    "text_scale": text_scale,
                    "imu_scale": imu_scale,
                    "text_source": prompt,
                    "imu_source": sample.motion_id,
                }
            )
    return {
        "name": name,
        "experiment": "same_head_imu_different_text",
        "seed": seed,
        "text_scale": text_scale,
        "imu_scale": imu_scale,
        "request": {"sensor_config": "head", "prompts": PROMPTS, "sample_count": len(samples)},
        "samples": {sample.motion_id: sample for sample in samples},
        "spec": spec,
    }


def guidance_sweep_run(
    *,
    name: str,
    split: str,
    samples: list[SampleRecord],
    sensor_config: str,
    seed: int,
    text_scale: float,
    imu_scale: float,
) -> dict[str, Any]:
    spec = [
        {
            "motion_id": sample.motion_id,
            "text": "a person walks",
            "label": f"text {text_scale:g} imu {imu_scale:g}",
            "sensor_config": sensor_config,
            "split": split,
            "text_scale": text_scale,
            "imu_scale": imu_scale,
            "text_source": "fixed_walk",
            "imu_source": sample.motion_id,
        }
        for sample in samples
    ]
    return {
        "name": name,
        "experiment": "guidance_sweep",
        "seed": seed,
        "text_scale": text_scale,
        "imu_scale": imu_scale,
        "request": {
            "sensor_config": sensor_config,
            "fixed_text": "a person walks",
            "sample_count": len(samples),
        },
        "samples": {sample.motion_id: sample for sample in samples},
        "spec": spec,
    }


def write_suite_summary(path: Path, suite: dict[str, Any]) -> None:
    lines = [
        "# MDM Control Experiment Suite",
        "",
        f"- split: `{suite['split']}`",
        f"- seed: `{suite['seed']}`",
        f"- checkpoint: `{suite['checkpoint']}`",
        f"- mdm checkpoint: `{suite['mdm_checkpoint']}`",
        f"- mdm args: `{suite['mdm_args']}`",
        f"- mdm root: `{suite['mdm_root']}`",
        f"- executed sampler: `{suite['execute']}`",
        "",
        "| run | status | cases | directory |",
        "| --- | --- | ---: | --- |",
    ]
    for run in suite["runs"]:
        lines.append(
            f"| {run['name']} | {run['status']} | {run['case_count']} | `{run['directory']}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
