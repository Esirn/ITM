#!/usr/bin/env python
"""Five bounded CPU diagnostic rebuilds; never replace or admit production caches."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

from audit_caption_intervals import artifact, write_json
from audit_humanml_imu_alignment import oracle_yaw_residual
from build_standard_imu_cache import _enable_legacy_chumpy_compatibility, _process_source
from itm.data.amass_bridge import load_humanml_amass_index
from itm.data.humanml_clock import PREFIX_TRIM_SECONDS
from itm.data.manifest import read_jsonl
from itm.data.physical_time import resample_positions_at_times


def overlap_indices(old_frames, new_frames, offset_frames, *, stencil_radius=3):
    """Return matching samples; interior requires both full differentiation stencils."""
    if offset_frames < 0 or min(old_frames, new_frames) < 1 or stencil_radius < 0:
        raise ValueError("Valid nonnegative offset/radius and nonempty arrays required")
    new = np.arange(max(0, min(new_frames, old_frames - offset_frames)))
    old = new + offset_frames
    interior = ((new >= stencil_radius) & (new < new_frames - stencil_radius)
                & (old >= stencil_radius) & (old < old_frames - stencil_radius))
    return old, new, interior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.cache_dir.resolve().relative_to(root / "outputs/itm_rework")
    if args.output_dir.exists() or args.cache_dir.exists():
        raise FileExistsError("Use new diagnostic output/cache directories")
    plan = json.loads(args.plan.read_text())
    selected = []
    for dataset in PREFIX_TRIM_SECONDS:
        selected.append(next(r for r in plan["cases"] if r["dataset"] == dataset))
    if len(selected) != 5 or any(r["split"] != "train" for r in selected):
        raise ValueError("Five fixed train clips required")
    args.output_dir.mkdir(parents=True)
    args.cache_dir.mkdir(parents=True)
    budget = {"clips": 5, "selection": "first already-declared case in each dataset; no metric selection",
              "motion_ids": [r["motion_id"] for r in selected], "device": "cpu", "torch_threads": 2,
              "maximum_output_frames_per_clip": 300, "body_model_forwards": 5,
              "diffusion_forwards": 0, "training_updates": 0, "gpu_jobs": 0}
    write_json(args.output_dir / "budget.json", budget)
    import torch
    torch.set_num_threads(2)
    _enable_legacy_chumpy_compatibility()
    imuposer = Path("/home/a200/0relatedworks/IMUPoser")
    sys.path.insert(0, str(imuposer / "src"))
    from imuposer import math as imu_math
    from imuposer.smpl.parametricModel import ParametricModel
    model_path = imuposer / "src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
    model = ParametricModel(model_path, device=torch.device("cpu"))
    mapping = load_humanml_amass_index("/home/a200/mount/a40/relatedworks/mdm/HumanML3D/index.csv",
                                     "/home/a200/mount/a40/datasets/AMASS")
    caches = {r["motion_id"]: r for r in read_jsonl("outputs/manifests_full/train_standard_imu.jsonl")}
    motions = {r["motion_id"]: r for r in read_jsonl("outputs/manifests_full/train.jsonl")}
    coordinate = torch.tensor([[[1., 0, 0], [0, 0, 1], [0, -1, 0]]])
    checks = []
    for case in selected:
        mid = case["motion_id"]
        source = mapping[mid]
        seconds = PREFIX_TRIM_SECONDS[case["dataset"]]
        with np.load(source.source_path, allow_pickle=False) as data:
            source_fps = float(data["mocap_framerate"])
        # This audit deliberately excludes the near-60-Hz stride ambiguity.
        if source_fps not in (60., 120.) or source_fps / int(source_fps / 20.) != 20.:
            raise ValueError("Fixed rebuild only permits exact 60/120 Hz, not an SSM clock correction")
        length = source.end_frame_20fps - source.start_frame_20fps
        if length < 2 or length * 1.5 > budget["maximum_output_frames_per_clip"]:
            raise ValueError("Body-model frame budget exceeded")
        shift = int(seconds * 20)
        corrected = replace(source, start_frame_20fps=source.start_frame_20fps + shift,
                            end_frame_20fps=source.end_frame_20fps + shift)
        arrays = _process_source(corrected, model, imu_math, coordinate, 30., "cpu")
        output = args.cache_dir / f"{mid}.npz"
        np.savez_compressed(output, motion_id=np.asarray(mid), **arrays,
                            schema=np.asarray("standard_imu_prefix_diagnostic_v1"),
                            source_mapping_version=np.asarray("notebook-prefix-exact-fps-diagnostic-v1"),
                            prefix_seconds=np.asarray(seconds), training_admissible=np.asarray(False))
        with np.load(caches[mid]["imu_path"], allow_pickle=False) as old:
            if float(old["fps"]) != 30.:
                raise ValueError("Expected 30 Hz old cache")
            old_idx, new_idx, interior = overlap_indices(len(old["acceleration"]), len(arrays["acceleration"]), int(seconds * 30))
            comparisons = {}
            for key in ("joints", "orientation", "acceleration"):
                a, b = old[key][old_idx], arrays[key][new_idx]
                use = interior if key == "acceleration" else np.ones(len(old_idx), dtype=bool)
                if not use.any():
                    raise ValueError("No valid overlap for fixed comparison")
                comparisons[key] = {"frames": int(use.sum()), "max_abs_difference": float(np.max(np.abs(a[use] - b[use]))),
                                    "allclose_atol_1e-4_rtol_1e-4": bool(np.allclose(a[use], b[use], atol=1e-4, rtol=1e-4))}
        target = np.load(motions[mid]["joints_path"], allow_pickle=False)[:, :22]
        target_times = np.arange(len(target)) / 20.
        new_times = np.arange(len(arrays["joints"])) / 30.
        resampled = resample_positions_at_times(arrays["joints"][:, :22], new_times, target_times)
        checks.append({"motion_id": mid, "dataset": case["dataset"], "prefix_seconds": seconds,
                       "feature_frames": len(target), "new_cache_frames": len(new_times),
                       "full_feature_time_support": True, "overlap_frames": len(new_idx),
                       "acceleration_stencil_radius_frames": 3, "overlap_comparisons": comparisons,
                       "full_clip_oracle_geometry": oracle_yaw_residual(resampled, target),
                       "diagnostic_cache": artifact(output), "source": artifact(source.source_path),
                       "old_cache": artifact(caches[mid]["imu_path"]), "target": artifact(motions[mid]["joints_path"]),
                       "original_index_crop": [source.start_frame_20fps, source.end_frame_20fps],
                       "explicit_shifted_crop": [corrected.start_frame_20fps, corrected.end_frame_20fps],
                       "training_admissible": False})
        print(json.dumps({"motion_id": mid, "full_time_support": True, "overlap": comparisons}), flush=True)
    report = {"kind": "five_prefix_diagnostic_cache_rebuilds", "budget": budget,
              "plan": artifact(args.plan), "body_model": artifact(model_path), "checks": checks,
              "code": [artifact(p) for p in (Path(__file__), Path("scripts/build_standard_imu_cache.py"),
                       Path("src/itm/data/amass_bridge.py"), Path("src/itm/data/standard_imu.py"))],
              "limitations": ["Diagnostic schema only; never a replacement manifest or training-admissible cache.",
                              "Full temporal support is not semantic, coordinate or generated-IMU closure certification.",
                              "Oracle yaw uses target GT and is forbidden at inference.",
                              "Five exact-FPS sources only; no full 2177-case migration or source FPS calibration."]}
    write_json(args.output_dir / "report.json", report)


if __name__ == "__main__":
    main()
