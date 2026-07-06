#!/usr/bin/env python
"""Build SMPL-based 12D virtual IMU caches for HumanML-AMASS records."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np

from itm.data.amass_bridge import load_humanml_amass_index, source_crop_at_fps
from itm.data.manifest import read_jsonl
from itm.data.standard_imu import synthesize_standard_imu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--amass-root", required=True)
    parser.add_argument("--output-dir", default="outputs/standard_imu")
    parser.add_argument("--cache-manifest", default="outputs/manifests/standard_imu.jsonl")
    parser.add_argument("--imuposer-root", default="/home/a200/0relatedworks/IMUPoser")
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    _enable_legacy_chumpy_compatibility()
    sys.path.insert(0, str(Path(args.imuposer_root) / "src"))
    from imuposer import math as imu_math
    from imuposer.smpl.parametricModel import ParametricModel

    mapping = load_humanml_amass_index(args.index, args.amass_root)
    smpl_path = Path(args.imuposer_root) / "src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
    body_model = ParametricModel(smpl_path, device=torch.device(args.device))
    coordinate_rotation = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]],
        device=args.device,
    )
    output_root = Path(args.output_dir).resolve()
    entries = []
    built = 0
    for record in read_jsonl(args.manifest):
        if args.max_records is not None and built >= args.max_records:
            break
        motion_id = str(record["motion_id"])
        source = mapping.get(motion_id)
        if source is None or not source.source_path.exists():
            continue
        output = output_root / str(record["split"]) / f"{motion_id}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite or not output.exists():
            arrays = _process_source(
                source,
                body_model,
                imu_math,
                coordinate_rotation,
                args.target_fps,
                args.device,
            )
            np.savez_compressed(output, motion_id=np.asarray(motion_id), **arrays)
        with np.load(output) as cached:
            entries.append(
                {
                    "motion_id": motion_id,
                    "split": record["split"],
                    "imu_path": str(output),
                    "source_path": str(source.source_path),
                    "num_frames": int(cached["acceleration"].shape[0]),
                    "fps": float(cached["fps"]),
                    "schema": "standard_imu_v1",
                }
            )
        built += 1
        print(f"[{built}] {motion_id}: {entries[-1]['num_frames']} frames")

    cache_manifest = Path(args.cache_manifest).resolve()
    cache_manifest.parent.mkdir(parents=True, exist_ok=True)
    cache_manifest.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )
    print(f"records: {len(entries)}")
    print(f"cache_manifest: {cache_manifest}")
    return 0


def _enable_legacy_chumpy_compatibility() -> None:
    """Restore aliases required only while unpickling the legacy SMPL model."""

    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec


def _process_source(source, body_model, imu_math, coordinate_rotation, target_fps, device):
    import torch

    with np.load(source.source_path) as data:
        source_fps = float(data["mocap_framerate"])
        crop = source_crop_at_fps(source, source_fps)
        poses = np.asarray(data["poses"][crop], dtype=np.float32)
        translations = np.asarray(data["trans"][crop], dtype=np.float32)
        shape = np.asarray(data["betas"][:10], dtype=np.float32)
    if len(poses) < 3:
        raise ValueError(f"Source crop is too short: {source.source_path}")
    poses = poses.reshape(len(poses), -1, 3)
    if poses.shape[1] > 37:
        poses[:, 23] = poses[:, 37]
    poses = poses[:, :24]
    poses, translations = _resample_amass(
        poses, translations, source_fps, target_fps
    )

    pose = torch.from_numpy(poses).to(device)
    translation = torch.from_numpy(translations).to(device)
    translation = coordinate_rotation.matmul(translation.unsqueeze(-1)).reshape_as(translation)
    local_rotation = imu_math.axis_angle_to_rotation_matrix(pose).reshape(-1, 24, 3, 3)
    local_rotation[:, 0] = coordinate_rotation.matmul(local_rotation[:, 0])
    with torch.inference_mode():
        global_rotation, joints, vertices = body_model.forward_kinematics(
            local_rotation,
            torch.from_numpy(shape).to(device),
            translation,
            calc_mesh=True,
        )
    imu = synthesize_standard_imu(
        vertices.cpu().numpy(),
        global_rotation.cpu().numpy(),
        fps=target_fps,
        smooth_window=5,
    )
    return {
        "acceleration": imu.acceleration,
        "orientation": imu.orientation,
        "sensor_mask": imu.sensor_mask,
        "fps": np.asarray(target_fps, dtype=np.float32),
        "local_rotation": local_rotation.cpu().numpy().astype(np.float32),
        "joints": joints[:, :24].cpu().numpy().astype(np.float32),
        "translation": translation.cpu().numpy().astype(np.float32),
        "shape": shape.astype(np.float32),
    }


def _resample_amass(poses, translations, source_fps, target_fps):
    from scipy.spatial.transform import Rotation, Slerp

    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("FPS values must be positive")
    if len(poses) < 2 or np.isclose(source_fps, target_fps):
        return poses, translations
    source_times = np.arange(len(poses), dtype=np.float64) / source_fps
    frame_count = int(np.floor(source_times[-1] * target_fps)) + 1
    target_times = np.arange(frame_count, dtype=np.float64) / target_fps
    target_times = np.minimum(target_times, source_times[-1])
    resampled_pose = np.empty((frame_count, poses.shape[1], 3), dtype=np.float32)
    for joint in range(poses.shape[1]):
        rotations = Rotation.from_rotvec(poses[:, joint])
        resampled_pose[:, joint] = Slerp(source_times, rotations)(target_times).as_rotvec()
    resampled_translation = np.stack(
        [np.interp(target_times, source_times, translations[:, axis]) for axis in range(3)],
        axis=-1,
    ).astype(np.float32)
    return resampled_pose, resampled_translation


if __name__ == "__main__":
    raise SystemExit(main())
