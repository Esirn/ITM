#!/usr/bin/env python3
"""Validate complete host IMU caches without loading PyTorch or using a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from itm.config import dataset_paths, load_toml
from itm.data.amass_bridge import load_humanml_amass_index
from itm.data.manifest import read_jsonl


def audit_split(manifest: Path, expected_ids: set[str], fps: float) -> dict:
    records = list(read_jsonl(manifest))
    ids = [str(r["motion_id"]) for r in records]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise ValueError(f"{manifest}: duplicate IDs or incomplete AMASS coverage")
    counts = []
    max_rotation_error = 0.0
    total_bytes = 0
    for index, record in enumerate(records):
        path = Path(record["imu_path"])
        if not Path(record["source_path"]).is_file():
            raise FileNotFoundError(record["source_path"])
        if record["schema"] != "standard_imu_v1" or float(record["fps"]) != fps:
            raise ValueError(f"{path}: unexpected schema or FPS")
        frames = int(record["num_frames"])
        if frames < 3:
            raise ValueError(f"{path}: fewer than three frames")
        shapes = {
            "acceleration": (frames, 6, 3),
            "orientation": (frames, 6, 3, 3),
            "sensor_mask": (6,),
            "local_rotation": (frames, 24, 3, 3),
            "joints": (frames, 24, 3),
            "translation": (frames, 3),
            "shape": (10,),
        }
        with np.load(path, allow_pickle=False) as data:
            if str(data["motion_id"].item()) != record["motion_id"] or float(data["fps"]) != fps:
                raise ValueError(f"{path}: NPZ metadata differs from manifest")
            for name, shape in shapes.items():
                value = data[name]
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"{path}: invalid {name} shape or nonfinite values")
            mask = data["sensor_mask"]
            if not np.isin(mask, [0, 1]).all() or not mask.any():
                raise ValueError(f"{path}: invalid sensor mask")
            for name in ("orientation", "local_rotation"):
                rotation = data[name]
                if name == "orientation":
                    rotation = rotation[:, mask.astype(bool)]
                orthogonal_error = np.max(np.abs(rotation @ rotation.swapaxes(-1, -2) - np.eye(3)))
                determinant_error = np.max(np.abs(np.linalg.det(rotation) - 1.0))
                error = float(max(orthogonal_error, determinant_error))
                max_rotation_error = max(max_rotation_error, error)
                if error > 1e-4:
                    raise ValueError(f"{path}: invalid {name} rotation matrices ({error})")
        counts.append(frames)
        total_bytes += path.stat().st_size
        if (index + 1) % 500 == 0:
            print(f"{manifest.name}: validated {index + 1}/{len(records)}", flush=True)
    return {
        "manifest": str(manifest.resolve()),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "records": len(records),
        "frames": sum(counts),
        "min_frames": min(counts) if counts else None,
        "max_frames": max(counts) if counts else None,
        "within_hymotion_20_360_frames": sum(20 <= n <= 360 for n in counts),
        "max_rotation_matrix_error": max_rotation_error,
        "cache_bytes": total_bytes,
        "fps": fps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=Path("outputs/manifests_full"))
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = dataset_paths(load_toml(args.config))
    mapping = load_humanml_amass_index(paths["humanml3d_index"], paths["amass"])
    report = {"config": str(args.config.resolve()), "splits": {}}
    seen = set()
    for split in args.splits:
        humanml = list(read_jsonl(args.manifest_dir / f"{split}.jsonl"))
        expected = {r["motion_id"] for r in humanml if r["motion_id"] in mapping}
        if seen.intersection(expected):
            raise ValueError("motion IDs overlap between splits")
        seen.update(expected)
        manifest = args.manifest_dir / f"{split}_standard_imu.jsonl"
        for record in read_jsonl(manifest):
            source = mapping[str(record["motion_id"])].source_path
            if record["split"] != split or Path(record["source_path"]).resolve() != source.resolve():
                raise ValueError(f"{manifest}: wrong split or AMASS source for {record['motion_id']}")
        report["splits"][split] = audit_split(manifest, expected, args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
