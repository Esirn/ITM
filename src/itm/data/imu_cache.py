"""Build cached synthetic IMU artifacts from manifest records."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from itm.data.manifest import read_jsonl
from itm.data.synthetic_imu import (
    DEFAULT_CHILD_JOINT_INDICES,
    DEFAULT_PARENT_JOINT_INDICES,
    DEFAULT_SENSOR_JOINT_INDICES,
    synthesize_sparse_imu,
)


@dataclass(frozen=True)
class IMUCacheEntry:
    """One cached synthetic IMU artifact."""

    motion_id: str
    split: str
    imu_path: str
    joints_path: str
    text_path: str
    acceleration_shape: tuple[int, ...]
    orientation_shape: tuple[int, ...] | None

    def to_json(self) -> str:
        return json.dumps(
            {
                "motion_id": self.motion_id,
                "split": self.split,
                "imu_path": self.imu_path,
                "joints_path": self.joints_path,
                "text_path": self.text_path,
                "acceleration_shape": list(self.acceleration_shape),
                "orientation_shape": (
                    list(self.orientation_shape)
                    if self.orientation_shape is not None
                    else None
                ),
            },
            ensure_ascii=False,
        )


def _array_shape(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(np.asarray(value).shape)


def build_imu_cache(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    include_orientation: bool = True,
    overwrite: bool = False,
) -> list[IMUCacheEntry]:
    """Generate compressed synthetic IMU files for a manifest.

    Args:
        manifest_path: JSONL manifest produced by ``build_manifest.py``.
        output_dir: Directory where ``<split>/<motion_id>.npz`` files are stored.
        include_orientation: Whether to compute default limb orientation vectors.
        overwrite: Recompute existing cache files when true.

    Returns:
        Cache manifest entries for all input records.
    """

    output_root = Path(output_dir)
    entries: list[IMUCacheEntry] = []
    for record in read_jsonl(manifest_path):
        motion_id = str(record["motion_id"])
        split = str(record["split"])
        imu_path = output_root / split / f"{motion_id}.npz"
        imu_path.parent.mkdir(parents=True, exist_ok=True)

        if overwrite or not imu_path.exists():
            joints = np.load(record["joints_path"])
            parent_indices = DEFAULT_PARENT_JOINT_INDICES if include_orientation else None
            child_indices = DEFAULT_CHILD_JOINT_INDICES if include_orientation else None
            imu = synthesize_sparse_imu(
                joints.tolist(),
                DEFAULT_SENSOR_JOINT_INDICES,
                parent_joint_indices=parent_indices,
                child_joint_indices=child_indices,
            )
            arrays = {
                "acceleration": np.asarray(imu.acceleration, dtype=np.float32),
                "sensor_joint_indices": np.asarray(
                    imu.sensor_joint_indices,
                    dtype=np.int64,
                ),
                "motion_id": np.asarray(motion_id),
                "split": np.asarray(split),
                "joints_path": np.asarray(str(record["joints_path"])),
                "text_path": np.asarray(str(record["text_path"])),
            }
            if imu.orientation_vectors is not None:
                arrays["orientation_vectors"] = np.asarray(
                    imu.orientation_vectors,
                    dtype=np.float32,
                )
                arrays["parent_joint_indices"] = np.asarray(
                    imu.parent_joint_indices,
                    dtype=np.int64,
                )
                arrays["child_joint_indices"] = np.asarray(
                    imu.child_joint_indices,
                    dtype=np.int64,
                )
            np.savez_compressed(imu_path, **arrays)

        with np.load(imu_path) as cached:
            orientation_shape = (
                tuple(cached["orientation_vectors"].shape)
                if "orientation_vectors" in cached.files
                else None
            )
            entries.append(
                IMUCacheEntry(
                    motion_id=motion_id,
                    split=split,
                    imu_path=str(imu_path),
                    joints_path=str(record["joints_path"]),
                    text_path=str(record["text_path"]),
                    acceleration_shape=tuple(cached["acceleration"].shape),
                    orientation_shape=orientation_shape,
                )
            )
    return entries


def write_imu_cache_manifest(
    entries: Iterable[IMUCacheEntry],
    output_path: str | Path,
) -> int:
    """Write cache entries as JSONL."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry.to_json())
            handle.write("\n")
            count += 1
    return count
