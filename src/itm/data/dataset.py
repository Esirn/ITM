"""Numpy dataset utilities for text, motion, and synthetic IMU records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from itm.data.manifest import read_jsonl
from itm.data.synthetic_imu import (
    DEFAULT_CHILD_JOINT_INDICES,
    DEFAULT_PARENT_JOINT_INDICES,
    DEFAULT_SENSOR_JOINT_INDICES,
    synthesize_sparse_imu,
)


@dataclass(frozen=True)
class CaptionRecord:
    """One HumanML3D-style caption line."""

    caption: str
    tokens: str
    start: float
    end: float


def parse_caption_line(line: str) -> CaptionRecord:
    """Parse a HumanML3D caption line.

    Expected format is ``caption#tokens#start#end``. Missing optional fields are
    tolerated so small hand-written fixtures remain easy to create.
    """

    parts = line.strip().split("#")
    caption = parts[0].strip() if parts else ""
    tokens = parts[1].strip() if len(parts) > 1 else ""
    start = _parse_float(parts[2]) if len(parts) > 2 else 0.0
    end = _parse_float(parts[3]) if len(parts) > 3 else 0.0
    return CaptionRecord(caption=caption, tokens=tokens, start=start, end=end)


def read_caption_records(text_path: str | Path) -> list[CaptionRecord]:
    """Read non-empty caption lines from a HumanML3D-style text file."""

    path = Path(text_path)
    return [
        parse_caption_line(line)
        for line in path.read_text(errors="ignore").splitlines()
        if line.strip()
    ]


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _cache_index(cache_manifest_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if cache_manifest_path is None:
        return {}
    return {str(record["motion_id"]): record for record in read_jsonl(cache_manifest_path)}


class TextIMUMotionDataset:
    """Load manifest records with text, motion arrays, and synthetic IMU arrays.

    This class intentionally returns numpy arrays and plain metadata. It keeps
    the data contract independent from PyTorch so early preprocessing and
    sanity checks work in a minimal conda environment.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        imu_cache_manifest_path: str | Path | None = None,
        text_index: int = 0,
        load_joints: bool = True,
        load_joint_vec: bool = True,
        load_imu: bool = True,
        synthesize_imu_if_missing: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = list(read_jsonl(self.manifest_path))
        self.imu_records = _cache_index(imu_cache_manifest_path)
        self.text_index = text_index
        self.load_joints = load_joints
        self.load_joint_vec = load_joint_vec
        self.load_imu = load_imu
        self.synthesize_imu_if_missing = synthesize_imu_if_missing

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        motion_id = str(record["motion_id"])
        captions = read_caption_records(record["text_path"])
        caption_record = _select_caption(captions, self.text_index)

        sample: dict[str, Any] = {
            "motion_id": motion_id,
            "split": str(record["split"]),
            "caption": caption_record.caption,
            "caption_record": caption_record,
            "captions": captions,
            "text_path": str(record["text_path"]),
            "joints_path": str(record["joints_path"]),
            "joint_vec_path": str(record["joint_vec_path"]),
        }

        joints = None
        needs_joints_for_imu = (
            self.load_imu
            and self.synthesize_imu_if_missing
            and motion_id not in self.imu_records
        )
        if self.load_joints or needs_joints_for_imu:
            joints = np.load(record["joints_path"]).astype(np.float32, copy=False)
            if self.load_joints:
                sample["joints"] = joints

        if self.load_joint_vec:
            sample["motion"] = np.load(record["joint_vec_path"]).astype(
                np.float32,
                copy=False,
            )

        if self.load_imu:
            sample.update(self._load_imu(record, joints))

        return sample

    def _load_imu(
        self,
        record: dict[str, Any],
        joints: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        motion_id = str(record["motion_id"])
        cache_record = self.imu_records.get(motion_id)
        if cache_record is not None:
            with np.load(cache_record["imu_path"]) as cached:
                data: dict[str, np.ndarray] = {
                    "imu_acceleration": cached["acceleration"].astype(
                        np.float32,
                        copy=False,
                    ),
                    "sensor_joint_indices": cached["sensor_joint_indices"].astype(
                        np.int64,
                        copy=False,
                    ),
                }
                if "orientation_vectors" in cached.files:
                    data["imu_orientation"] = cached["orientation_vectors"].astype(
                        np.float32,
                        copy=False,
                    )
                    data["parent_joint_indices"] = cached["parent_joint_indices"].astype(
                        np.int64,
                        copy=False,
                    )
                    data["child_joint_indices"] = cached["child_joint_indices"].astype(
                        np.int64,
                        copy=False,
                    )
                return data

        if not self.synthesize_imu_if_missing:
            raise KeyError(f"No IMU cache entry for motion_id={motion_id}")
        if joints is None:
            joints = np.load(record["joints_path"]).astype(np.float32, copy=False)

        imu = synthesize_sparse_imu(
            joints.tolist(),
            DEFAULT_SENSOR_JOINT_INDICES,
            parent_joint_indices=DEFAULT_PARENT_JOINT_INDICES,
            child_joint_indices=DEFAULT_CHILD_JOINT_INDICES,
        )
        data = {
            "imu_acceleration": np.asarray(imu.acceleration, dtype=np.float32),
            "sensor_joint_indices": np.asarray(imu.sensor_joint_indices, dtype=np.int64),
        }
        if imu.orientation_vectors is not None:
            data["imu_orientation"] = np.asarray(imu.orientation_vectors, dtype=np.float32)
            data["parent_joint_indices"] = np.asarray(
                imu.parent_joint_indices,
                dtype=np.int64,
            )
            data["child_joint_indices"] = np.asarray(
                imu.child_joint_indices,
                dtype=np.int64,
            )
        return data


def _select_caption(captions: list[CaptionRecord], text_index: int) -> CaptionRecord:
    if not captions:
        return CaptionRecord(caption="", tokens="", start=0.0, end=0.0)
    return captions[text_index % len(captions)]


def pad_sequence(
    arrays: Iterable[np.ndarray],
    *,
    pad_value: float = 0.0,
    dtype: np.dtype | type | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad variable-length arrays on axis 0.

    Returns ``padded, mask, lengths`` where mask is true for valid frames.
    """

    items = [np.asarray(array) for array in arrays]
    if not items:
        raise ValueError("pad_sequence requires at least one array")
    lengths = np.asarray([item.shape[0] for item in items], dtype=np.int64)
    max_len = int(lengths.max(initial=0))
    trailing_shape = items[0].shape[1:]
    output_dtype = dtype if dtype is not None else items[0].dtype
    padded = np.full(
        (len(items), max_len, *trailing_shape),
        pad_value,
        dtype=output_dtype,
    )
    mask = np.zeros((len(items), max_len), dtype=bool)
    for idx, item in enumerate(items):
        if item.shape[1:] != trailing_shape:
            raise ValueError(
                f"Inconsistent trailing shape at index {idx}: "
                f"{item.shape[1:]} != {trailing_shape}"
            )
        length = item.shape[0]
        padded[idx, :length] = item
        mask[idx, :length] = True
    return padded, mask, lengths


def collate_text_imu_motion(
    samples: list[dict[str, Any]],
    *,
    pad_value: float = 0.0,
) -> dict[str, Any]:
    """Collate dataset samples into a padded numpy batch."""

    if not samples:
        raise ValueError("collate_text_imu_motion requires at least one sample")

    batch: dict[str, Any] = {
        "motion_id": [sample["motion_id"] for sample in samples],
        "split": [sample["split"] for sample in samples],
        "caption": [sample["caption"] for sample in samples],
        "caption_record": [sample["caption_record"] for sample in samples],
        "text_path": [sample["text_path"] for sample in samples],
    }
    _maybe_pad(samples, batch, "motion", "motion", pad_value)
    _maybe_pad(samples, batch, "joints", "joints", pad_value)
    _maybe_pad(samples, batch, "imu_acceleration", "imu_acceleration", pad_value)
    _maybe_pad(samples, batch, "imu_orientation", "imu_orientation", pad_value)

    if "sensor_joint_indices" in samples[0]:
        batch["sensor_joint_indices"] = samples[0]["sensor_joint_indices"]
    if "parent_joint_indices" in samples[0]:
        batch["parent_joint_indices"] = samples[0]["parent_joint_indices"]
    if "child_joint_indices" in samples[0]:
        batch["child_joint_indices"] = samples[0]["child_joint_indices"]
    return batch


def _maybe_pad(
    samples: list[dict[str, Any]],
    batch: dict[str, Any],
    sample_key: str,
    batch_key: str,
    pad_value: float,
) -> None:
    if sample_key not in samples[0]:
        return
    padded, mask, lengths = pad_sequence(
        [sample[sample_key] for sample in samples],
        pad_value=pad_value,
    )
    batch[batch_key] = padded
    batch[f"{batch_key}_mask"] = mask
    batch[f"{batch_key}_length"] = lengths
