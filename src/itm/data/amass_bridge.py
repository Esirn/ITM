"""Map HumanML3D records back to their source AMASS motions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HumanMLAMASSRecord:
    motion_id: str
    source_path: Path
    start_frame_20fps: int
    end_frame_20fps: int


def load_humanml_amass_index(
    index_path: str | Path,
    amass_root: str | Path,
) -> dict[str, HumanMLAMASSRecord]:
    """Load AMASS-backed HumanML rows, excluding KIT and HumanAct12."""

    root = Path(amass_root)
    records: dict[str, HumanMLAMASSRecord] = {}
    with Path(index_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = str(row["source_path"])
            if "/KIT/" in source or "/humanact12/" in source:
                continue
            relative = source.removeprefix("./pose_data/")
            source_path = root / Path(relative).with_suffix(".npz")
            motion_id = Path(row["new_name"]).stem
            records[motion_id] = HumanMLAMASSRecord(
                motion_id=motion_id,
                source_path=source_path,
                start_frame_20fps=int(row["start_frame"]),
                end_frame_20fps=int(row["end_frame"]),
            )
    return records


def source_crop_at_fps(
    record: HumanMLAMASSRecord,
    source_fps: float,
    *,
    target_fps: float = 20.0,
) -> slice:
    """Convert HumanML's 20 FPS crop to source-frame indices."""

    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("FPS values must be positive")
    scale = source_fps / target_fps
    start = round(record.start_frame_20fps * scale)
    stop = None if record.end_frame_20fps < 0 else round(record.end_frame_20fps * scale)
    return slice(start, stop)

