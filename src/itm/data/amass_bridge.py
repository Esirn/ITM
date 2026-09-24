"""Map HumanML3D records back to their source AMASS motions."""

from __future__ import annotations

import csv
import math
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
    source_frame_count: int | None = None,
) -> slice:
    """Convert HumanML's 20 FPS crop, optionally rejecting source-end clamping."""

    if not math.isfinite(source_fps) or not math.isfinite(target_fps) or source_fps <= 0 or target_fps <= 0:
        raise ValueError("FPS values must be positive")
    if record.start_frame_20fps < 0:
        raise ValueError("Source crop start must be nonnegative")
    scale = source_fps / target_fps
    start = round(record.start_frame_20fps * scale)
    stop = None if record.end_frame_20fps < 0 else round(record.end_frame_20fps * scale)
    if stop is not None and stop <= start:
        raise ValueError("Source crop must be nonempty")
    if source_frame_count is not None:
        if source_frame_count < 1 or start >= source_frame_count or (stop is not None and stop > source_frame_count):
            raise ValueError(f"Source crop [{start}, {stop}) exceeds {source_frame_count} available frames; no silent clamping")
    return slice(start, stop)
