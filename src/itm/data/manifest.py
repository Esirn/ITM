"""Manifest utilities for HumanML3D-style text-to-motion data."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator, Optional


@dataclass(frozen=True)
class ManifestEntry:
    """One text-motion sample entry."""

    motion_id: str
    split: str
    text_path: str
    joints_path: str
    joint_vec_path: str
    num_texts: int

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def read_split_ids(split_file: str | Path, limit: Optional[int] = None) -> list[str]:
    """Read motion ids from a split file."""

    ids: list[str] = []
    for line in Path(split_file).read_text().splitlines():
        motion_id = line.strip()
        if not motion_id:
            continue
        ids.append(motion_id)
        if limit is not None and len(ids) >= limit:
            break
    return ids


def count_text_lines(text_path: str | Path) -> int:
    """Count non-empty HumanML3D caption lines."""

    path = Path(text_path)
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(errors="ignore").splitlines() if line.strip())


def build_manifest_entries(
    *,
    split: str,
    split_file: str | Path,
    texts_dir: str | Path,
    joints_dir: str | Path,
    joint_vecs_dir: str | Path,
    limit: Optional[int] = None,
    require_all: bool = True,
) -> list[ManifestEntry]:
    """Build manifest entries for a HumanML3D-style split."""

    entries: list[ManifestEntry] = []
    missing: list[str] = []
    for motion_id in read_split_ids(split_file, limit=limit):
        text_path = Path(texts_dir) / f"{motion_id}.txt"
        joints_path = Path(joints_dir) / f"{motion_id}.npy"
        joint_vec_path = Path(joint_vecs_dir) / f"{motion_id}.npy"
        required_paths = [text_path, joints_path, joint_vec_path]
        if require_all and not all(path.exists() for path in required_paths):
            missing.append(motion_id)
            continue
        entries.append(
            ManifestEntry(
                motion_id=motion_id,
                split=split,
                text_path=str(text_path),
                joints_path=str(joints_path),
                joint_vec_path=str(joint_vec_path),
                num_texts=count_text_lines(text_path),
            )
        )
    if require_all and missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} split ids are missing files: {preview}")
    return entries


def write_jsonl(entries: Iterable[ManifestEntry], output_path: str | Path) -> int:
    """Write manifest entries as JSONL."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry.to_json())
            handle.write("\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Read JSONL records."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)

