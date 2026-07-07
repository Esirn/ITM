#!/usr/bin/env python
"""Create a writable MDM runtime root with local SMPL assets."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/home/a200/0relatedworks/motion-diffusion-model")
DEFAULT_SMPL = Path("/home/a200/0proj/datasets/mdm-need/motion-diffusion-model/body_models/smpl")
DEFAULT_OUTPUT = ROOT / "outputs/mdm/runtime_root"
LINK_NAMES = (
    "assets",
    "data_loaders",
    "dataset",
    "diffusion",
    "eval",
    "model",
    "prepare",
    "sample",
    "train",
    "utils",
    "visualize",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE))
    parser.add_argument("--smpl-source", default=str(DEFAULT_SMPL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source_root).resolve()
    smpl = Path(args.smpl_source).resolve()
    output = Path(args.output_root).resolve()
    if not (source / "model/mdm.py").exists():
        raise FileNotFoundError(f"Invalid MDM source root: {source}")
    if not (smpl / "SMPL_NEUTRAL.pkl").exists():
        raise FileNotFoundError(f"Missing SMPL_NEUTRAL.pkl under: {smpl}")
    output.mkdir(parents=True, exist_ok=True)
    for name in LINK_NAMES:
        _link(source / name, output / name)
    body_models = output / "body_models"
    body_models.mkdir(exist_ok=True)
    _link(smpl, body_models / "smpl")
    for filename in ("LICENSE", "README.md"):
        if (source / filename).exists():
            _link(source / filename, output / filename)
    print(f"runtime root: {output}")
    return 0


def _link(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.resolve() == source:
            return
        raise FileExistsError(f"{target} already exists and does not point to {source}")
    target.symlink_to(source, target_is_directory=source.is_dir())


if __name__ == "__main__":
    raise SystemExit(main())
