#!/usr/bin/env python3
"""Build HumanML3D-style JSONL manifests for ITM."""

from __future__ import annotations

import argparse
from pathlib import Path

from itm.config import dataset_paths, load_toml
from itm.data.manifest import build_manifest_entries, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paths.toml"))
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/manifests"))
    args = parser.parse_args()

    paths = dataset_paths(load_toml(args.config))
    split_file = Path(paths["motion_splits"]) / f"{args.split}.txt"
    entries = build_manifest_entries(
        split=args.split,
        split_file=split_file,
        texts_dir=paths["humanml3d_texts"],
        joints_dir=paths["humanml3d_joints"],
        joint_vecs_dir=paths["humanml3d_joint_vecs"],
        limit=args.limit,
    )
    output_path = args.output_dir / f"{args.split}.jsonl"
    count = write_jsonl(entries, output_path)
    print(f"wrote {count} entries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

