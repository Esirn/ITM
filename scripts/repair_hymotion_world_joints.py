#!/usr/bin/env python3
"""Write a corrected copy of legacy HY-Motion results without resampling."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from itm.data.hymotion import world_joints_from_result, forward_kinematics, rotation_6d_to_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wooden-assets", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        raise ValueError("Choose a new output path; original results are preserved")
    with np.load(args.input, allow_pickle=False) as data:
        arrays = dict(data)
    metadata = json.loads(arrays.pop("metadata").item())
    if metadata.get("joint_coordinate_system") == "world":
        raise ValueError("Result already uses world joints")
    template_path = args.wooden_assets / "j_template.bin"
    parents_path = args.wooden_assets / "kintree.bin"
    template = np.fromfile(template_path, dtype=np.float32).reshape(-1, 3)
    parents = np.fromfile(parents_path, dtype=np.int32)
    arrays["keypoints3d_official"] = arrays["keypoints3d"]
    arrays["keypoints3d"] = world_joints_from_result(arrays, template, parents)
    arrays["global_rotations_mat_legacy"] = arrays["global_rotations_mat"]
    arrays["global_rotations_mat"] = forward_kinematics(
        rotation_6d_to_matrix(arrays["rot6d"]), arrays["transl"], template[:22], parents[:22],
    )[1]
    metadata.update(
        joint_coordinate_system="world",
        rotation_6d_convention="interleaved first two columns",
        joint_reconstruction="WoodenMesh FK + ground-aligned transl",
        original_result=str(args.input.resolve()),
        original_result_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        template_sha256=hashlib.sha256(template_path.read_bytes()).hexdigest(),
        parents_sha256=hashlib.sha256(parents_path.read_bytes()).hexdigest(),
        output_shapes={key: list(value.shape) for key, value in arrays.items()},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, metadata=np.array(json.dumps(metadata)))
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
