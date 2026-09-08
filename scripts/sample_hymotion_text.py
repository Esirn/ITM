#!/usr/bin/env python3
"""Minimal official HY-Motion Text-to-Motion sampler for ITM migration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml


BODY_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int64,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hymotion-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=Path("outputs/hymotion/runtime_root"))
    parser.add_argument("--model", choices=("lite", "full"), default="lite")
    parser.add_argument("--text", required=True)
    parser.add_argument("--frames", type=int, default=120, help="Output frames at 30 FPS")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rot6d_to_matrix(value: np.ndarray) -> np.ndarray:
    """Convert HY-Motion's row-based 6D rotations to proper matrices."""

    rotation = np.asarray(value, dtype=np.float32)
    first = rotation[..., :3]
    second = rotation[..., 3:6]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-2)


def local_to_global(local: np.ndarray, parents: np.ndarray = BODY_PARENTS) -> np.ndarray:
    """Accumulate local joint rotations along the WoodenMesh body hierarchy."""

    local = np.asarray(local, dtype=np.float32)
    if local.shape[-3:] != (len(parents), 3, 3):
        raise ValueError(f"unexpected local rotation shape: {local.shape}")
    global_rotation = np.empty_like(local)
    global_rotation[..., 0, :, :] = local[..., 0, :, :]
    for joint in range(1, len(parents)):
        global_rotation[..., joint, :, :] = np.matmul(
            global_rotation[..., parents[joint], :, :], local[..., joint, :, :]
        )
    return global_rotation


def main() -> int:
    args = parse_args()
    if not 20 <= args.frames <= 360:
        raise ValueError("frames must be within the official 20..360 training range")
    source = args.hymotion_root.resolve()
    runtime = args.runtime_root.resolve()
    workspace = json.loads((runtime / "workspace.json").read_text(encoding="utf-8"))
    if workspace["mounted_text_encoder_matches_commit"]:
        print("mounted text encoder matches commit")
    else:
        print("mounted text encoder is modified; using clean extracted runtime")
    model_name = "HY-Motion-1.0-Lite" if args.model == "lite" else "HY-Motion-1.0"
    model_dir = source / "ckpts" / "tencent" / model_name
    config_path = model_dir / "config.yml"
    checkpoint_path = model_dir / "latest.ckpt"
    for path in (config_path, checkpoint_path, source / "stats", source / "ckpts/Qwen3-8B", source / "ckpts/clip-vit-large-patch14"):
        if not path.exists():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; HY-Motion sampling was not started")

    sys.path.insert(0, str(runtime))
    old_cwd = Path.cwd()
    os.chdir(source)
    try:
        from hymotion.utils.loaders import load_object

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        pipeline = load_object(
            config["train_pipeline"],
            config["train_pipeline_args"],
            network_module=config["network_module"],
            network_module_args=config["network_module_args"],
        )
        pipeline.to(device)
        pipeline.load_in_demo(str(checkpoint_path), build_text_encoder=True)
        pipeline.eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.inference_mode():
            result = pipeline.generate(
                args.text,
                [args.seed],
                args.frames / 30.0,
                cfg_scale=args.cfg_scale,
                length=args.frames,
            )
        elapsed = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    finally:
        os.chdir(old_cwd)

    arrays = {
        key: value.detach().cpu().numpy() if torch.is_tensor(value) else value
        for key, value in result.items()
        if key != "text"
    }
    arrays["global_rotations_mat"] = local_to_global(rot6d_to_matrix(arrays["rot6d"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": model_name,
        "text": args.text,
        "frames": args.frames,
        "fps": 30,
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "device": str(device),
        "sampling_seconds": elapsed,
        "peak_cuda_memory_bytes": peak,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "source_commit": workspace["commit"],
        "clean_runtime": True,
        "prompt_rewrite": False,
        "duration_estimation": False,
        "output_shapes": {key: list(np.asarray(value).shape) for key, value in arrays.items()},
    }
    np.savez_compressed(args.output, **arrays, metadata=np.array(json.dumps(metadata)))
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
