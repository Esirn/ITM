#!/usr/bin/env python
"""Sample controlled MDM cases with shared diffusion noise."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import sys

import numpy as np

from itm.data.manifest import read_jsonl
from itm.data.standard_imu import sensor_mask
from itm.backbones import MDMBackbone, MotionConditions
from itm.models.mdm_imu_control import (
    MDMIMUControlConfig,
    install_imu_control,
    make_text_imu_guidance_model,
)
from itm.models.torch_frame_baseline import require_torch


SENSOR_CONFIGS = {"head": (4,), "wrists": (0, 1)}
ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--mdm-args", required=True)
    parser.add_argument("--standard-imu-manifest", required=True)
    parser.add_argument("--spec", required=True, help="JSON list of case objects")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mdm-checkpoint",
        default=None,
        help="Optional override for the MDM checkpoint path stored in the control checkpoint.",
    )
    parser.add_argument("--mean", default="/home/a200/0proj/datasets/all/Mean.npy")
    parser.add_argument("--std", default="/home/a200/0proj/datasets/all/Std.npy")
    parser.add_argument("--mdm-root", default=str(ROOT / "outputs/mdm/runtime_root"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--text-scale", type=float, default=2.5)
    parser.add_argument("--imu-scale", type=float, default=1.0)
    parser.add_argument("--joint-scale", type=float, default=1.0)
    parser.add_argument(
        "--guidance-mode", choices=("legacy", "factorized"), default="legacy"
    )
    parser.add_argument(
        "--branch-execution", choices=("sequential", "batched"), default="sequential"
    )
    parser.add_argument(
        "--no-condition-cache",
        action="store_true",
        help="Disable text/IMU caching for numerical compatibility diagnostics.",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch = require_torch()
    _legacy_compatibility()
    device = torch.device(args.device)
    _check_device(torch, device, args.min_free_gib)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    control_checkpoint_path = Path(args.control_checkpoint).resolve()
    mdm_args_path = Path(args.mdm_args).resolve()
    manifest_path = Path(args.standard_imu_manifest).resolve()
    spec_path = Path(args.spec).resolve()
    output = Path(args.output).resolve()
    mean_path = Path(args.mean).resolve()
    std_path = Path(args.std).resolve()
    control_state = torch.load(control_checkpoint_path, map_location="cpu", weights_only=True)
    mdm_checkpoint_path = (
        Path(args.mdm_checkpoint).resolve()
        if args.mdm_checkpoint
        else Path(control_state["mdm_checkpoint"]).resolve()
    )
    mdm_options = json.loads(mdm_args_path.read_text())
    _set_mdm_compatibility_defaults(mdm_options)
    root = Path(args.mdm_root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    from utils.model_util import create_model_and_diffusion, load_model_wo_clip

    mdm_args = SimpleNamespace(**mdm_options)
    data_stub = SimpleNamespace(dataset=SimpleNamespace(num_actions=1))
    model, diffusion = create_model_and_diffusion(mdm_args, data_stub)
    mdm_state = torch.load(mdm_checkpoint_path, map_location="cpu", weights_only=True)
    load_model_wo_clip(model, mdm_state)
    config = MDMIMUControlConfig(**control_state["control_config"])
    controlled, encoder, hook = install_imu_control(model, config)
    encoder.load_state_dict(control_state["imu_encoder"])
    controlled.adapters.load_state_dict(control_state["adapters"])
    model.to(device)
    model.eval()
    guided_model = make_text_imu_guidance_model(
        model, mode=args.guidance_mode, branch_execution=args.branch_execution
    )
    guided_model.to(device)
    guided_model.eval()

    cases = json.loads(spec_path.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("spec must contain a non-empty JSON list")
    records = {str(row["motion_id"]): row for row in read_jsonl(manifest_path)}
    loaded = [_load_case(case, records, control_state["imu_normalization"]) for case in cases]
    original_lengths = [len(case["imu"]) for case in loaded]
    effective_lengths = [min(length, 196) for length in original_lengths]
    frame_count = max(effective_lengths)
    batch = len(loaded)
    imu = torch.from_numpy(
        np.stack([_pad_frames(case["imu"], frame_count) for case in loaded])
    ).to(device)
    masks = torch.from_numpy(np.stack([case["sensor_mask"] for case in loaded])).to(device)
    lengths = torch.tensor(effective_lengths, dtype=torch.long, device=device)
    frame_mask = MDMBackbone.frame_mask(lengths, frame_count)
    y = {
        "imu": imu,
        "sensor_mask": masks,
        "imu_frame_mask": frame_mask,
        "text_scale": torch.tensor(
            [case.get("text_scale", args.text_scale) for case in loaded], device=device
        ),
        "imu_scale": torch.tensor(
            [case.get("imu_scale", args.imu_scale) for case in loaded], device=device
        ),
        "joint_scale": torch.tensor(
            [case.get("joint_scale", args.joint_scale) for case in loaded], device=device
        ),
    }
    mean = torch.from_numpy(np.load(mean_path)).to(device)
    std = torch.from_numpy(np.load(std_path)).to(device)
    backbone = MDMBackbone(
        model=model,
        diffusion=diffusion,
        mean=mean,
        std=std,
        recover_from_ric=recover_from_ric,
        guidance_model=guided_model,
    )
    conditions = MotionConditions(
        text=[case["text"] for case in loaded], lengths=lengths, extra=y
    )
    if not args.no_condition_cache:
        backbone.cache_conditions(conditions, encoder)
    elif args.branch_execution == "batched":
        raise ValueError("Batched branch execution requires condition caching")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        sample = backbone.sample(conditions, seed=args.seed)
        joints = backbone.decode_motion(sample).cpu()
    elapsed = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "control_checkpoint": str(control_checkpoint_path),
        "mdm_checkpoint": str(mdm_checkpoint_path),
        "seed": args.seed,
        "text_scale": args.text_scale,
        "imu_scale": args.imu_scale,
        "joint_scale": args.joint_scale,
        "guidance_mode": args.guidance_mode,
        "branch_execution": args.branch_execution,
        "condition_cache": not args.no_condition_cache,
        "device": str(device),
        "frame_count": frame_count,
        "original_lengths": original_lengths,
        "effective_lengths": effective_lengths,
        "sampling_seconds": elapsed,
        "peak_cuda_memory_bytes": peak_memory,
        "control_checkpoint_sha256": _sha256(control_checkpoint_path),
        "mdm_checkpoint_sha256": _sha256(mdm_checkpoint_path),
        "cases": [{k: v for k, v in case.items() if k not in {"imu", "gt", "raw_acceleration", "raw_orientation", "sensor_mask"}} for case in loaded],
    }
    np.savez_compressed(
        output,
        motion=joints.numpy(),
        gt=np.stack([_pad_frames(case["gt"][:, :22], frame_count) for case in loaded]),
        acceleration=np.stack([_pad_frames(case["raw_acceleration"], frame_count) for case in loaded]),
        orientation=np.stack([_pad_frames(case["raw_orientation"], frame_count) for case in loaded]),
        sensor_mask=np.stack([case["sensor_mask"] for case in loaded]),
        metadata=np.array(json.dumps(metadata)),
    )
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    hook.remove()
    print(f"results: {output}")
    return 0


def _load_case(case, records, normalization):
    motion_id = str(case.get("target_motion_id", case["motion_id"]))
    imu_motion_id = str(case.get("imu_motion_id", case["motion_id"]))
    for role, value in (("target", motion_id), ("IMU", imu_motion_id)):
        if value not in records:
            raise KeyError(f"{role} motion {value} is absent from the IMU manifest")
    config_name = case.get("sensor_config", "head")
    if config_name not in SENSOR_CONFIGS:
        raise ValueError(f"Unknown sensor config: {config_name}")
    with np.load(records[imu_motion_id]["imu_path"]) as data:
        acceleration = data["acceleration"].astype(np.float32)
        orientation = data["orientation"].astype(np.float32)
        fps = float(data["fps"])
    with np.load(records[motion_id]["imu_path"]) as data:
        gt = data["joints"].astype(np.float32)
        gt_fps = float(data["fps"])
    acceleration = _resample_linear(acceleration, fps, 20.0)
    orientation = _resample_orientation(orientation, fps, 20.0)
    gt = _resample_linear(gt, gt_fps, 20.0)
    target_length = min(int(case.get("length", len(gt))), len(gt), 196)
    acceleration = _fit_length_linear(acceleration, target_length)
    orientation = _fit_length_orientation(orientation, target_length)
    gt = gt[:target_length]
    mean = np.asarray(normalization["acceleration_mean"], dtype=np.float32)
    std = np.asarray(normalization["acceleration_std"], dtype=np.float32)
    normalized_acceleration = (acceleration - mean[None]) / std[None]
    imu = np.concatenate(
        (normalized_acceleration, orientation.reshape(len(orientation), 6, 9)), axis=-1
    ).astype(np.float32)
    imu_mode = str(case.get("imu_mode", "paired"))
    if imu_mode == "zero":
        imu = np.zeros_like(imu)
    elif imu_mode not in {"paired", "shuffled"}:
        raise ValueError(f"Unknown IMU mode: {imu_mode}")
    return {
        **case,
        "motion_id": motion_id,
        "target_motion_id": motion_id,
        "imu_motion_id": imu_motion_id,
        "imu_mode": imu_mode,
        "text": str(case.get("text", "")),
        "label": str(case.get("label", motion_id)),
        "sensor_config": config_name,
        "sensor_mask": sensor_mask(SENSOR_CONFIGS[config_name], sensor_count=6),
        "imu": imu,
        "raw_acceleration": acceleration,
        "raw_orientation": orientation,
        "gt": gt,
    }


def _fit_length_linear(values, target_length):
    if len(values) == target_length:
        return values
    source = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target = np.linspace(0.0, 1.0, target_length, dtype=np.float64)
    flat = values.reshape(len(values), -1)
    result = np.stack(
        [np.interp(target, source, flat[:, index]) for index in range(flat.shape[1])],
        axis=-1,
    )
    return result.reshape((target_length,) + values.shape[1:]).astype(np.float32)


def _fit_length_orientation(values, target_length):
    if len(values) == target_length:
        return values
    from scipy.spatial.transform import Rotation, Slerp

    source = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target = np.linspace(0.0, 1.0, target_length, dtype=np.float64)
    result = np.empty((target_length, values.shape[1], 3, 3), dtype=np.float32)
    for sensor in range(values.shape[1]):
        result[:, sensor] = Slerp(source, Rotation.from_matrix(values[:, sensor]))(
            target
        ).as_matrix()
    return result


def _resample_linear(values, source_fps, target_fps):
    if np.isclose(source_fps, target_fps):
        return values
    source_time = np.arange(len(values), dtype=np.float64) / source_fps
    count = max(1, int(round(len(values) * target_fps / source_fps)))
    target_time = np.minimum(np.arange(count, dtype=np.float64) / target_fps, source_time[-1])
    flat = values.reshape(len(values), -1)
    result = np.stack([np.interp(target_time, source_time, flat[:, i]) for i in range(flat.shape[1])], axis=-1)
    return result.reshape((count,) + values.shape[1:]).astype(np.float32)


def _resample_orientation(values, source_fps, target_fps):
    if np.isclose(source_fps, target_fps):
        return values
    from scipy.spatial.transform import Rotation, Slerp

    source_time = np.arange(len(values), dtype=np.float64) / source_fps
    count = max(1, int(round(len(values) * target_fps / source_fps)))
    target_time = np.minimum(np.arange(count, dtype=np.float64) / target_fps, source_time[-1])
    result = np.empty((count, values.shape[1], 3, 3), dtype=np.float32)
    for sensor in range(values.shape[1]):
        result[:, sensor] = Slerp(source_time, Rotation.from_matrix(values[:, sensor]))(target_time).as_matrix()
    return result


def _pad_frames(values, frame_count):
    values = np.asarray(values)[:frame_count]
    if len(values) == frame_count:
        return values
    padding = np.zeros((frame_count - len(values),) + values.shape[1:], dtype=values.dtype)
    return np.concatenate((values, padding), axis=0)


def _sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _check_device(torch, device, min_free_gib):
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.cuda.device(index):
        free, total = torch.cuda.mem_get_info()
    free_gib = free / 1024**3
    print(f"device=cuda:{index} free={free_gib:.2f}GiB total={total / 1024**3:.2f}GiB")
    if free_gib < min_free_gib:
        raise RuntimeError(f"cuda:{index} has only {free_gib:.2f} GiB free")


def _legacy_compatibility():
    aliases = {"bool": bool, "int": int, "float": float, "complex": complex, "object": object, "unicode": str, "str": str}
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec


def _set_mdm_compatibility_defaults(options):
    options.setdefault("unconstrained", False)
    options.setdefault("text_encoder_type", "clip")
    options.setdefault("pos_embed_max_len", 5000)
    options.setdefault("mask_frames", False)


if __name__ == "__main__":
    raise SystemExit(main())
