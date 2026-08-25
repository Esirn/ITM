#!/usr/bin/env python3
"""Minimal, renderer-free sampler for MotionLab's official MotionFlow model."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motionlab-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--trajectory-hints",
        type=Path,
        help="NPZ with joints [B,T,22,3] and optional mask; enables oracle Text+Hint",
    )
    parser.add_argument(
        "--control-tokens",
        type=Path,
        help="NPZ with learned tokens [B,T,66]; enables direct Text+IMU control",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--guidance-mode", choices=("legacy", "factorized"), default="legacy")
    parser.add_argument("--text-scale", type=float, default=1.75)
    parser.add_argument("--imu-scale", type=float, default=1.0)
    parser.add_argument("--joint-scale", type=float, default=1.0)
    return parser.parse_args()


def load_config(root: Path):
    from omegaconf import OmegaConf
    from rfmotion.config import get_module_config

    base = OmegaConf.load(root / "configs/base.yaml")
    experiment = OmegaConf.load(root / "configs/config_rfmotion.yaml")
    cfg = OmegaConf.merge(base, experiment)
    cfg.model = get_module_config(cfg.model, cfg.model.target)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(root / "configs/assets.yaml"))
    cfg.DATASET.NFEATS = 263
    cfg.DATASET.NJOINTS = 22
    cfg.METRIC.TYPE = []
    cfg.DEMO.RENDER = False
    return cfg


class MinimalHumanMLDataModule:
    """Only the normalization and 263D decoding needed during generation."""

    def __init__(self, root: Path):
        import torch
        from rfmotion.data.humanml.scripts.motion_process import recover_from_ric

        dataset = root / "datasets/all"
        self.mean = torch.from_numpy(np.load(dataset / "Mean.npy")).float()
        self.std = torch.from_numpy(np.load(dataset / "Std.npy")).float()
        self.mean_motion = torch.from_numpy(np.load(dataset / "mean_motion.npy")).float()
        self.std_motion = torch.from_numpy(np.load(dataset / "std_motion.npy")).float()
        self.nfeats = 263
        self.njoints = 22
        self._recover_from_ric = recover_from_ric

    def feats2joints(self, features):
        mean = self.mean.to(features)
        std = self.std.to(features)
        return self._recover_from_ric(features * std + mean, self.njoints)


def disable_unrelated_initializers() -> None:
    """Keep the unified checkpoint architecture while omitting unrelated tasks."""
    from rfmotion.models.modeltype.rfmotion import RFMOTION

    RFMOTION.get_style_encoder = lambda self, cfg: None
    RFMOTION.get_content_encoder = lambda self, cfg: None
    RFMOTION.get_style_test_dataset = lambda self, cfg: None
    RFMOTION.get_t2m_evaluator = lambda self, cfg: None
    RFMOTION.configure_metrics = lambda self: None


def sample_batch(model, datamodule, texts: list[str], lengths: list[int], device: str):
    """Equivalent to demo_text(), but preserves every sample in the batch."""
    import torch

    batch_size = len(texts)
    frames = max(lengths)
    target_motion = torch.zeros(batch_size, frames, 263, device=device)

    instruction_uncond = model.instructions["uncond"].repeat(batch_size, 1)
    instruction_text = model.instructions["text"].repeat(batch_size, 1)
    instructions = torch.cat([instruction_uncond, instruction_text], dim=0)
    encoded_text = model.text_encoder([""] * batch_size + texts)
    text_lengths = [0] * batch_size + [77] * batch_size

    with torch.inference_mode():
        generated = model.diffusion_reverse(
            stage="demo",
            condition_type="text",
            instructions=instructions,
            text=encoded_text,
            text_lengths=text_lengths,
            target_motion=target_motion,
            target_lengths=lengths,
            target_lengths_z=lengths,
        )
        joints = datamodule.feats2joints(generated)
    return generated.cpu().numpy(), joints.cpu().numpy()


def sample_text_hint_batch(
    model,
    datamodule,
    texts: list[str],
    lengths: list[int],
    device: str,
    hint_joints: np.ndarray,
    hint_mask: np.ndarray,
):
    """Run MotionLab's native Text+Hint path with caller-supplied trajectories."""
    import torch

    batch_size = len(texts)
    frames = max(lengths)
    target_motion = torch.zeros(batch_size, frames, 263, device=device)
    joints = torch.as_tensor(hint_joints[:, :frames], dtype=torch.float32, device=device)
    mask = torch.as_tensor(hint_mask[:, :frames], dtype=torch.bool, device=device)
    if joints.shape != (batch_size, frames, 22, 3) or mask.shape != joints.shape:
        raise ValueError(
            "trajectory hint joints/mask must have shape [batch,max(lengths),22,3]"
        )
    frame_indices = torch.arange(frames, device=device)[None, :, None, None]
    valid_frames = frame_indices < torch.as_tensor(lengths, device=device)[:, None, None, None]
    mask = mask & valid_frames

    mean = datamodule.mean_motion.to(device).view(1, 1, 22, 3)
    std = datamodule.std_motion.to(device).view(1, 1, 22, 3)
    encoded_hint = (((joints - mean) / std) * mask).reshape(batch_size, frames, 66)
    hint = torch.cat([torch.zeros_like(encoded_hint), encoded_hint], dim=0)
    hint_lengths_cond = mask.any(dim=-1).any(dim=-1)
    hint_lengths = torch.cat([torch.zeros_like(hint_lengths_cond), hint_lengths_cond], dim=0)

    instructions = torch.cat(
        [
            model.instructions["uncond"].repeat(batch_size, 1),
            model.instructions["text_hint"].repeat(batch_size, 1),
        ],
        dim=0,
    )
    encoded_text = model.text_encoder([""] * batch_size + texts)
    text_lengths = [0] * batch_size + [77] * batch_size
    with torch.inference_mode():
        generated = model.diffusion_reverse(
            stage="demo",
            condition_type="text_hint",
            instructions=instructions,
            text=encoded_text,
            text_lengths=text_lengths,
            hint=hint,
            hint_lengths=hint_lengths,
            hint_masks=mask,
            target_motion=target_motion,
            target_lengths=lengths,
            target_lengths_z=lengths,
        )
        generated_joints = datamodule.feats2joints(generated)
    return generated.cpu().numpy(), generated_joints.cpu().numpy()


def sample_text_control_tokens(
    model, datamodule, texts, lengths, device, control_tokens, control_mask,
    guidance_mode="legacy", text_scale=1.75, imu_scale=1.0, joint_scale=1.0,
):
    """Run native hint attention with learned IMU control tokens."""
    import torch

    batch_size = len(texts)
    frames = max(lengths)
    target_motion = torch.zeros(batch_size, frames, 263, device=device)
    tokens = torch.as_tensor(control_tokens[:, :frames], dtype=torch.float32, device=device)
    mask = torch.as_tensor(control_mask[:, :frames], dtype=torch.bool, device=device)
    if tokens.shape[:2] != (batch_size, frames) or tokens.shape[-1] not in (
        66, model.denoiser.token_dim
    ):
        raise ValueError(
            f"control tokens must have shape [B,T,66|{model.denoiser.token_dim}]"
        )
    if mask.shape != (batch_size, frames):
        raise ValueError("control mask must have shape [B,T]")
    valid = torch.arange(frames, device=device)[None] < torch.as_tensor(
        lengths, device=device
    )[:, None]
    mask = mask & valid
    if tokens.shape[-1] == model.denoiser.token_dim:
        model.denoiser.hint_embed1 = torch.nn.Identity()
    if guidance_mode == "factorized":
        return sample_factorized_control(
            model, datamodule, texts, lengths, target_motion, tokens, mask,
            text_scale, imu_scale, joint_scale,
        )
    hint = torch.cat([torch.zeros_like(tokens), tokens], dim=0)
    hint_lengths = torch.cat([torch.zeros_like(mask), mask], dim=0)
    instructions = torch.cat(
        [
            model.instructions["uncond"].repeat(batch_size, 1),
            model.instructions["text_hint"].repeat(batch_size, 1),
        ],
        dim=0,
    )
    encoded_text = model.text_encoder([""] * batch_size + texts)
    with torch.inference_mode():
        generated = model.diffusion_reverse(
            stage="demo",
            condition_type="text_hint",
            instructions=instructions,
            text=encoded_text,
            text_lengths=[0] * batch_size + [77] * batch_size,
            hint=hint,
            hint_lengths=hint_lengths,
            target_motion=target_motion,
            target_lengths=lengths,
            target_lengths_z=lengths,
        )
        generated_joints = datamodule.feats2joints(generated)
    return generated.cpu().numpy(), generated_joints.cpu().numpy()


def sample_factorized_control(
    model, datamodule, texts, lengths, target_motion, tokens, mask,
    text_scale, imu_scale, joint_scale,
):
    """Sample explicit unconditional, text, IMU, and joint branches."""
    import torch

    batch_size = len(texts)
    instructions = torch.cat(
        [model.instructions[name].repeat(batch_size, 1)
         for name in ("uncond", "text", "hint", "text_hint")],
        dim=0,
    )
    encoded_text = model.text_encoder(
        [""] * batch_size + texts + [""] * batch_size + texts
    )
    text_lengths = (
        [0] * batch_size + [77] * batch_size
        + [0] * batch_size + [77] * batch_size
    )
    zeros = torch.zeros_like(tokens)
    zero_mask = torch.zeros_like(mask)
    hints = torch.cat([zeros, zeros, tokens, tokens], dim=0)
    hint_lengths = torch.cat([zero_mask, zero_mask, mask, mask], dim=0)
    noisy = torch.randn_like(target_motion)
    model.scheduler.set_timesteps(
        num_inference_steps=model.cfg.model.scheduler.num_demo_steps,
        device=target_motion.device,
    )
    with torch.inference_mode():
        for timestep in model.scheduler.timesteps.to(torch.int32):
            if timestep == 0:
                continue
            velocity = model.denoiser(
                instructions=instructions,
                hidden_states=torch.cat([noisy] * 4),
                timestep=timestep,
                text=encoded_text,
                text_lengths=text_lengths,
                hint=hints,
                hint_lengths=hint_lengths,
                target_lengths=lengths * 4,
                target_lengths_z=lengths * 4,
                return_dict=False,
            )[0]
            f00, f10, f01, f11 = velocity.chunk(4)
            velocity = (
                f00
                + text_scale * (f10 - f00)
                + imu_scale * (f01 - f00)
                + joint_scale * (f11 - f10 - f01 + f00)
            )
            noisy = model.scheduler.step(
                velocity, timestep, noisy, return_dict=False
            )[0]
        joints = datamodule.feats2joints(noisy)
    return noisy.cpu().numpy(), joints.cpu().numpy()


def main() -> None:
    args = parse_args()
    root = args.motionlab_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    request_path = args.request.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    trajectory_path = (
        args.trajectory_hints.expanduser().resolve() if args.trajectory_hints else None
    )
    control_path = args.control_tokens.expanduser().resolve() if args.control_tokens else None
    if trajectory_path is not None and control_path is not None:
        raise ValueError("trajectory hints and learned control tokens are mutually exclusive")
    if not root.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("MotionLab root or checkpoint is unavailable")

    os.chdir(root)
    sys.path.insert(0, str(root))
    import torch
    from rfmotion.models.get_model import get_model

    request = json.loads(request_path.read_text(encoding="utf-8"))
    texts = [str(text) for text in request["text"]]
    lengths = [int(length) for length in request["lengths"]]
    if any(length < 1 or length > 196 for length in lengths):
        raise ValueError("MotionLab lengths must be in [1, 196]")
    if not args.device.startswith("cuda") and args.device != "cpu":
        raise ValueError("device must be cpu or cuda:N")

    seed = int(request["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = load_config(root)
    if args.num_steps is not None:
        cfg.model.scheduler.num_demo_steps = args.num_steps
    disable_unrelated_initializers()
    datamodule = MinimalHumanMLDataModule(root)

    started = time.perf_counter()
    model = get_model(cfg, datamodule)
    state_dict = torch.load(checkpoint, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)
    model.to(args.device).eval()
    loaded = time.perf_counter()
    mode = "text"
    active_joints = None
    if trajectory_path is None and control_path is None:
        features, joints = sample_batch(model, datamodule, texts, lengths, args.device)
    elif trajectory_path is not None:
        trajectory = np.load(trajectory_path)
        hint_joints = np.asarray(trajectory["joints"], dtype=np.float32)
        if "mask" in trajectory:
            hint_mask = np.asarray(trajectory["mask"], dtype=bool)
        else:
            active_joints = [int(value) for value in request.get("active_joints", [])]
            if not active_joints:
                raise ValueError("request.active_joints is required when hint NPZ has no mask")
            hint_mask = np.zeros_like(hint_joints, dtype=bool)
            hint_mask[:, :, active_joints, :] = True
        features, joints = sample_text_hint_batch(
            model,
            datamodule,
            texts,
            lengths,
            args.device,
            hint_joints,
            hint_mask,
        )
        hint_source = str(request.get("hint_source", "oracle_trajectory"))
        mode = (
            "text_imu_adapter"
            if hint_source == "imu_adapter"
            else "text_trajectory_hint_oracle"
        )
    else:
        control = np.load(control_path)
        features, joints = sample_text_control_tokens(
            model,
            datamodule,
            texts,
            lengths,
            args.device,
            np.asarray(control["tokens"], dtype=np.float32),
            np.asarray(control["mask"], dtype=bool),
            guidance_mode=args.guidance_mode,
            text_scale=args.text_scale,
            imu_scale=args.imu_scale,
            joint_scale=args.joint_scale,
        )
        mode = "text_imu_direct_control"
    finished = time.perf_counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features,
        joints=joints,
        lengths=np.asarray(lengths, dtype=np.int64),
    )
    metadata = {
        "backbone": "MotionLab MotionFlow",
        "mode": mode,
        "motionlab_root": str(root),
        "motionlab_commit": "98cc88c8e31e43be9f94c2d1edf4c21e4715c683",
        "checkpoint": str(checkpoint),
        "checkpoint_size": checkpoint.stat().st_size,
        "device": args.device,
        "seed": seed,
        "texts": texts,
        "lengths": lengths,
        "num_steps": int(cfg.model.scheduler.num_demo_steps),
        "guidance_mode": args.guidance_mode,
        "text_scale": args.text_scale,
        "imu_scale": args.imu_scale,
        "joint_scale": args.joint_scale,
        "text_guidance_scale": float(model.text_guidance_scale),
        "text_hint_guidance_scale": float(model.text_hint_guidance_scale),
        "trajectory_hints": str(trajectory_path) if trajectory_path else None,
        "control_tokens": str(control_path) if control_path else None,
        "hint_source": request.get("hint_source") if trajectory_path else None,
        "active_joints": active_joints,
        "load_seconds": loaded - started,
        "sample_seconds": finished - loaded,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
