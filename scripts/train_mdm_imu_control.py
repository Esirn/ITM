#!/usr/bin/env python
"""Stage-1 training for frozen MDM with zero-initialized IMU control adapters."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import sys

import numpy as np

from itm.data.dataset import TextIMUMotionDataset
from itm.data.manifest import read_jsonl
from itm.data.standard_imu import sensor_mask
from itm.models.mdm_imu_control import MDMIMUControlConfig, install_imu_control
from itm.models.torch_frame_baseline import require_torch


SENSOR_CONFIGS = {"head": (4,), "wrists": (0, 1)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mdm-root", default="/home/a200/mount/a40/relatedworks/mdm/motion-diffusion-model")
    parser.add_argument("--mdm-checkpoint", required=True)
    parser.add_argument("--mdm-args", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--standard-imu-manifest", required=True)
    parser.add_argument("--mean", default="/home/a200/0proj/datasets/all/Mean.npy")
    parser.add_argument("--std", default="/home/a200/0proj/datasets/all/Std.npy")
    parser.add_argument("--sensor-configs", default="head,wrists")
    parser.add_argument("--output", default="outputs/mdm_control/stage1.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch = require_torch()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    _legacy_compatibility()
    workspace_root = Path.cwd()
    manifest_path = Path(args.manifest).resolve()
    imu_manifest_path = Path(args.standard_imu_manifest).resolve()
    mean_path = Path(args.mean).resolve()
    std_path = Path(args.std).resolve()
    output_path = Path(args.output).resolve()
    mdm_checkpoint_path = Path(args.mdm_checkpoint).resolve()
    mdm_args_path = Path(args.mdm_args).resolve()
    root = Path(args.mdm_root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    from utils.model_util import create_model_and_diffusion, load_model_wo_clip

    mdm_options = json.loads(mdm_args_path.read_text())
    mdm_options.setdefault("unconstrained", False)
    mdm_args = SimpleNamespace(**mdm_options)
    data_stub = SimpleNamespace(dataset=SimpleNamespace(num_actions=1))
    model, diffusion = create_model_and_diffusion(mdm_args, data_stub)
    state = torch.load(mdm_checkpoint_path, map_location="cpu", weights_only=True)
    load_model_wo_clip(model, state)
    control_config = MDMIMUControlConfig(latent_dim=mdm_args.latent_dim)
    controlled, imu_encoder, hook = install_imu_control(model, control_config)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in imu_encoder.parameters():
        parameter.requires_grad = True
    for parameter in controlled.adapters.parameters():
        parameter.requires_grad = True
    device = torch.device(args.device)
    _check_device(torch, device, args.min_free_gib)
    model.to(device)
    model.train()
    model.clip_model.eval()

    configs = _parse_configs(args.sensor_configs)
    dataset = _PairedDataset(
        manifest_path,
        imu_manifest_path,
        np.load(mean_path).astype(np.float32),
        np.load(std_path).astype(np.float32),
        configs,
        args.max_records,
        workspace_root,
    )
    normalization = dataset.compute_normalization()
    dataset.set_normalization(normalization)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    history = []
    start_epoch = 1
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=True)
        imu_encoder.load_state_dict(resume["imu_encoder"])
        controlled.adapters.load_state_dict(resume["adapters"])
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
        history = list(resume.get("history", []))
        start_epoch = int(resume.get("epoch", len(history))) + 1
        normalization = resume["imu_normalization"]
        dataset.set_normalization(normalization)
    for epoch in range(start_epoch, args.epochs + 1):
        total_loss = 0.0
        total_items = 0
        for motion, y in loader:
            motion = motion.to(device)
            y = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in y.items()}
            timesteps = torch.randint(0, diffusion.num_timesteps, (len(motion),), device=device)
            losses = diffusion.training_losses(
                model,
                motion,
                timesteps,
                model_kwargs={"y": y},
            )
            loss = losses["loss"].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(motion)
            total_items += len(motion)
        epoch_loss = total_loss / total_items
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"epoch={epoch} loss={epoch_loss:.6f}")
        _save_checkpoint(
            output_path.with_name(f"{output_path.stem}_epoch{epoch:03d}{output_path.suffix}"),
            torch,
            imu_encoder,
            controlled,
            optimizer,
            control_config,
            configs,
            mdm_checkpoint_path,
            normalization,
            history,
            epoch,
            args,
        )
    hook.remove()
    output = output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(
        output, torch, imu_encoder, controlled, optimizer, control_config, configs,
        mdm_checkpoint_path, normalization, history, args.epochs, args,
    )
    print(f"control_checkpoint: {output}")
    return 0


class _PairedDataset:
    def __init__(self, manifest, imu_manifest, mean, std, configs, limit, workspace_root):
        motion_dataset = TextIMUMotionDataset(
            manifest, load_joints=False, load_joint_vec=True, load_imu=False
        )
        samples = {str(motion_dataset[index]["motion_id"]): motion_dataset[index] for index in range(len(motion_dataset))}
        imu_records = {str(record["motion_id"]): record for record in read_jsonl(imu_manifest)}
        common = sorted(set(samples) & set(imu_records))
        if limit is not None:
            common = common[:limit]
        self.items = [(samples[motion_id], imu_records[motion_id], name, slots) for motion_id in common for name, slots in configs.items()]
        self.mean = mean
        self.std = std
        self.workspace_root = workspace_root
        self.normalization = None
        if not self.items:
            raise ValueError("No overlapping HumanML and standard IMU records")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        sample, imu_record, config_name, slots = self.items[index]
        motion = (np.asarray(sample["motion"], dtype=np.float32) - self.mean) / self.std
        imu_path = Path(imu_record["imu_path"])
        if not imu_path.is_absolute():
            imu_path = self.workspace_root / imu_path
        with np.load(imu_path) as data:
            fps = float(data["fps"])
            acceleration = data["acceleration"].astype(np.float32)
            orientation = data["orientation"].astype(np.float32)
        acceleration, orientation = _resample_imu(acceleration, orientation, fps, 20.0)
        if self.normalization is not None:
            mean = np.asarray(self.normalization["acceleration_mean"], dtype=np.float32)
            std = np.asarray(self.normalization["acceleration_std"], dtype=np.float32)
            acceleration = (acceleration - mean[None]) / std[None]
        imu = np.concatenate(
            (acceleration, orientation.reshape(len(orientation), 6, 9)), axis=-1
        ).astype(np.float32)
        length = min(len(motion), len(imu), 196)
        return {
            "motion": motion[:length],
            "imu": imu[:length],
            "sensor_mask": sensor_mask(slots, sensor_count=6),
            "caption": str(sample["caption"]),
            "sensor_config": config_name,
        }

    def compute_normalization(self):
        total = np.zeros((6, 3), dtype=np.float64)
        squared = np.zeros((6, 3), dtype=np.float64)
        count = 0
        seen = set()
        for _, record, _, _ in self.items:
            path = Path(record["imu_path"])
            if not path.is_absolute():
                path = self.workspace_root / path
            if path in seen:
                continue
            seen.add(path)
            with np.load(path) as data:
                acceleration = data["acceleration"].astype(np.float64)
            total += acceleration.sum(0)
            squared += np.square(acceleration).sum(0)
            count += len(acceleration)
        mean = total / count
        variance = np.maximum(squared / count - np.square(mean), 1e-8)
        return {
            "acceleration_mean": mean.astype(np.float32).tolist(),
            "acceleration_std": np.sqrt(variance).astype(np.float32).tolist(),
            "orientation": "rotation_matrix_unscaled",
        }

    def set_normalization(self, normalization):
        self.normalization = normalization


def _collate(items):
    torch = require_torch()
    lengths = torch.tensor([len(item["motion"]) for item in items], dtype=torch.long)
    frames = int(lengths.max())
    motion = torch.zeros(len(items), 263, 1, frames)
    imu = torch.zeros(len(items), frames, 6, 12)
    frame_mask = torch.arange(frames)[None] < lengths[:, None]
    for index, item in enumerate(items):
        length = len(item["motion"])
        motion[index, :, 0, :length] = torch.from_numpy(item["motion"].T)
        imu[index, :length] = torch.from_numpy(item["imu"])
    y = {
        "mask": frame_mask[:, None, None],
        "lengths": lengths,
        "text": [item["caption"] for item in items],
        "imu": imu,
        "sensor_mask": torch.from_numpy(np.stack([item["sensor_mask"] for item in items])),
        "imu_frame_mask": frame_mask,
    }
    return motion, y


def _parse_configs(value):
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in SENSOR_CONFIGS]
    if unknown or not names:
        raise ValueError(f"Invalid sensor configs {unknown}; choose from {sorted(SENSOR_CONFIGS)}")
    return {name: SENSOR_CONFIGS[name] for name in names}


def _resample_imu(acceleration, orientation, source_fps, target_fps):
    if np.isclose(source_fps, target_fps):
        return acceleration, orientation
    from scipy.spatial.transform import Rotation, Slerp

    source_time = np.arange(len(acceleration), dtype=np.float64) / float(source_fps)
    target_count = max(1, int(round(len(acceleration) * target_fps / source_fps)))
    target_time = np.arange(target_count, dtype=np.float64) / float(target_fps)
    target_time = np.minimum(target_time, source_time[-1])
    resampled_acceleration = np.stack(
        [
            np.interp(target_time, source_time, acceleration[:, sensor, axis])
            for sensor in range(acceleration.shape[1])
            for axis in range(3)
        ],
        axis=-1,
    ).reshape(target_count, acceleration.shape[1], 3)
    resampled_orientation = np.empty(
        (target_count, orientation.shape[1], 3, 3), dtype=np.float32
    )
    for sensor in range(orientation.shape[1]):
        rotations = Rotation.from_matrix(orientation[:, sensor])
        resampled_orientation[:, sensor] = Slerp(source_time, rotations)(target_time).as_matrix()
    return resampled_acceleration.astype(np.float32), resampled_orientation


def _save_checkpoint(
    path, torch, imu_encoder, controlled, optimizer, control_config, configs,
    mdm_checkpoint_path, normalization, history, epoch, args,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "imu_encoder": imu_encoder.state_dict(),
            "adapters": controlled.adapters.state_dict(),
            "optimizer": optimizer.state_dict(),
            "control_config": control_config.__dict__,
            "sensor_configs": {name: list(slots) for name, slots in configs.items()},
            "mdm_checkpoint": str(mdm_checkpoint_path),
            "imu_normalization": normalization,
            "history": history,
            "epoch": epoch,
            "training_args": vars(args),
        },
        path,
    )


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
        raise RuntimeError(
            f"cuda:{index} has only {free_gib:.2f} GiB free; "
            f"required {min_free_gib:.2f} GiB"
        )


def _legacy_compatibility():
    aliases = {"bool": bool, "int": int, "float": float, "complex": complex, "object": object, "unicode": str, "str": str}
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec


if __name__ == "__main__":
    raise SystemExit(main())
