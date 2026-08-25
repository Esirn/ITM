#!/usr/bin/env python3
"""Train direct IMU control tokens against a frozen MotionLab MotionFlow model."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

from sample_motionlab_text import (
    MinimalHumanMLDataModule,
    disable_unrelated_initializers,
    load_config,
)


SENSOR_CONFIGS = {"head": (4,), "wrists": (0, 1)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motionlab-root", type=Path, required=True)
    parser.add_argument("--motionlab-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--imu-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--val-imu-manifest", type=Path)
    parser.add_argument("--sensor-configs", default="head,wrists")
    parser.add_argument("--sensor-fusion", choices=("mean", "concat"), default="mean")
    parser.add_argument("--control-dim", type=int, choices=(66, 256, 512), default=66)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--text-drop-probability", type=float, default=0.1)
    parser.add_argument("--imu-drop-probability", type=float, default=0.1)
    parser.add_argument("--active-joint-loss-weight", type=float, default=0.0)
    parser.add_argument("--active-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--ranking-margin", type=float, default=0.01)
    parser.add_argument("--zero-init-output", action="store_true")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def make_adapter(torch, hidden_dim=256, layers=2, heads=8, max_frames=196, sensor_fusion="mean", output_dim=66):
    class DirectIMUControl(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_projection = torch.nn.Linear(12, hidden_dim)
            self.sensor_embedding = torch.nn.Embedding(6, hidden_dim)
            self.sensor_fusion = (
                torch.nn.Linear(6 * hidden_dim, hidden_dim)
                if sensor_fusion == "concat"
                else None
            )
            layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal_encoder = torch.nn.TransformerEncoder(layer, layers)
            self.output_projection = torch.nn.Linear(hidden_dim, output_dim)
            position = torch.arange(max_frames, dtype=torch.float32)[:, None]
            divisor = torch.exp(
                torch.arange(0, hidden_dim, 2, dtype=torch.float32)
                * (-math.log(10000.0) / hidden_dim)
            )
            encoding = torch.zeros(max_frames, hidden_dim)
            encoding[:, 0::2] = torch.sin(position * divisor)
            encoding[:, 1::2] = torch.cos(position * divisor)
            self.register_buffer("position_encoding", encoding, persistent=False)

        def forward(self, imu, sensor_mask, frame_mask):
            frames = imu.shape[1]
            ids = torch.arange(imu.shape[2], device=imu.device)
            hidden = self.feature_projection(imu) + self.sensor_embedding(ids)[None, None]
            active = sensor_mask[:, None, :, None].to(hidden.dtype)
            hidden = hidden * active
            if self.sensor_fusion is None:
                hidden = hidden.sum(2) / active.sum(2).clamp_min(1.0)
            else:
                hidden = self.sensor_fusion(hidden.flatten(2))
            hidden = hidden + self.position_encoding[:frames].to(hidden)[None]
            hidden = self.temporal_encoder(hidden, src_key_padding_mask=~frame_mask)
            return self.output_projection(hidden) * frame_mask[:, :, None].to(hidden.dtype)

    return DirectIMUControl()


class PairedDataset:
    def __init__(self, manifest, imu_manifest, configs, normalization=None, limit=None):
        motion_records = {}
        for line in manifest.read_text().splitlines():
            record = json.loads(line)
            motion = np.load(record["joint_vec_path"])
            if motion.ndim == 2 and motion.shape[1] == 263 and len(motion) >= 2:
                motion_records[str(record["motion_id"])] = record
        imu_records = {
            str(record["motion_id"]): record
            for record in (json.loads(line) for line in imu_manifest.read_text().splitlines())
        }
        ids = sorted(set(motion_records) & set(imu_records))
        if limit is not None:
            ids = ids[:limit]
        self.items = [
            (motion_records[motion_id], imu_records[motion_id], name, slots)
            for motion_id in ids
            for name, slots in configs.items()
        ]
        self.normalization = normalization
        if not self.items:
            raise ValueError("no aligned MotionLab/IMU records")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        motion_record, imu_record, config_name, slots = self.items[index]
        motion = np.load(motion_record["joint_vec_path"]).astype(np.float32)
        captions = [
            line.split("#", 1)[0].strip()
            for line in Path(motion_record["text_path"]).read_text(errors="ignore").splitlines()
            if line.strip()
        ]
        caption = captions[index % len(captions)] if captions else ""
        with np.load(imu_record["imu_path"]) as data:
            acceleration, orientation = resample_imu(
                data["acceleration"], data["orientation"], float(data["fps"]), 20.0
            )
        mean = np.asarray(self.normalization["acceleration_mean"], dtype=np.float32)
        std = np.asarray(self.normalization["acceleration_std"], dtype=np.float32)
        acceleration = (acceleration - mean[None]) / std[None]
        imu = np.concatenate(
            [acceleration, orientation.reshape(len(orientation), 6, 9)], axis=-1
        ).astype(np.float32)
        length = min(len(motion), len(imu), 196)
        sensors = np.zeros(6, dtype=bool)
        sensors[list(slots)] = True
        return motion[:length], imu[:length], sensors, caption, config_name

    def compute_normalization(self):
        total = np.zeros((6, 3), dtype=np.float64)
        squared = np.zeros((6, 3), dtype=np.float64)
        count = 0
        seen = set()
        for _, record, _, _ in self.items:
            path = Path(record["imu_path"])
            if path in seen:
                continue
            seen.add(path)
            with np.load(path) as data:
                values = np.asarray(data["acceleration"], dtype=np.float64)
            total += values.sum(0)
            squared += np.square(values).sum(0)
            count += len(values)
        mean = total / count
        variance = np.maximum(squared / count - mean**2, 1e-8)
        return {
            "acceleration_mean": mean.astype(np.float32).tolist(),
            "acceleration_std": np.sqrt(variance).astype(np.float32).tolist(),
            "orientation": "rotation_matrix_unscaled",
        }


def collate(torch, items):
    lengths = [len(item[0]) for item in items]
    frames = max(lengths)
    motion = torch.zeros(len(items), frames, 263)
    imu = torch.zeros(len(items), frames, 6, 12)
    mask = torch.arange(frames)[None] < torch.tensor(lengths)[:, None]
    for index, item in enumerate(items):
        motion[index, : lengths[index]] = torch.from_numpy(item[0])
        imu[index, : lengths[index]] = torch.from_numpy(item[1])
    return {
        "motion": motion,
        "imu": imu,
        "sensor_mask": torch.from_numpy(np.stack([item[2] for item in items])),
        "frame_mask": mask,
        "lengths": lengths,
        "text": [item[3] for item in items],
    }


def resample_imu(acceleration, orientation, source_fps, target_fps):
    if np.isclose(source_fps, target_fps):
        return acceleration.astype(np.float32), orientation.astype(np.float32)
    from scipy.spatial.transform import Rotation, Slerp

    source_time = np.arange(len(acceleration), dtype=np.float64) / source_fps
    count = max(1, int(round(len(acceleration) * target_fps / source_fps)))
    target_time = np.minimum(np.arange(count, dtype=np.float64) / target_fps, source_time[-1])
    acc = np.stack(
        [
            np.interp(target_time, source_time, acceleration[:, sensor, axis])
            for sensor in range(6)
            for axis in range(3)
        ],
        axis=-1,
    ).reshape(count, 6, 3)
    ori = np.empty((count, 6, 3, 3), dtype=np.float32)
    for sensor in range(6):
        ori[:, sensor] = Slerp(
            source_time, Rotation.from_matrix(orientation[:, sensor])
        )(target_time).as_matrix()
    return acc.astype(np.float32), ori


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    root = args.motionlab_root.resolve()
    motionlab_checkpoint = args.motionlab_checkpoint.resolve()
    manifest = args.manifest.resolve()
    imu_manifest = args.imu_manifest.resolve()
    val_manifest = args.val_manifest.resolve() if args.val_manifest else None
    val_imu_manifest = args.val_imu_manifest.resolve() if args.val_imu_manifest else None
    if (val_manifest is None) != (val_imu_manifest is None):
        raise ValueError("val manifest and val IMU manifest must be provided together")
    output = args.output.resolve()
    resume_path = args.resume.resolve() if args.resume else None
    os.chdir(root)
    sys.path.insert(0, str(root))
    import torch
    from rfmotion.models.get_model import get_model

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_config(root)
    disable_unrelated_initializers()
    datamodule = MinimalHumanMLDataModule(root)
    model = get_model(cfg, datamodule)
    state = torch.load(motionlab_checkpoint, map_location="cpu")["state_dict"]
    model.load_state_dict(state)
    for parameter in model.parameters():
        parameter.requires_grad = False
    adapter = make_adapter(
        torch, sensor_fusion=args.sensor_fusion, output_dim=args.control_dim
    )
    if args.zero_init_output and args.resume is None:
        torch.nn.init.zeros_(adapter.output_projection.weight)
        torch.nn.init.zeros_(adapter.output_projection.bias)
    device = torch.device(args.device)
    model.to(device).eval()
    if args.control_dim not in (66, model.denoiser.token_dim):
        raise ValueError(
            f"control dim must be 66 or checkpoint token_dim={model.denoiser.token_dim}; "
            f"got {args.control_dim}"
        )
    if args.control_dim == model.denoiser.token_dim:
        model.denoiser.hint_embed1 = torch.nn.Identity()
    adapter.to(device).train()

    configs = {
        name: SENSOR_CONFIGS[name]
        for name in args.sensor_configs.split(",")
        if name
    }
    dataset = PairedDataset(manifest, imu_manifest, configs, limit=args.max_records)
    normalization = dataset.compute_normalization()
    dataset.normalization = normalization
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda items: collate(torch, items),
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = None
    if val_manifest is not None:
        val_dataset = PairedDataset(
            val_manifest,
            val_imu_manifest,
            configs,
            normalization=normalization,
            limit=args.max_records,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda items: collate(torch, items),
        )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate)
    history = []
    start_epoch = 1
    best_val = float("inf")
    if resume_path:
        resume = torch.load(resume_path, map_location="cpu")
        adapter.load_state_dict(resume["adapter"])
        normalization = resume["imu_normalization"]
        dataset.normalization = normalization
        history = list(resume.get("history", []))
        start_epoch = int(resume.get("epoch", 0)) + 1
        previous = [item.get("val_flow_matching_loss") for item in history]
        previous = [value for value in previous if value is not None]
        best_val = min(previous, default=float("inf"))
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs + 1):
        total = 0.0
        active_total = 0.0
        velocity_total = 0.0
        ranking_total = 0.0
        items = 0
        for batch in loader:
            motion = batch["motion"].to(device)
            motion = (
                motion - datamodule.mean.to(device)[None, None]
            ) / datamodule.std.to(device)[None, None]
            imu = batch["imu"].to(device)
            sensors = batch["sensor_mask"].to(device)
            frame_mask = batch["frame_mask"].to(device)
            text = list(batch["text"])
            text_drop = torch.rand(len(text), device=device) < args.text_drop_probability
            text = ["" if bool(text_drop[i]) else value for i, value in enumerate(text)]
            control = adapter(imu, sensors, frame_mask)
            imu_keep = (
                torch.rand(len(text), 1, 1, device=device) >= args.imu_drop_probability
            ).to(control.dtype)
            control = control * imu_keep
            encoded_text = model.text_encoder(text)
            instructions = model.instructions["text_hint"].repeat(len(text), 1)
            losses = direct_control_losses(
                torch,
                model,
                datamodule,
                instructions,
                motion,
                batch["lengths"],
                encoded_text,
                [0 if value == "" else 77 for value in text],
                control,
                frame_mask,
                sensors,
            )
            ranking_loss = motion.new_zeros(())
            if args.ranking_loss_weight > 0 and len(text) > 1:
                permutation = torch.roll(torch.arange(len(text), device=device), 1)
                negative_losses = direct_control_losses(
                    torch,
                    model,
                    datamodule,
                    instructions,
                    motion,
                    batch["lengths"],
                    encoded_text,
                    [0 if value == "" else 77 for value in text],
                    control[permutation],
                    frame_mask,
                    sensors,
                    timesteps=losses["timesteps"],
                    noise=losses["noise"],
                )
                ranking_loss = torch.relu(
                    losses["active_joint_loss"]
                    - negative_losses["active_joint_loss"]
                    + args.ranking_margin
                )
            loss = (
                losses["flow_matching_loss"]
                + args.active_joint_loss_weight * losses["active_joint_loss"]
                + args.active_velocity_loss_weight * losses["active_velocity_loss"]
                + args.ranking_loss_weight * ranking_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            total += float(losses["flow_matching_loss"]) * len(text)
            active_total += float(losses["active_joint_loss"]) * len(text)
            velocity_total += float(losses["active_velocity_loss"]) * len(text)
            ranking_total += float(ranking_loss) * len(text)
            items += len(text)
        record = {
            "epoch": epoch,
            "flow_matching_loss": total / items,
            "active_joint_loss": active_total / items,
            "active_velocity_loss": velocity_total / items,
            "ranking_loss": ranking_total / items,
        }
        if val_loader is not None:
            record["val_flow_matching_loss"] = evaluate_flow_loss(
                torch, model, adapter, val_loader, datamodule, device
            )
        history.append(record)
        print(json.dumps(record))
        save_checkpoint(torch, output.with_name(f"{output.stem}_epoch{epoch:03d}{output.suffix}"), adapter, optimizer, normalization, configs, history, epoch, args)
        selection_value = record.get("val_flow_matching_loss", record["flow_matching_loss"])
        if selection_value < best_val:
            best_val = selection_value
            save_checkpoint(torch, output.with_name(f"{output.stem}_best{output.suffix}"), adapter, optimizer, normalization, configs, history, epoch, args)
    save_checkpoint(torch, output, adapter, optimizer, normalization, configs, history, args.epochs, args)
    print(f"checkpoint: {output}")
    return 0


def direct_control_losses(
    torch,
    model,
    datamodule,
    instructions,
    target_motion,
    lengths,
    text,
    text_lengths,
    hint,
    frame_mask,
    sensor_mask,
    timesteps=None,
    noise=None,
):
    batch = len(target_motion)
    if timesteps is None:
        timesteps = torch.randint(
            0,
            model.noise_scheduler.config.num_train_timesteps + 1,
            (batch,),
            device=target_motion.device,
            dtype=torch.long,
        )
    if noise is None:
        noise = torch.randn_like(target_motion)
    noisy = model.noise_scheduler.scale_noise(
        sample=target_motion.clone(), noise=noise, timestep=timesteps
    )
    predicted_velocity = model.denoiser(
        instructions=instructions,
        hidden_states=noisy,
        timestep=timesteps,
        text=text,
        text_lengths=text_lengths,
        hint=hint,
        hint_lengths=frame_mask,
        target_lengths=lengths,
        target_lengths_z=lengths,
        return_dict=False,
    )[0]
    target_velocity = noise - target_motion
    feature_mask = frame_mask[:, :, None].expand_as(target_motion)
    denominator = feature_mask.sum().clamp_min(1).to(target_motion.dtype)
    flow_loss = (
        ((predicted_velocity - target_velocity) ** 2) * feature_mask
    ).sum() / denominator

    sigma = timesteps.to(target_motion.dtype) / float(
        model.noise_scheduler.config.num_train_timesteps
    )
    predicted_x0 = noisy - sigma[:, None, None] * predicted_velocity
    predicted_joints = datamodule.feats2joints(predicted_x0)
    target_joints = datamodule.feats2joints(target_motion)
    predicted_joints = predicted_joints - predicted_joints[:, :, :1]
    target_joints = target_joints - target_joints[:, :, :1]
    joint_mask = torch.zeros(batch, 22, dtype=torch.bool, device=target_motion.device)
    joint_mask[:, 15] = sensor_mask[:, 4]
    joint_mask[:, 20] = sensor_mask[:, 0]
    joint_mask[:, 21] = sensor_mask[:, 1]
    active = frame_mask[:, :, None, None] & joint_mask[:, None, :, None]
    active = active.expand(-1, -1, -1, 3)
    active_denominator = active.sum().clamp_min(1).to(target_motion.dtype)
    position_error = predicted_joints - target_joints
    active_joint_loss = ((position_error**2) * active).sum() / active_denominator
    if target_motion.shape[1] < 2:
        active_velocity_loss = target_motion.new_zeros(())
    else:
        velocity_active = active[:, 1:] & active[:, :-1]
        velocity_denominator = velocity_active.sum().clamp_min(1).to(target_motion.dtype)
        velocity_error = position_error[:, 1:] - position_error[:, :-1]
        active_velocity_loss = (
            (velocity_error**2) * velocity_active
        ).sum() / velocity_denominator
    return {
        "flow_matching_loss": flow_loss,
        "active_joint_loss": active_joint_loss,
        "active_velocity_loss": active_velocity_loss,
        "timesteps": timesteps,
        "noise": noise,
    }


def evaluate_flow_loss(torch, model, adapter, loader, datamodule, device):
    adapter.eval()
    total = 0.0
    items = 0
    cuda_devices = [] if device.type != "cuda" else [device.index or 0]
    with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
        torch.manual_seed(4321)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(4321)
        for batch in loader:
            motion = batch["motion"].to(device)
            motion = (
                motion - datamodule.mean.to(device)[None, None]
            ) / datamodule.std.to(device)[None, None]
            imu = batch["imu"].to(device)
            sensors = batch["sensor_mask"].to(device)
            frame_mask = batch["frame_mask"].to(device)
            text = list(batch["text"])
            control = adapter(imu, sensors, frame_mask)
            result = model.diffusion_process(
                instructions=model.instructions["text_hint"].repeat(len(text), 1),
                target_motion=motion,
                target_lengths=batch["lengths"],
                target_lengths_z=batch["lengths"],
                text=model.text_encoder(text),
                text_lengths=[77] * len(text),
                hint=control,
                hint_lengths=frame_mask,
            )
            denominator = (frame_mask.sum() * 263).clamp_min(1).to(motion.dtype)
            loss = ((result["v_pred"] - result["v_gt"]) ** 2).sum() / denominator
            total += float(loss) * len(text)
            items += len(text)
    adapter.train()
    return total / items


def save_checkpoint(torch, path, adapter, optimizer, normalization, configs, history, epoch, args):
    torch.save(
        {
            "adapter": adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "imu_normalization": normalization,
            "sensor_configs": {key: list(value) for key, value in configs.items()},
            "history": history,
            "epoch": epoch,
            "training_args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "control_space": (
                f"preembedded MotionLab {args.control_dim}D hint tokens"
                if args.control_dim != 66
                else "learned MotionLab 66D hint tokens"
            ),
            "adapter_config": {
                "sensor_fusion": args.sensor_fusion,
                "output_dim": args.control_dim,
            },
        },
        path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
