#!/usr/bin/env python3
"""Deterministically select MotionLab direct-IMU checkpoints on validation data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sample_motionlab_text import (
    MinimalHumanMLDataModule,
    disable_unrelated_initializers,
    load_config,
)
from train_motionlab_direct_imu_control import (
    PairedDataset,
    SENSOR_CONFIGS,
    collate,
    make_adapter,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motionlab-root", type=Path, required=True)
    parser.add_argument("--motionlab-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--imu-manifest", type=Path, required=True)
    parser.add_argument("--sensor-configs", default="head,wrists")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def evaluate(torch, model, adapter, loader, datamodule, device, seed, mode):
    if mode not in {"paired", "zero", "shuffled"}:
        raise ValueError(mode)
    adapter.eval()
    total = 0.0
    count = 0
    cuda_devices = [] if device.type != "cuda" else [device.index or 0]
    with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        for batch in loader:
            motion = batch["motion"].to(device)
            motion = (
                motion - datamodule.mean.to(device)[None, None]
            ) / datamodule.std.to(device)[None, None]
            frame_mask = batch["frame_mask"].to(device)
            control = adapter(
                batch["imu"].to(device),
                batch["sensor_mask"].to(device),
                frame_mask,
            )
            if mode == "zero":
                control = torch.zeros_like(control)
            elif mode == "shuffled":
                control = torch.roll(control, shifts=1, dims=0)
            text = list(batch["text"])
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
            count += len(text)
    return total / count


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    root = args.motionlab_root.resolve()
    motionlab_checkpoint = args.motionlab_checkpoint.resolve()
    checkpoints = [path.resolve() for path in args.checkpoints]
    manifest = args.manifest.resolve()
    imu_manifest = args.imu_manifest.resolve()
    output = args.output.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    import torch
    from rfmotion.models.get_model import get_model

    cfg = load_config(root)
    disable_unrelated_initializers()
    datamodule = MinimalHumanMLDataModule(root)
    model = get_model(cfg, datamodule)
    model.load_state_dict(torch.load(motionlab_checkpoint, map_location="cpu")["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad = False
    device = torch.device(args.device)
    model.to(device).eval()
    checkpoint_payloads = [torch.load(path, map_location="cpu") for path in checkpoints]
    normalization = checkpoint_payloads[0]["imu_normalization"]
    names = [value.strip() for value in args.sensor_configs.split(",") if value.strip()]
    loaders = {}
    for name in names:
        dataset = PairedDataset(
            manifest,
            imu_manifest,
            {name: SENSOR_CONFIGS[name]},
            normalization=normalization,
            limit=args.max_records,
        )
        loaders[name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda items: collate(torch, items),
        )
    results = []
    for path, payload in zip(checkpoints, checkpoint_payloads):
        adapter_config = payload.get("adapter_config", {"sensor_fusion": "mean"})
        adapter = make_adapter(torch, **adapter_config).to(device)
        adapter.load_state_dict(payload["adapter"])
        output_dim = int(adapter_config.get("output_dim", 66))
        if output_dim == model.denoiser.token_dim:
            model.denoiser.hint_embed1 = torch.nn.Identity()
        elif output_dim != 66:
            raise ValueError(
                f"checkpoint control dim {output_dim} does not match "
                f"MotionFlow token dim {model.denoiser.token_dim}"
            )
        record = {"checkpoint": str(path), "epoch": int(payload["epoch"]), "configs": {}}
        for name, loader in loaders.items():
            values = {
                mode: evaluate(
                    torch, model, adapter, loader, datamodule, device, args.seed, mode
                )
                for mode in ("paired", "zero", "shuffled")
            }
            values["gain_vs_zero"] = values["zero"] - values["paired"]
            values["gain_vs_shuffled"] = values["shuffled"] - values["paired"]
            record["configs"][name] = values
        results.append(record)
        print(json.dumps(record))
    ranked = sorted(
        results,
        key=lambda item: sum(item["configs"][name]["paired"] for name in names),
    )
    summary = {
        "protocol": {
            "seed": args.seed,
            "manifest": str(manifest),
            "samples_per_config": len(loaders[names[0]].dataset),
            "modes": ["paired", "zero", "shuffled"],
        },
        "results": results,
        "selected_checkpoint": ranked[0]["checkpoint"],
        "selected_epoch": ranked[0]["epoch"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"selected_checkpoint": summary["selected_checkpoint"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
