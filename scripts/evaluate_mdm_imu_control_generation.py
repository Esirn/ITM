#!/usr/bin/env python
"""Evaluate ITM generations with the official HumanML text-motion evaluator."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import types
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from itm.data.manifest import read_jsonl
from itm.data.standard_imu import sensor_mask
from itm.models.mdm_imu_control import (
    MDMIMUControlConfig,
    install_imu_control,
    make_text_imu_guidance_model,
)
from itm.models.torch_frame_baseline import require_torch


SENSOR_CONFIGS = {"head": (4,), "wrists": (0, 1)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--mdm-root", default="/home/a200/0relatedworks/motion-diffusion-model")
    parser.add_argument("--mdm-checkpoint", default=str(ROOT / "outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/model000475000.pt"))
    parser.add_argument("--mdm-args", default=str(ROOT / "outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/args.json"))
    parser.add_argument("--manifest", default=str(ROOT / "outputs/manifests_full/test.jsonl"))
    parser.add_argument("--standard-imu-manifest", default=str(ROOT / "outputs/manifests_full/test_standard_imu.jsonl"))
    parser.add_argument("--mean", default="/home/a200/0proj/datasets/all/Mean.npy")
    parser.add_argument("--std", default="/home/a200/0proj/datasets/all/Std.npy")
    parser.add_argument("--t2m-mean", default="/home/a200/0relatedworks/motion-diffusion-model/dataset/t2m_mean.npy")
    parser.add_argument("--t2m-std", default="/home/a200/0relatedworks/motion-diffusion-model/dataset/t2m_std.npy")
    parser.add_argument("--sensor-config", choices=tuple(SENSOR_CONFIGS), default="head")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--replication-times", type=int, default=1)
    parser.add_argument("--diversity-times", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--text-scale", type=float, default=2.5)
    parser.add_argument("--imu-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", default=str(ROOT / "outputs/mdm_control/comparisons/itm_generation_eval/stage2b_head_debug.log"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _legacy_compatibility()
    _install_optional_dependency_stubs()
    torch = require_torch()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    control_checkpoint = Path(args.control_checkpoint).resolve()
    mdm_checkpoint = Path(args.mdm_checkpoint).resolve()
    mdm_args_path = Path(args.mdm_args).resolve()
    output_path = Path(args.output).resolve()
    args.control_checkpoint = str(control_checkpoint)
    args.mdm_checkpoint = str(mdm_checkpoint)
    args.mdm_args = str(mdm_args_path)
    args.output = str(output_path)

    mdm_root = Path(args.mdm_root).resolve()
    if not (mdm_root / "model/mdm.py").exists():
        raise FileNotFoundError(f"Invalid MDM root: {mdm_root}")
    os.chdir(mdm_root)
    sys.path.insert(0, str(mdm_root))

    from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
    from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
    from torch.utils.data import DataLoader
    from utils.model_util import create_model_and_diffusion, load_model_wo_clip

    mdm_options = json.loads(mdm_args_path.read_text())
    _set_mdm_compatibility_defaults(mdm_options)
    mdm_args = SimpleNamespace(**mdm_options)
    data_stub = SimpleNamespace(dataset=SimpleNamespace(num_actions=1))
    model, diffusion = create_model_and_diffusion(mdm_args, data_stub)
    mdm_state = torch.load(mdm_checkpoint, map_location="cpu", weights_only=True)
    load_model_wo_clip(model, mdm_state)

    control_state = torch.load(control_checkpoint, map_location="cpu", weights_only=True)
    config = MDMIMUControlConfig(**control_state["control_config"])
    controlled, encoder, hook = install_imu_control(model, config)
    encoder.load_state_dict(control_state["imu_encoder"])
    controlled.adapters.load_state_dict(control_state["adapters"])
    model.to(device)
    model.eval()
    guided_model = make_text_imu_guidance_model(model)
    guided_model.to(device)
    guided_model.eval()

    records = load_eval_records(args)
    if len(records) < args.batch_size:
        raise RuntimeError(f"Only {len(records)} usable records; need at least batch_size={args.batch_size}")
    usable = (len(records) // args.batch_size) * args.batch_size
    records = records[:usable]
    print(f"usable_records={len(records)} sensor={args.sensor_config}")

    word_vectorizer = WordVectorizer("./glove", "our_vab")
    generated_motion: list[dict[str, Any]] = []
    gt_motion: list[dict[str, Any]] = []
    mean = np.load(args.mean).astype(np.float32)
    std = np.load(args.std).astype(np.float32)
    t2m_mean = np.load(args.t2m_mean).astype(np.float32)
    t2m_std = np.load(args.t2m_std).astype(np.float32)

    eval_wrapper = EvaluatorMDMWrapper("humanml", device)
    replication_stats = []
    for replication in range(args.replication_times):
        print(f"replication {replication + 1}/{args.replication_times}")
        generated_motion.clear()
        gt_motion.clear()
        generator = torch.Generator(device=device).manual_seed(args.seed + replication)
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            sample = sample_batch(
                batch_records,
                args,
                control_state,
                guided_model,
                diffusion,
                model,
                device,
                generator,
            )
            generated = sample.cpu().squeeze(2).permute(0, 2, 1).numpy()
            raw_generated = generated * std[None, None] + mean[None, None]
            eval_generated = (raw_generated - t2m_mean[None, None]) / t2m_std[None, None]
            for index, record in enumerate(batch_records):
                length = int(record["length"])
                raw_gt = record["motion"][:196]
                eval_gt = (raw_gt - t2m_mean[None]) / t2m_std[None]
                generated_motion.append(
                    motion_item(record, eval_generated[index], length)
                )
                gt_motion.append(motion_item(record, eval_gt, length))

        gen_loader = DataLoader(
            EvalMotionDataset(generated_motion, word_vectorizer),
            batch_size=args.batch_size,
            collate_fn=collate_eval_batch,
            drop_last=True,
            shuffle=False,
            num_workers=0,
        )
        gt_loader = DataLoader(
            EvalMotionDataset(gt_motion, word_vectorizer),
            batch_size=args.batch_size,
            collate_fn=collate_eval_batch,
            drop_last=True,
            shuffle=False,
            num_workers=0,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        stats = compute_metrics(
            eval_wrapper,
            gt_loader,
            gen_loader,
            diversity_times=min(args.diversity_times, len(generated_motion) - 1),
        )
        replication_stats.append(stats)
        write_report(output, args, records, aggregate_stats(replication_stats), replication_stats)
        print(f"partial_report: {output}")
    summary_stats = aggregate_stats(replication_stats)
    write_report(output, args, records, summary_stats, replication_stats)
    print(f"report: {output}")
    hook.remove()
    return 0


def load_eval_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = {str(row["motion_id"]): row for row in read_jsonl(Path(args.manifest))}
    imu_manifest = {str(row["motion_id"]): row for row in read_jsonl(Path(args.standard_imu_manifest))}
    records = []
    for motion_id, row in manifest.items():
        if motion_id not in imu_manifest:
            continue
        motion = np.load(row["joint_vec_path"]).astype(np.float32)
        if len(motion) < 40 or len(motion) > 196:
            continue
        caption, tokens = first_caption(row["text_path"])
        if not caption:
            continue
        records.append(
            {
                "motion_id": motion_id,
                "motion": motion,
                "length": len(motion),
                "caption": caption,
                "tokens": tokens,
                "imu_path": imu_manifest[motion_id]["imu_path"],
                "fps": float(imu_manifest[motion_id]["fps"]),
            }
        )
        if len(records) >= args.num_samples:
            break
    return records


def first_caption(path: str) -> tuple[str, list[str]]:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("#")
        if len(parts) >= 2:
            return parts[0], parts[1].split()
    return "", []


def sample_batch(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    control_state: dict[str, Any],
    guided_model,
    diffusion,
    model,
    device,
    generator,
):
    torch = require_torch()
    max_frames = 196
    imus = []
    frame_masks = []
    for record in records:
        imu = load_normalized_imu(record, args.sensor_config, control_state["imu_normalization"])
        length = min(int(record["length"]), max_frames)
        padded = np.zeros((max_frames, 6, 12), dtype=np.float32)
        padded[: min(length, len(imu))] = imu[: min(length, len(imu))]
        imus.append(padded)
        mask = np.zeros(max_frames, dtype=bool)
        mask[:length] = True
        frame_masks.append(mask)
    lengths = torch.tensor([min(int(r["length"]), max_frames) for r in records], dtype=torch.long, device=device)
    y = {
        "text": [record["caption"] for record in records],
        "lengths": lengths,
        "mask": lengths_to_mask(lengths, max_frames).view(len(records), 1, 1, max_frames),
        "imu": torch.from_numpy(np.stack(imus)).to(device),
        "sensor_mask": torch.from_numpy(np.stack([sensor_mask(SENSOR_CONFIGS[args.sensor_config], sensor_count=6) for _ in records])).to(device),
        "imu_frame_mask": torch.from_numpy(np.stack(frame_masks)).to(device),
        "text_scale": torch.full((len(records),), args.text_scale, device=device),
        "imu_scale": torch.full((len(records),), args.imu_scale, device=device),
    }
    noise = torch.randn(
        len(records), model.njoints, model.nfeats, max_frames, generator=generator, device=device
    )
    with torch.inference_mode():
        return diffusion.p_sample_loop(
            guided_model,
            (len(records), model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs={"y": y},
            progress=False,
            noise=noise,
            const_noise=False,
        )


def load_normalized_imu(record: dict[str, Any], sensor_config: str, normalization: dict[str, Any]) -> np.ndarray:
    with np.load(record["imu_path"]) as data:
        acceleration = data["acceleration"].astype(np.float32)
        orientation = data["orientation"].astype(np.float32)
        fps = float(data["fps"])
    acceleration = resample_linear(acceleration, fps, 20.0)
    orientation = resample_orientation(orientation, fps, 20.0)
    mean = np.asarray(normalization["acceleration_mean"], dtype=np.float32)
    std = np.asarray(normalization["acceleration_std"], dtype=np.float32)
    acceleration = (acceleration - mean[None]) / std[None]
    return np.concatenate((acceleration, orientation.reshape(len(orientation), 6, 9)), axis=-1)


def resample_linear(values: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if np.isclose(source_fps, target_fps):
        return values
    source_time = np.arange(len(values), dtype=np.float64) / source_fps
    count = max(1, int(round(len(values) * target_fps / source_fps)))
    target_time = np.minimum(np.arange(count, dtype=np.float64) / target_fps, source_time[-1])
    flat = values.reshape(len(values), -1)
    result = np.stack([np.interp(target_time, source_time, flat[:, i]) for i in range(flat.shape[1])], axis=-1)
    return result.reshape((count,) + values.shape[1:]).astype(np.float32)


def resample_orientation(values: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
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


def lengths_to_mask(lengths, max_len: int):
    torch = require_torch()
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)


def motion_item(record: dict[str, Any], motion: np.ndarray, length: int) -> dict[str, Any]:
    padded = np.zeros((196, motion.shape[-1]), dtype=np.float32)
    padded[: min(len(motion), 196)] = motion[:196]
    tokens = prepare_tokens(record["tokens"])
    return {
        "motion": padded,
        "length": min(length, 196),
        "caption": record["caption"],
        "tokens": tokens,
        "cap_len": tokens.index("eos/OTHER") + 1,
    }


def prepare_tokens(tokens: list[str], max_text_len: int = 20) -> list[str]:
    if len(tokens) < max_text_len:
        output = ["sos/OTHER", *tokens, "eos/OTHER"]
        output.extend(["unk/OTHER"] * (max_text_len + 2 - len(output)))
        return output
    return ["sos/OTHER", *tokens[:max_text_len], "eos/OTHER"]


class EvalMotionDataset:
    def __init__(self, items: list[dict[str, Any]], w_vectorizer):
        self.items = items
        self.w_vectorizer = w_vectorizer

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        word_embeddings = []
        pos_one_hots = []
        for token in item["tokens"]:
            word_emb, pos_oh = self.w_vectorizer[token]
            word_embeddings.append(word_emb[None])
            pos_one_hots.append(pos_oh[None])
        return (
            np.concatenate(word_embeddings, axis=0),
            np.concatenate(pos_one_hots, axis=0),
            item["caption"],
            item["cap_len"],
            item["motion"],
            item["length"],
            "_".join(item["tokens"]),
        )


def collate_eval_batch(batch):
    from torch.utils.data._utils.collate import default_collate

    batch = sorted(batch, key=lambda item: item[3], reverse=True)
    return default_collate(batch)


def compute_metrics(eval_wrapper, gt_loader, gen_loader, *, diversity_times: int) -> dict[str, Any]:
    from data_loaders.humanml.utils.metrics import (
        calculate_activation_statistics,
        calculate_diversity,
        calculate_frechet_distance,
    )

    torch = require_torch()
    with torch.no_grad():
        gt_embeddings = collect_embeddings(eval_wrapper, gt_loader)
        gen_embeddings = collect_embeddings(eval_wrapper, gen_loader)
        match, r_precision = matching(eval_wrapper, gen_loader)
    gt_mu, gt_cov = calculate_activation_statistics(gt_embeddings)
    gen_mu, gen_cov = calculate_activation_statistics(gen_embeddings)
    return {
        "matching_score": match,
        "r_precision": r_precision.tolist(),
        "fid": float(calculate_frechet_distance(gt_mu, gt_cov, gen_mu, gen_cov)),
        "diversity": float(calculate_diversity(gen_embeddings, diversity_times)),
        "gt_diversity": float(calculate_diversity(gt_embeddings, diversity_times)),
        "num_samples": len(gen_embeddings),
    }


def collect_embeddings(eval_wrapper, loader) -> np.ndarray:
    torch = require_torch()
    chunks = []
    with torch.no_grad():
        for batch in loader:
            _, _, _, sent_lens, motions, m_lens, _ = batch
            embeddings = eval_wrapper.get_motion_embeddings(motions, m_lens)
            chunks.append(embeddings.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def matching(eval_wrapper, loader) -> tuple[float, np.ndarray]:
    from data_loaders.humanml.utils.metrics import calculate_top_k, euclidean_distance_matrix

    torch = require_torch()
    matching_score_sum = 0.0
    top_k_count = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
            text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                word_embs=word_embeddings,
                pos_ohot=pos_one_hots,
                cap_lens=sent_lens,
                motions=motions,
                m_lens=m_lens,
            )
            dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(), motion_embeddings.cpu().numpy())
            matching_score_sum += float(dist_mat.trace())
            top_k_count += calculate_top_k(np.argsort(dist_mat), top_k=3).sum(axis=0)
            total += text_embeddings.shape[0]
    return matching_score_sum / total, top_k_count / total


def aggregate_stats(replication_stats: list[dict[str, Any]]) -> dict[str, Any]:
    if not replication_stats:
        raise ValueError("replication_stats must be non-empty")
    keys = ("matching_score", "fid", "diversity", "gt_diversity")
    summary: dict[str, Any] = {
        "num_samples": replication_stats[0]["num_samples"],
        "replications": len(replication_stats),
    }
    for key in keys:
        values = np.asarray([stats[key] for stats in replication_stats], dtype=np.float64)
        summary[key] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    r_values = np.asarray([stats["r_precision"] for stats in replication_stats], dtype=np.float64)
    summary["r_precision"] = r_values.mean(axis=0).tolist()
    summary["r_precision_std"] = r_values.std(axis=0).tolist()
    return summary


def write_report(
    path: Path,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    stats: dict[str, Any],
    replication_stats: list[dict[str, Any]],
) -> None:
    payload = {
        "control_checkpoint": str(Path(args.control_checkpoint).resolve()),
        "sensor_config": args.sensor_config,
        "num_records": len(records),
        "seed": args.seed,
        "text_scale": args.text_scale,
        "imu_scale": args.imu_scale,
        "metrics": stats,
        "replication_metrics": replication_stats,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path = path.with_suffix(".md")
    md_path.write_text(
        "\n".join(
            [
                "# ITM Generation Evaluator",
                "",
                f"- checkpoint: `{payload['control_checkpoint']}`",
                f"- sensor: `{args.sensor_config}`",
                f"- samples: `{stats['num_samples']}`",
                f"- matching score: `{stats['matching_score']:.4f}`",
                f"- R-precision: `{stats['r_precision'][0]:.4f}, {stats['r_precision'][1]:.4f}, {stats['r_precision'][2]:.4f}`",
                f"- FID: `{stats['fid']:.4f}`",
                f"- diversity: `{stats['diversity']:.4f}`",
                f"- GT diversity: `{stats['gt_diversity']:.4f}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _set_mdm_compatibility_defaults(options: dict[str, Any]) -> None:
    options.setdefault("unconstrained", False)
    options.setdefault("text_encoder_type", "clip")
    options.setdefault("pos_embed_max_len", 5000)
    options.setdefault("mask_frames", False)


def _legacy_compatibility() -> None:
    aliases = {"bool": bool, "int": int, "float": float, "complex": complex, "object": object, "unicode": str, "str": str}
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec


def _install_optional_dependency_stubs() -> None:
    if "wandb" not in sys.modules:
        wandb = types.ModuleType("wandb")
        wandb.login = lambda *args, **kwargs: None
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None
        wandb.finish = lambda *args, **kwargs: None
        wandb.watch = lambda *args, **kwargs: None
        wandb.Video = lambda *args, **kwargs: None
        wandb.config = types.SimpleNamespace(update=lambda *args, **kwargs: None)
        sys.modules["wandb"] = wandb


if __name__ == "__main__":
    raise SystemExit(main())
