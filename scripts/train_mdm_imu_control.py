#!/usr/bin/env python
"""Train frozen MDM with zero-initialized IMU control adapters."""

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
from itm.data.humanml_torch import recover_from_ric_torch
from itm.data.manifest import read_jsonl
from itm.data.standard_imu import sensor_mask
from itm.models.mdm_conditions import (
    INPUT_CONTRACT_VERSION, INPUT_PIPELINE_VERSION, load_method_cache,
    sample_imu_at_times, normalize_imu,
)
from itm.models.mdm_imu_control import (MDMIMUControlConfig, install_imu_control,
    configure_control_training, evaluating_preserving_modes)
from itm.models.torch_frame_baseline import require_torch
from itm.models.mdm_training_provenance import build_training_input_provenance, validate_training_resume
from itm.models.mdm_canonical_conditions import (CanonicalB0Package, CANONICAL_PIPELINE_VERSION,
    CANONICAL_CONTRACT_VERSION, CANONICAL_SCOPE, CANONICAL_ALIGNMENT)
from itm.models.mdm_caption_conditions import (
    CAPTION_PIPELINE_VERSION, read_caption_plans, prepare_caption_input, require_caption_training_admission,
)


SENSOR_CONFIGS = {"head": (4,), "wrists": (0, 1)}
SENSOR_SLOT_TO_JOINT = {0: 20, 1: 21, 4: 15}
UPPER_BODY_JOINTS = (16, 17, 18, 19, 20, 21)


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
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--trajectory-loss-weight", type=float, default=0.0)
    parser.add_argument("--velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--jerk-loss-weight", type=float, default=0.0)
    parser.add_argument("--upper-body-loss-weight", type=float, default=0.0)
    parser.add_argument("--upper-body-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--text-anchor-loss-weight", type=float, default=0.0)
    parser.add_argument("--upper-body-text-anchor-loss-weight", type=float, default=0.0)
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--encoder-variant", choices=("legacy_mean", "ordered_concat"), default="ordered_concat")
    parser.add_argument("--adapter-variant", choices=("additive", "film"), default="additive")
    parser.add_argument("--null-mode", choices=("legacy_null", "strict"), default="strict")
    parser.add_argument("--time-encoding", choices=("none", "physical_sinusoidal"), default="physical_sinusoidal")
    parser.add_argument("--backbone-mode", choices=("frozen_eval", "legacy_train"), default="frozen_eval")
    parser.add_argument("--text-drop-probability", type=float, default=0.1)
    parser.add_argument("--imu-drop-probability", type=float, default=0.1)
    parser.add_argument("--diffusion-loss-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-space", choices=("global_joint_proxy", "root_relative_joint_proxy"), default="global_joint_proxy")
    parser.add_argument("--derivative-units", choices=("legacy_frame", "seconds"), default="seconds")
    parser.add_argument("--caption-plan", help="Explicit owner-02 caption plan; unadmitted plans are input-check only")
    parser.add_argument("--canonical-package", help="Opt in to owner-02's admitted synthetic B0 package")
    parser.add_argument("--input-only-report", help="Write normalization/provenance on CPU and exit before MDM setup")
    parser.add_argument("--response-manifest", help="Owner-02 train-only compatible arm_spread_m witness JSONL")
    parser.add_argument("--response-loss-weight", type=float, default=0.0)
    parser.add_argument("--response-proxy-weight", type=float, default=0.0)
    parser.add_argument("--response-mode", choices=("bounded_margin", "exact_delta"), default="bounded_margin")
    parser.add_argument("--response-margin", type=float, default=0.02)
    parser.add_argument("--teacher-min-delta", type=float, default=0.01)
    parser.add_argument("--no-compatibility-weight", action="store_true")
    parser.add_argument("--contract-version", default=INPUT_CONTRACT_VERSION)
    parser.add_argument("--allow-unsynchronized-diagnostic", action="store_true",
                        help="Acknowledge that the current loader alignment is not certified; diagnostic only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    canonical = None
    if getattr(args, 'canonical_package', None):
        canonical = CanonicalB0Package(args.canonical_package, workspace_root=Path.cwd())
        canonical.require_manifests('train', args.manifest, args.standard_imu_manifest, args.caption_plan)
        if args.resume or args.max_records is not None or args.response_manifest:
            raise ValueError('Canonical B0 requires all admitted train clips, a fresh controller and no edit witnesses')
        if args.contract_version != CANONICAL_CONTRACT_VERSION:
            raise ValueError('Canonical B0 requires explicit --contract-version '+CANONICAL_CONTRACT_VERSION)
    if canonical is None and not args.allow_unsynchronized_diagnostic:
        raise ValueError("Training blocked: owner 02 has not certified loader time/world alignment. "
                         "Use --allow-unsynchronized-diagnostic only for explicitly labelled diagnostics.")
    if not 0 <= args.text_drop_probability <= 1:
        raise ValueError("Invalid text dropout probability")
    caption_plans = None
    if args.caption_plan:
        caption_plans = {mid: row for mid, row in read_caption_plans(Path(args.caption_plan).resolve()).items()
                         if row["split"] == "train"}
        require_caption_training_admission(caption_plans)
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
    motion_mean = np.load(mean_path).astype(np.float32)
    motion_std = np.load(std_path).astype(np.float32)
    output_path = Path(args.output).resolve()
    if output_path.exists() or any(output_path.parent.glob(f"{output_path.stem}_epoch*{output_path.suffix}")):
        raise FileExistsError("Choose a new run output; existing checkpoints will not be overwritten")
    mdm_checkpoint_path = Path(args.mdm_checkpoint).resolve()
    mdm_args_path = Path(args.mdm_args).resolve()
    resume_path = Path(args.resume).resolve() if args.resume else None
    response_records = load_response_manifest(args.response_manifest) if args.response_manifest else {}
    if args.response_loss_weight > 0 and not response_records:
        raise ValueError("Response training requires nonempty train-only verified witnesses")
    configs = _parse_configs(args.sensor_configs)
    dataset = _PairedDataset(
        manifest_path,
        imu_manifest_path,
        motion_mean,
        motion_std,
        configs,
        args.max_records,
        workspace_root,
        caption_plans=caption_plans,
        canonical_package=canonical,
    )
    normalization = dataset.compute_normalization()
    dataset.set_normalization(normalization)
    input_provenance = build_training_input_provenance(
        dataset, normalization,
        artifacts={"motion_manifest": manifest_path, "imu_manifest": imu_manifest_path,
                   "motion_mean": mean_path, "motion_std": std_path,
                   "mdm_checkpoint": mdm_checkpoint_path, "mdm_args": mdm_args_path,
                   "caption_plan": Path(args.caption_plan).resolve() if args.caption_plan else None,
                   **(canonical.provenance_artifacts if canonical else {})},
        pipeline_version=_input_pipeline(args),
        contract_version=args.contract_version,
    )
    resume = torch.load(resume_path, map_location="cpu", weights_only=True) if resume_path else None
    if resume is not None:
        validate_training_resume(resume, input_provenance, normalization)
    if getattr(args, 'input_only_report', None):
        report = {'imu_normalization': normalization, 'training_input_provenance': input_provenance,
                  'training_args': vars(args), 'inputs': [dataset[i]['input_metadata'] for i in range(len(dataset))],
                  'model_forwards': 0, 'training_updates': 0, 'generated_samples': 0, 'gpu_jobs': 0}
        report_path = Path(args.input_only_report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
        print(f'input_only_report: {report_path}')
        return 0

    mdm_options = json.loads(mdm_args_path.read_text())
    _set_mdm_compatibility_defaults(mdm_options)
    if not mdm_options.get("predict_xstart", False):
        raise ValueError("ITM auxiliary losses require an MDM checkpoint with predict_xstart=true")
    mdm_args = SimpleNamespace(**mdm_options)
    control_config = MDMIMUControlConfig(
        latent_dim=mdm_args.latent_dim, encoder_variant=args.encoder_variant, adapter_variant=args.adapter_variant,
        null_mode=args.null_mode, time_encoding=args.time_encoding,
        imu_drop_probability=args.imu_drop_probability,
    )
    if resume is not None and MDMIMUControlConfig(**resume["control_config"]) != control_config:
        raise ValueError("Resume config differs; legacy weights require explicit legacy settings. No silent migration.")
    root = Path(args.mdm_root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    from utils.model_util import create_model_and_diffusion, load_model_wo_clip

    data_stub = SimpleNamespace(dataset=SimpleNamespace(num_actions=1))
    model, diffusion = create_model_and_diffusion(mdm_args, data_stub)
    state = torch.load(mdm_checkpoint_path, map_location="cpu", weights_only=True)
    load_model_wo_clip(model, state)
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
    if args.backbone_mode == "frozen_eval":
        configure_control_training(model)
    else:
        model.train()
        model.clip_model.eval()

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    history = []
    start_epoch = 1
    if resume_path is not None:
        imu_encoder.load_state_dict(resume["imu_encoder"])
        controlled.adapters.load_state_dict(resume["adapters"])
        history = list(resume.get("history", []))
        start_epoch = int(resume.get("epoch", len(history))) + 1
        # Current normalization has been recomputed and verified against the
        # bound checkpoint; never silently replace it with unverified statistics.
        # Stage-2 commonly changes the loss mix and learning rate, so keep the
        # weights but start with a fresh optimizer unless explicitly extended later.
    mean_tensor = torch.from_numpy(motion_mean).view(1, -1, 1, 1).to(device)
    std_tensor = torch.from_numpy(motion_std).view(1, -1, 1, 1).to(device)
    if start_epoch > args.epochs:
        raise ValueError("epochs must exceed the resumed epoch")
    combination_counts = {"00": 0, "10": 0, "01": 0, "11": 0}
    for epoch in range(start_epoch, args.epochs + 1):
        totals = {
            "response_loss": 0.0,
            "response_joint_proxy_loss": 0.0,
            "total_loss": 0.0,
            "diffusion_loss": 0.0,
            "trajectory_loss": 0.0,
            "velocity_loss": 0.0,
            "jerk_loss": 0.0,
            "upper_body_loss": 0.0,
            "upper_body_velocity_loss": 0.0,
            "text_anchor_loss": 0.0,
            "upper_body_text_anchor_loss": 0.0,
        }
        total_items = 0
        response_coverage = {"selected_examples": 0, "accepted_attributes": 0, "total_attributes": 0,
                             "student_forwards": 0, "teacher_forwards": 0, "timestep_buckets": {}}
        for motion, y in loader:
            motion = motion.to(device)
            y = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in y.items()}
            if args.backbone_mode == "frozen_eval":
                # Independent Bernoulli masks, once per sample. MDM is eval so its
                # internal text dropout cannot silently redraw this decision.
                y, counts = training_condition_dropout(model, y, args.text_drop_probability, args.imu_drop_probability)
                for key, count in counts.items():
                    combination_counts[key] += count
            timesteps = torch.randint(0, diffusion.num_timesteps, (len(motion),), device=device)
            noise = torch.randn_like(motion)
            model_output, diffusion_loss = _predict_xstart_and_diffusion_loss(
                model, diffusion, motion, timesteps, noise, y
            )
            auxiliary = _stage2_auxiliary_losses(
                model,
                diffusion,
                motion,
                timesteps,
                noise,
                y,
                mean_tensor,
                std_tensor,
                model_output=model_output,
                trajectory_space=args.trajectory_space,
                derivative_fps=(20.0 if args.derivative_units == "seconds" else 1.0),
                use_text_anchor=(
                    args.text_anchor_loss_weight > 0
                    or args.upper_body_text_anchor_loss_weight > 0
                ),
            )
            trajectory_loss = auxiliary["trajectory_loss"]
            velocity_loss = auxiliary["velocity_loss"]
            jerk_loss = auxiliary["jerk_loss"]
            upper_body_loss = auxiliary["upper_body_loss"]
            upper_body_velocity_loss = auxiliary["upper_body_velocity_loss"]
            text_anchor_loss = auxiliary["text_anchor_loss"]
            upper_body_text_anchor_loss = auxiliary["upper_body_text_anchor_loss"]
            response_loss, response_proxy = motion.new_zeros(()), motion.new_zeros(())
            if args.response_loss_weight > 0:
                from itm.models.semantic_response import semantic_response_step
                selected = [(i, response_records[(mid, cfg)])
                            for i, (mid, cfg) in enumerate(zip(y["motion_id"], y["sensor_config"]))
                            if (mid, cfg) in response_records]
                if selected:
                    indices = torch.tensor([i for i, _ in selected], device=device)
                    rows = [row for _, row in selected]
                    validate_response_intervals(rows, [y["input_metadata"][i] for i, _ in selected])
                    edit_y = {key: (value[indices] if torch.is_tensor(value) and key != "text_embed" else value)
                              for key, value in y.items() if key != "text_embed"}
                    edit_losses, diagnostic = semantic_response_step(
                        model, diffusion.q_sample(motion[indices], timesteps[indices], noise=noise[indices]),
                        diffusion._scale_timesteps(timesteps[indices]), edit_y,
                        caption0=[r["base_caption_text"] for r in rows], caption1=[r["edit_caption_text"] for r in rows],
                        witnesses=[{"split": "train", "compatibility_evidence": r, "attribute_spec": "arm_spread_m"} for r in rows],
                        decode_joints=lambda out: _recover_joints_from_normalized_motion(out, mean_tensor, std_tensor),
                        target_proxy_joints=_recover_joints_from_normalized_motion(motion[indices], mean_tensor, std_tensor),
                        compatibility_weight=motion.new_tensor([r.get("compatibility_weight", 1.0) for r in rows]),
                        mode=args.response_mode, margin=args.response_margin, teacher_min_delta=args.teacher_min_delta,
                        use_compatibility_weight=not args.no_compatibility_weight,
                    )
                    response_loss = edit_losses["relative_response"]
                    response_proxy = edit_losses["edited_observation_joint_proxy"]
                    response_coverage["selected_examples"] += len(rows)
                    for key in ("accepted_attributes", "total_attributes", "student_forwards", "teacher_forwards"):
                        response_coverage[key] += diagnostic[key]
                    for t in timesteps[indices].tolist():
                        bucket = str(min(9, int(t * 10 / diffusion.num_timesteps)))
                        response_coverage["timestep_buckets"][bucket] = response_coverage["timestep_buckets"].get(bucket, 0) + 1
            loss = (
                args.response_loss_weight * response_loss
                + args.response_proxy_weight * response_proxy
                + args.diffusion_loss_weight * diffusion_loss
                + args.trajectory_loss_weight * trajectory_loss
                + args.velocity_loss_weight * velocity_loss
                + args.jerk_loss_weight * jerk_loss
                + args.upper_body_loss_weight * upper_body_loss
                + args.upper_body_velocity_loss_weight * upper_body_velocity_loss
                + args.text_anchor_loss_weight * text_anchor_loss
                + args.upper_body_text_anchor_loss_weight * upper_body_text_anchor_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            batch = len(motion)
            totals["response_loss"] += float(response_loss.item()) * batch
            totals["response_joint_proxy_loss"] += float(response_proxy.item()) * batch
            totals["total_loss"] += float(loss.item()) * batch
            totals["diffusion_loss"] += float(diffusion_loss.item()) * batch
            totals["trajectory_loss"] += float(trajectory_loss.item()) * batch
            totals["velocity_loss"] += float(velocity_loss.item()) * batch
            totals["jerk_loss"] += float(jerk_loss.item()) * batch
            totals["upper_body_loss"] += float(upper_body_loss.item()) * batch
            totals["upper_body_velocity_loss"] += float(upper_body_velocity_loss.item()) * batch
            totals["text_anchor_loss"] += float(text_anchor_loss.item()) * batch
            totals["upper_body_text_anchor_loss"] += float(upper_body_text_anchor_loss.item()) * batch
            total_items += len(motion)
        epoch_values = {key: value / total_items for key, value in totals.items()}
        history.append({"epoch": epoch, "loss": epoch_values["total_loss"], **epoch_values, "condition_counts_cumulative": dict(combination_counts), "response_coverage": response_coverage})
        print(
            f"epoch={epoch} loss={epoch_values['total_loss']:.6f} "
            f"diffusion={epoch_values['diffusion_loss']:.6f} "
            f"trajectory={epoch_values['trajectory_loss']:.6f} "
            f"velocity={epoch_values['velocity_loss']:.6f} "
            f"jerk={epoch_values['jerk_loss']:.6f} "
            f"upper_body={epoch_values['upper_body_loss']:.6f} "
            f"upper_body_velocity={epoch_values['upper_body_velocity_loss']:.6f} "
            f"text_anchor={epoch_values['text_anchor_loss']:.6f} "
            f"upper_body_text_anchor={epoch_values['upper_body_text_anchor_loss']:.6f}"
        )
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
            input_provenance,
        )
    hook.remove()
    output = output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(
        output, torch, imu_encoder, controlled, optimizer, control_config, configs,
        mdm_checkpoint_path, normalization, history, args.epochs, args, input_provenance,
    )
    print(f"control_checkpoint: {output}")
    return 0


class _PairedDataset:
    def __init__(self, manifest, imu_manifest, mean, std, configs, limit, workspace_root, *, caption_plans=None,
                 canonical_package=None):
        self.canonical_package = canonical_package
        if canonical_package is not None:
            canonical_package.require_manifests('train', manifest, imu_manifest,
                                                canonical_package.manifests['train_caption_plans'])
            if caption_plans != canonical_package.plans['train'] or limit is not None:
                raise ValueError('Use the complete delivered canonical training plan')
        motion_dataset = TextIMUMotionDataset(
            manifest, load_joints=False, load_joint_vec=True, load_imu=False
        )
        if any(record.get("split") != "train" for record in motion_dataset.records):
            raise ValueError("Training and normalization require a train-only manifest")
        sample_indices = {str(record["motion_id"]): index for index, record in enumerate(motion_dataset.records)}
        imu_records = {str(record["motion_id"]): record for record in read_jsonl(imu_manifest)}
        common = sorted(set(sample_indices) & set(imu_records))
        if caption_plans is not None:
            if set(caption_plans) - set(common):
                raise ValueError("Caption plan references unavailable train/IMU records")
            common = [mid for mid in common if mid in caption_plans]
        self.caption_plans = caption_plans
        self.motion_records = {str(r["motion_id"]): r for r in motion_dataset.records}
        if limit is not None:
            common = common[:limit]
        samples = {mid: motion_dataset[sample_indices[mid]] for mid in common}
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
        if getattr(self, "caption_plans", None) is not None:
            mid = str(sample["motion_id"])
            plan = self.caption_plans[mid]
            if config_name not in plan["sensor_configs"]:
                raise ValueError("Sensor configuration absent from caption plan")
            canonical = getattr(self, 'canonical_package', None)
            item = (canonical.prepare('train', mid, self.normalization) if canonical else
                    prepare_caption_input(plan, self.motion_records[mid], imu_record, self.normalization,
                                          workspace_root=self.workspace_root))
            if not item["cache_sensor_mask"][list(slots)].all():
                raise ValueError("Requested sensors are absent from cache")
            return {"motion_id": mid, "motion": (item["motion"] - self.mean) / self.std,
                    "imu": item["imu"], "sensor_mask": sensor_mask(slots, sensor_count=6),
                    "caption": item["caption"], "sensor_config": config_name,
                    "timestamps_seconds": item["model_times_seconds"], "input_metadata": item["input_metadata"]}
        motion = (np.asarray(sample["motion"], dtype=np.float32) - self.mean) / self.std
        cache = load_method_cache(imu_record, workspace_root=self.workspace_root)
        length = min(len(motion), 196)
        if length < 1:
            raise ValueError("Empty motion")
        times = np.arange(length, dtype=np.float64) / 20.0
        if not cache["sensor_mask"][list(slots)].all():
            raise ValueError("Requested sensors are absent from cache")
        acceleration, orientation = sample_imu_at_times(cache, times)
        normalization = self.normalization or {
            "acceleration_mean": np.zeros((6, 3)), "acceleration_std": np.ones((6, 3))}
        imu = normalize_imu(acceleration, orientation, normalization)
        return {
            "motion_id": str(sample["motion_id"]),
            "timestamps_seconds": times,
            "input_metadata": {
                "input_pipeline_version": INPUT_PIPELINE_VERSION,
                "original_motion_frames": len(motion), "effective_frames": length,
                "time_interval_seconds": [float(times[0]), float(times[-1])],
                "clock_reference": "cache_crop_local_uncertified_against_HumanML",
                "caption_interval_seconds": [sample["caption_record"].start, sample["caption_record"].end],
                "alignment_status": "uncertified_diagnostic",
            },
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
            if record.get("split") != "train":
                raise ValueError("IMU normalization requires train-only records")
            if self.caption_plans is not None:
                mid = str(record["motion_id"])
                canonical = getattr(self, 'canonical_package', None)
                item = (canonical.prepare('train', mid) if canonical else
                        prepare_caption_input(self.caption_plans[mid], self.motion_records[mid], record, None,
                                              workspace_root=self.workspace_root))
                acceleration = item["acceleration"].astype(np.float64)
            else:
                cache = load_method_cache(record, workspace_root=self.workspace_root)
                acceleration = cache["acceleration"].astype(np.float64)
            total += acceleration.sum(0)
            squared += np.square(acceleration).sum(0)
            count += len(acceleration)
        mean = total / count
        variance = np.maximum(squared / count - np.square(mean), 1e-8)
        result = {
            "acceleration_mean": mean.astype(np.float32).tolist(),
            "acceleration_std": np.sqrt(variance).astype(np.float32).tolist(),
            "orientation": "rotation_matrix_unscaled",
        }
        if getattr(self, 'canonical_package', None):
            result['source'] = {'split': 'train', 'unique_motion_ids': sorted(self.caption_plans),
                                'fps': 20., 'unique_frames': count, 'configuration_duplicates_counted': False,
                                'pipeline': CANONICAL_PIPELINE_VERSION}
        return result

    def set_normalization(self, normalization):
        self.normalization = normalization


def _collate(items):
    torch = require_torch()
    lengths = torch.tensor([len(item["motion"]) for item in items], dtype=torch.long)
    frames = int(lengths.max())
    motion = torch.zeros(len(items), 263, 1, frames)
    imu = torch.zeros(len(items), frames, 6, 12)
    frame_mask = torch.arange(frames)[None] < lengths[:, None]
    timestamps = torch.zeros(len(items), frames, dtype=torch.float64)
    for index, item in enumerate(items):
        length = len(item["motion"])
        motion[index, :, 0, :length] = torch.from_numpy(item["motion"].T)
        imu[index, :length] = torch.from_numpy(item["imu"])
        timestamps[index, :length] = torch.from_numpy(item["timestamps_seconds"])
    y = {
        "motion_id": [item.get("motion_id", "") for item in items],
        "sensor_config": [item["sensor_config"] for item in items],
        "mask": frame_mask[:, None, None],
        "lengths": lengths,
        "text": [item["caption"] for item in items],
        "imu": imu,
        "sensor_mask": torch.from_numpy(np.stack([item["sensor_mask"] for item in items])),
        "imu_frame_mask": frame_mask,
        "imu_timestamps": timestamps,
        "input_metadata": [item["input_metadata"] for item in items],
    }
    return motion, y


def _parse_configs(value):
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in SENSOR_CONFIGS]
    if unknown or not names:
        raise ValueError(f"Invalid sensor configs {unknown}; choose from {sorted(SENSOR_CONFIGS)}")
    return {name: SENSOR_CONFIGS[name] for name in names}


def _stage2_auxiliary_losses(
    model,
    diffusion,
    motion,
    timesteps,
    noise,
    y,
    mean,
    std,
    *,
    model_output=None,
    use_text_anchor=False,
    trajectory_space="global_joint_proxy",
    derivative_fps=1.0,
):
    torch = require_torch()
    x_t = diffusion.q_sample(motion, timesteps, noise=noise)
    if model_output is None:
        model_output = model(x_t, diffusion._scale_timesteps(timesteps), y=y)
    predicted = _recover_joints_from_normalized_motion(model_output, mean, std)
    target = _recover_joints_from_normalized_motion(motion, mean, std)
    frame_mask = y["mask"][:, 0, 0].bool()
    joint_weights = active_sensor_joint_weights(y["sensor_mask"], joint_count=22).to(
        predicted.device
    )
    head_only = head_only_mask(y["sensor_mask"]).to(predicted.device)
    anchor = None
    if use_text_anchor and getattr(model, "cond_mode", None) == "text":
        anchor_y = text_only_anchor_kwargs(y)
        with evaluating_preserving_modes(model), torch.no_grad():
            anchor_output = model(x_t, diffusion._scale_timesteps(timesteps), y=anchor_y)
        anchor = _recover_joints_from_normalized_motion(anchor_output, mean, std)
    return stage2_control_losses(
        predicted,
        target,
        frame_mask,
        joint_weights,
        head_only,
        text_anchor=anchor,
        trajectory_space=trajectory_space, derivative_fps=derivative_fps,
    )


def load_response_manifest(path):
    rows = list(read_jsonl(Path(path).resolve()))
    if not rows:
        raise ValueError("No verified compatible training witnesses; B is blocked")
    result = {}
    for row in rows:
        if (row.get("split_role") != "train" or row.get("compatibility_label") != "compatible"
                or row.get("edited_attribute") != "arm_spread_m" or not row.get("evidence_type")
                or not row.get("witness_observation_residuals") or not row.get("quality_checks")
                or not row.get("reviewer_or_rule_version") or not row.get("tolerances")
                or not row.get("witness_motion_path") or not Path(row["witness_motion_path"]).is_file()
                or row.get("training_admissible") is not True or row.get("blockers")
                or row.get("sensor_config") != "head"):
            raise ValueError("B requires owner-02 compatible train head arm_spread_m witnesses, training_admissible=true and checked residuals/quality")
        with np.load(row["witness_motion_path"], allow_pickle=False) as witness:
            if "timestamps_seconds" not in witness.files:
                raise ValueError("Witness artifact requires explicit timestamps_seconds")
            row = dict(row, _witness_timestamps_seconds=witness["timestamps_seconds"].copy())
        key = (str(row["source_motion_id"]), row["sensor_config"])
        if key in result:
            raise ValueError("One edit per motion/config in this initial bounded implementation")
        result[key] = row
    return result


def validate_response_intervals(rows, metadata):
    """Do not silently crop a full-clip witness to the training prefix.

    The initial B implementation only supports witnesses already checked for the
    exact prefix on the HumanML clip-local 20Hz grid. No universal frame shift.
    """
    if len(rows) != len(metadata):
        raise ValueError("One witness interval per training sample required")
    for row, item in zip(rows, metadata):
        if item.get("source_time_origin_seconds", 0.0) != 0.0:
            raise ValueError("Nonzero caption crop requires separate witness/root-frame certification")
        times = row.get("timestamps", {})
        count = times.get("frames")
        if (times.get("clock") != "HumanML_clip_local_seconds" or times.get("fps") != 20.0
                or not isinstance(count, int) or count != item["effective_frames"]
                or times.get("start_seconds", 0.0) != item["time_interval_seconds"][0]
                or not np.isclose((count - 1) / 20.0, item["time_interval_seconds"][1], rtol=0, atol=1e-10)):
            raise ValueError("Witness timestamps must exactly match the checked training interval; no silent crop")
        actual = np.asarray(row.get("_witness_timestamps_seconds", []), dtype=np.float64)
        expected = np.arange(count, dtype=np.float64) / 20.0
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=1e-10):
            raise ValueError("Witness artifact time axis differs from training interval; no silent crop")


def training_condition_dropout(model, y, text_probability, imu_probability):
    torch = require_torch()
    batch, device = len(y["imu"]), y["imu"].device
    text_present = torch.rand(batch, device=device) >= text_probability
    imu_present = ((torch.rand(batch, device=device) >= imu_probability)
                   & y["sensor_mask"].bool().any(-1) & y["imu_frame_mask"].bool().any(-1))
    result = dict(y)
    with torch.no_grad():
        text = model.encode_text(y["text"])
    result["text_embed"] = text * text_present[None, :, None]
    result["imu_condition_present"] = imu_present
    result["uncond"] = False
    counts = {f"{t}{i}": int(((text_present == bool(t)) & (imu_present == bool(i))).sum())
              for t in range(2) for i in range(2)}
    return result, counts


def _predict_xstart_and_diffusion_loss(model, diffusion, motion, timesteps, noise, y):
    """Run one conditioned forward pass and reuse it for every training loss."""

    x_t = diffusion.q_sample(motion, timesteps, noise=noise)
    model_output = model(x_t, diffusion._scale_timesteps(timesteps), y=y)
    diffusion_loss = diffusion.masked_l2(motion, model_output, y["mask"]).mean()
    return model_output, diffusion_loss


def text_only_anchor_kwargs(y):
    """Return MDM text-only kwargs without arming IMU control adapters."""

    anchor = {
        key: value
        for key, value in y.items()
        if key not in {"imu", "sensor_mask", "imu_frame_mask", "imu_uncond", "imu_control", "imu_condition_present", "text_embed"}
    }
    anchor["uncond"] = False
    return anchor


def _recover_joints_from_normalized_motion(motion, mean, std):
    features = (motion * std + mean).squeeze(2).transpose(1, 2)
    return recover_from_ric_torch(features, joints_num=22)


def active_sensor_joint_weights(sensor_mask, *, joint_count: int = 22):
    """Map active standard IMU slots to HumanML3D joints."""

    torch = require_torch()
    weights = torch.zeros(
        sensor_mask.shape[0],
        joint_count,
        dtype=torch.float32,
        device=sensor_mask.device,
    )
    for slot, joint in SENSOR_SLOT_TO_JOINT.items():
        if slot < sensor_mask.shape[1] and joint < joint_count:
            weights[:, joint] = torch.maximum(
                weights[:, joint], sensor_mask[:, slot].to(weights.dtype)
            )
    return weights


def head_only_mask(sensor_mask):
    """Return samples whose only trained active slot is head."""

    active = sensor_mask > 0.5
    has_head = active[:, 4] if active.shape[1] > 4 else active.new_zeros(active.shape[0])
    has_wrist = active[:, 0] if active.shape[1] > 0 else active.new_zeros(active.shape[0])
    if active.shape[1] > 1:
        has_wrist = has_wrist | active[:, 1]
    return has_head & ~has_wrist


def stage2_control_losses(
    predicted,
    target,
    frame_mask,
    joint_weights,
    head_only=None,
    text_anchor=None,
    *, trajectory_space="global_joint_proxy", derivative_fps=1.0,
):
    """Compute active sensor, upper-body, and excess-jerk losses."""

    if trajectory_space not in {"global_joint_proxy", "root_relative_joint_proxy"}:
        raise ValueError("Unknown trajectory space")
    trajectory_pred, trajectory_target = predicted, target
    if trajectory_space == "root_relative_joint_proxy":
        trajectory_pred = predicted - predicted[:, :, :1]
        trajectory_target = target - target[:, :, :1]
    trajectory = _masked_joint_mse(trajectory_pred, trajectory_target, frame_mask, joint_weights)
    velocity = _masked_joint_mse(
        predicted[:, 1:] - predicted[:, :-1],
        target[:, 1:] - target[:, :-1],
        frame_mask[:, 1:] & frame_mask[:, :-1],
        joint_weights,
    )
    velocity = velocity * derivative_fps**2
    jerk = _excess_jerk_loss(predicted, target, frame_mask) * derivative_fps**6
    if head_only is None:
        head_only = frame_mask.new_zeros((frame_mask.shape[0],), dtype=frame_mask.dtype)
    upper_body_weights = upper_body_joint_weights(
        head_only.to(predicted.device), joint_count=predicted.shape[2]
    )
    pred_rel = predicted - predicted[:, :, :1]
    target_rel = target - target[:, :, :1]
    upper_body = _masked_joint_mse(pred_rel, target_rel, frame_mask, upper_body_weights)
    upper_body_velocity = _masked_joint_mse(
        pred_rel[:, 1:] - pred_rel[:, :-1],
        target_rel[:, 1:] - target_rel[:, :-1],
        frame_mask[:, 1:] & frame_mask[:, :-1],
        upper_body_weights,
    )
    text_anchor_loss = predicted.new_zeros(())
    upper_body_text_anchor = predicted.new_zeros(())
    if text_anchor is not None:
        anchor_rel = text_anchor - text_anchor[:, :, :1]
        non_active_weights = non_active_joint_weights(joint_weights)
        text_anchor_loss = _masked_joint_mse(
            pred_rel,
            anchor_rel,
            frame_mask,
            non_active_weights,
        )
        upper_body_text_anchor = _masked_joint_mse(
            pred_rel,
            anchor_rel,
            frame_mask,
            upper_body_text_anchor_weights(joint_weights),
        )
    return {
        "trajectory_loss": trajectory,
        "velocity_loss": velocity,
        "jerk_loss": jerk,
        "upper_body_loss": upper_body,
        "upper_body_velocity_loss": upper_body_velocity * derivative_fps**2,
        "text_anchor_loss": text_anchor_loss,
        "upper_body_text_anchor_loss": upper_body_text_anchor,
    }


def upper_body_joint_weights(head_only, *, joint_count: int = 22):
    torch = require_torch()
    weights = torch.zeros(
        head_only.shape[0],
        joint_count,
        dtype=torch.float32,
        device=head_only.device,
    )
    for joint in UPPER_BODY_JOINTS:
        if joint < joint_count:
            weights[:, joint] = head_only.to(weights.dtype)
    return weights


def non_active_joint_weights(active_weights):
    weights = (active_weights <= 0.5).to(active_weights.dtype)
    if weights.shape[1] > 0:
        weights[:, 0] = 0.0
    return weights


def upper_body_text_anchor_weights(active_weights):
    torch = require_torch()
    weights = torch.zeros_like(active_weights)
    for joint in UPPER_BODY_JOINTS:
        if joint < weights.shape[1]:
            weights[:, joint] = 1.0
    weights = weights * (active_weights <= 0.5).to(weights.dtype)
    return weights


def _masked_joint_mse(predicted, target, frame_mask, joint_weights):
    weight = frame_mask[:, :, None].to(predicted.dtype) * joint_weights[:, None]
    squared = (predicted - target).pow(2).sum(dim=-1)
    denominator = weight.sum().clamp_min(1.0)
    return (squared * weight).sum() / denominator


def _excess_jerk_loss(predicted, target, frame_mask):
    torch = require_torch()
    if predicted.shape[1] < 4:
        return predicted.new_zeros(())
    pred_jerk = predicted[:, 3:] - 3 * predicted[:, 2:-1] + 3 * predicted[:, 1:-2] - predicted[:, :-3]
    target_jerk = target[:, 3:] - 3 * target[:, 2:-1] + 3 * target[:, 1:-2] - target[:, :-3]
    pred_norm = torch.linalg.norm(pred_jerk, dim=-1)
    target_norm = torch.linalg.norm(target_jerk, dim=-1)
    excess = torch.relu(pred_norm - target_norm)
    valid = (
        frame_mask[:, 3:]
        & frame_mask[:, 2:-1]
        & frame_mask[:, 1:-2]
        & frame_mask[:, :-3]
    )
    weight = valid[:, :, None].to(predicted.dtype)
    denominator = (weight.sum() * predicted.shape[2]).clamp_min(1.0)
    return (excess.pow(2) * weight).sum() / denominator


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


def _input_pipeline(args):
    if getattr(args, 'canonical_package', None):
        return CANONICAL_PIPELINE_VERSION
    return CAPTION_PIPELINE_VERSION if args.caption_plan else INPUT_PIPELINE_VERSION


def _save_checkpoint(
    path, torch, imu_encoder, controlled, optimizer, control_config, configs,
    mdm_checkpoint_path, normalization, history, epoch, args, input_provenance,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    torch.save(
        {
            "checkpoint_schema": "itm-control-v2",
            "input_pipeline_version": _input_pipeline(args),
            "contract_version": args.contract_version,
            "alignment_status": CANONICAL_ALIGNMENT if getattr(args, 'canonical_package', None) else "uncertified_diagnostic",
            "world_alignment_scope": CANONICAL_SCOPE if getattr(args, 'canonical_package', None) else "uncertified",
            "resume_semantics": "input_bound_weights_verified_normalization_fresh_optimizer",
            "training_input_provenance": input_provenance,
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


def _set_mdm_compatibility_defaults(options):
    options.setdefault("unconstrained", False)
    options.setdefault("text_encoder_type", "clip")
    options.setdefault("pos_embed_max_len", 5000)
    options.setdefault("mask_frames", False)


if __name__ == "__main__":
    raise SystemExit(main())
