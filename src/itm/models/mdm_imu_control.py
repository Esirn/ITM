"""ControlNet-style IMU adapters for a pretrained MDM Transformer encoder."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import math

from itm.models.torch_frame_baseline import require_torch


@dataclass(frozen=True)
class MDMIMUControlConfig:
    sensor_feature_dim: int = 12
    sensor_count: int = 6
    latent_dim: int = 512
    encoder_layers: int = 2
    encoder_heads: int = 8
    dropout: float = 0.1
    imu_drop_probability: float = 0.1
    # Missing fields in historical checkpoints retain their original semantics.
    encoder_variant: str = "legacy_mean"
    null_mode: str = "legacy_null"
    time_encoding: str = "none"
    model_fps: float = 20.0
    adapter_variant: str = "additive"

    def __post_init__(self):
        if self.adapter_variant not in {"additive", "film"}:
            raise ValueError("Unknown adapter_variant")
        if self.encoder_variant not in {"legacy_mean", "ordered_concat"}:
            raise ValueError("Unknown encoder_variant")
        if self.null_mode not in {"legacy_null", "strict"}:
            raise ValueError("Unknown null_mode")
        if self.time_encoding not in {"none", "physical_sinusoidal"}:
            raise ValueError("Unknown time_encoding")
        if not math.isfinite(self.model_fps) or self.model_fps <= 0:
            raise ValueError("model_fps must be positive")
        if not 0 <= self.imu_drop_probability <= 1:
            raise ValueError("Invalid IMU dropout probability")


def make_imu_control_encoder(config: MDMIMUControlConfig):
    torch = require_torch()

    class IMUControlEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            if config.encoder_variant == "legacy_mean":
                self.feature_projection = torch.nn.Linear(config.sensor_feature_dim, config.latent_dim)
                self.sensor_embedding = torch.nn.Embedding(config.sensor_count, config.latent_dim)
            else:
                self.fusion = torch.nn.Sequential(
                    torch.nn.Linear(config.sensor_count * (config.sensor_feature_dim + 1), config.latent_dim),
                    torch.nn.GELU(), torch.nn.Linear(config.latent_dim, config.latent_dim),
                )
            layer = torch.nn.TransformerEncoderLayer(
                d_model=config.latent_dim,
                nhead=config.encoder_heads,
                dim_feedforward=config.latent_dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal_encoder = torch.nn.TransformerEncoder(
                layer, config.encoder_layers
            )

        def forward(self, imu, sensor_mask, frame_mask=None, force_mask=False,
                    *, condition_present=None, timestamps=None, return_presence=False):
            if imu.ndim != 4 or imu.shape[2:] != (config.sensor_count, config.sensor_feature_dim):
                raise ValueError("imu must have shape (B,T,sensor_count,sensor_feature_dim)")
            batch, frames, sensors, _ = imu.shape
            if frames == 0 or sensor_mask.shape != (batch, sensors):
                raise ValueError("Nonempty frames and batch-aligned sensor_mask required")
            active = sensor_mask.to(device=imu.device, dtype=torch.bool)
            valid = (torch.ones((batch, frames), device=imu.device, dtype=torch.bool)
                     if frame_mask is None else frame_mask.to(device=imu.device, dtype=torch.bool))
            if valid.shape != (batch, frames):
                raise ValueError("frame_mask must have shape (B,T)")
            present = (active.any(-1) & valid.any(-1) if config.null_mode == "strict"
                       else torch.ones(batch, dtype=torch.bool, device=imu.device))
            if condition_present is not None:
                present = present & _batch_flag(condition_present, batch, imu.device)
            present = present & ~_batch_flag(force_mask, batch, imu.device)
            if self.training and config.imu_drop_probability > 0 and condition_present is None:
                present = present & (torch.rand(batch, device=imu.device) >= config.imu_drop_probability)
            if config.encoder_variant == "legacy_mean":
                ids = torch.arange(sensors, device=imu.device)
                features = (torch.where(active[:, None, :, None] & valid[:, :, None, None], imu, 0)
                            if config.null_mode == "strict" else imu)
                hidden = self.feature_projection(features) + self.sensor_embedding(ids)[None, None]
                weight = active[:, None, :, None].to(hidden.dtype)
                hidden = (hidden * weight).sum(2) / weight.sum(2).clamp_min(1.0)
            else:
                # where, rather than multiplication, also excludes NaNs in unused slots.
                clean = torch.where(active[:, None, :, None] & valid[:, :, None, None], imu, 0)
                features = torch.cat((clean.flatten(2), active[:, None].expand(-1, frames, -1).to(imu.dtype)), -1)
                hidden = self.fusion(features)
            if config.time_encoding == "physical_sinusoidal":
                seconds = (torch.arange(frames, device=imu.device, dtype=imu.dtype)[None].expand(batch, -1)
                           / config.model_fps if timestamps is None else timestamps.to(imu))
                if seconds.shape != (batch, frames) or not torch.isfinite(seconds[valid]).all():
                    raise ValueError("timestamps must be finite seconds with shape (B,T)")
                seconds = torch.where(valid, seconds, 0)
                freq = torch.exp(torch.arange(0, config.latent_dim, 2, device=imu.device,
                                             dtype=imu.dtype) * (-math.log(10000.0) / config.latent_dim))
                phase = seconds[..., None] * freq
                pos = torch.zeros_like(hidden)
                pos[..., 0::2] = phase.sin()
                pos[..., 1::2] = phase[..., :pos[..., 1::2].shape[-1]].cos()
                hidden = hidden + pos
            if config.null_mode == "strict" or config.encoder_variant != "legacy_mean":
                hidden = torch.where(valid[..., None], hidden, 0)
            # Attention over an entirely masked sequence is undefined. Temporarily
            # expose a zero token, then clear all invalid outputs below.
            safe_valid = valid.clone()
            safe_valid[~valid.any(-1), 0] = True
            encoded = self.temporal_encoder(hidden, src_key_padding_mask=~safe_valid)
            if config.null_mode == "strict":
                encoded = torch.where(valid[..., None] & present[:, None, None], encoded, 0)
            else:
                encoded = encoded * present[:, None, None].to(encoded.dtype)
            return (encoded, present) if return_presence else encoded

    return IMUControlEncoder()


def attach_zero_control_adapters(mdm_model, config: MDMIMUControlConfig):
    """Replace a loaded MDM TransformerEncoder while preserving its layers."""

    torch = require_torch()
    original = mdm_model.seqTransEncoder
    if not isinstance(original, torch.nn.TransformerEncoder):
        raise TypeError("Control adapters currently require MDM arch=trans_enc")
    controlled = _make_controlled_encoder(original, config.latent_dim, config.null_mode, config.adapter_variant)
    mdm_model.seqTransEncoder = controlled
    return controlled


def install_imu_control(mdm_model, config: MDMIMUControlConfig):
    """Install registered IMU modules and a hook compatible with MDM diffusion."""

    imu_encoder = make_imu_control_encoder(config)
    controlled = attach_zero_control_adapters(mdm_model, config)
    mdm_model.imu_control_encoder = imu_encoder

    def prepare_control(module, args, kwargs):
        y = kwargs.get("y")
        if y is None and len(args) >= 3:
            y = args[2]
        if y and y.get('_raw_backbone_bypass', False):
            controlled.set_control(None)
            return
        set_mdm_imu_control(
            module.seqTransEncoder,
            module.imu_control_encoder,
            y or {},
        )

    def clear_control(module, args, output):
        controlled.set_control(None)

    pre = mdm_model.register_forward_pre_hook(prepare_control, with_kwargs=True)
    post = mdm_model.register_forward_hook(clear_control, always_call=True)

    class Handles:
        def remove(self):
            pre.remove()
            post.remove()
            controlled.set_control(None)

    return controlled, imu_encoder, Handles()


def _make_controlled_encoder(original, latent_dim, null_mode, adapter_variant):
    torch = require_torch()

    class ControlledTransformerEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = original.layers
            self.norm = original.norm
            self.adapters = torch.nn.ModuleList(
                [torch.nn.Linear(latent_dim, latent_dim * (2 if adapter_variant == "film" else 1)) for _ in self.layers]
            )
            for adapter in self.adapters:
                torch.nn.init.zeros_(adapter.weight)
                torch.nn.init.zeros_(adapter.bias)
            self.control = None
            self.presence = None
            self.frame_mask = None
            self.null_mode = null_mode

        def set_control(self, control, presence=None, frame_mask=None):
            self.control, self.presence, self.frame_mask = control, presence, frame_mask

        def forward(self, src, mask=None, src_key_padding_mask=None, **kwargs):
            # Consume before validation/layers so exceptions cannot leave stale state.
            control, present, valid = self.control, self.presence, self.frame_mask
            self.set_control(None)
            output = src
            gate = None
            if control is not None:
                gate = torch.ones_like(control[..., :1], dtype=torch.bool)
                if present is not None:
                    gate = gate & present[None, :, None]
                if valid is not None:
                    if valid.shape != (control.shape[1], control.shape[0]):
                        raise ValueError("Cached control and frame mask lengths differ")
                    gate = gate & valid.transpose(0, 1)[..., None]
                if control.shape[0] == src.shape[0] - 1:
                    control = torch.cat((torch.zeros_like(control[:1]), control), dim=0)
                    # Strict mode never modifies the prepended text/timestep token.
                    gate = torch.cat((torch.zeros_like(gate[:1]), gate), dim=0)
                if control.shape != src.shape:
                    raise ValueError(f"Control shape {tuple(control.shape)} does not match MDM sequence {tuple(src.shape)}")
            for index, layer in enumerate(self.layers):
                output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, **kwargs)
                if control is not None:
                    residual = self.adapters[index](control)
                    if adapter_variant == "film":
                        scale, shift = residual.chunk(2, dim=-1)
                        residual = scale * output + shift
                    if self.null_mode == "strict":
                        residual = torch.where(gate, residual, 0)
                    output = output + residual
            return self.norm(output) if self.norm is not None else output

    return ControlledTransformerEncoder()


def set_mdm_imu_control(controlled_encoder, imu_encoder, y):
    """Encode ``y['imu']`` and arm the MDM encoder for its next forward call."""

    controlled_encoder.set_control(None)
    cached = y.get("imu_control")
    strict = controlled_encoder.null_mode == "strict"
    if cached is not None:
        torch = require_torch()
        batch = len(cached)
        if strict and "imu_condition_present" not in y:
            raise ValueError("Strict cached control requires imu_condition_present")
        present = _batch_flag(y.get("imu_condition_present", True), batch, cached.device)
        present = present & ~_batch_flag(y.get("imu_uncond", False), batch, cached.device)
        control = torch.where(present[:, None, None], cached, 0)
        valid = y.get("imu_frame_mask")
    elif "imu" in y and "sensor_mask" in y:
        valid = y.get("imu_frame_mask")
        control, present = imu_encoder(
            y["imu"], y["sensor_mask"], valid,
            force_mask=y.get("imu_uncond", False),
            condition_present=y.get("imu_condition_present"),
            timestamps=y.get("imu_timestamps"), return_presence=True,
        )
    else:
        return None
    controlled_encoder.set_control(control.transpose(0, 1), present, valid)
    return control


def _batch_flag(value, batch, device):
    torch = require_torch()
    flag = torch.as_tensor(value, dtype=torch.bool, device=device)
    if flag.ndim == 0:
        return flag.expand(batch)
    if flag.shape != (batch,):
        raise ValueError("Presence/unconditional flags must be scalar or shape (B,)")
    return flag


def configure_control_training(model):
    """Freeze and eval backbone; train only IMU encoder/adapters (including dropout)."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.imu_control_encoder, model.seqTransEncoder.adapters):
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad_(True)


@contextmanager
def evaluating_preserving_modes(model):
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


def compose_text_imu_guidance(unconditional, text_only, text_imu, *, text_scale, imu_scale):
    """Independent classifier-free guidance for text and incremental IMU control."""

    return (
        unconditional
        + text_scale * (text_only - unconditional)
        + imu_scale * (text_imu - text_only)
    )


def compose_factorized_text_imu_guidance(
    unconditional,
    text_only,
    imu_only,
    text_imu,
    *,
    text_scale,
    imu_scale,
    joint_scale,
):
    """Factorized guidance with explicit text, IMU, and interaction effects."""

    return (
        unconditional
        + text_scale * (text_only - unconditional)
        + imu_scale * (imu_only - unconditional)
        + joint_scale * (text_imu - text_only - imu_only + unconditional)
    )


def make_text_imu_guidance_model(model, *, mode="legacy", branch_execution="sequential", bypass_encoder=None):
    """Wrap MDM with independent text and incremental IMU guidance."""

    torch = require_torch()
    if mode not in {"legacy", "factorized"}:
        raise ValueError(f"Unknown guidance mode: {mode}")
    if branch_execution not in {"sequential", "batched"}:
        raise ValueError(f"Unknown branch execution mode: {branch_execution}")

    class TextIMUGuidanceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            for name in (
                "rot2xyz",
                "translation",
                "njoints",
                "nfeats",
                "data_rep",
                "cond_mode",
                "encode_text",
            ):
                setattr(self, name, getattr(model, name))

        def forward(self, x, timesteps, y=None):
            if y is None:
                raise ValueError("Text/IMU guidance requires a condition dictionary")
            if 'text_condition_present' in y:
                y = dict(y)
                text = y.get('text_embed')
                if text is None:
                    text = self.model.encode_text(y['text'])
                present = _batch_flag(y['text_condition_present'], len(x), x.device)
                y['text_embed'] = torch.where(present[None, :, None], text, 0)
            text_scale = y.get("text_scale", 1.0)
            imu_scale = y.get("imu_scale", 1.0)
            joint_scale = y.get("joint_scale", 1.0)
            def scale(value):
                value = torch.as_tensor(value, device=x.device, dtype=x.dtype)
                if value.ndim > 1 or value.numel() not in (1, len(x)):
                    raise ValueError("Guidance scale must be scalar or shape (B,)")
                return value.reshape(-1, *([1] * (x.ndim - 1)))

            text_scale, imu_scale, joint_scale = map(scale, (text_scale, imu_scale, joint_scale))

            unconditional_y = dict(y)
            unconditional_y["uncond"] = True
            unconditional_y["imu_uncond"] = True
            text_only_y = dict(y)
            text_only_y["uncond"] = False
            text_only_y["imu_uncond"] = True
            text_imu_y = dict(y)
            text_imu_y["uncond"] = False
            text_imu_y["imu_uncond"] = False
            branches = [unconditional_y, text_only_y]
            if mode == "factorized":
                imu_only_y = dict(y)
                imu_only_y["uncond"] = True
                imu_only_y["imu_uncond"] = False
                branches.append(imu_only_y)
            branches.append(text_imu_y)
            outputs = self._run_branches(x, timesteps, branches)
            unconditional, text_only = outputs[:2]
            text_imu = outputs[-1]
            if mode == "factorized":
                return compose_factorized_text_imu_guidance(
                    unconditional,
                    text_only,
                    outputs[2],
                    text_imu,
                    text_scale=text_scale,
                    imu_scale=imu_scale,
                    joint_scale=joint_scale,
                )
            return compose_text_imu_guidance(
                unconditional,
                text_only,
                text_imu,
                text_scale=text_scale,
                imu_scale=imu_scale,
            )

        def _run_branches(self, x, timesteps, branches):
            bypass = _batch_flag(branches[0].get('backbone_bypass', False), len(x), x.device)
            if bypass.any():
                if not bypass.all() or bypass_encoder is None:
                    raise ValueError('Raw backbone bypass requires its original encoder and a homogeneous clip batch')
                current = self.model.seqTransEncoder
                self.model.seqTransEncoder = bypass_encoder
                try:
                    return self._execute_branches(x, timesteps, [dict(b, _raw_backbone_bypass=True) for b in branches])
                finally:
                    self.model.seqTransEncoder = current
            return self._execute_branches(x, timesteps, branches)

        def _execute_branches(self, x, timesteps, branches):
            if branch_execution == "sequential":
                return [self.model(x, timesteps, branch) for branch in branches]
            return self._run_batched(x, timesteps, branches)

        def _run_batched(self, x, timesteps, branches):
            # Batched CFG uses already cached conditions so per-branch masking can
            # be represented as tensor values instead of scalar MDM flags.
            if "text_embed" not in branches[0] or "imu_control" not in branches[0]:
                raise ValueError(
                    "Batched guidance requires cached text_embed and imu_control"
                )
            conditioned = []
            for branch in branches:
                item = dict(branch)
                text = item["text_embed"]
                control = item["imu_control"]
                if item.pop("uncond", False):
                    text = torch.zeros_like(text)
                present = _batch_flag(item.get("imu_condition_present", True), len(x), x.device)
                if item.pop("imu_uncond", False):
                    control = torch.zeros_like(control)
                    present = torch.zeros_like(present)
                item["imu_condition_present"] = present
                item["text_embed"] = text
                item["imu_control"] = control
                conditioned.append(item)
            merged = _concatenate_condition_branches(conditioned, len(x), torch)
            output = self.model(
                torch.cat([x] * len(branches), dim=0),
                torch.cat([timesteps] * len(branches), dim=0),
                merged,
            )
            return output.chunk(len(branches), dim=0)

    return TextIMUGuidanceModel()


def _concatenate_condition_branches(branches, batch_size, torch):
    """Concatenate equivalent MDM condition dictionaries along batch axes."""

    merged = {}
    ignored = {
        "text_scale",
        "imu_scale",
        "joint_scale",
        "imu",
        "sensor_mask",
    }
    for key in branches[0]:
        if key in ignored:
            continue
        values = [branch[key] for branch in branches]
        first = values[0]
        if torch.is_tensor(first):
            # CLIP embeddings are [tokens, B, D]; all other cached conditions
            # used here are batch-first.
            dim = 1 if key == "text_embed" else 0
            merged[key] = torch.cat(values, dim=dim)
        elif isinstance(first, list):
            merged[key] = [item for value in values for item in value]
        else:
            merged[key] = first
    merged["uncond"] = False
    merged["imu_uncond"] = False
    return merged
