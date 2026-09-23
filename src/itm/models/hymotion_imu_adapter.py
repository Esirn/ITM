"""Trainable motion-only residual control for a frozen official HY-Motion DiT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from itm.models.torch_frame_baseline import require_torch
from itm.models.motionlab_imu_adapter import _sinusoidal_encoding
from itm.data.hymotion import BODY_PARENTS, SENSOR_JOINTS


def pad_hymotion_inputs(latent, imu, max_frames=360):
    """Keep motion/text RoPE positions identical to official padded inference."""
    torch = require_torch()
    if latent.ndim != 3 or imu.ndim != 4 or latent.shape[:2] != imu.shape[:2]:
        raise ValueError('Motion and IMU batch/time dimensions must match')
    b, frames = latent.shape[:2]
    if not 20 <= frames <= max_frames:
        raise ValueError('Motion length must be within 20..max_frames')
    mask = torch.arange(max_frames, device=latent.device)[None].expand(b, -1) < frames
    return (torch.nn.functional.pad(latent, (0, 0, 0, max_frames - frames)),
            torch.nn.functional.pad(imu, (0, 0, 0, 0, 0, max_frames - frames)), mask)


def place_hymotion_rotary_tables(backbone, device):
    """Place immutable official RoPE tables once; they are not registered buffers."""
    count = 0
    for module in backbone.modules():
        if module.__class__.__name__ == 'RotaryEmbedding' and hasattr(module, 'freqs_cis'):
            value = module.freqs_cis
            module.freqs_cis = tuple(v.to(device) for v in value) if isinstance(value, tuple) else value.to(device)
            count += 1
    return count


def control_residual_curvature_loss(residuals, valid):
    """Penalize frame-to-frame curvature of the residuals actually injected."""
    torch = require_torch()
    losses = []
    for residual in residuals:
        for value, mask in zip(residual, valid):
            frames = int(mask.sum().item())
            if frames >= 3:
                losses.append(torch.diff(value[:frames], n=2, dim=0).square().mean())
    if losses:
        return torch.stack(losses).mean()
    reference = residuals[0] if residuals else valid
    return reference.new_zeros((), dtype=torch.float32)


def sensor_chain_motion_mask(sensor_mask):
    """Select translation and local rotations on each active sensor's FK chain."""
    torch = require_torch()
    if sensor_mask.ndim != 2 or sensor_mask.shape[1] != len(SENSOR_JOINTS):
        raise ValueError("sensor_mask must have shape [B, 6]")
    result = torch.zeros(sensor_mask.shape[0], 201, dtype=torch.bool,
                         device=sensor_mask.device)
    result[:, :3] = True
    for batch, slot in torch.nonzero(sensor_mask.bool(), as_tuple=False).tolist():
        joint = SENSOR_JOINTS[slot]
        while joint >= 0:
            result[batch, 3 + joint * 6:3 + (joint + 1) * 6] = True
            joint = int(BODY_PARENTS[joint])
    return result


@dataclass(frozen=True)
class HYMotionIMUConfig:
    sensor_count: int = 6
    sensor_feature_dim: int = 12
    hidden_dim: int = 256
    output_dim: int = 1024
    encoder_heads: int = 8
    sensor_layers: int = 1
    temporal_layers: int = 2
    double_blocks: int = 6
    single_blocks: int = 12
    max_frames: int = 360
    dropout: float = 0.1
    imu_drop_probability: float = 0.1
    control_token_smoothing_passes: int = 0
    control_token_stride: int = 1
    control_injection_start: int = 0
    control_injection_end: int | None = None
    control_timestep_power: float = 0.0
    control_mode: str = "hidden"
    motion_dim: int = 201
    output_velocity_smoothing_passes: int = 0

    def to_dict(self):
        return asdict(self)


def make_hymotion_imu_adapter(config=HYMotionIMUConfig()):
    torch = require_torch()
    if config.hidden_dim % config.encoder_heads or config.hidden_dim % 2:
        raise ValueError("hidden_dim must be even and divisible by encoder_heads")
    if config.control_mode not in ("hidden", "output") or config.motion_dim < 1:
        raise ValueError("invalid control mode or motion dimension")
    if not 0 <= config.imu_drop_probability <= 1:
        raise ValueError("invalid IMU dropout probability")
    total_blocks = config.double_blocks + config.single_blocks
    if (config.control_token_smoothing_passes < 0 or config.control_token_stride < 1
            or not 0 <= config.output_velocity_smoothing_passes <= 16
            or not math.isfinite(config.control_timestep_power)
            or config.control_timestep_power < 0):
        raise ValueError("invalid control token smoothing configuration")
    injection_end = total_blocks if config.control_injection_end is None else config.control_injection_end
    if not 0 <= config.control_injection_start < injection_end <= total_blocks:
        raise ValueError("invalid control injection window")

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.feature_projection = torch.nn.Linear(config.sensor_feature_dim, config.hidden_dim)
            self.sensor_embedding = torch.nn.Embedding(config.sensor_count, config.hidden_dim)
            def transformer(layers):
                layer = torch.nn.TransformerEncoderLayer(
                    config.hidden_dim, config.encoder_heads, config.hidden_dim * 4,
                    dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
                )
                return torch.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.sensor_encoder = transformer(config.sensor_layers)
            self.temporal_encoder = transformer(config.temporal_layers)
            self.token_projection = torch.nn.Linear(config.hidden_dim, config.output_dim)
            if config.control_mode == "hidden":
                self.residuals = torch.nn.ModuleList([
                    torch.nn.Linear(config.output_dim, config.output_dim)
                    for _ in range(config.double_blocks + config.single_blocks)
                ])
            else:
                self.state_projection = torch.nn.Linear(config.motion_dim, config.output_dim)
                self.timestep_projection = torch.nn.Linear(1, config.output_dim)
                self.velocity_norm = torch.nn.LayerNorm(config.output_dim)
                self.velocity_projection = torch.nn.Linear(config.output_dim, config.motion_dim)
                self.residuals = torch.nn.ModuleList()
            for layer in self.residuals:
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
            if config.control_mode == "output":
                torch.nn.init.zeros_(self.velocity_projection.weight)
                torch.nn.init.zeros_(self.velocity_projection.bias)
            self.register_buffer("position_encoding", _sinusoidal_encoding(torch, config.max_frames, config.hidden_dim), persistent=False)

        def forward(self, imu, sensor_mask, frame_mask, keep=None):
            if imu.ndim != 4 or imu.shape[2:] != (config.sensor_count, config.sensor_feature_dim):
                raise ValueError("expected IMU [B,T,6,12]")
            b, t, s, _ = imu.shape
            if t < 1 or t > config.max_frames or sensor_mask.shape != (b, s) or frame_mask.shape != (b, t):
                raise ValueError("invalid frame length or mask shapes")
            sensor_mask, frame_mask = sensor_mask.bool(), frame_mask.bool()
            active = sensor_mask[:, None, :] & frame_mask[:, :, None]
            clean = torch.where(active[..., None], imu, 0)
            hidden = self.feature_projection(clean)
            hidden = hidden + self.sensor_embedding(torch.arange(s, device=imu.device))[None, None]
            hidden = torch.where(active[..., None], hidden, 0)
            flat_mask = active.reshape(b * t, s)
            # Attention needs one visible key even for fully absent conditions.
            safe_mask = flat_mask.clone()
            safe_mask[:, 0] |= ~safe_mask.any(-1)
            encoded = self.sensor_encoder(hidden.reshape(b * t, s, -1), src_key_padding_mask=~safe_mask)
            encoded = torch.where(flat_mask[..., None], encoded, 0)
            encoded = encoded.sum(1) / flat_mask.sum(1, keepdim=True).clamp_min(1)
            encoded = encoded.reshape(b, t, -1) + self.position_encoding[:t].to(encoded)[None]
            safe_frames = frame_mask.clone()
            safe_frames[:, 0] |= ~safe_frames.any(-1)
            encoded = self.temporal_encoder(encoded, src_key_padding_mask=~safe_frames)
            if keep is None:
                keep = torch.ones(b, dtype=torch.bool, device=imu.device)
                if self.training and config.imu_drop_probability:
                    keep = torch.rand(b, device=imu.device) >= config.imu_drop_probability
            if keep.shape != (b,):
                raise ValueError("keep must have shape [B]")
            valid = frame_mask & sensor_mask.any(-1)[:, None] & keep.bool()[:, None]
            tokens = self.token_projection(encoded)
            if config.control_token_stride > 1:
                resampled = tokens.clone()
                for index, mask in enumerate(valid):
                    frames = int(mask.sum().item())
                    if frames < 2:
                        continue
                    value = tokens[index:index + 1, :frames]
                    groups = (frames + config.control_token_stride - 1) // config.control_token_stride
                    pooled = torch.stack([
                        value[:, start:min(start + config.control_token_stride, frames)].mean(1)
                        for start in range(0, frames, config.control_token_stride)
                    ], dim=1)
                    restored = torch.nn.functional.interpolate(
                        pooled.transpose(1, 2), size=frames, mode="linear", align_corners=True
                    ).transpose(1, 2)
                    if pooled.shape[1] != groups:
                        raise RuntimeError("invalid token pooling")
                    resampled[index:index + 1, :frames] = restored
                tokens = resampled
            for _ in range(config.control_token_smoothing_passes):
                filtered = tokens.clone()
                for index, mask in enumerate(valid):
                    frames = int(mask.sum().item())
                    if frames < 3:
                        continue
                    value = tokens[index:index + 1, :frames]
                    padded = torch.cat((value[:, :1], value, value[:, -1:]), dim=1)
                    filtered[index:index + 1, :frames] = (padded[:, :-2] + 2 * padded[:, 1:-1] + padded[:, 2:]) / 4
                tokens = filtered
            return torch.where(valid[..., None], tokens, 0), valid

        def output_velocity(self, tokens, state, timesteps):
            if config.control_mode != "output":
                raise RuntimeError("output velocity is unavailable in hidden control mode")
            if (state.shape != tokens.shape[:2] + (config.motion_dim,)
                    or timesteps.shape != (state.shape[0],)):
                raise ValueError("invalid state or timestep dimensions for output control")
            hidden = (tokens + self.state_projection(state)
                      + self.timestep_projection(timesteps[:, None])[:, None])
            return self.velocity_projection(torch.nn.functional.gelu(self.velocity_norm(hidden)))

    return Adapter()


def control_hymotion(backbone, adapter):
    """Wrap an in-memory official DiT. No external source files are changed."""
    torch = require_torch()
    cfg = adapter.config
    injection_end = (cfg.double_blocks + cfg.single_blocks
                     if cfg.control_injection_end is None else cfg.control_injection_end)
    if (len(backbone.double_blocks), len(backbone.single_blocks), backbone.feat_dim) != (
        cfg.double_blocks, cfg.single_blocks, cfg.output_dim,
    ) or getattr(backbone, "insert_start_token", False):
        raise ValueError("backbone layout does not match the control configuration")
    if getattr(backbone, "_itm_imu_control_installed", False):
        raise ValueError("backbone already has IMU hooks")

    class Controlled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone.requires_grad_(False).eval()
            self.adapter = adapter
            self._condition = None
            self._injected_residuals = None
            self._last_curvature_loss = None
            self._handles = []
            for index, block in enumerate(backbone.double_blocks if cfg.control_mode == "hidden" else ()):
                def inject_double(module, args, kwargs, output, index=index):
                    if (self._condition is None or index < cfg.control_injection_start
                            or index >= injection_end):
                        return output
                    motion, text = output
                    return motion + self._residual(index, motion), text
                self._handles.append(block.register_forward_hook(inject_double, with_kwargs=True))
            single_blocks = backbone.single_blocks if cfg.control_mode == "hidden" else ()
            for index, block in enumerate(single_blocks, start=cfg.double_blocks):
                def inject_single(module, args, kwargs, output, index=index):
                    if (self._condition is None or index < cfg.control_injection_start
                            or index >= injection_end):
                        return output
                    split = kwargs["split_len"] if "split_len" in kwargs else args[1]
                    motion = output[:, :split]
                    return torch.cat((motion + self._residual(index, motion), output[:, split:]), dim=1)
                self._handles.append(block.register_forward_hook(inject_single, with_kwargs=True))
            backbone._itm_imu_control_installed = True

        def _residual(self, index, motion):
            tokens, valid, strength, timestep_gate = self._condition
            if motion.shape != tokens.shape:
                raise ValueError("control tokens and motion prefix have different shapes")
            value = self.adapter.residuals[index](tokens).to(motion)
            if timestep_gate is not None:
                value = value * timestep_gate[:, None, None].to(value)
            if self._injected_residuals is not None:
                self._injected_residuals.append(value)
            return torch.where(valid[..., None], value * strength, 0)

        def take_control_residual_curvature_loss(self):
            if self._last_curvature_loss is None:
                raise RuntimeError("controlled forward must run before reading curvature loss")
            value = self._last_curvature_loss
            self._last_curvature_loss = None
            return value

        def train(self, mode=True):
            super().train(mode)
            self.backbone.eval()
            return self

        def _smooth_velocity(self, output, frame_mask, strength):
            if strength == 0:
                return output
            motion_frames = frame_mask.shape[1]
            motion = output[:, :motion_frames]
            filtered = motion.clone()
            for index, valid in enumerate(frame_mask):
                frames = int(valid.sum().item())
                if frames < 3:
                    continue
                value = motion[index:index + 1, :frames]
                padded = torch.cat((value[:, :1], value, value[:, -1:]), dim=1)
                local_mean = (padded[:, :-2] + 2 * padded[:, 1:-1] + padded[:, 2:]) / 4
                filtered[index:index + 1, :frames] = torch.lerp(value, local_mean, strength)
            return torch.cat((filtered, output[:, motion_frames:]), dim=1)

        def _output_control(self, tokens, valid, strength, timestep_gate, kwargs):
            delta = self.adapter.output_velocity(tokens, kwargs["x"], kwargs["timesteps"])
            if timestep_gate is not None:
                delta = delta * timestep_gate[:, None, None].to(delta)
            delta = torch.where(valid[..., None], delta * strength, 0)
            for _ in range(cfg.output_velocity_smoothing_passes):
                delta = self._smooth_velocity(delta, kwargs["x_mask_temporal"], 1.0)
            return delta

        def forward(self, *, imu=None, sensor_mask=None, imu_keep=None, control_strength=1.0,
                    control_timestep_cutoff=1.0, temporal_velocity_smoothing=0.0,
                    control_delta_smoothing=0.0, control_delta_smoothing_passes=1,
                    control_delta_projection="all", **kwargs):
            if self._condition is not None:
                raise RuntimeError("nested controlled forward is unsupported")
            if not 0 <= control_strength <= 1:
                raise ValueError("control_strength must be within [0, 1]")
            if not 0 <= temporal_velocity_smoothing <= 1:
                raise ValueError("temporal_velocity_smoothing must be within [0, 1]")
            if not 0 <= control_delta_smoothing <= 1:
                raise ValueError("control_delta_smoothing must be within [0, 1]")
            if (not isinstance(control_delta_smoothing_passes, int)
                    or not 1 <= control_delta_smoothing_passes <= 16):
                raise ValueError("control_delta_smoothing_passes must be within [1, 16]")
            if control_delta_projection not in ("all", "sensor_chains"):
                raise ValueError("control_delta_projection must be all or sensor_chains")
            if control_delta_projection != "all" and not control_delta_smoothing:
                raise ValueError("control_delta_projection requires control_delta_smoothing")
            if not 0 <= control_timestep_cutoff <= 1:
                raise ValueError("control_timestep_cutoff must be within [0, 1]")
            if self.training and control_delta_smoothing:
                raise ValueError("control_delta_smoothing is inference-only")
            try:
                self._injected_residuals = [] if self.training else None
                if imu is not None:
                    tokens, valid = self.adapter(imu, sensor_mask, kwargs["x_mask_temporal"], keep=imu_keep)
                    timestep_gate = None
                    if cfg.control_timestep_power:
                        timesteps = kwargs.get("timesteps")
                        if timesteps is None or timesteps.shape != (tokens.shape[0],):
                            raise ValueError("timesteps must have shape [B] for timestep-gated control")
                        timestep_gate = (1 - timesteps).clamp(0, 1).pow(cfg.control_timestep_power)
                    if control_timestep_cutoff < 1:
                        timesteps = kwargs.get("timesteps")
                        if timesteps is None or timesteps.shape != (tokens.shape[0],):
                            raise ValueError("timesteps must have shape [B] for cutoff control")
                        cutoff_gate = (timesteps < control_timestep_cutoff).to(tokens)
                        timestep_gate = (cutoff_gate if timestep_gate is None
                                         else timestep_gate * cutoff_gate)
                    self._condition = tokens, valid, float(control_strength), timestep_gate
                if cfg.control_mode == "output":
                    output = self.backbone(**kwargs)
                    if self._condition is not None:
                        delta = self._output_control(*self._condition, kwargs)
                        if control_delta_smoothing:
                            for _ in range(control_delta_smoothing_passes):
                                delta = self._smooth_velocity(
                                    delta, kwargs["x_mask_temporal"],
                                    float(control_delta_smoothing))
                        if control_delta_projection == "sensor_chains":
                            delta = delta * sensor_chain_motion_mask(sensor_mask)[:, None].to(delta)
                        if self._injected_residuals is not None:
                            self._injected_residuals.append(delta)
                        output = output + delta
                else:
                    output = self.backbone(**kwargs)
                if (cfg.control_mode == "hidden" and self._condition is not None
                        and control_delta_smoothing):
                    condition = self._condition
                    self._condition = None
                    baseline = self.backbone(**kwargs)
                    delta = output - baseline
                    for _ in range(control_delta_smoothing_passes):
                        delta = self._smooth_velocity(
                            delta, kwargs["x_mask_temporal"],
                            float(control_delta_smoothing))
                    if control_delta_projection == "sensor_chains":
                        if delta.shape[-1] != 201:
                            raise ValueError("sensor-chain projection requires 201 motion channels")
                        delta = delta * sensor_chain_motion_mask(sensor_mask)[:, None].to(delta)
                    output = baseline + delta
                    self._condition = condition
                if self.training:
                    valid = (self._condition[1] if self._condition is not None
                             else torch.zeros_like(kwargs["x_mask_temporal"]))
                    self._last_curvature_loss = control_residual_curvature_loss(
                        self._injected_residuals, valid)
                return self._smooth_velocity(output, kwargs["x_mask_temporal"], float(temporal_velocity_smoothing))
            finally:
                self._condition = None
                self._injected_residuals = None

        def remove_hooks(self):
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self.backbone._itm_imu_control_installed = False

    return Controlled()
