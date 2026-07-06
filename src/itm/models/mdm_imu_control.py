"""ControlNet-style IMU adapters for a pretrained MDM Transformer encoder."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

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


def make_imu_control_encoder(config: MDMIMUControlConfig):
    torch = require_torch()

    class IMUControlEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_projection = torch.nn.Linear(
                config.sensor_feature_dim, config.latent_dim
            )
            self.sensor_embedding = torch.nn.Embedding(
                config.sensor_count, config.latent_dim
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

        def forward(self, imu, sensor_mask, frame_mask=None, force_mask=False):
            if imu.ndim != 4:
                raise ValueError("imu must have shape (B, T, S, 12)")
            batch, frames, sensors, _ = imu.shape
            if sensors != config.sensor_count:
                raise ValueError(
                    f"Expected {config.sensor_count} sensor slots, got {sensors}"
                )
            ids = torch.arange(sensors, device=imu.device)
            hidden = self.feature_projection(imu)
            hidden = hidden + self.sensor_embedding(ids)[None, None]
            active = sensor_mask[:, None, :, None].to(hidden.dtype)
            hidden = (hidden * active).sum(2) / active.sum(2).clamp_min(1.0)
            keep = None
            if force_mask:
                keep = hidden.new_zeros((batch, 1, 1))
            elif self.training and config.imu_drop_probability > 0:
                keep = torch.rand(batch, 1, 1, device=imu.device)
                keep = (keep >= config.imu_drop_probability).to(hidden.dtype)
            encoded = self.temporal_encoder(
                hidden,
                src_key_padding_mask=(~frame_mask if frame_mask is not None else None),
            )
            return encoded if keep is None else encoded * keep

    return IMUControlEncoder()


def attach_zero_control_adapters(mdm_model, config: MDMIMUControlConfig):
    """Replace a loaded MDM TransformerEncoder while preserving its layers."""

    torch = require_torch()
    original = mdm_model.seqTransEncoder
    if not isinstance(original, torch.nn.TransformerEncoder):
        raise TypeError("Control adapters currently require MDM arch=trans_enc")
    controlled = _make_controlled_encoder(original, config.latent_dim)
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
        set_mdm_imu_control(
            module.seqTransEncoder,
            module.imu_control_encoder,
            y or {},
        )

    handle = mdm_model.register_forward_pre_hook(prepare_control, with_kwargs=True)
    return controlled, imu_encoder, handle


def _make_controlled_encoder(original, latent_dim):
    torch = require_torch()

    class ControlledTransformerEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = original.layers
            self.norm = original.norm
            self.adapters = torch.nn.ModuleList(
                [torch.nn.Linear(latent_dim, latent_dim) for _ in self.layers]
            )
            for adapter in self.adapters:
                torch.nn.init.zeros_(adapter.weight)
                torch.nn.init.zeros_(adapter.bias)
            self.control = None

        def set_control(self, control):
            self.control = control

        def forward(self, src, mask=None, src_key_padding_mask=None, **kwargs):
            output = src
            control = self.control
            if control is not None:
                if control.shape[0] == src.shape[0] - 1:
                    control = torch.cat((torch.zeros_like(control[:1]), control), dim=0)
                if control.shape != src.shape:
                    raise ValueError(
                        f"Control shape {tuple(control.shape)} does not match MDM sequence {tuple(src.shape)}"
                    )
            for index, layer in enumerate(self.layers):
                output = layer(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    **kwargs,
                )
                if control is not None:
                    output = output + self.adapters[index](control)
            self.control = None
            return self.norm(output) if self.norm is not None else output

    return ControlledTransformerEncoder()


def set_mdm_imu_control(controlled_encoder, imu_encoder, y):
    """Encode ``y['imu']`` and arm the MDM encoder for its next forward call."""

    if "imu" not in y or "sensor_mask" not in y:
        controlled_encoder.set_control(None)
        return None
    frame_mask = y.get("imu_frame_mask")
    control = imu_encoder(
        y["imu"],
        y["sensor_mask"],
        frame_mask,
        force_mask=bool(y.get("imu_uncond", False)),
    )
    controlled_encoder.set_control(control.transpose(0, 1))
    return control


def compose_text_imu_guidance(unconditional, text_only, text_imu, *, text_scale, imu_scale):
    """Independent classifier-free guidance for text and incremental IMU control."""

    return (
        unconditional
        + text_scale * (text_only - unconditional)
        + imu_scale * (text_imu - text_only)
    )


def make_text_imu_guidance_model(model):
    """Wrap MDM with independent text and incremental IMU guidance."""

    torch = require_torch()

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
            text_scale = y.get("text_scale", 1.0)
            imu_scale = y.get("imu_scale", 1.0)
            if not torch.is_tensor(text_scale):
                text_scale = torch.full(
                    (len(x),), float(text_scale), device=x.device, dtype=x.dtype
                )
            if not torch.is_tensor(imu_scale):
                imu_scale = torch.full(
                    (len(x),), float(imu_scale), device=x.device, dtype=x.dtype
                )
            text_scale = text_scale.view(-1, 1, 1, 1)
            imu_scale = imu_scale.view(-1, 1, 1, 1)

            unconditional_y = deepcopy(y)
            unconditional_y["uncond"] = True
            unconditional_y["imu_uncond"] = True
            text_only_y = deepcopy(y)
            text_only_y["uncond"] = False
            text_only_y["imu_uncond"] = True
            text_imu_y = deepcopy(y)
            text_imu_y["uncond"] = False
            text_imu_y["imu_uncond"] = False
            unconditional = self.model(x, timesteps, unconditional_y)
            text_only = self.model(x, timesteps, text_only_y)
            text_imu = self.model(x, timesteps, text_imu_y)
            return compose_text_imu_guidance(
                unconditional,
                text_only,
                text_imu,
                text_scale=text_scale,
                imu_scale=imu_scale,
            )

    return TextIMUGuidanceModel()
