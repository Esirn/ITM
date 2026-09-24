"""Addressed DDPM random streams; delegates posterior math to the loaded MDM."""
from __future__ import annotations

import hashlib
import json
import numpy as np

from itm.data.random_plan import sampling_noise, VERSION
from itm.models.torch_frame_baseline import require_torch


def sample_addressed_ddpm(model, diffusion, *, y, lengths, case_ids, sample_seed, backbone_id):
    """Sample each clip at its own length (MDM legacy padding is not invariant).

    case_ids identify a noise group, shared across paired/shuffled/null/text edits.
    Returns normalized HumanML features plus per-clip complete noise provenance.
    No monkey-patching/global RNG reset and no auxiliary condition gradients.
    """
    torch = require_torch()
    if len(lengths) != len(case_ids) or not case_ids:
        raise ValueError("One random-plan case id per length required")
    if model.training:
        raise ValueError("Sampling requires eval mode")
    batch = len(lengths)
    frames = int(lengths.max())
    if int(lengths.min()) < 1 or frames > 196:
        raise ValueError("Supported clip length is 1..196; explicitly crop before sampling")
    outputs, plans = [], []
    with torch.no_grad():
        for index, case_id in enumerate(case_ids):
            length = int(lengths[index])
            item = slice_condition(y, index, length, batch)
            shape = (1, model.njoints, model.nfeats, length)
            addresses = []

            def draw(step):
                value, metadata = sampling_noise(shape, backbone_id=backbone_id, case_id=str(case_id),
                                                  sample_seed=int(sample_seed), step=step)
                addresses.append(metadata)
                return torch.from_numpy(value).to(device=lengths.device)

            x = draw(-1)
            for step in reversed(range(diffusion.num_timesteps)):
                t = torch.full((1,), step, device=x.device, dtype=torch.long)
                posterior = diffusion.p_mean_variance(model, x, t, clip_denoised=False, model_kwargs={"y": item})
                x = posterior["mean"]
                if step:
                    x = x + torch.exp(0.5 * posterior["log_variance"]) * draw(step)
                if not torch.isfinite(x).all():
                    raise FloatingPointError(f"Nonfinite DDPM sample at step {step}, case {case_id}")
            outputs.append(torch.nn.functional.pad(x, (0, frames - length)))
            payload = json.dumps(addresses, sort_keys=True, separators=(",", ":")).encode()
            plans.append({"random_plan_id": hashlib.sha256(payload).hexdigest(), "version": VERSION,
                          "sampler": "ddpm", "steps": diffusion.num_timesteps,
                          "clip_denoised": False, "noise_draws": addresses,
                          "timestep_map": list(getattr(diffusion, "timestep_map", range(diffusion.num_timesteps)))})
    return torch.cat(outputs), plans


def slice_condition(y, index, length, batch):
    torch = require_torch()
    result = {}
    for key, value in y.items():
        if torch.is_tensor(value) and value.ndim:
            axis = 1 if key == "text_embed" else 0
            if value.shape[axis] != batch:
                raise ValueError(f"Condition {key} has an invalid batch dimension")
            value = value[:, index:index + 1] if axis == 1 else value[index:index + 1]
            if key in {"imu", "imu_control", "imu_frame_mask", "imu_timestamps"}:
                value = value[:, :length]
            elif key == "mask":
                value = value[..., :length]
        elif isinstance(value, list):
            value = value[index:index + 1]
        result[key] = value
    return result
