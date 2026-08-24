"""MDM implementation of the common Text-to-Motion backbone interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from itm.backbones.base import MotionConditions
from itm.models.torch_frame_baseline import require_torch


@dataclass
class MDMBackbone:
    """Thin adapter around an already loaded MDM model and diffusion object."""

    model: Any
    diffusion: Any
    mean: Any
    std: Any
    recover_from_ric: Callable[[Any, int], Any]
    guidance_model: Any | None = None

    def encode_text(self, text: list[str]):
        return self.model.encode_text(text)

    def predict(self, x_t, timestep, conditions: MotionConditions):
        target = self.guidance_model or self.model
        return target(x_t, timestep, y=self._model_kwargs(conditions))

    def sample(self, conditions: MotionConditions, *, seed: int):
        torch = require_torch()
        y = self._model_kwargs(conditions)
        lengths = conditions.lengths
        batch = len(conditions.text)
        frames = int(lengths.max().item())
        generator = torch.Generator(device=lengths.device).manual_seed(seed)
        base_noise = torch.randn(
            1,
            self.model.njoints,
            self.model.nfeats,
            frames,
            generator=generator,
            device=lengths.device,
        )
        return self.diffusion.p_sample_loop(
            self.guidance_model or self.model,
            (batch, self.model.njoints, self.model.nfeats, frames),
            clip_denoised=False,
            model_kwargs={"y": y},
            progress=True,
            noise=base_noise.repeat(batch, 1, 1, 1),
            const_noise=False,
        )

    def decode_motion(self, motion):
        batch, _, _, frames = motion.shape
        vectors = motion.permute(0, 2, 3, 1)
        vectors = vectors * self.std.to(vectors) + self.mean.to(vectors)
        return self.recover_from_ric(vectors.float(), 22).view(batch, frames, 22, 3)

    @staticmethod
    def frame_mask(lengths, frames):
        torch = require_torch()
        indices = torch.arange(frames, device=lengths.device)
        return indices[None] < lengths[:, None]

    def cache_conditions(self, conditions: MotionConditions, imu_encoder=None):
        """Cache timestep-invariant CLIP and IMU encoder outputs in-place."""

        torch = require_torch()
        extra = conditions.extra
        with torch.inference_mode():
            extra.setdefault("text_embed", self.encode_text(conditions.text))
            if imu_encoder is not None and "imu_control" not in extra:
                extra["imu_control"] = imu_encoder(
                    extra["imu"],
                    extra["sensor_mask"],
                    extra.get("imu_frame_mask"),
                    force_mask=False,
                )
        return conditions

    def _model_kwargs(self, conditions: MotionConditions):
        lengths = conditions.lengths
        frames = max(
            int(lengths.max().item()),
            int(conditions.extra.get("imu", ()).shape[1]) if "imu" in conditions.extra else 0,
        )
        y = dict(conditions.extra)
        y.update(
            {
                "text": conditions.text,
                "lengths": lengths,
                "mask": self.frame_mask(lengths, frames)[:, None, None],
            }
        )
        return y
