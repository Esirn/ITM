"""Process-isolated MotionLab Text-to-Motion backbone adapter."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from itm.backbones.base import MotionConditions


@dataclass
class MotionLabBackbone:
    """Run MotionLab in its own conda environment.

    MotionLab and ITM intentionally use different dependency stacks. Keeping the
    official model in a subprocess avoids silently changing either environment.
    ``predict`` is unavailable because the process boundary exposes complete
    generation rather than individual flow-matching steps.
    """

    repo_root: Path
    checkpoint: Path
    conda_env: str = "rfmotion"
    device: str = "cuda:0"
    num_steps: int | None = None
    sampler_script: Path | None = None

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).expanduser().resolve()
        self.checkpoint = Path(self.checkpoint).expanduser().resolve()
        if self.sampler_script is None:
            self.sampler_script = (
                Path(__file__).resolve().parents[3] / "scripts" / "sample_motionlab_text.py"
            )

    def encode_text(self, text: list[str]) -> Any:
        raise NotImplementedError(
            "MotionLab text encoding is process-local; call sample() instead"
        )

    def predict(self, x_t: Any, timestep: Any, conditions: MotionConditions) -> Any:
        raise NotImplementedError(
            "MotionLab flow prediction is process-local; call sample() instead"
        )

    def sample(self, conditions: MotionConditions, *, seed: int) -> dict[str, Any]:
        texts = list(conditions.text)
        lengths = self._integer_lengths(conditions.lengths)
        if len(texts) != len(lengths):
            raise ValueError("text and lengths must contain the same number of samples")
        if not texts:
            raise ValueError("at least one text prompt is required")

        self._validate_assets()
        request = {"text": texts, "lengths": lengths, "seed": int(seed)}
        if "active_joints" in conditions.extra:
            request["active_joints"] = [
                int(value) for value in conditions.extra["active_joints"]
            ]
        with tempfile.TemporaryDirectory(prefix="itm-motionlab-") as directory:
            directory = Path(directory)
            request_path = directory / "request.json"
            output_path = directory / "result.npz"
            metadata_path = directory / "metadata.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")

            command = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                self.conda_env,
                "python",
                str(self.sampler_script),
                "--motionlab-root",
                str(self.repo_root),
                "--checkpoint",
                str(self.checkpoint),
                "--request",
                str(request_path),
                "--output",
                str(output_path),
                "--metadata",
                str(metadata_path),
                "--device",
                self.device,
            ]
            if self.num_steps is not None:
                command.extend(["--num-steps", str(self.num_steps)])
            if "trajectory_hints" in conditions.extra:
                command.extend(
                    ["--trajectory-hints", str(Path(conditions.extra["trajectory_hints"]).resolve())]
                )
            subprocess.run(command, check=True)

            arrays = np.load(output_path)
            return {
                "joints": arrays["joints"],
                "features": arrays["features"],
                "lengths": arrays["lengths"],
                "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
            }

    def decode_motion(self, motion: Any) -> Any:
        if isinstance(motion, dict) and "joints" in motion:
            return motion["joints"]
        return motion

    def _validate_assets(self) -> None:
        missing = [
            path
            for path in (self.repo_root, self.checkpoint, Path(self.sampler_script))
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError("missing MotionLab assets: " + ", ".join(map(str, missing)))

    @staticmethod
    def _integer_lengths(lengths: Any) -> list[int]:
        if hasattr(lengths, "detach"):
            lengths = lengths.detach().cpu().tolist()
        elif hasattr(lengths, "tolist"):
            lengths = lengths.tolist()
        return [int(value) for value in lengths]
