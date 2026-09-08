"""Process-isolated HY-Motion Text-to-Motion backbone adapter."""

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
class HYMotionBackbone:
    """Run the clean official HY-Motion sampler in an isolated environment."""

    repo_root: Path
    runtime_root: Path
    model: str = "lite"
    conda_env: str = "hymotion"
    device: str = "cuda:0"
    cfg_scale: float = 5.0
    sampler_script: Path | None = None

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).expanduser().resolve()
        self.runtime_root = Path(self.runtime_root).expanduser().resolve()
        if self.sampler_script is None:
            self.sampler_script = (
                Path(__file__).resolve().parents[3] / "scripts" / "sample_hymotion_text.py"
            )

    def encode_text(self, text: list[str]) -> Any:
        raise NotImplementedError("HY-Motion text encoding is process-local; call sample()")

    def predict(self, x_t: Any, timestep: Any, conditions: MotionConditions) -> Any:
        raise NotImplementedError("HY-Motion flow prediction is process-local; call sample()")

    def sample(self, conditions: MotionConditions, *, seed: int) -> dict[str, Any]:
        texts = list(conditions.text)
        lengths = self._integer_lengths(conditions.lengths)
        if len(texts) != 1 or len(lengths) != 1:
            raise ValueError("the initial HY-Motion adapter accepts exactly one sample")
        if not 20 <= lengths[0] <= 360:
            raise ValueError("HY-Motion length must be within 20..360 frames at 30 FPS")
        self._validate_assets()
        with tempfile.TemporaryDirectory(prefix="itm-hymotion-") as directory:
            output = Path(directory) / "result.npz"
            command = [
                "conda", "run", "--no-capture-output", "-n", self.conda_env,
                "python", str(self.sampler_script),
                "--hymotion-root", str(self.repo_root),
                "--runtime-root", str(self.runtime_root),
                "--model", self.model,
                "--text", texts[0],
                "--frames", str(lengths[0]),
                "--seed", str(int(seed)),
                "--cfg-scale", str(self.cfg_scale),
                "--device", self.device,
                "--output", str(output),
            ]
            subprocess.run(command, check=True)
            with np.load(output) as arrays:
                return {
                    key: arrays[key].copy()
                    for key in (
                        "latent_denorm", "keypoints3d", "rot6d", "transl",
                        "root_rotations_mat", "global_rotations_mat",
                    )
                } | {"metadata": json.loads(arrays["metadata"].item())}

    def decode_motion(self, motion: Any) -> Any:
        if isinstance(motion, dict) and "keypoints3d" in motion:
            return motion["keypoints3d"][:, :, :22]
        return motion

    def _validate_assets(self) -> None:
        paths = (self.repo_root, self.runtime_root / "workspace.json", Path(self.sampler_script))
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError("missing HY-Motion assets: " + ", ".join(map(str, missing)))

    @staticmethod
    def _integer_lengths(lengths: Any) -> list[int]:
        if hasattr(lengths, "detach"):
            lengths = lengths.detach().cpu().tolist()
        elif hasattr(lengths, "tolist"):
            lengths = lengths.tolist()
        return [int(value) for value in lengths]
