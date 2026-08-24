from pathlib import Path

import numpy as np
import pytest

from itm.backbones.base import MotionConditions
from itm.backbones.motionlab import MotionLabBackbone


def test_integer_lengths_accepts_numpy_and_lists():
    assert MotionLabBackbone._integer_lengths(np.asarray([40, 80])) == [40, 80]
    assert MotionLabBackbone._integer_lengths([12.0, 24]) == [12, 24]


def test_sample_rejects_mismatched_conditions(tmp_path: Path):
    adapter = MotionLabBackbone(tmp_path, tmp_path / "model.ckpt")
    with pytest.raises(ValueError, match="same number"):
        adapter.sample(MotionConditions(["walk"], np.asarray([40, 50])), seed=1)


def test_decode_motion_extracts_joint_array(tmp_path: Path):
    adapter = MotionLabBackbone(tmp_path, tmp_path / "model.ckpt")
    joints = np.zeros((1, 40, 22, 3), dtype=np.float32)
    assert adapter.decode_motion({"joints": joints}) is joints


def test_integer_lengths_rejects_no_values_only_at_sample_boundary(tmp_path: Path):
    adapter = MotionLabBackbone(tmp_path, tmp_path / "model.ckpt")
    with pytest.raises(ValueError, match="at least one"):
        adapter.sample(MotionConditions([], np.asarray([])), seed=1)
