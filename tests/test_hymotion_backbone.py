from pathlib import Path

import numpy as np
import pytest

from itm.backbones import HYMotionBackbone, MotionConditions


def test_hymotion_lengths_and_decode(tmp_path):
    backbone = HYMotionBackbone(tmp_path, tmp_path)
    assert backbone._integer_lengths(np.asarray([60])) == [60]
    joints = np.zeros((1, 60, 52, 3), dtype=np.float32)
    assert backbone.decode_motion({"keypoints3d": joints}).shape == (1, 60, 22, 3)


def test_hymotion_initial_adapter_rejects_batches(tmp_path):
    backbone = HYMotionBackbone(tmp_path, tmp_path)
    conditions = MotionConditions(text=["walk", "run"], lengths=[60, 60])
    with pytest.raises(ValueError, match="exactly one"):
        backbone.sample(conditions, seed=1)
