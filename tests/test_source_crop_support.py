import importlib.util
from pathlib import Path

import numpy as np
import pytest

from itm.data.amass_bridge import HumanMLAMASSRecord, source_crop_at_fps


def test_source_boundaries_and_open_end():
    row = HumanMLAMASSRecord("fixture", Path("unused"), 10, 30)
    assert source_crop_at_fps(row, 60., source_frame_count=90) == slice(30, 90)
    with pytest.raises(ValueError, match="no silent clamping"):
        source_crop_at_fps(row, 60., source_frame_count=89)
    row = HumanMLAMASSRecord("fixture", Path("unused"), 10, -1)
    assert source_crop_at_fps(row, 60., source_frame_count=90) == slice(30, None)
    with pytest.raises(ValueError, match="no silent clamping"):
        source_crop_at_fps(row, 60., source_frame_count=30)


def test_cache_builder_rejects_short_source_before_body_model(tmp_path):
    path = tmp_path / "source.npz"
    np.savez(path, mocap_framerate=60., poses=np.zeros((6, 72)), trans=np.zeros((6, 3)), betas=np.zeros(10))
    row = HumanMLAMASSRecord("fixture", path, 0, 10)
    script_path = Path(__file__).resolve().parents[1] / "scripts/build_standard_imu_cache.py"
    spec = importlib.util.spec_from_file_location("cache_crop_test", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="no silent clamping"):
        module._process_source(row, None, None, None, 30., "cpu")
