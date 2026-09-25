import importlib.util
from pathlib import Path

import numpy as np
import pytest


def helper(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("prefix_audit_test", scripts / "audit_humanml_prefix.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_common_frames_are_identical_for_all_three_clocks(monkeypatch):
    module = helper(monkeypatch)
    clocks, common, metadata = module.comparison_clocks(path="AMASS/MPI_HDM05/x.npz", fps=60.,
        source_frames=900, start=0, end=100, feature_frames=99, cache_times=np.arange(150) / 30)
    np.testing.assert_allclose(clocks["nominal_20hz"], clocks["stride_only"])
    np.testing.assert_allclose(clocks["stride_and_prefix"] - clocks["stride_only"], 3.)
    np.testing.assert_array_equal(np.flatnonzero(common), np.arange(40))
    assert metadata["stride_and_prefix"]["dataset_prefix_trim_nominal_seconds"] == 3.


def test_frame_mismatch_is_not_silently_cropped(monkeypatch):
    module = helper(monkeypatch)
    with pytest.raises(ValueError, match="predicted feature length"):
        module.comparison_clocks(path="MPI_HDM05/x.npz", fps=60., source_frames=210,
            start=0, end=100, feature_frames=99, cache_times=np.arange(100) / 30)
