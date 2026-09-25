import importlib.util
from pathlib import Path

import numpy as np


def test_acceleration_overlap_keeps_both_full_stencils(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("prefix_rebuild_test", scripts / "check_prefix_cache_rebuild.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    old, new, valid = module.overlap_indices(12, 12, 4)
    np.testing.assert_array_equal(old, np.arange(4, 12))
    np.testing.assert_array_equal(new, np.arange(8))
    # New frames 0..2 and old frames 9..11 require unavailable stencil samples.
    np.testing.assert_array_equal(new[valid], [3, 4])
