import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_mdm_multiseed_control.py"
SPEC = importlib.util.spec_from_file_location("prepare_mdm_multiseed_control", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_stratified_selection_and_derangement():
    subset = {
        "motion_ids": [str(index) for index in range(9)],
        "captions": ["x"] * 9,
        "lengths": [40, 50, 60, 90, 100, 110, 150, 160, 170],
    }
    selected = MODULE.choose_stratified(subset, 2, 7)
    lengths = [subset["lengths"][index] for index in selected]
    assert sum(length <= 80 for length in lengths) == 2
    assert sum(80 < length <= 140 for length in lengths) == 2
    assert sum(length > 140 for length in lengths) == 2
    permutation = MODULE.derangement(20, 9)
    assert not np.any(permutation == np.arange(20))
