from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_motionlab_control_statistics.py"
SPEC = importlib.util.spec_from_file_location("motionlab_control_statistics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_bootstrap_mean_ci_contains_constant():
    interval = MODULE.bootstrap_mean_ci(
        np.ones(8), rng=np.random.default_rng(1), samples=100
    )
    assert interval == (1.0, 1.0)


def test_sign_flip_detects_consistent_difference():
    pvalue = MODULE.sign_flip_pvalue(
        np.full(20, -0.1), rng=np.random.default_rng(1), permutations=5_000
    )
    assert pvalue < 0.01


def test_summary_keeps_paired_comparisons_separate():
    records = []
    for index in range(10):
        config = {
            "active_error_m": {
                "text": 0.3,
                "paired": 0.2,
                "zero": 0.25,
                "shuffled": 0.35,
            },
            "text_jerk_ratio": 2.0,
            "paired_jerk_ratio": 1.5,
        }
        records.append({"configs": {"head": config, "wrists": config}})
    summary = MODULE.summarize_config(
        records,
        "head",
        rng=np.random.default_rng(2),
        bootstrap_samples=100,
        permutations=100,
    )
    assert summary["active_error_comparisons"]["text"]["improved_count"] == 10
    np.testing.assert_allclose(
        summary["active_error_comparisons"]["zero"][
            "paired_minus_reference_mean_m"
        ],
        -0.05,
    )
