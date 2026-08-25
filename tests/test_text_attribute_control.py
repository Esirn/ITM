from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_text_attribute_control.py"
SPEC = importlib.util.spec_from_file_location("text_attribute_control", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def skeleton(frames=20):
    joints = np.zeros((frames, 22, 3), dtype=np.float32)
    joints[:, 1, 0] = -0.1
    joints[:, 2, 0] = 0.1
    return joints


def test_path_speed_uses_traveled_path():
    joints = skeleton()
    joints[:, 0, 2] = np.linspace(0, 1, len(joints))
    np.testing.assert_allclose(MODULE.path_speed(joints), 20 / 19, rtol=1e-5)


def test_turning_amount_detects_body_yaw():
    joints = skeleton()
    angles = np.linspace(0, np.pi / 2, len(joints))
    joints[:, 1, 0] = -0.1 * np.cos(angles)
    joints[:, 1, 2] = -0.1 * np.sin(angles)
    joints[:, 2, 0] = 0.1 * np.cos(angles)
    joints[:, 2, 2] = 0.1 * np.sin(angles)
    np.testing.assert_allclose(MODULE.turning_amount(joints), np.pi / 2, rtol=1e-5)
    net, efficiency = MODULE.turning_net_and_efficiency(joints)
    np.testing.assert_allclose(net, np.pi / 2, rtol=1e-5)
    np.testing.assert_allclose(efficiency, 1.0, rtol=1e-5)


def test_stride_span_increases_with_foot_excursion():
    small = skeleton()
    large = skeleton()
    phase = np.sin(np.linspace(0, 2 * np.pi, len(small)))
    small[:, 10, 2] = 0.1 * phase
    small[:, 11, 2] = -0.1 * phase
    large[:, 10, 2] = 0.4 * phase
    large[:, 11, 2] = -0.4 * phase
    assert MODULE.stride_span(large) > MODULE.stride_span(small)
    assert MODULE.foot_separation(large) > MODULE.foot_separation(small)
