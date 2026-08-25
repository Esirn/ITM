from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT = Path(__file__).parents[1] / "scripts" / "sample_mdm_imu_control.py"
SPEC = importlib.util.spec_from_file_location("sample_mdm_imu_control", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_fit_length_linear_preserves_endpoints():
    values = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
    result = MODULE._fit_length_linear(values, 5)
    assert result.shape == (5, 1)
    np.testing.assert_allclose(result[[0, -1], 0], [0.0, 2.0])


def test_fit_length_orientation_stays_on_so3():
    rotations = Rotation.from_euler("z", [0.0, 90.0], degrees=True).as_matrix()
    values = np.repeat(rotations[:, None], 6, axis=1).astype(np.float32)
    result = MODULE._fit_length_orientation(values, 5)
    assert result.shape == (5, 6, 3, 3)
    identity = np.swapaxes(result, -1, -2) @ result
    np.testing.assert_allclose(
        identity, np.broadcast_to(np.eye(3), identity.shape), atol=1e-5
    )
    np.testing.assert_allclose(np.linalg.det(result), 1.0, atol=1e-5)
