import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "export_hymotion_virtual_imu.py"
SPEC = importlib.util.spec_from_file_location("export_hymotion_virtual_imu", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_constant_velocity_has_zero_acceleration():
    time = np.arange(6, dtype=np.float32)
    joints = np.stack((time, 2 * time, -time), axis=-1)[:, None, :]
    acceleration = MODULE.joint_acceleration(joints, fps=30.0)
    np.testing.assert_allclose(acceleration, 0.0, atol=1e-5)
