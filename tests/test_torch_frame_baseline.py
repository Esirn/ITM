import importlib.util
import unittest

import numpy as np

from itm.baselines.linear_reconstruct import LinearBaselineConfig
from itm.models.torch_frame_baseline import build_frame_arrays


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
class TorchFrameBaselineTest(unittest.TestCase):
    def test_build_frame_arrays(self):
        config = LinearBaselineConfig(text_dim=8)
        samples = [
            {
                "motion_id": "000001",
                "caption": "walk forward",
                "motion": np.zeros((4, 3), dtype=np.float32),
                "imu_acceleration": np.zeros((2, 2, 3), dtype=np.float32),
                "imu_orientation": np.zeros((4, 2, 3), dtype=np.float32),
            },
            {
                "motion_id": "000002",
                "caption": "turn left",
                "motion": np.ones((5, 3), dtype=np.float32),
                "imu_acceleration": np.zeros((3, 2, 3), dtype=np.float32),
                "imu_orientation": np.zeros((5, 2, 3), dtype=np.float32),
            },
        ]
        x, y = build_frame_arrays(samples, config)
        self.assertEqual(x.shape[0], 9)
        self.assertEqual(y.shape, (9, 3))


if __name__ == "__main__":
    unittest.main()
