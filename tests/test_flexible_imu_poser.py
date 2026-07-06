import importlib.util
import unittest

import numpy as np

from itm.metrics.motion_quality import foot_skating, jerk_ratio, root_relative_mpjpe
from itm.models.flexible_imu_poser import (
    FlexibleIMUPoserConfig,
    make_flexible_imu_poser,
    masked_pose_loss,
    r6d_to_rotation_matrix,
    rotation_matrix_to_r6d,
)


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
class FlexibleIMUPoserTest(unittest.TestCase):
    def test_forward_and_rotation_roundtrip(self):
        import torch

        config = FlexibleIMUPoserConfig(hidden_dim=16, num_layers=1)
        model = make_flexible_imu_poser(config)
        output = model(torch.zeros(2, 5, 60), torch.tensor([5, 3]))
        self.assertEqual(tuple(output.shape), (2, 5, 24, 6))
        identity = torch.eye(3).reshape(1, 3, 3)
        recovered = r6d_to_rotation_matrix(rotation_matrix_to_r6d(identity))
        torch.testing.assert_close(recovered, identity)

    def test_masked_temporal_loss_is_finite(self):
        import torch

        target = torch.randn(2, 5, 24, 6)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
        loss = masked_pose_loss(target + 0.1, target, mask)
        self.assertTrue(torch.isfinite(loss))

    def test_motion_metrics(self):
        joints = np.zeros((6, 24, 3), dtype=np.float32)
        translated = joints + np.asarray([2.0, 0.0, -3.0])
        self.assertAlmostEqual(root_relative_mpjpe(translated, joints), 0.0)
        self.assertAlmostEqual(jerk_ratio(joints, joints, fps=30), 0.0)
        self.assertAlmostEqual(foot_skating(joints, fps=30), 0.0)


if __name__ == "__main__":
    unittest.main()
