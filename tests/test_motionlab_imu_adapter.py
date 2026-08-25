import unittest

from itm.models.motionlab_imu_adapter import (
    MotionLabIMUAdapterConfig,
    active_joint_mask,
    make_motionlab_imu_adapter,
    masked_trajectory_losses,
)
from itm.models.torch_frame_baseline import require_torch


class MotionLabIMUAdapterTest(unittest.TestCase):
    def setUp(self):
        self.torch = require_torch()

    def test_shapes_padding_and_active_joint_mapping(self):
        torch = self.torch
        config = MotionLabIMUAdapterConfig(hidden_dim=32, encoder_heads=4, encoder_layers=1)
        model = make_motionlab_imu_adapter(config).eval()
        imu = torch.randn(2, 8, 6, 12)
        sensors = torch.tensor(
            [[False, False, False, False, True, False], [True, True, False, False, False, False]]
        )
        frames = torch.tensor([[True] * 8, [True] * 5 + [False] * 3])
        output = model(imu, sensors, frames)
        self.assertEqual(tuple(output.shape), (2, 8, 66))
        self.assertTrue(torch.equal(output[1, 5:], torch.zeros_like(output[1, 5:])))
        joints = active_joint_mask(sensors)
        self.assertEqual(torch.where(joints[0, :, 0])[0].tolist(), [15])
        self.assertEqual(torch.where(joints[1, :, 0])[0].tolist(), [20, 21])

    def test_masked_losses_are_zero_for_matching_active_trajectories(self):
        torch = self.torch
        target = torch.randn(2, 6, 66)
        predicted = target.clone()
        frames = torch.ones(2, 6, dtype=torch.bool)
        sensors = torch.tensor(
            [[False, False, False, False, True, False], [True, True, False, False, False, False]]
        )
        trajectory, velocity = masked_trajectory_losses(
            predicted, target, frames, active_joint_mask(sensors)
        )
        self.assertEqual(float(trajectory), 0.0)
        self.assertEqual(float(velocity), 0.0)


if __name__ == "__main__":
    unittest.main()
