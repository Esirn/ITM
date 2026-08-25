import unittest

from itm.models.motionlab_imu_adapter import (
    MotionLabIMUAdapterConfig,
    active_joint_mask,
    make_motionlab_imu_adapter,
    masked_trajectory_losses,
    factorized_guidance,
    grouped_contrastive_matching_loss,
    paired_control_ranking_loss,
    same_group_derangement,
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

    def test_concat_fusion_preserves_output_contract(self):
        torch = self.torch
        config = MotionLabIMUAdapterConfig(
            hidden_dim=32, encoder_heads=4, encoder_layers=1, sensor_fusion="concat"
        )
        model = make_motionlab_imu_adapter(config).eval()
        imu = torch.randn(1, 5, 6, 12)
        sensors = torch.tensor([[True, True, False, False, False, False]])
        frames = torch.ones(1, 5, dtype=torch.bool)
        self.assertEqual(tuple(model(imu, sensors, frames).shape), (1, 5, 66))

    def test_preembedded_control_dimension(self):
        torch = self.torch
        config = MotionLabIMUAdapterConfig(
            hidden_dim=32, encoder_heads=4, encoder_layers=1, output_dim=512
        )
        model = make_motionlab_imu_adapter(config).eval()
        output = model(
            torch.randn(1, 5, 6, 12),
            torch.tensor([[False, False, False, False, True, False]]),
            torch.ones(1, 5, dtype=torch.bool),
        )
        self.assertEqual(tuple(output.shape), (1, 5, 512))

    def test_factorized_guidance_separates_main_and_interaction_effects(self):
        torch = self.torch
        f00 = torch.tensor(1.0)
        f10 = torch.tensor(3.0)
        f01 = torch.tensor(4.0)
        f11 = torch.tensor(8.0)
        result = factorized_guidance(f00, f10, f01, f11, 2.0, 0.5, 1.5)
        self.assertAlmostEqual(float(result), 9.5)

    def test_factorized_text_only_disables_imu_and_interaction(self):
        torch = self.torch
        f00 = torch.tensor(1.0)
        f10 = torch.tensor(3.0)
        result = factorized_guidance(
            f00, f10, torch.tensor(100.0), torch.tensor(200.0),
            text_scale=2.5, imu_scale=0.0, joint_scale=0.0,
        )
        self.assertAlmostEqual(float(result), 6.0)

    def test_ranking_loss_rewards_paired_control(self):
        torch = self.torch
        self.assertEqual(float(paired_control_ranking_loss(torch.tensor(0.1), torch.tensor(0.2))), 0.0)
        self.assertGreater(float(paired_control_ranking_loss(torch.tensor(0.2), torch.tensor(0.1))), 0.0)

    def test_attention_fusion_preserves_shape_and_padding(self):
        torch = self.torch
        config = MotionLabIMUAdapterConfig(
            hidden_dim=32, encoder_heads=4, encoder_layers=1,
            output_dim=512, sensor_fusion="attention",
        )
        model = make_motionlab_imu_adapter(config).eval()
        frames = torch.tensor([[True] * 4 + [False] * 2])
        output = model(
            torch.randn(1, 6, 6, 12),
            torch.tensor([[True, True, False, False, False, False]]),
            frames,
        )
        self.assertEqual(tuple(output.shape), (1, 6, 512))
        self.assertTrue(torch.equal(output[:, 4:], torch.zeros_like(output[:, 4:])))

    def test_same_group_derangement_never_crosses_sensor_config(self):
        permutation, valid = same_group_derangement(["head", "wrists", "head", "single"])
        self.assertEqual(permutation.tolist(), [2, 1, 0, 3])
        self.assertEqual(valid.tolist(), [True, False, True, False])

    def test_grouped_contrastive_loss_uses_only_same_config_negatives(self):
        torch = self.torch
        embeddings = torch.eye(4)
        loss, correct, compared = grouped_contrastive_matching_loss(
            embeddings,
            embeddings,
            ["head", "head", "wrists", "wrists"],
            temperature=0.1,
        )
        self.assertLess(float(loss), 0.001)
        self.assertEqual(correct, 8)
        self.assertEqual(compared, 8)

    def test_grouped_contrastive_loss_skips_singletons(self):
        torch = self.torch
        control = torch.randn(2, 8, requires_grad=True)
        motion = torch.randn(2, 8)
        loss, correct, compared = grouped_contrastive_matching_loss(
            control, motion, ["head", "wrists"]
        )
        loss.backward()
        self.assertEqual(correct, 0)
        self.assertEqual(compared, 0)
        self.assertTrue(torch.equal(control.grad, torch.zeros_like(control)))


if __name__ == "__main__":
    unittest.main()
