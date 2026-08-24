import importlib.util
from pathlib import Path
import unittest

import numpy as np

from itm.data.humanml import recover_from_ric
from itm.data.humanml_torch import recover_from_ric_torch
from itm.models.mdm_imu_control import (
    MDMIMUControlConfig,
    attach_zero_control_adapters,
    compose_text_imu_guidance,
    compose_factorized_text_imu_guidance,
    install_imu_control,
    make_text_imu_guidance_model,
    make_imu_control_encoder,
)


def _load_train_helpers():
    spec = importlib.util.spec_from_file_location(
        "train_mdm_imu_control", Path(__file__).resolve().parents[1] / "scripts/train_mdm_imu_control.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
class MDMIMUControlTest(unittest.TestCase):
    def test_encoder_mask_and_zero_adapters(self):
        import torch

        config = MDMIMUControlConfig(
            latent_dim=16, encoder_layers=1, encoder_heads=4, dropout=0.0
        )
        encoder = make_imu_control_encoder(config).eval()
        imu = torch.randn(2, 5, 6, 12)
        mask = torch.tensor([[1, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
        control = encoder(imu, mask)
        self.assertEqual(tuple(control.shape), (2, 5, 16))
        dropped = encoder(imu, mask, force_mask=True)
        self.assertEqual(int(torch.count_nonzero(dropped)), 0)

        class DummyMDM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                layer = torch.nn.TransformerEncoderLayer(16, 4, dropout=0.0)
                self.seqTransEncoder = torch.nn.TransformerEncoder(layer, 2)

        controlled = attach_zero_control_adapters(DummyMDM(), config)
        for adapter in controlled.adapters:
            self.assertEqual(int(torch.count_nonzero(adapter.weight)), 0)
        controlled.set_control(control.transpose(0, 1))
        output = controlled(torch.randn(6, 2, 16))
        self.assertEqual(tuple(output.shape), (6, 2, 16))

    def test_independent_guidance(self):
        import torch

        unconditional = torch.tensor(1.0)
        text = torch.tensor(3.0)
        joint = torch.tensor(4.0)
        result = compose_text_imu_guidance(
            unconditional, text, joint, text_scale=2.0, imu_scale=3.0
        )
        self.assertEqual(float(result), 8.0)

    def test_factorized_guidance(self):
        import torch

        result = compose_factorized_text_imu_guidance(
            torch.tensor(1.0),
            torch.tensor(3.0),
            torch.tensor(4.0),
            torch.tensor(8.0),
            text_scale=2.0,
            imu_scale=3.0,
            joint_scale=0.5,
        )
        self.assertEqual(float(result), 15.0)

    def test_install_hook_preserves_mdm_call_signature(self):
        import torch

        config = MDMIMUControlConfig(
            latent_dim=16, encoder_layers=1, encoder_heads=4, dropout=0.0
        )

        class DummyMDM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                layer = torch.nn.TransformerEncoderLayer(16, 4, dropout=0.0)
                self.seqTransEncoder = torch.nn.TransformerEncoder(layer, 1)

            def forward(self, x, timesteps, y=None):
                return self.seqTransEncoder(x)

        model = DummyMDM().eval()
        _, _, handle = install_imu_control(model, config)
        y = {
            "imu": torch.randn(2, 5, 6, 12),
            "sensor_mask": torch.ones(2, 6, dtype=torch.bool),
        }
        output = model(torch.randn(6, 2, 16), torch.zeros(2), y=y)
        self.assertEqual(tuple(output.shape), (6, 2, 16))
        handle.remove()

    def test_guidance_wrapper_uses_three_condition_branches(self):
        import torch

        class Dummy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rot2xyz = None
                self.translation = True
                self.njoints = 1
                self.nfeats = 1
                self.data_rep = "hml_vec"
                self.cond_mode = "text"
                self.encode_text = None
                self.calls = []

            def forward(self, x, timesteps, y=None):
                self.calls.append((y["uncond"], y["imu_uncond"]))
                value = 0.0 if y["uncond"] else (1.0 if y["imu_uncond"] else 3.0)
                return torch.full_like(x, value)

        base = Dummy()
        guided = make_text_imu_guidance_model(base)
        value = guided(
            torch.zeros(2, 1, 1, 1),
            torch.zeros(2),
            {"text_scale": 2.0, "imu_scale": 4.0},
        )
        self.assertTrue(torch.all(value == 10.0))
        self.assertEqual(base.calls, [(True, True), (False, True), (False, False)])

    def test_factorized_wrapper_uses_four_condition_branches(self):
        import torch

        class Dummy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rot2xyz = None
                self.translation = True
                self.njoints = self.nfeats = 1
                self.data_rep = "hml_vec"
                self.cond_mode = "text"
                self.encode_text = None
                self.calls = []

            def forward(self, x, timesteps, y=None):
                key = (y["uncond"], y["imu_uncond"])
                self.calls.append(key)
                values = {(True, True): 1.0, (False, True): 3.0, (True, False): 4.0, (False, False): 8.0}
                return torch.full_like(x, values[key])

        base = Dummy()
        guided = make_text_imu_guidance_model(base, mode="factorized")
        result = guided(
            torch.zeros(1, 1, 1, 1),
            torch.zeros(1),
            {"text_scale": 2.0, "imu_scale": 3.0, "joint_scale": 0.5},
        )
        self.assertEqual(float(result), 15.0)
        self.assertEqual(
            base.calls,
            [(True, True), (False, True), (True, False), (False, False)],
        )

    def test_batched_guidance_matches_sequential_factorization(self):
        import torch

        class Dummy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rot2xyz = None
                self.translation = True
                self.njoints = self.nfeats = 1
                self.data_rep = "hml_vec"
                self.cond_mode = "text"
                self.encode_text = None

            def forward(self, x, timesteps, y=None):
                text = y["text_embed"].transpose(0, 1).reshape(len(x), 1, 1, 1)
                imu = y["imu_control"].mean((1, 2)).reshape(len(x), 1, 1, 1)
                return text + imu

        y = {
            "text": ["walk", "turn"],
            "lengths": torch.tensor([3, 3]),
            "mask": torch.ones(2, 1, 1, 3, dtype=torch.bool),
            "text_embed": torch.tensor([[[2.0], [3.0]]]),
            "imu_control": torch.tensor([[[4.0]], [[5.0]]]),
            "text_scale": 2.0,
            "imu_scale": 3.0,
            "joint_scale": 0.5,
        }
        guided = make_text_imu_guidance_model(
            Dummy(), mode="factorized", branch_execution="batched"
        )
        result = guided(torch.zeros(2, 1, 1, 3), torch.zeros(2), y)
        expected = 2.0 * y["text_embed"].transpose(0, 1).reshape(2, 1, 1, 1)
        expected = expected + 3.0 * y["imu_control"].mean((1, 2)).reshape(2, 1, 1, 1)
        self.assertTrue(torch.allclose(result, expected))

    def test_torch_recover_from_ric_matches_numpy(self):
        import torch

        features = np.random.default_rng(1234).normal(size=(2, 8, 263)).astype(np.float32)
        numpy_joints = recover_from_ric(features)
        torch_joints = recover_from_ric_torch(torch.from_numpy(features)).numpy()
        self.assertTrue(np.allclose(torch_joints, numpy_joints, atol=1e-5))

    def test_stage2_auxiliary_losses_are_finite_with_padding(self):
        import torch

        helpers = _load_train_helpers()
        predicted = torch.zeros(2, 5, 22, 3)
        target = torch.zeros_like(predicted)
        anchor = torch.zeros_like(predicted)
        anchor[:, :, 20, 0] = 0.1
        anchor[:, :, 21, 0] = 0.1
        predicted[0, :, 15, 0] = torch.linspace(0.0, 0.4, 5)
        predicted[1, :, 20, 1] = torch.linspace(0.0, 0.2, 5)
        frame_mask = torch.tensor(
            [[True, True, True, True, True], [True, True, True, False, False]]
        )
        sensor_mask = torch.zeros(2, 6)
        sensor_mask[0, 4] = 1.0
        sensor_mask[1, 0] = 1.0
        sensor_mask[1, 1] = 1.0
        weights = helpers.active_sensor_joint_weights(sensor_mask)
        head_only = helpers.head_only_mask(sensor_mask)
        upper_body_weights = helpers.upper_body_joint_weights(head_only)
        self.assertEqual(float(weights[0, 15]), 1.0)
        self.assertEqual(float(weights[1, 20]), 1.0)
        self.assertEqual(float(weights[1, 21]), 1.0)
        self.assertTrue(bool(head_only[0]))
        self.assertFalse(bool(head_only[1]))
        self.assertGreater(float(upper_body_weights[0, 20]), 0.0)
        self.assertEqual(float(upper_body_weights[1, 20]), 0.0)
        non_active = helpers.non_active_joint_weights(weights)
        upper_anchor = helpers.upper_body_text_anchor_weights(weights)
        self.assertEqual(float(non_active[0, 15]), 0.0)
        self.assertGreater(float(non_active[0, 20]), 0.0)
        self.assertGreater(float(upper_anchor[0, 20]), 0.0)
        self.assertEqual(float(upper_anchor[1, 20]), 0.0)
        losses = helpers.stage2_control_losses(
            predicted,
            target,
            frame_mask,
            weights,
            head_only,
            text_anchor=anchor,
        )
        for value in losses.values():
            self.assertTrue(torch.isfinite(value))
        self.assertGreater(float(losses["trajectory_loss"]), 0.0)
        self.assertGreaterEqual(float(losses["upper_body_loss"]), 0.0)
        self.assertGreater(float(losses["text_anchor_loss"]), 0.0)
        self.assertGreater(float(losses["upper_body_text_anchor_loss"]), 0.0)

    def test_stage2_auxiliary_losses_handle_short_sequences(self):
        import torch

        helpers = _load_train_helpers()
        predicted = torch.zeros(1, 3, 22, 3)
        target = torch.zeros_like(predicted)
        frame_mask = torch.ones(1, 3, dtype=torch.bool)
        sensor_mask = torch.zeros(1, 6)
        sensor_mask[0, 4] = 1.0
        weights = helpers.active_sensor_joint_weights(sensor_mask)
        head_only = helpers.head_only_mask(sensor_mask)
        losses = helpers.stage2_control_losses(predicted, target, frame_mask, weights, head_only)
        self.assertEqual(float(losses["jerk_loss"]), 0.0)

    def test_training_prediction_is_reused_for_diffusion_loss(self):
        import torch

        helpers = _load_train_helpers()

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, x, timesteps, y=None):
                self.calls += 1
                return x * 0.5

        class DummyDiffusion:
            @staticmethod
            def q_sample(motion, timesteps, noise=None):
                return motion + noise

            @staticmethod
            def _scale_timesteps(timesteps):
                return timesteps

            @staticmethod
            def masked_l2(target, predicted, mask):
                return ((target - predicted).pow(2) * mask).flatten(1).mean(1)

        model = DummyModel()
        diffusion = DummyDiffusion()
        motion = torch.ones(2, 3, 1, 4)
        noise = torch.zeros_like(motion)
        timesteps = torch.tensor([1, 2])
        y = {"mask": torch.ones(2, 1, 1, 4)}
        output, loss = helpers._predict_xstart_and_diffusion_loss(
            model, diffusion, motion, timesteps, noise, y
        )
        self.assertEqual(model.calls, 1)
        self.assertTrue(torch.allclose(output, motion * 0.5))
        self.assertAlmostEqual(float(loss), 0.25)

    def test_text_only_anchor_kwargs_disable_imu_control(self):
        import torch

        helpers = _load_train_helpers()
        y = {
            "mask": torch.ones(2, 1, 1, 5, dtype=torch.bool),
            "lengths": torch.tensor([5, 5]),
            "text": ["walk", "run"],
            "imu": torch.randn(2, 5, 6, 12),
            "sensor_mask": torch.ones(2, 6),
            "imu_frame_mask": torch.ones(2, 5, dtype=torch.bool),
            "imu_uncond": False,
        }
        anchor = helpers.text_only_anchor_kwargs(y)
        self.assertNotIn("imu", anchor)
        self.assertNotIn("sensor_mask", anchor)
        self.assertNotIn("imu_frame_mask", anchor)
        self.assertFalse(anchor["uncond"])
        self.assertEqual(anchor["text"], ["walk", "run"])


if __name__ == "__main__":
    unittest.main()
