import importlib.util
import unittest

from itm.models.mdm_imu_control import (
    MDMIMUControlConfig,
    attach_zero_control_adapters,
    compose_text_imu_guidance,
    install_imu_control,
    make_text_imu_guidance_model,
    make_imu_control_encoder,
)


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


if __name__ == "__main__":
    unittest.main()
