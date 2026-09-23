import numpy as np
import torch

from itm.data.hymotion import forward_kinematics, world_joints_from_result, joint_imu_features
from itm.models.hymotion_imu_adapter import (HYMotionIMUConfig, make_hymotion_imu_adapter,
                                             control_hymotion, sensor_chain_motion_mask)


class Double(torch.nn.Module):
    def forward(self, *, motion_feat, text_feat):
        return motion_feat * 1.1, text_feat * 1.2


class Single(torch.nn.Module):
    def forward(self, *, x, split_len):
        return x * 1.1


class Backbone(torch.nn.Module):
    feat_dim = 8

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.double_blocks = torch.nn.ModuleList([Double()])
        self.single_blocks = torch.nn.ModuleList([Single()])

    def forward(self, *, x, text, x_mask_temporal, timesteps=None):
        motion, text = self.double_blocks[0](motion_feat=x * self.scale, text_feat=text)
        return self.single_blocks[0](x=torch.cat((motion, text), dim=1), split_len=x.shape[1])


class OutputBackbone(torch.nn.Module):
    feat_dim = 8

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.double_blocks = torch.nn.ModuleList([Double()])
        self.single_blocks = torch.nn.ModuleList([Single()])

    def forward(self, *, x, **kwargs):
        return x * self.scale


def setup_control():
    torch.manual_seed(3)
    cfg = HYMotionIMUConfig(hidden_dim=8, output_dim=8, encoder_heads=2,
                            double_blocks=1, single_blocks=1, temporal_layers=1,
                            max_frames=8, dropout=0, imu_drop_probability=0)
    adapter = make_hymotion_imu_adapter(cfg)
    model = control_hymotion(Backbone(), adapter)
    inputs = dict(x=torch.randn(2, 4, 8), text=torch.randn(2, 3, 8),
                  x_mask_temporal=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]).bool())
    imu = torch.randn(2, 4, 6, 12)
    mask = torch.tensor([[0, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0]]).bool()
    return model, inputs, imu, mask


def test_zero_initialization_and_frozen_backbone():
    model, inputs, imu, mask = setup_control()
    model.train()
    expected = model(**inputs)
    actual = model(imu=imu, sensor_mask=mask, **inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    assert not model.backbone.training
    assert all(not p.requires_grad and p.grad is None for p in model.backbone.parameters())
    assert model.adapter.residuals[0].weight.grad.abs().sum() > 0
    assert model.adapter.feature_projection.weight.grad.abs().sum() == 0


def test_motion_only_masking_and_gradient_after_update():
    model, inputs, imu, mask = setup_control()
    with torch.no_grad():
        for layer in model.adapter.residuals:
            layer.weight.fill_(0.02)
            layer.bias.fill_(0.1)
    base = model(**inputs)
    actual = model(imu=imu, sensor_mask=mask, **inputs)
    torch.testing.assert_close(actual[:, 4:], base[:, 4:], rtol=0, atol=0)
    torch.testing.assert_close(actual[:, :4][~inputs['x_mask_temporal']], base[:, :4][~inputs['x_mask_temporal']], rtol=0, atol=0)
    for absent in (dict(sensor_mask=torch.zeros_like(mask)), dict(sensor_mask=mask, imu_keep=torch.zeros(2).bool())):
        torch.testing.assert_close(model(imu=imu, **absent, **inputs), base, rtol=0, atol=0)
    actual.square().mean().backward()
    assert model.adapter.feature_projection.weight.grad.abs().sum() > 0
    model.remove_hooks()
    assert not model.backbone._itm_imu_control_installed


def test_control_injection_start_skips_early_blocks():
    model, inputs, imu, mask = setup_control()
    model.remove_hooks()
    cfg = HYMotionIMUConfig(hidden_dim=8, output_dim=8, encoder_heads=2,
                            double_blocks=1, single_blocks=1, temporal_layers=1,
                            max_frames=8, dropout=0, imu_drop_probability=0,
                            control_injection_start=1)
    model = control_hymotion(Backbone(), make_hymotion_imu_adapter(cfg))
    with torch.no_grad():
        for layer in model.adapter.residuals:
            layer.weight.fill_(0.02)
            layer.bias.fill_(0.1)
    base = model(**inputs)
    controlled = model(imu=imu, sensor_mask=mask, **inputs)
    assert not torch.equal(controlled[:, :4], base[:, :4])
    controlled.square().mean().backward()
    assert model.adapter.residuals[0].weight.grad is None
    assert model.adapter.residuals[1].weight.grad.abs().sum() > 0


def test_control_injection_end_skips_late_blocks():
    _, inputs, imu, mask = setup_control()
    cfg = HYMotionIMUConfig(hidden_dim=8, output_dim=8, encoder_heads=2,
                            double_blocks=1, single_blocks=1, temporal_layers=1,
                            max_frames=8, dropout=0, imu_drop_probability=0,
                            control_injection_end=1)
    model = control_hymotion(Backbone(), make_hymotion_imu_adapter(cfg))
    with torch.no_grad():
        for layer in model.adapter.residuals:
            layer.weight.fill_(0.02)
            layer.bias.fill_(0.1)
    base = model(**inputs)
    controlled = model(imu=imu, sensor_mask=mask, **inputs)
    assert not torch.equal(controlled[:, :4], base[:, :4])
    controlled.square().mean().backward()
    assert model.adapter.residuals[0].weight.grad.abs().sum() > 0
    assert model.adapter.residuals[1].weight.grad is None


def test_zero_control_strength_is_exact_unconditional():
    model, inputs, imu, mask = setup_control()
    with torch.no_grad():
        for layer in model.adapter.residuals:
            layer.weight.fill_(0.02)
            layer.bias.fill_(0.1)
    expected = model(**inputs)
    actual = model(imu=imu, sensor_mask=mask, control_strength=0, **inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with __import__('pytest').raises(ValueError, match='control_strength'):
        model(imu=imu, sensor_mask=mask, control_strength=1.1, **inputs)


def test_timestep_gate_removes_control_at_data_endpoint():
    _, inputs, imu, mask = setup_control()
    cfg = HYMotionIMUConfig(hidden_dim=8, output_dim=8, encoder_heads=2,
                            double_blocks=1, single_blocks=1, temporal_layers=1,
                            max_frames=8, dropout=0, imu_drop_probability=0,
                            control_timestep_power=1)
    model = control_hymotion(Backbone(), make_hymotion_imu_adapter(cfg))
    with torch.no_grad():
        for layer in model.adapter.residuals:
            layer.weight.fill_(0.02)
            layer.bias.fill_(0.1)
    base = model(**inputs)
    at_noise = model(imu=imu, sensor_mask=mask, timesteps=torch.zeros(2), **inputs)
    at_data = model(imu=imu, sensor_mask=mask, timesteps=torch.ones(2), **inputs)
    assert not torch.equal(at_noise[:, :4], base[:, :4])
    torch.testing.assert_close(at_data, base, rtol=0, atol=0)
    with __import__('pytest').raises(ValueError, match='timesteps'):
        model(imu=imu, sensor_mask=mask, **inputs)


def test_timestep_cutoff_disables_late_control():
    model, inputs, imu, mask = setup_control()
    with torch.no_grad():
        for layer in model.adapter.residuals:
            layer.weight.fill_(0.02)
            layer.bias.fill_(0.1)
    base = model(**inputs)
    early = model(imu=imu, sensor_mask=mask, timesteps=torch.full((2,), 0.25),
                  control_timestep_cutoff=0.5, **inputs)
    late = model(imu=imu, sensor_mask=mask, timesteps=torch.full((2,), 0.75),
                 control_timestep_cutoff=0.5, **inputs)
    assert not torch.equal(early[:, :4], base[:, :4])
    torch.testing.assert_close(late, base, rtol=0, atol=0)


def test_control_residual_curvature_reuses_injected_residuals():
    model, inputs, imu, mask = setup_control()
    with torch.no_grad():
        for layer in model.adapter.residuals:
            layer.weight.fill_(0.02)
            layer.bias.fill_(0.1)
    model.train()
    model(imu=imu, sensor_mask=mask, **inputs)
    curvature = model.take_control_residual_curvature_loss()
    assert curvature.ndim == 0 and curvature > 0
    curvature.backward()
    assert model.adapter.feature_projection.weight.grad.abs().sum() > 0


def test_temporal_velocity_smoothing_is_motion_only_and_preserves_padding():
    model, inputs, _, _ = setup_control()
    base = model(**inputs)
    exact = model(temporal_velocity_smoothing=0, **inputs)
    smoothed = model(temporal_velocity_smoothing=1, **inputs)
    torch.testing.assert_close(exact, base, rtol=0, atol=0)
    torch.testing.assert_close(smoothed[:, 4:], base[:, 4:], rtol=0, atol=0)
    torch.testing.assert_close(smoothed[:, :4][~inputs['x_mask_temporal']], base[:, :4][~inputs['x_mask_temporal']], rtol=0, atol=0)
    assert not torch.equal(smoothed[:, :4][inputs['x_mask_temporal']], base[:, :4][inputs['x_mask_temporal']])
    with __import__('pytest').raises(ValueError, match='temporal_velocity_smoothing'):
        model(temporal_velocity_smoothing=1.1, **inputs)


def test_control_delta_smoothing_filters_only_condition_delta():
    model, inputs, imu, mask = setup_control()
    with torch.no_grad():
        for layer in model.adapter.residuals:
            layer.weight.fill_(0.02)
            layer.bias.fill_(0.1)
    model.eval()
    base = model(**inputs)
    controlled = model(imu=imu, sensor_mask=mask, **inputs)
    exact = model(imu=imu, sensor_mask=mask, control_delta_smoothing=0, **inputs)
    smoothed = model(imu=imu, sensor_mask=mask, control_delta_smoothing=1, **inputs)
    smoothed_twice = model(imu=imu, sensor_mask=mask, control_delta_smoothing=1,
                           control_delta_smoothing_passes=2, **inputs)
    expected = base + model._smooth_velocity(
        controlled - base, inputs['x_mask_temporal'], 1)
    expected_delta = model._smooth_velocity(
        controlled - base, inputs['x_mask_temporal'], 1)
    expected_twice = base + model._smooth_velocity(
        expected_delta, inputs['x_mask_temporal'], 1)
    torch.testing.assert_close(exact, controlled, rtol=0, atol=0)
    torch.testing.assert_close(smoothed, expected, rtol=0, atol=0)
    torch.testing.assert_close(smoothed_twice, expected_twice, rtol=0, atol=0)
    torch.testing.assert_close(
        smoothed[:, :4][~inputs['x_mask_temporal']],
        base[:, :4][~inputs['x_mask_temporal']], rtol=0, atol=0)
    with __import__('pytest').raises(ValueError, match='control_delta_smoothing'):
        model(imu=imu, sensor_mask=mask, control_delta_smoothing=1.1, **inputs)
    with __import__('pytest').raises(ValueError, match='control_delta_smoothing_passes'):
        model(imu=imu, sensor_mask=mask, control_delta_smoothing_passes=0, **inputs)


def test_sensor_chain_motion_mask_selects_only_fk_ancestors():
    sensor_mask = torch.zeros(2, 6, dtype=torch.bool)
    sensor_mask[0, 4] = True
    sensor_mask[1, :2] = True
    mask = sensor_chain_motion_mask(sensor_mask)
    assert mask.shape == (2, 201)
    assert mask[0].sum() == 39
    assert mask[1].sum() == 75
    assert mask[:, :3].all() and not mask[:, 135:].any()
    head_joints = {joint for joint in range(22)
                   if mask[0, 3 + joint * 6:3 + (joint + 1) * 6].all()}
    wrist_joints = {joint for joint in range(22)
                    if mask[1, 3 + joint * 6:3 + (joint + 1) * 6].all()}
    assert head_joints == {0, 3, 6, 9, 12, 15}
    assert wrist_joints == {0, 3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21}


def test_output_velocity_control_zero_init_and_gradients():
    torch.manual_seed(7)
    cfg = HYMotionIMUConfig(hidden_dim=8, output_dim=8, motion_dim=8,
                            encoder_heads=2, double_blocks=1, single_blocks=1,
                            temporal_layers=1, max_frames=8, dropout=0,
                            imu_drop_probability=0, control_mode='output',
                            output_velocity_smoothing_passes=2)
    model = control_hymotion(OutputBackbone(), make_hymotion_imu_adapter(cfg)).train()
    inputs = dict(x=torch.randn(2, 4, 8), timesteps=torch.tensor([0.2, 0.7]),
                  x_mask_temporal=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]).bool())
    imu = torch.randn(2, 4, 6, 12)
    mask = torch.tensor([[0, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0]]).bool()
    expected = model(**inputs)
    actual = model(imu=imu, sensor_mask=mask, **inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    assert model.adapter.velocity_projection.weight.grad.abs().sum() > 0
    assert model.adapter.feature_projection.weight.grad.abs().sum() == 0
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in model.backbone.parameters())


def test_output_velocity_control_masks_and_reports_actual_curvature():
    torch.manual_seed(11)
    cfg = HYMotionIMUConfig(hidden_dim=8, output_dim=8, motion_dim=8,
                            encoder_heads=2, double_blocks=1, single_blocks=1,
                            temporal_layers=1, max_frames=8, dropout=0,
                            imu_drop_probability=0, control_mode='output',
                            output_velocity_smoothing_passes=2)
    model = control_hymotion(OutputBackbone(), make_hymotion_imu_adapter(cfg)).train()
    with torch.no_grad():
        model.adapter.velocity_projection.weight.fill_(0.02)
        model.adapter.velocity_projection.bias.fill_(0.1)
    inputs = dict(x=torch.randn(2, 4, 8), timesteps=torch.tensor([0.2, 0.7]),
                  x_mask_temporal=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]).bool())
    imu = torch.randn(2, 4, 6, 12)
    mask = torch.tensor([[0, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0]]).bool()
    base = model(**inputs)
    controlled = model(imu=imu, sensor_mask=mask, **inputs)
    assert not torch.equal(controlled[inputs['x_mask_temporal']], base[inputs['x_mask_temporal']])
    torch.testing.assert_close(controlled[~inputs['x_mask_temporal']],
                               base[~inputs['x_mask_temporal']], rtol=0, atol=0)
    curvature = model.take_control_residual_curvature_loss()
    assert curvature.ndim == 0 and curvature >= 0
    controlled.square().mean().backward()
    assert model.adapter.feature_projection.weight.grad.abs().sum() > 0


def test_masked_nan_and_padding_cannot_change_condition():
    model, inputs, imu, mask = setup_control()
    adapter = model.adapter.eval()
    frames = inputs['x_mask_temporal']
    expected, _ = adapter(imu, mask, frames)
    dirty = imu.clone()
    dirty[~(mask[:, None, :] & frames[:, :, None])] = float('nan')
    actual, _ = adapter(dirty, mask, frames)
    torch.testing.assert_close(actual, expected)
    empty, valid = adapter(dirty, torch.zeros_like(mask), torch.zeros_like(frames))
    assert torch.isfinite(empty).all() and not empty.any() and not valid.any()


def test_world_fk_translation_drives_acceleration():
    t = np.arange(20, dtype=np.float32) / 30
    translation = np.stack((t * t, t * 0, t * 0), axis=-1)
    rotations = np.broadcast_to(np.eye(3), (20, 22, 3, 3)).copy()
    template = np.zeros((22, 3), dtype=np.float32)
    joints, global_rotation = forward_kinematics(rotations, translation, template)
    imu = joint_imu_features(joints, global_rotation)
    np.testing.assert_allclose(imu[4:-4, :, 0], 2, atol=1e-4)
    result = dict(rot6d=rotations[..., :, :2].reshape(20, 22, 6), transl=translation)
    from itm.data.hymotion import BODY_PARENTS
    np.testing.assert_allclose(world_joints_from_result(result, template, BODY_PARENTS), joints)


def test_official_padding_keeps_text_prefix_at_360_and_masks_loss():
    from itm.models.hymotion_imu_adapter import pad_hymotion_inputs
    latent, imu = torch.randn(1, 20, 201), torch.randn(1, 20, 6, 12)
    padded, padded_imu, mask = pad_hymotion_inputs(latent, imu)
    assert padded.shape == (1, 360, 201) and padded_imu.shape == (1, 360, 6, 12)
    assert mask.shape == (1, 360) and mask.sum() == 20
    torch.testing.assert_close(padded[:, :20], latent)
    torch.testing.assert_close(padded_imu[:, :20], imu)
    assert not padded[:, 20:].any() and not padded_imu[:, 20:].any()
    prediction = torch.randn_like(padded, requires_grad=True)
    (prediction[mask] - padded[mask]).square().mean().backward()
    assert prediction.grad[:, :20].abs().sum() > 0
    assert not prediction.grad[:, 20:].any()
