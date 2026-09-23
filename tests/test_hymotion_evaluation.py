import importlib.util
from pathlib import Path

import numpy as np

from itm.data.hymotion import forward_kinematics, joint_imu_features


def module(monkeypatch):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('hymotion_evaluation_test', scripts / 'evaluate_hymotion_imu.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def training_module(monkeypatch):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('hymotion_training_test', scripts / 'train_hymotion_imu.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_identical_motion_has_zero_imu_score(monkeypatch):
    evaluator = module(monkeypatch)
    rotations = np.broadcast_to(np.eye(3), (20, 22, 3, 3)).copy()
    joints, global_rotation = forward_kinematics(rotations, np.zeros((20, 3)), np.zeros((22, 3)))
    target = joint_imu_features(joints, global_rotation)
    result = evaluator.scores(joints, global_rotation, target, [0, 1], np.ones(3))
    assert all(value == 0 for value in result.values())


def test_shuffle_resampling_preserves_rotation_manifold(monkeypatch):
    evaluator = module(monkeypatch)
    from scipy.spatial.transform import Rotation
    imu = np.zeros((20, 6, 12), dtype=np.float32)
    r = Rotation.from_euler('y', np.linspace(0, 1, 20)).as_matrix()
    imu[..., 3:] = r.reshape(20, 1, 9)
    output = evaluator.resample_imu(imu, 37)
    matrices = output[..., 3:].reshape(37, 6, 3, 3)
    np.testing.assert_allclose(np.linalg.det(matrices), 1, atol=1e-6)
    np.testing.assert_allclose(output[[0, -1]], imu[[0, -1]], atol=1e-6)


def test_gate_rejects_jerk_regression_even_when_paired_wins(monkeypatch):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    from merge_hymotion_evaluation import aggregate
    records = []
    for sensor in ('head', 'wrists'):
        for mode in ('text_only', 'paired', 'zero', 'shuffled'):
            records.append(dict(motion_id='x', sensor=sensor, mode=mode,
                                imu_score=1 if mode == 'paired' else 2,
                                acceleration_rmse_m_s2=1, orientation_mean_deg=1,
                                jerk_mean_m_s3=2 if mode == 'paired' else 1))
    result = aggregate(records)
    assert result['head']['paired_vs_shuffled_gate']
    assert not result['head']['jerk_gate']
    assert result['wrists']['paired_to_text_only_jerk_ratio'] == 2
    assert result['head']['paired_beats_zero_fraction'] == 1
    assert result['head']['paired_minus_text_only_jerk']['mean'] == 1
    assert result['head']['paired_minus_shuffled_imu_score']['mean'] == -1


def test_predeclared_relative_jerk_noninferiority_is_separate_from_strict_gate(monkeypatch):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    from merge_hymotion_evaluation import aggregate
    records = []
    for motion_id in ('a', 'b', 'c', 'd'):
        for sensor in ('head', 'wrists'):
            for mode in ('text_only', 'paired', 'zero', 'shuffled'):
                records.append(dict(motion_id=motion_id, sensor=sensor, mode=mode,
                                    imu_score=1 if mode == 'paired' else 2,
                                    acceleration_rmse_m_s2=1, orientation_mean_deg=1,
                                    jerk_mean_m_s3=101 if mode == 'paired' else 100))
    result = aggregate(records, jerk_noninferiority_relative_margin=0.02)
    assert not result['head']['jerk_gate']
    assert result['head']['jerk_noninferiority_gate']
    assert result['head']['jerk_noninferiority_absolute_margin_m_s3'] == 2
    assert result['head']['paired_vs_shuffled_ci_gate']
    assert result['wrists']['jerk_noninferiority_relative_margin'] == 0.02


def test_position_jerk_matching_loss_uses_only_valid_world_joint_frames(monkeypatch):
    trainer = training_module(monkeypatch)
    import torch
    frames = 6
    target = torch.zeros(1, frames, 201)
    target[..., 135:] = torch.arange(frames, dtype=torch.float32).pow(3)[None, :, None]
    prediction = target.clone()
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    kwargs = dict(noisy_latent=torch.zeros_like(target), target_latent=target,
                  timestep=torch.zeros(1), frame_mask=mask,
                  mean=torch.zeros(1, 1, 201), std=torch.ones(1, 1, 201))
    assert trainer.position_jerk_matching_loss(prediction, **kwargs).item() == 0
    prediction[:, 2, 135:] += 1
    assert trainer.position_jerk_matching_loss(prediction, **kwargs).item() > 0


def test_prepared_extension_requires_exact_checkpoint_prefix(monkeypatch):
    trainer = training_module(monkeypatch)
    rows = [dict(motion_id=value) for value in ('a', 'b', 'c')]
    prepared = dict(train_records=rows[:2], checkpoint_sha256='checkpoint')
    assert trainer.prepared_extension_start(prepared, rows, 'checkpoint') == 2
    import pytest
    with pytest.raises(ValueError, match='exact nonempty prefix'):
        trainer.prepared_extension_start(dict(prepared, checkpoint_sha256='other'), rows, 'checkpoint')
    with pytest.raises(ValueError, match='exact nonempty prefix'):
        trainer.prepared_extension_start(dict(prepared, train_records=[rows[1]]), rows, 'checkpoint')


def test_position_acceleration_matching_loss_matches_raw_sensor_imu(monkeypatch):
    trainer = training_module(monkeypatch)
    import torch
    frames = 8
    endpoint = torch.zeros(1, frames, 201)
    time = torch.arange(frames, dtype=torch.float32) / 30
    endpoint[..., :3] = torch.stack((time.square(), time * 0, time * 0), dim=-1)
    physical = endpoint[..., 135:].reshape(1, frames, 22, 3) + endpoint[..., :3].unsqueeze(-2)
    target_imu = torch.zeros(1, frames, 6, 12)
    target_imu[..., :3] = trainer.physical_sensor_acceleration(
        physical[:, :, list(trainer.SENSOR_JOINTS)])
    kwargs = dict(noisy_latent=torch.zeros_like(endpoint), timestep=torch.zeros(1),
                  target_imu=target_imu, sensor_mask=torch.ones(1, 6, dtype=torch.bool),
                  frame_mask=torch.ones(1, frames, dtype=torch.bool),
                  mean=torch.zeros(1, 1, 201), std=torch.ones(1, 1, 201),
                  condition_active=torch.ones(1, dtype=torch.bool))
    assert trainer.position_acceleration_matching_loss(endpoint, **kwargs).item() == 0
    target_imu[:, 3, 4, 0] += 2
    assert trainer.position_acceleration_matching_loss(endpoint, **kwargs).item() > 0
    kwargs['condition_active'][:] = False
    assert trainer.position_acceleration_matching_loss(endpoint, **kwargs).item() == 0


def test_fk_orientation_matching_loss_matches_raw_sensor_imu(monkeypatch):
    trainer = training_module(monkeypatch)
    import torch
    frames = 5
    endpoint = torch.zeros(1, frames, 201)
    endpoint[..., 3:135] = torch.tensor([1, 0, 0, 1, 0, 0] * 22)
    target_imu = torch.zeros(1, frames, 6, 12)
    target_imu[..., 3:] = torch.eye(3).reshape(9)
    kwargs = dict(noisy_latent=torch.zeros_like(endpoint), timestep=torch.zeros(1),
                  target_imu=target_imu, sensor_mask=torch.ones(1, 6, dtype=torch.bool),
                  frame_mask=torch.ones(1, frames, dtype=torch.bool),
                  mean=torch.zeros(1, 1, 201), std=torch.ones(1, 1, 201),
                  condition_active=torch.ones(1, dtype=torch.bool))
    assert trainer.fk_orientation_matching_loss(endpoint, **kwargs).item() == 0
    target_imu[:, 2, 4, 3:] = torch.tensor([0, -1, 0, 1, 0, 0, 0, 0, 1.])
    assert trainer.fk_orientation_matching_loss(endpoint, **kwargs).item() > 0
    kwargs['condition_active'][:] = False
    assert trainer.fk_orientation_matching_loss(endpoint, **kwargs).item() == 0
