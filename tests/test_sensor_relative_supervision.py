import numpy as np
import pytest
import torch

from itm.models.sensor_relative_supervision import fit_sensor_relative_scales, sensor_relative_auxiliary_loss


def records():
    return [dict(motion_id='train',source_group_id='a',split='train',frames=10,training_admissible=True),
            dict(motion_id='val',source_group_id='b',split='val',frames=10,training_admissible=False)]


def test_normalization_never_reads_validation():
    touched = []
    def load(r):
        assert r['split'] == 'train'
        touched.append(r['motion_id'])
        g = np.zeros((10,22,3)); g[:,15,0] = np.arange(10)*.1
        return g
    s = fit_sensor_relative_scales(records(),load)
    assert touched == ['train'] and s['unique_train_frames'] == 10
    assert s['configurations']['head']['position_variance_m2'] == pytest.approx(.0275)
    assert s['configurations']['head']['velocity_mean_square_m2_s2'] == pytest.approx(4/3)


def test_source_leakage_rejected_before_loading():
    rr = records(); rr[1]['source_group_id'] = 'a'
    with pytest.raises(ValueError): fit_sensor_relative_scales(rr,lambda r: None)


def tensors():
    x = torch.zeros(12,263)
    x[:,3] = 1
    x[:,4+(20-1)*3] = torch.linspace(0,1,12)
    x[:,4+(21-1)*3+2] = torch.linspace(1,0,12)
    return x, torch.zeros(263), torch.ones(263), dict(configurations={
        'wrists':dict(position_variance_m2=.1,velocity_mean_square_m2_s2=1.)})


def test_static_mean_has_velocity_penalty_and_finite_gradients():
    gt,mean,std,scales = tensors()
    p = gt.mean(0,keepdim=True).expand_as(gt).clone().requires_grad_()
    loss,parts = sensor_relative_auxiliary_loss(p,gt,mean,std,'wrists',scales,True)
    assert parts['velocity'] > 0
    loss.backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_root_translation_is_not_auxiliary_target():
    gt,mean,std,scales = tensors()
    p = gt.clone(); p[:,1:3] = .4; p.requires_grad_()
    loss,_ = sensor_relative_auxiliary_loss(p,gt,mean,std,'wrists',scales,True)
    assert loss.item() < 1e-9
    loss.backward()
    torch.testing.assert_close(p.grad[:,1:3],torch.zeros_like(p.grad[:,1:3]),atol=1e-7,rtol=0)


def test_absent_imu_disables_auxiliary_gradient():
    gt,mean,std,scales = tensors()
    p = (gt+1).requires_grad_()
    loss,_ = sensor_relative_auxiliary_loss(p,gt,mean,std,'wrists',scales,False)
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(p.grad) == 0
