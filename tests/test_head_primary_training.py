import numpy as np
import pytest
import torch

from run_head_primary_training import training_objective


@pytest.mark.parametrize('sensor,present', [('wrists',True),('wrists',False),('head',False)])
def test_inactive_auxiliary_preserves_original_gradient(sensor,present):
    p = torch.randn(1,263,1,9,requires_grad=True)
    gt = torch.zeros_like(p)
    loss,parts,active = training_objective(p,gt,dict(sensor_config=sensor,imu_present=present),
        np.zeros(263),np.ones(263),{},dict(position=.1,velocity=.1))
    loss.backward()
    assert not active and parts['auxiliary'].item()==0
    torch.testing.assert_close(p.grad,2*p.detach()/p.numel(),rtol=0,atol=1e-9)


def test_head_velocity_penalizes_static_prediction_in_true_length_layout():
    gt=torch.zeros(1,263,1,11)
    gt[0,4+(15-1)*3,0]=torch.linspace(0,1,11)
    p=gt.mean(-1,keepdim=True).expand_as(gt).clone().requires_grad_()
    scales=dict(configurations=dict(head=dict(position_variance_m2=.1,velocity_mean_square_m2_s2=1.)))
    loss,parts,active=training_objective(p,gt,dict(sensor_config='head',imu_present=True),
        np.zeros(263),np.ones(263),scales,dict(position=.1,velocity=.1))
    loss.backward()
    assert active and parts['velocity']>0 and loss>parts['base']
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 and gt.grad is None


def test_batch_layout_rejected():
    p=torch.zeros(2,263,1,8)
    with pytest.raises(ValueError):training_objective(p,p,{},None,None,{}, {})
