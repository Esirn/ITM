import numpy as np
import pytest
from itm.metrics.canonical_b0 import evaluate_motion


def test_identical_and_translation_without_fit():
    g = np.zeros((5,22,3)); g[:, :, 0] = np.arange(5)[:, None]*.05
    m = evaluate_motion(g,g,'head')
    assert m['global_MPJPE_m'] == m['velocity_joint_proxy_RMSE_m_s'] == 0
    m = evaluate_motion(g+[3,0,4],g,'wrists')
    assert m['global_MPJPE_m'] == m['root_path_RMSE_m'] == 5
    assert m['root_relative_MPJPE_m'] == 0
    assert m['velocity_joint_proxy_RMSE_m_s'] < 1e-13


def test_cubic_jerk_and_euclidean_velocity():
    t=np.arange(6)/20; g=np.zeros((6,22,3));p=g.copy();p[:,:,0]=t[:,None]**3
    m=evaluate_motion(p,g,'head')
    assert m['jerk_absolute_m_s3'] == pytest.approx(6)
    assert m['GT_jerk_absolute_m_s3'] == 0
    assert m['velocity_joint_proxy_RMSE_m_s'] == pytest.approx(np.sqrt(np.mean((np.diff(t**3)*20)**2)))


def test_fixed_GT_contacts_and_no_endpoint():
    g=np.zeros((5,22,3));p=g.copy();p[:,7,0]=np.arange(5)/20;p[:,10,1]=np.arange(5)*2/20
    c=np.zeros((5,4));c[0,0]=1;c[2,1]=1;c[-1,:]=1
    m=evaluate_motion(p,g,'head',c)
    assert m['foot_contact_stencils']==2
    assert m['foot_contact_speed_proxy_m_s']==pytest.approx(1.5)
    assert m['GT_foot_contact_speed_proxy_m_s']==0
    assert evaluate_motion(p,g,'head')['foot_contact_speed_proxy_m_s'] is None


@pytest.mark.parametrize('kind',['length','nan','contacts'])
def test_invalid_input_rejected(kind):
    g=np.zeros((5,22,3));p=g.copy();c=np.zeros((5,4))
    if kind=='length':p=p[:-1]
    if kind=='nan':p[0,0,0]=np.nan
    if kind=='contacts':c[0,0]=.5
    with pytest.raises(ValueError):evaluate_motion(p,g,'head',c)
