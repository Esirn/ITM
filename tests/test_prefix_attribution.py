import numpy as np

from itm.metrics.prefix_attribution import two_component_delta


def test_no_state_component_when_intermediate_equals_reference():
    r=np.zeros((8,263));p=np.arange(8)[:,None]*np.ones((1,263));g=p*.7
    v=two_component_delta(p,r,r,g)['uniform']
    assert v['state_energy']==0 and v['state_error_cross']==0
    assert v['direct_energy']==v['total_difference_energy']


def test_large_components_cancel_without_any_total_change():
    r=np.zeros((8,263));i=np.arange(8)[:,None]*np.ones((1,263));g=i*.5
    v=two_component_delta(r,i,r,g)['root_ric67']
    assert v['direct_energy']>0 and v['state_energy']>0
    assert v['twice_direct_state_covariance']==-2*v['direct_energy']
    assert v['delta_mse']==v['total_difference_energy']==v['delta_temporal_energy']==0


def test_mean_pose_response_does_not_create_temporal_energy():
    r=np.zeros((7,263));p=np.full_like(r,3.);g=np.ones_like(r)
    v=two_component_delta(p,r,r,g)['uniform']
    assert v['delta_mse']==3 and v['direct_energy']==9
    assert v['direct_centered_energy']==v['delta_temporal_energy']==0


def test_ordering_changes_components_but_not_total():
    r=np.zeros((8,263));p=np.ones_like(r);g=np.full_like(r,.3)
    a=two_component_delta(p,np.full_like(r,.4),r,g)['uniform']
    b=two_component_delta(p,np.full_like(r,.6),r,g)['uniform']
    assert a['delta_mse']==b['delta_mse'] and a['total_difference_energy']==b['total_difference_energy']
    assert a['direct_energy']!=b['direct_energy']


def test_invalid_padding_is_selected_before_arithmetic():
    p=np.arange(5)[:,None]*np.ones((1,263));i=p*.6;r=p*.3;g=p*.2
    mask=np.array([True,False,True,True,False])
    expected=two_component_delta(p[mask],i[mask],r[mask],g[mask])
    for a in (p,i,r,g):a[~mask]=np.nan
    assert two_component_delta(p,i,r,g,mask)==expected
