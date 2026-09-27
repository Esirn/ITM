"""Independent evaluator algebra/sign checks, not checks of model effectiveness."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('alignment_eval',SCRIPTS/'evaluate_b0_sampling_alignment.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

@pytest.mark.parametrize('f10,expected_cross,expected_delta',[(1.,3.,5.25),(-1.,-3.,-.75)])
def test_Q_signed_interaction_can_help_or_hurt(f10,expected_cross,expected_delta):
    r=e.decompose(np.zeros(3),np.full(3,f10),np.ones(3),np.zeros(3))
    assert r['unit_mse']==1 and r['text_energy']==2.25
    assert r['twice_error_text_inner_product']==expected_cross
    assert r['legacy_minus_unit']==expected_delta
    assert r['decomposition_error']<=r['decomposition_bound']


def test_common_text_increment_cancels_raw_difference_not_error_contrast():
    zero=np.zeros(2);text=np.ones(2);a=np.full(2,2.);b=np.ones(2)
    pa=e.decompose(zero,text,a,zero);pb=e.decompose(zero,text,b,zero)
    np.testing.assert_array_equal((a+1.5*text)-(b+1.5*text),a-b)
    assert pa['unit_mse']-pb['unit_mse']==3
    assert pa['legacy_reduced_mse']-pb['legacy_reduced_mse']==6
    assert (pa['unit_mse']-pb['unit_mse'])-(pa['legacy_reduced_mse']-pb['legacy_reduced_mse'])==-3


def test_extent_differences_keep_axes_and_path_with_signed_DID():
    a=dict(min_xyz=[1,2,3],max_xyz=[4,5,6],range_xyz=[3,3,3],pelvis_path_length_m=2.)
    b=dict(min_xyz=[2,1,4],max_xyz=[6,4,9],range_xyz=[4,3,5],pelvis_path_length_m=5.)
    d=e.extent_delta(a,b)
    assert d['range_xyz']==[-1,0,-2] and d['pelvis_path_length_m']==-3
    assert e.extent_delta(d,d)==dict(min_xyz=[0,0,0],max_xyz=[0,0,0],range_xyz=[0,0,0],pelvis_path_length_m=0.)


def test_full_motion_contrast_is_new_minus_old_and_contact_stays_NA():
    p=np.zeros((4,22,3));q=p+[3.,0.,4.]
    a=dict(case_id='new',metrics={k:2. for k in e.KEYS})
    b=dict(case_id='old',metrics={k:5. for k in e.KEYS})
    ext=dict(min_xyz=[0,0,0],max_xyz=[1,1,1],range_xyz=[1,1,1],pelvis_path_length_m=0.)
    for r in (a,b):r['metrics']['motion_extent_m']=ext
    a['metrics']['foot_contact_speed_proxy_m_s']=None
    r=e.contrast(a,b,p,q)
    assert r['metrics']['global_MPJPE_m']==-3
    assert r['metrics']['foot_contact_speed_proxy_m_s'] is None
    assert r['mean_joint_distance_m']==5 and r['max_abs_m']==4


def test_identity_respacing_preserves_frozen_runtime_beta_identity():
    assert e.ah(e.frozen_betas())=='9e50a88ff4dcc6a477e9c621aafe1ca7a29f5bb0eae9111ce86b61795ef4f871'
