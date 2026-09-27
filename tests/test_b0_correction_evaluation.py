"""Analytic checks for evaluation sign, missing denominators and hierarchical means."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('correction_eval',SCRIPTS/'evaluate_b0_correction.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


def test_group_mean_preserves_failure_and_na_denominators():
    rows=[dict(case_id=str(i),arm='R',status='valid',metrics={k:float(v) for k in e.KEYS}) for i,v in enumerate([2,6])]
    rows[0]['metrics']['foot_contact_speed_proxy_m_s']=None
    rows.append(dict(case_id='missing',arm='R',status='failure',reason='file absent'))
    r=e.averages(rows,('arm',))[0]
    assert (r['planned'],r['valid'],r['partial'],r['missing'])==(3,2,True,['missing'])
    assert r['metrics']['global_MPJPE_m']['mean']==4
    assert r['metrics']['foot_contact_speed_proxy_m_s']['n']==1
    assert r['metrics']['foot_contact_speed_proxy_m_s']['mean']==6


def test_probe_source_average_first_and_timestep_isolation():
    # Unequal replicate counts stress source weighting: (mean(0,2)+10)/2=5.5, not 4.
    rows=[dict(motion_id=mid,timestep=t,values={'x':v,'zero_denominator':None})
          for mid,t,v in [('a',50,0),('a',50,2),('b',50,10),('a',900,100)]]
    a,b=e.probe_aggregate(rows,('timestep',))
    assert a['values']['x']['mean']==5.5 and a['values']['x']['n']==2
    assert a['values']['zero_denominator']['mean'] is None
    assert b['values']['x']['mean']==100


def test_delta_direction_and_na_propagation():
    a={k:2. for k in e.KEYS};b={k:5. for k in e.KEYS}
    a['foot_contact_speed_proxy_m_s']=None
    d=e.metric_delta(a,b)
    assert d['global_MPJPE_m']==-3 and d['foot_contact_speed_proxy_m_s'] is None


def test_output_distance_is_euclidean_not_component_average():
    a=np.zeros((7,22,3),np.float32);b=a+[3.,0.,4.]
    d=e.distance(a,b)
    assert d['mean_joint_distance_m']==5 and d['max_abs_m']==4 and not d['bitwise_equal']


def test_noise_tampering_rejected():
    address=dict(version='itm-pcg64-addressed-noise-v1',backbone_id='test',case_id='case',sample_seed=2,step=5,shape=[1,3,1,2])
    h=e.digest(address);x=np.random.Generator(np.random.PCG64(int(h,16))).standard_normal(address['shape']).astype(np.float32)
    draw=dict(address,address_sha256=h,noise_sha256=e.ah(x))
    np.testing.assert_array_equal(e.reconstruct_noise(draw),x)
    draw['case_id']='changed'
    with pytest.raises(AssertionError):e.reconstruct_noise(draw)


def test_probe_reductions_analytical():
    s=e.stats(np.array([3.,4.]));assert s['count']==s['finite_count']==2
    assert s['sums_sq']==25 and s['rms']==pytest.approx(np.sqrt(12.5)) and s['max_abs']==4
