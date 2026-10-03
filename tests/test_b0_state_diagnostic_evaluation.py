"""Owner02 signed algebra, source-weighting and final-ledger acceptance checks."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('state_eval',SCRIPTS/'evaluate_b0_state_diagnostic.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


def array(value):return np.full((263,1,2),value,dtype=np.float32)


@pytest.mark.parametrize('prediction,delta,cross,energy',[(0.,-1.,-2.,1.),(2.,3.,2.,1.),(1.,0.,0.,0.)])
def test_signed_residual_help_harm_null(prediction,delta,cross,energy):
    r=e.contrast_metrics(array(prediction),array(1),array(0))['all']
    assert (r['delta_mse'],r['cross'],r['residual_energy'])==(delta,cross,energy)
    assert r['prediction_difference']['rms']==np.sqrt(energy)
    assert r['prediction_difference']['count']==526


def test_partition_weight_and_absolute_feature_units():
    p=array(0);p[e.BLOCKS['ric_legs']]=2
    result=e.branch_metrics(p,array(0));delta=e.contrast_metrics(p,array(0),array(0))
    assert result['ric_legs']['mse']==4 and result['ric_arms']['mse']==0
    assert result['all']['mse']==pytest.approx(96/263)
    assert sum(delta[k]['weighted_delta'] for k in e.PRIMARY)==pytest.approx(delta['all']['delta_mse'])
    assert delta['ric_legs']['weighted_delta']==delta['ric']['weighted_delta']
    assert result['ric_legs']['prediction']['max_abs']==2


def test_bad_shape_nonfinite_and_empty_are_rejected():
    for a in (np.zeros((2,263)),np.zeros((263,1,0)),array(np.nan),array(np.inf)):
        with pytest.raises(ValueError):e.branch_metrics(a,array(0))
    with pytest.raises(ValueError):e.contrast_metrics(np.zeros((263,1,3)),array(0),array(0))


def row(mid,rep,prediction,t=50):
    return dict(data_split='train',sensor_config='head',timestep=t,motion_id=mid,replicate=rep,contrast='P-R',blocks=e.contrast_metrics(array(prediction),array(0),array(0)))


def test_average_Q_replicates_before_equal_source_means_and_keep_t_separate():
    q=[row(mid,r,value) for mid,values in [('a',(0,2)),('b',(1,1)),('c',(3,3))] for r,value in enumerate(values)]
    means=e.source_means(q,2)
    # mean of squared errors (2,1,9), not squared mean predictions (1,1,9).
    a=e.aggregate(means)[0]
    assert a['blocks']['all']['delta_mse']['mean']==4
    assert a['blocks']['all']['delta_mse']['min']==1
    assert a['blocks']['all']['delta_mse']['max']==9
    assert a['n_sources']==a['planned_sources']==3
    other=[row(mid,r,1,900) for mid in ('a','b','c') for r in range(2)]
    assert len(e.aggregate(e.source_means(q+other,2)))==2
    with pytest.raises(ValueError):e.source_means(q[:-1],2)
    with pytest.raises(ValueError):e.source_means([q[0],q[0]],2)
    with pytest.raises(ValueError):e.aggregate(means+means)


def test_V_minus_Q_pairing_sign_and_missing_source_rejection():
    v=e.source_means([row('a',0,2)],1)
    q=e.source_means([row('a',0,1),row('a',1,1)],2)
    r=e.paired_state_comparison(v,q)[0]
    assert r['blocks']['all']['V_minus_Q_delta_mse']==3
    assert r['V_replicates']==1 and r['Q_replicates']==2
    bad=deepcopy(q);bad[0]['motion_id']='b'
    with pytest.raises(ValueError):e.paired_state_comparison(v,bad)


def test_pending_sidecar_requires_completed_final_chain():
    good=dict(status='completed',branches=[dict(slot_id='V001-P',status='completed',executed=True,diagnostic_only=False)])
    e.completed_chain(good,good,good,'V001-P')
    for mutation in ('status','executed','diagnostic_only'):
        bad=deepcopy(good);bad['branches'][0][mutation]={'status':'failed','executed':False,'diagnostic_only':True}[mutation]
        with pytest.raises(ValueError):e.completed_chain(good,good,bad,'V001-P')
    bad=deepcopy(good);bad['status']='failed'
    with pytest.raises(ValueError):e.completed_chain(bad,good,good,'V001-P')
