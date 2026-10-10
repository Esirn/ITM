"""Reject incorrect original T-prefix paired source roles before freezing."""
from copy import deepcopy
import pytest
from freeze_b0_imu_main_knockout import select_on,first_anchor


def rows():
    return [dict(trajectory_id='Hx-'+c,condition='switch_imu_on',sensor_config=c,imu_present=True,text_present=True,frames=136,checkpoint='F200',motion_id='000506',data_split='train',diffusion_steps=list(range(899,-1,-1))) for c in ('head','wrists')]


def test_selection_unique_roles():
    r=rows();assert len(select_on(r))==2
    with pytest.raises(ValueError):select_on(r+[deepcopy(r[0])])
    r[0]['imu_present']=False
    with pytest.raises(ValueError):select_on(r)


def test_wrong_first_anchor_rejected():
    r=rows()[0];j=dict(trajectory_id=r['trajectory_id'],timestep=899,sensor_config='head',condition='switch_imu_on',prefix_history='T_text_only',suffix_condition='paired',imu_present=True,raw={'fixture':True},metadata={'fixture':True})
    assert first_anchor(r,j)['arrays']==['x_t','f00','f10','f01','f11']
    for k,v in [('condition','keep_text_only'),('prefix_history','F_paired'),('suffix_condition','text_only'),('timestep',500),('sensor_config','wrists')]:
        wrong=dict(j);wrong[k]=v
        with pytest.raises(ValueError):first_anchor(r,wrong)


def test_schedule_or_weight_drift_rejected():
    for k,v in [('checkpoint','U200'),('diffusion_steps',list(range(898,-1,-1))),('frames',135)]:
        r=rows();r[0][k]=v
        with pytest.raises(ValueError):select_on(r)
