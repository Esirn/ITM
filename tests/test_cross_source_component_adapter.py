"""Synthetic metadata rejection and full denominator retention, never model evidence."""
from copy import deepcopy
import numpy as np
import pytest
from analyze_b0_cross_source_components import validate_raw,pair_rows,WEIGHTS,NAMES
from evaluate_b0_native_training_sufficiency import descriptor


def raw_fixture():
    j=dict(trace_id='synthetic',case_id='synthetic',timestep=50,condition='paired',arm='F',frames=3)
    z={k:np.ones((1,263,1,3),np.float32) for k in NAMES};z['frame_mask']=np.ones((1,3),bool)
    slot=dict(guidance={'fixture':True},historical_random_plan_sha256='fixture')
    m=dict(j,status='completed',scaled_t=[50],checkpoint_sha256=WEIGHTS['F'],guidance=slot['guidance'],random_plan_sha256='fixture',input_hashes={'fixture':True},arrays={k:descriptor(v) for k,v in z.items()},actual_return_sha256=descriptor(z['actual_combined_prediction'])['sha256'])
    return j,m,z,slot,{'fixture':True}


def test_metadata_accepts_native_fixture():validate_raw(*raw_fixture())


@pytest.mark.parametrize('change',['arm_weight','time','return_hash','inputs','dtype','mask','nonfinite'])
def test_source_shape_or_role_drift_rejected(change):
    j,m,z,s,i=raw_fixture()
    if change=='arm_weight':m['checkpoint_sha256']=WEIGHTS['U']
    elif change=='time':m['scaled_t']=[49]
    elif change=='return_hash':m['actual_return_sha256']='different'
    elif change=='inputs':m['input_hashes']={'other':True}
    elif change=='dtype':z['f10']=z['f10'].astype(np.float64)
    elif change=='mask':z['frame_mask'][0,1]=False
    else:z['f10'][0,0,0,1]=np.nan
    m['arrays']={k:descriptor(v) for k,v in z.items()}
    with pytest.raises(ValueError):validate_raw(j,m,z,s,i)


def pair_fixture():
    return [dict(pair_case_id='synthetic-'+str(c),timestep=50,arm=a,row_id=a+str(c),status='valid',data_split='val',motion_id=str(c),sensor_config='head',condition='paired',frames=3,component_terms={'fixture':{'delta':2. if a=='F' else 1.,'undefined':None}},hypotheses={'fixture':{'mse':2. if a=='F' else 1.}}) for c in range(240) for a in ('F','U')]


def test_missing_invalid_and_sign_retained():
    rows=pair_fixture();rows[0]['status']='invalid';p=pair_rows(rows)
    assert len(p)==240 and sum(r['status']=='inconclusive' for r in p)==1
    valid=next(r for r in p if r['status']=='valid');assert valid['component_terms']['fixture']=={'delta':1.,'undefined':None}
    with pytest.raises(ValueError):pair_rows(rows[:-1])
    rows=pair_fixture();rows[1]['condition']='text_only'
    with pytest.raises(ValueError):pair_rows(rows)
