"""Synthetic fixtures reject role drift and prevent dropped factorial denominators."""
import pytest
from test_suffix_response_adapter import fixture
from analyze_b0_history_factorial_responses import validate_raw,role,complete_table,build_contrasts,FIELDS


def test_text_role_checks_strict_absence():
    j,m,z=fixture();j.update(condition='keep_text_only',prefix_history='T_text_only',suffix_condition='text_only',imu_present=False)
    validate_raw(j,m,z)
    z['f11'][0,0,0,1]+=1
    from evaluate_b0_native_training_sufficiency import descriptor
    m['arrays']={k:descriptor(v) for k,v in z.items()}
    with pytest.raises(ValueError):validate_raw(j,m,z)


@pytest.mark.parametrize('field,value',[('prefix_history','F_paired'),('suffix_condition','paired'),('imu_present',True),('condition','keep_paired')])
def test_wrong_history_suffix_role_rejected(field,value):
    r=dict(condition='keep_text_only',prefix_history='T_text_only',suffix_condition='text_only',imu_present=False);r[field]=value
    with pytest.raises(ValueError):role(r)


def rows():
    return [dict(prefix_history=h,sensor_config=c,timestep=t,suffix_condition=s,row_id=f'{h}-{c}-{t}-{s}',status='valid',temporal={'actual_combined_prediction':{'fixture':dict.fromkeys(FIELDS,(2 if h=='F_paired' else 5) if s=='paired' else 1)}}) for h in ('F_paired','T_text_only') for c in ('head','wrists') for t in range(900) for s in ('paired','text_only')]


def test_duplicate_or_missing_cells_rejected():
    r=rows()
    with pytest.raises(ValueError):complete_table(r[:-1])
    with pytest.raises(ValueError):complete_table(r+[r[0]])


def test_factorial_sign_and_missing_denominator():
    r=rows();r[0]['status']='invalid'
    rms={(h,c,t):dict(root_ric67=0.) for h in ('F_paired','T_text_only') for c in ('head','wrists') for t in range(900)}
    a,b=build_contrasts(r,rms)
    assert len(a)==3600 and len(b)==1800
    assert sum(x['status']=='inconclusive' for x in a)==sum(x['status']=='inconclusive' for x in b)==1
    assert next(x for x in b if x['status']=='valid')['actual_prediction']['fixture']['prediction_temporal_energy']==3
    assert next(x for x in a if x['prefix_history']=='F_paired' and x['status']=='valid')['actual_prediction']['fixture']['mse']==1
