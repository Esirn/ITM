"""Population rejection uses synthetic role-only metadata, never metric evidence."""
import pytest
from freeze_b0_cross_source_component_CPU import validate_population


def fixture():
    sources=[('train',m) for m in ('000027','000506','004728')]+[('val',m) for m in ('000139','000382','000433')];rows=[]
    for split,motion in sources:
        conditions=['paired','text_only','physical_static']+(['imu_only','backbone_bypass','numeric_zero','tail_reversed'] if split=='val' else [])
        for c in ('head','wrists'):
            for condition in conditions:
                for t in (900,500,250,50):
                    for a in ('F','U'):rows.append(dict(status='valid',arm=a,pair_case_id=split+motion+c+condition,timestep=t,condition=condition,data_split=split,motion_id=motion,sensor_config=c))
    return rows


def test_all_conditions_and_pairs_accept():assert validate_population(fixture())['paired']==96


def test_missing_or_duplicate_pair_rejected():
    rows=fixture()
    with pytest.raises(ValueError):validate_population(rows[:-1])
    rows[0]=dict(rows[1])
    with pytest.raises(ValueError):validate_population(rows)


def test_negative_source_or_invalid_row_rejected():
    rows=fixture();rows[0]['motion_id']='new-source'
    with pytest.raises(ValueError):validate_population(rows)
    rows=fixture();rows[0]['status']='invalid'
    with pytest.raises(ValueError):validate_population(rows)
