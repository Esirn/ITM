"""Role/time rejection for the next fixed-state probe; synthetic metadata only."""
import pytest
from freeze_b0_control_mean_probe import select_states


def fixture():
    return [dict(arm=a,condition='paired',status='valid',pair_case_id=f'fixture-{c}',timestep=t) for a in ('F','U') for c in range(12) for t in (900,500,250,50)]


def test_exact_F_paired_selection():
    p=select_states(fixture());assert len(p)==48 and all(r['arm']=='F' for r in p)


def test_missing_duplicate_or_wrong_time_rejected():
    rows=fixture()
    with pytest.raises(ValueError):select_states(rows[1:])
    rows=fixture();rows[0]=dict(rows[1])
    with pytest.raises(ValueError):select_states(rows)
    rows=fixture();rows[0]['timestep']=51
    with pytest.raises(ValueError):select_states(rows)


def test_wrong_arm_condition_or_failed_status_rejected():
    for k,v in [('arm','U'),('condition','text_only'),('status','diagnostic_only')]:
        rows=fixture();rows[0][k]=v
        with pytest.raises(ValueError):select_states(rows)
