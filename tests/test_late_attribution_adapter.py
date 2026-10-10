from copy import deepcopy
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_b0_late_attribution import validate_roles


def fixture(first):
    j=dict(data_split='train',motion_id='CPUfixture',sensor_config='wrists',timestep=50,
        direct_is_first=first,ordering='direct_at_F_state' if first else 'direct_at_reference_state')
    roles=[('F200','F_paired_trace'),('U200','F_paired_trace') if first else ('F200','U_paired_trace'),('U200','U_paired_trace')]
    m=[dict(status='completed',diagnostic_only=False,state=dict(**{k:j[k] for k in ['data_split','motion_id','sensor_config','timestep']},
        state_family=family,frames=3),job=dict(snapshot_id=w,branch_order=['T','P','S','R'],caption_batch=1,input_batch_shape=[4,263,1,3])) for w,family in roles]
    return j,m


@pytest.mark.parametrize('first',[True,False])
def test_both_orderings_keep_checkpoint_and_state_roles(first):
    j,m=fixture(first);validate_roles(j,m)


@pytest.mark.parametrize('fault',['family','weight','case','time','batch','partial'])
def test_rejects_role_or_unverified_slot_drift(fault):
    j,m=fixture(True);m=deepcopy(m)
    if fault=='family':m[1]['state']['state_family']='U_paired_trace'
    elif fault=='weight':m[1]['job']['snapshot_id']='F200'
    elif fault=='case':m[1]['state']['motion_id']='wrong'
    elif fault=='time':m[1]['state']['timestep']=250
    elif fault=='batch':m[1]['job']['input_batch_shape'][0]=5
    else:m[1]['diagnostic_only']=True
    with pytest.raises(ValueError):validate_roles(j,m)
