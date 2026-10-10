"""Saved-array identity rejection; synthetic data are never experiment results."""
from copy import deepcopy

import numpy as np
import pytest

from analyze_b0_suffix_responses import validate_raw,F200_SHA,NAMES
from evaluate_b0_native_training_sufficiency import descriptor


def fixture():
    job=dict(trajectory_id='CPUfixture',timestep=50,frames=3,condition='switch_imu_off',raw=dict(path='CPUfixture',sha256='CPUfixture'))
    z={k:np.ones((1,263,1,3),np.float32) for k in NAMES};z['frame_mask']=np.ones((1,3),bool)
    meta=dict(status='completed',trajectory_id=job['trajectory_id'],timestep=50,checkpoint=dict(sha256=F200_SHA),raw_artifact=job['raw'],arrays={k:descriptor(v) for k,v in z.items()})
    return job,meta,z


def test_native_saved_identity_accepts():validate_raw(*fixture())


@pytest.mark.parametrize('change',['time','weight','hash','dtype','mask','nonfinite','strict_absence'])
def test_saved_drift_rejected(change):
    j,m,z=fixture()
    if change=='time':m['timestep']=49
    elif change=='weight':m['checkpoint']['sha256']='U200'
    elif change=='hash':m['raw_artifact']=dict(path='other',sha256='other')
    elif change=='dtype':z['f11']=z['f11'].astype(np.float64)
    elif change=='mask':z['frame_mask'][0,1]=False
    elif change=='nonfinite':z['x_t'][0,0,0,1]=np.nan
    else:z['f11'][0,0,0,1]+=1
    m['arrays']={k:descriptor(v) for k,v in z.items()}
    with pytest.raises(ValueError):validate_raw(j,m,z)
