import numpy as np
import pytest

from itm.data.physical_time import resample_positions_at_times


def test_crop_preserves_physical_velocity_without_full_clip_stretch():
    times=np.arange(60)/30
    values=np.stack([3*times,np.zeros_like(times),times],-1)
    target=np.arange(12)/20
    actual=resample_positions_at_times(values,times,target)
    np.testing.assert_allclose(actual[:,0],3*target)
    assert actual[-1,0]<values[-1,0]


@pytest.mark.parametrize("target",[[-.01,0],[0,3],[0,0]])
def test_outside_or_duplicate_target_times_rejected(target):
    with pytest.raises(ValueError):resample_positions_at_times(np.zeros((3,3)),[0,1,2],target)
