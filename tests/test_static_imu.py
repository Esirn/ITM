import numpy as np
import pytest

from itm.data.static_imu import static_imu_fixture


def normalization():
    return {"acceleration_mean":np.ones((6,3)).tolist(),"acceleration_std":np.full((6,3),2).tolist()}


@pytest.mark.parametrize("config,slots", [("head",[4]),("wrists",[0,1])])
def test_static_fixture_is_physical_static_not_normalized_zero(config, slots):
    fixture = static_imu_fixture(frames=12,fps=20,sensor_config=config,normalization=normalization())
    np.testing.assert_array_equal(fixture["acceleration"],0)
    np.testing.assert_array_equal(fixture["imu"][...,:3],-.5)
    np.testing.assert_array_equal(fixture["orientation"],np.broadcast_to(np.eye(3),(12,6,3,3)))
    assert np.flatnonzero(fixture["sensor_mask"]).tolist() == slots
    assert fixture["timestamps_seconds"][-1] == .55


def test_reflection_or_bad_normalization_rejected():
    with pytest.raises(ValueError,match="proper"):
        static_imu_fixture(frames=12,fps=20,sensor_config="head",normalization=normalization(),
                           orientations=np.broadcast_to(np.diag([-1,1,1]),(6,3,3)))
    bad=normalization(); bad["acceleration_std"][0][0]=0
    with pytest.raises(ValueError,match="positive std"):
        static_imu_fixture(frames=12,fps=20,sensor_config="head",normalization=bad)
