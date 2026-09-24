import numpy as np
import pytest

from itm.data.observation import (
    device_acceleration_to_world, load_standard_cache, observe_sensor_positions,
)
from itm.data.standard_imu import moving_average, second_difference


def observe(p, *, fps=100, rotations=None, **kwargs):
    rotations = np.broadcast_to(np.eye(3), (*p.shape[:2], 3, 3)).copy() if rotations is None else rotations
    return observe_sensor_positions(p, rotations, np.arange(len(p)) / fps,
                                    np.ones(p.shape[1], dtype=bool), **kwargs)


@pytest.mark.parametrize("velocity,acceleration", [(0., 0.), (2., 0.), (1., 3.)])
def test_physical_translation(velocity, acceleration):
    t = np.arange(60) / 100
    p = np.zeros((60, 1, 3))
    p[:, 0, 0] = velocity*t + .5*acceleration*t*t
    out = observe(p)
    np.testing.assert_allclose(out.acceleration[3:-3, 0, 0], acceleration, atol=1e-10)
    assert not out.acceleration_valid[:3].any()
    assert not out.acceleration_valid[-3:].any()
    assert out.orientation_valid.all()


def test_offset_rotating_sensor_has_centripetal_and_tangential_acceleration():
    t = np.arange(200) / 1000
    omega, alpha, radius = 2., 3., .5
    angle = omega*t + .5*alpha*t*t
    p = radius * np.stack([np.cos(angle), np.sin(angle), np.zeros_like(t)], -1)[:, None]
    expected = -(omega+alpha*t)[:, None]**2 * p[:, 0]
    expected += alpha * radius * np.stack([-np.sin(angle), np.cos(angle), np.zeros_like(t)], -1)
    out = observe(p, fps=1000, smooth_window=1)
    np.testing.assert_allclose(out.acceleration[1:-1, 0], expected[1:-1], atol=2e-5)


def test_global_transform_and_legacy_interior():
    rng = np.random.default_rng(23)
    p = rng.normal(size=(25, 2, 3)).cumsum(0)
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    out = observe(p)
    transformed = observe(p @ rotation.T + [3, 2, 1], rotations=np.broadcast_to(rotation, (25, 2, 3, 3)))
    np.testing.assert_allclose(transformed.acceleration[3:-3], out.acceleration[3:-3] @ rotation.T, atol=1e-9)
    legacy = moving_average(second_difference(p, fps=100), 5)
    np.testing.assert_allclose(out.acceleration[3:-3], legacy[3:-3], atol=.03)


def test_gaps_invalidate_entire_stencil_and_inactive_data_is_ignored():
    p = np.zeros((25, 2, 3)); p[12, 0] = np.nan; p[:, 1] = np.nan
    rot = np.broadcast_to(np.eye(3), (25, 2, 3, 3)).copy(); rot[:, 1] = np.nan
    out = observe_sensor_positions(p, rot, np.arange(25)/20, np.array([True, False]))
    assert not out.acceleration_valid[9:16, 0].any()
    assert out.acceleration_valid[8, 0] and out.acceleration_valid[16, 0]
    assert not out.orientation_valid[:, 1].any()
    assert not out.acceleration_valid[:, 1].any()


def test_reflection_not_a_rotation_and_time_warp_rejected():
    p = np.zeros((15, 1, 3))
    r = np.broadcast_to(np.diag([-1., 1, 1]), (15, 1, 3, 3))
    assert not observe(p, rotations=r).orientation_valid.any()
    with pytest.raises(ValueError, match="uniform"):
        observe_sensor_positions(p, r, np.arange(15)**2, np.array([True]))


def test_conditional_device_bridge():
    f = np.array([[0., 9.81, 0.]])
    r = np.eye(3)[None]
    np.testing.assert_allclose(device_acceleration_to_world(f, r, quantity="body_specific_force", gravity_world=[0,-9.81,0]), 0)
    np.testing.assert_allclose(device_acceleration_to_world(f, r, quantity="body_linear_acceleration"), f)
    with pytest.raises(ValueError, match="forbids"):
        device_acceleration_to_world(f, r, quantity="body_linear_acceleration", gravity_world=[0,-9.81,0])
    with pytest.raises(ValueError, match="requires"):
        device_acceleration_to_world(f, r, quantity="body_specific_force")


def test_legacy_vector_schema_rejected_before_loading():
    with pytest.raises(ValueError, match="schema"):
        load_standard_cache({"schema": "joint_vector_legacy", "imu_path": "nonexistent"})
    with pytest.raises(ValueError, match="physical metadata"):
        load_standard_cache({"schema": "standard_imu_v1", "acceleration_unit": "position_per_frame_squared"})


def test_crop_boundaries_and_aliasing_are_visible():
    t = np.arange(100)/100
    p = np.zeros((100, 1, 3)); p[:,0,0] = np.sin(2*np.pi*20*t)
    full = observe(p, smooth_window=1)
    crop = observe(p[20:80], smooth_window=1)
    np.testing.assert_allclose(crop.acceleration[1:-1], full.acceleration[21:79])
    # Sampling a 20 Hz sine at 20 Hz destroys it; this operator does not disguise
    # unfiltered decimation as an equivalent physical observation.
    low = observe(p[::5], fps=20, smooth_window=1)
    assert np.nanmax(np.abs(low.acceleration)) < 1e-8
    assert np.nanmax(np.abs(full.acceleration)) > 1000
