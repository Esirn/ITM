import numpy as np
import pytest

from itm.metrics.sensor_relative import evaluate_sensor_relative


def example():
    t = np.linspace(0, 1, 24)
    g = np.zeros((len(t), 22, 3))
    g[:, 0, 0] = t
    g[:, 15, 1] = 1.5
    g[:, 20, 0] = t + np.sin(t * 2 * np.pi)
    g[:, 21, 2] = np.cos(t * 2 * np.pi)
    return g


@pytest.mark.parametrize('sensor', ['head', 'wrists'])
def test_time_varying_whole_body_translation_is_removed(sensor):
    g = example()
    translation = np.arange(len(g))[:, None, None] * np.array([.1, -.04, .2])
    r = evaluate_sensor_relative(g + translation, g, sensor)
    assert r['observed_relative_position_mean_m'] < 1e-14
    assert r['observed_relative_velocity_RMSE_m_s'] < 1e-12
    assert r['observed_global_position_MSE_m2'] > 1


def test_static_mean_cannot_pass_dynamic_control():
    g = example()
    p = g.copy()
    local = g[:, [20, 21]] - g[:, :1]
    p[:, [20, 21]] = g[:, :1] + local.mean(axis=0, keepdims=True)
    r = evaluate_sensor_relative(p, g, 'wrists')
    assert r['observed_relative_static_error_RMS_m'] < 1e-14
    assert r['observed_relative_velocity_RMSE_m_s'] > 1
    assert r['observed_relative_spread_RMS_m'] < 1e-14
    assert r['GT_observed_relative_spread_RMS_m'] > .1


def test_only_observed_joints_and_root_determine_proxies():
    g = example()
    p = g.copy()
    p[:, 5] += 100
    assert evaluate_sensor_relative(p, g, 'wrists') == evaluate_sensor_relative(g, g, 'wrists')
    p[:, 20, 1] += .5
    assert evaluate_sensor_relative(p, g, 'head')['observed_relative_position_mean_m'] == 0
    assert evaluate_sensor_relative(p, g, 'wrists')['observed_relative_position_mean_m'] == .25


@pytest.mark.parametrize('bad', ['short', 'nonfinite', 'shape', 'config'])
def test_invalid_sequences_are_rejected(bad):
    p = g = example()
    config = 'wrists'
    if bad == 'short': p = g = g[:3]
    if bad == 'nonfinite': p = p.copy(); p[0, 0, 0] = np.nan
    if bad == 'shape': p = p[:, :21]
    if bad == 'config': config = 'pelvis'
    with pytest.raises(ValueError): evaluate_sensor_relative(p, g, config)
