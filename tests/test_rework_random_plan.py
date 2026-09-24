import numpy as np

from itm.data.random_plan import sampling_noise


def test_shared_noise_survives_extra_branch_draws_and_call_order():
    params = dict(backbone_id="mdm-test", case_id="clip", sample_seed=42)
    a, meta = sampling_noise((3, 20), **params, step=-1)
    sampling_noise((3, 20), **params, step=500)
    np.random.standard_normal(123)
    b, again = sampling_noise((3, 20), **params, step=-1)
    np.testing.assert_array_equal(a, b)
    assert meta == again
    next_noise, _ = sampling_noise((3, 20), **params, step=0)
    assert not np.array_equal(a, next_noise)
