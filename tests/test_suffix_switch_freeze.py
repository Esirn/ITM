import numpy as np
import pytest

from freeze_b0_suffix_switch import suffix_noise,validate_terminal


def test_suffix_uses_no_new_initial_or_zero_noise():
    records=[dict(step=t,noise_hash=str(t)) for t in [-1]+list(range(999,0,-1))]
    result=suffix_noise(records)
    assert len(result)==899 and result[0] is records[101] and result[-1] is records[-1]
    assert [r['step'] for r in result]==list(range(899,0,-1))
    assert len(result)*4==3596


@pytest.mark.parametrize('change',['missing','duplicate','zero'])
def test_reject_altered_historical_plan(change):
    records=[dict(step=t) for t in [-1]+list(range(999,0,-1))]
    if change=='missing':records.pop()
    elif change=='duplicate':records[101]['step']=900
    else:records[-1]['step']=0
    with pytest.raises(ValueError):suffix_noise(records)


def test_terminal_full_native_identity_required():
    z=dict(x_899=np.zeros((1,263,1,7),np.float32),frame_mask=np.ones((1,7),bool))
    validate_terminal(z,7)
    for replacement in [np.zeros((1,263,1,7),np.float64),np.zeros((263,7),np.float32),np.full((1,263,1,7),np.nan,np.float32)]:
        with pytest.raises(ValueError):validate_terminal(dict(z,x_899=replacement),7)
    with pytest.raises(ValueError):validate_terminal(dict(z,frame_mask=np.array([[True]*6+[False]])),7)
