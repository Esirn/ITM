import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from freeze_b0_prefix_mechanism import prefix_noise


def test_selects_initial_draw_and_all100_posterior_updates():
    rows=[dict(step=t,noise_sha256=str(t)) for t in [-1]+list(range(999,0,-1))]
    prefix=prefix_noise(rows)
    assert len(prefix)==101
    assert prefix[-1]['step']==900
    assert prefix==rows[:101]
    assert prefix[0] is rows[0]


@pytest.mark.parametrize('fault',['missing','reordered','wronginitial'])
def test_rejects_incomplete_or_changed_noise_schedule(fault):
    rows=[dict(step=t) for t in [-1]+list(range(999,0,-1))]
    if fault=='missing':rows.pop()
    elif fault=='reordered':rows[1],rows[2]=rows[2],rows[1]
    else:rows[0]['step']=0
    with pytest.raises(ValueError):prefix_noise(rows)
