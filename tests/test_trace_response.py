import numpy as np
import pytest

from itm.metrics.trace_response import paired_rows, response_decomposition


def branches(base, text=0, imu=0, interaction=0):
    return dict(f00=base,f10=base+text,f01=base+imu,
                f11=base+text+imu+interaction)


def test_random_component_identity():
    r=np.random.default_rng(12)
    f=branches(*(r.normal(size=(41,263)) for _ in range(4)))
    for v in response_decomposition(f).values():
        assert v['identity_error']<=v['identity_bound']
        for c in v['conditional_changes'].values():
            assert c['identity_error']<=c['identity_bound']


def test_large_conditional_response_can_cancel_dynamic_base():
    base=np.arange(10,dtype=float)[:,None]*np.ones((1,263))
    v=response_decomposition(branches(base,imu=-base))['root_ric67']
    assert v['branch_energy']['f11']==0
    assert v['component_energy']['imu']>0
    c=v['conditional_changes']['imu_at_text']
    assert c['delta_temporal_energy']==-v['branch_energy']['f10']
    assert c['twice_base_delta_covariance']==-2*c['delta_centered_energy']


@pytest.mark.parametrize('absent',['text','imu'])
def test_absent_branch_has_zero_energy(absent):
    base=np.arange(6)[:,None]*np.ones((1,263))
    v=response_decomposition(branches(base))['uniform']
    assert v['component_energy'][absent]==0
    assert v['component_energy']['interaction']==0


def test_mask_discards_nonfinite_padding_before_centering():
    r=np.random.default_rng(5);f=branches(r.normal(size=(4,263)),imu=0.4)
    f={k:v.copy() for k,v in f.items()}
    for v in f.values():v[1]=np.nan
    mask=np.array([True,False,True,True])
    assert response_decomposition(f,mask)==response_decomposition({k:v[mask] for k,v in f.items()})


def test_mean_response_is_not_temporal_response():
    f=branches(np.zeros((5,263)),text=3)
    c=response_decomposition(f)['uniform']['conditional_changes']['text_at_no_imu']
    assert c['delta_raw_rms']==3
    assert c['delta_mean_energy']==9
    assert c['delta_centered_energy']==0


def test_missing_or_invalid_pair_remains_in_denominator():
    specs=[dict(trace_id='F1',historical_trace_id='U1',timestep=50,case_id='c')]
    rows=[dict(arm='F',trace_id='F1',status='valid')]
    assert paired_rows(rows,specs)[0]['U_status']=='missing'
    rows.append(dict(arm='U',trace_id='U1',status='invalid'))
    assert len(paired_rows(rows,specs))==1
    assert paired_rows(rows,specs)[0]['status']=='incomplete'


def test_rejects_invalid_valid_frame_or_shape():
    f=branches(np.zeros((3,263)));f['f01'][0,0]=np.inf
    with pytest.raises(ValueError):response_decomposition(f)
    with pytest.raises(ValueError):response_decomposition(branches(np.zeros((3,262))))
