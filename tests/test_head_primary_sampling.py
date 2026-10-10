import pytest
from freeze_head_primary_sampling import head_rows


def base():
    return dict(rows=[dict(case_id=f'{split}{i}{sensor}',noise_group=f'noise{i}{sensor}',split=split,
        sensor_config=sensor,condition='paired',imu_present=True,text_scale=2.5)
        for split,n in [('val',8),('train',3)] for i in range(n) for sensor in ['head','wrists']])


def test_head_selection_keeps_original_random_addresses_and_guidance():
    b=base();rows=head_rows(b)
    assert len(rows)==11 and all(r['sensor_config']=='head' for r in rows)
    for new,old in zip(rows,[r for r in b['rows'] if r['sensor_config']=='head']):
        assert {k:v for k,v in new.items() if k!='case_id'}=={k:v for k,v in old.items() if k!='case_id'}


def test_missing_head_or_unmatched_guidance_is_rejected():
    b=base();b['rows']=b['rows'][2:]
    with pytest.raises(ValueError):head_rows(b)
    b=base();b['rows'][0]['text_scale']=1
    with pytest.raises(ValueError):head_rows(b)
