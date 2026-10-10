import pytest

from run_expanded_canonical_sampling import sampling_rows


def population():
    return [dict(motion_id=f'v{i}', split='val', frames=40 + i) for i in range(32)] + [
        dict(motion_id=mid, split='train', frames=104) for mid in ('000027', '000506', '004728')]


def test_fixed_population_and_matched_noise_without_test_sources():
    rows = sampling_rows(population())
    assert len(rows) == len({r['case_id'] for r in rows}) == 38
    val = [r for r in rows if r['split'] == 'val']
    assert len(val) == 32 and {r['motion_id'] for r in val} == {f'v{i}' for i in range(8)}
    for mid in {r['motion_id'] for r in val}:
        for sensor in ('head', 'wrists'):
            pair = [r for r in val if r['motion_id'] == mid and r['sensor_config'] == sensor]
            assert len({r['noise_group'] for r in pair}) == len({r['frames'] for r in pair}) == 1
            assert {r['condition'] for r in pair} == {'paired', 'mdm_text_g2_5'}
            assert all(r['text_present'] for r in pair)
            assert [r['imu_present'] for r in pair] == [True, False]
    assert all(r['condition'] == 'paired' for r in rows if r['split'] == 'train')


def test_no_silent_missing_anchor_or_validation_replacement():
    with pytest.raises(ValueError, match='Fixed 8 validation/3 training anchor'):
        sampling_rows(population()[:-1])
