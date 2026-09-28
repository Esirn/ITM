"""CPU-only regression of immutable saved-state population and signed evaluation."""
import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('freeze_v', ROOT / 'scripts/freeze_b0_state_diagnostic.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
PACKAGE = ROOT / v.REL


def read(name):
    return v.read(PACKAGE / (name + '.json'))


def test_signed_residual_and_weighted_partition():
    gt = np.zeros((263, 2))
    baseline = np.ones_like(gt)
    helpful = v.decomposition(baseline, gt, gt)
    harmful = v.decomposition(baseline, 2 * baseline, gt)
    assert helpful['all']['delta_mse'] == -1
    assert helpful['all']['cross'] == -2
    assert helpful['all']['residual_energy'] == 1
    assert harmful['all']['delta_mse'] == 3
    # Only 24 leg RIC coordinates change: weighting must not count the RIC twice.
    changed = gt.copy()
    changed[v.BLOCKS['ric_legs']] = 2
    result = v.decomposition(gt, changed, gt)
    assert result['ric_arms']['delta_mse'] == 0
    assert result['all']['delta_mse'] == pytest.approx(4 * 24 / 263)
    assert sum(result[k]['weighted_delta'] for k in v.PRIMARY) == pytest.approx(result['all']['delta_mse'])
    assert sorted(sum(v.PRIMARY.values(), [])) == list(range(263))
    assert sorted(sum([v.BLOCKS[k] for k in ('ric_legs', 'ric_trunk_head', 'ric_arms')], [])) == list(range(4, 67))


def test_bad_numeric_inputs_rejected():
    good = np.zeros((263, 2))
    for bad in (np.empty((263, 0)), np.zeros((263, 3)), np.zeros((262, 2)), np.full_like(good, np.nan), np.full_like(good, np.inf)):
        with pytest.raises(ValueError):
            v.decomposition(good, bad, good)


def test_all_paired_states_preserved_without_error_selection():
    groups = read('state_groups')
    old_samples = {r['case_id']: r for r in v.read(v.OLD / 'sampling_slots.json')}
    old = [t for t in v.read(v.OLD / 'trace_slots.json') if old_samples[t['case_id']]['condition'] == 'paired']
    assert len(groups) == len(old) == 48
    assert [g['trace_id'] for g in groups] == [t['trace_id'] for t in old]
    assert Counter(g['data_split'] for g in groups) == {'train': 24, 'val': 24}
    populations = {}
    for g in groups:
        populations.setdefault((g['data_split'], g['motion_id'], g['sensor_config']), []).append(g['timestep'])
        z = v.arrays(ROOT / g['saved_state']['path'])
        assert v.describe(z['x_t']) == g['state_descriptor']
        assert z['x_t'].shape == (1, 263, 1, g['frames'])
        assert z['frame_mask'].all()
        assert g['checkpoint']['sha256'] == v.CHECKPOINT_SHA
    assert len(populations) == 12
    assert all(ts == [900, 500, 250, 50] for ts in populations.values())


def test_same_state_four_branch_slots_and_cache_counts():
    groups, slots, cases = read('state_groups'), read('branch_slots'), read('input_cases')
    assert len(slots) == len({s['slot_id'] for s in slots}) == 192
    assert len(cases) == 12
    for i, g in enumerate(groups):
        batch = slots[4*i:4*i+4]
        assert [s['branch'] for s in batch] == ['T', 'P', 'S', 'R']
        assert [s['slot_id'] for s in batch] == g['slot_ids']
        assert {s['group_id'] for s in batch} == {g['group_id']}
        assert all(s['text_present'] for s in batch)
        assert [s['imu_present'] for s in batch] == [False, True, True, True]
        assert g['encoder_examples_max'] == (6 if i % 4 == 0 else 0)
        assert g['caption_encoder_calls'] == (1 if i % 4 == 0 else 0)
    assert sum(g['encoder_examples_max'] for g in groups) == 72
    assert sum(g['caption_encoder_calls'] for g in groups) == 12


def test_transforms_preserve_original_root_reference_and_clock():
    normal = read('design')['common']['imu_normalization']
    mean = np.array(normal['acceleration_mean'], dtype=np.float32)
    std = np.array(normal['acceleration_std'], dtype=np.float32)
    for case in read('input_cases'):
        z = v.arrays(ROOT / case['original_sample']['path'])
        a, o = z['acceleration'][0], z['orientation'][0]
        t = case['frames']
        # Index 0 is never reversed: the reference remains the original O[0].
        indices = [0] + list(range(t - 1, 0, -1))
        static_o = np.repeat(z['original_orientation0'][None], t, axis=0)
        ref = np.concatenate([(np.zeros_like(a) - mean) / std, static_o.reshape(t, 6, 9)], axis=-1)[None]
        for branch in case['branches']:
            name, desc = branch['branch'], branch['arrays']
            aa = np.zeros_like(a) if name == 'S' else a[indices] if name == 'R' else a
            oo = static_o if name == 'S' else o[indices] if name == 'R' else o
            norm = np.concatenate([(aa - mean) / std, oo.reshape(t, 6, 9)], axis=-1)[None]
            assert desc['acceleration'] == v.describe(aa[None])
            assert desc['orientation'] == v.describe(oo[None])
            assert desc['normalized_imu'] == v.describe(norm)
            assert desc['normalized_reference_imu'] == v.describe(ref)
            for key in ('timestamps_seconds', 'frame_mask', 'sensor_mask', 'original_orientation0', 'gt'):
                assert desc[key] == v.describe(z[key])
            if name == 'S':
                assert desc['normalized_imu'] == desc['normalized_reference_imu']


def test_anchors_are_distinct_from_effectiveness_contrasts():
    checks = read('software_checks')
    assert len(checks) == len({c['check_id'] for c in checks}) == 144
    assert Counter(c['branch'] for c in checks) == {'P': 48, 'T': 48, 'S': 48}
    assert all(c['atol'] == 1e-6 and c['rtol'] == 1e-6 for c in checks)
    assert {c['reference'] for c in checks} == {'f11', 'f10', 'T'}
    ev = read('evaluation')
    assert [(r['name'], r['role']) for r in ev['contrasts']] == [('P-S', 'primary'), ('P-R', 'primary'), ('P-T', 'supporting'), ('R-S', 'supporting')]
    assert ev['contrast_counts'] == {'rows': 192, 'block_cells': 1728, 'primary_rows': 96, 'supporting_rows': 96}


def test_historical_comparators_match_source_and_state_time():
    old = {g['group_id']: g for g in v.read(v.OLD / 'probe_groups.json')}
    rows, groups = read('historical_Q'), read('state_groups')
    assert len(rows) == 48
    for row, state in zip(rows, groups):
        assert row['group_id'] == state['group_id']
        assert len(row['existing_Q_groups']) == 2
        assert len({q['noise_seed'] for q in row['existing_Q_groups']}) == 2
        for q in row['existing_Q_groups']:
            original = old[q['group_id']]
            for key in ('data_split', 'motion_id', 'sensor_config', 'timestep'):
                assert original[key] == state[key]
            assert [b['probe_id'] for b in q['branches']] == original['probe_ids'][1:]
            assert [b['branch'] for b in q['branches']] == ['T', 'P', 'S', 'R']


def test_budget_is_bounded_and_not_authorized():
    b, p = read('budget'), read('package')
    assert b['authorized'] is False and b['authorization'] is None
    assert p['execution_authorization'] is None and p['run_directory'] is None and p['GPU_UUID'] is None
    assert b['total_seconds'] == 300 and b['GPUs'] == 1 and b['allocated_memory_bytes'] == 24 * 1024**3
    assert b['MDM_calls'] == 48 and b['MDM_branch_examples'] == 192
    assert b['encoder_examples_max'] == 72 and b['caption_encoder_calls_max'] == 12
    for key in ('training_updates', 'new_checkpoints', 'new_samples', 'new_diffusion_steps', 'new_noise_draws'):
        assert b[key] == 0
    assert b['deadline_starts'] == 'Before preflight/load'


def test_frozen_hashes_and_no_overwrite(tmp_path):
    package = read('package')
    assert len(package['artifacts']) == 12
    for entry in package['artifacts'] + read('sources'):
        assert v.sha(ROOT / entry['path']) == entry['sha256'], entry['path']
    with pytest.raises(FileExistsError):
        v.guarded_build(tmp_path)
