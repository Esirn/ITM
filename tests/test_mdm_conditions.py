"""Public v0.2 input integration, without certifying world alignment."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from itm.models.mdm_conditions import load_method_cache, sample_imu_at_times, supported_frame_times


ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache_record(tmp_path, *, frames=3, fps=10.):
    path = tmp_path / 'cache.npz'
    np.savez(path, acceleration=np.arange(frames)[:, None, None] * np.ones((frames, 6, 3)),
             orientation=np.broadcast_to(np.eye(3), (frames, 6, 3, 3)),
             joints=np.zeros((frames, 24, 3)), motion_id=np.array('fixture'),
             sensor_mask=np.ones(6, dtype=bool), fps=fps)
    return dict(imu_path=str(path), schema='standard_imu_v1', motion_id='fixture',
                num_frames=frames, fps=fps, split='train')


def normalization():
    return dict(acceleration_mean=np.ones((6, 3)).tolist(),
                acceleration_std=np.full((6, 3), 2.).tolist())


def test_seconds_sampling_uses_support_without_clamping(tmp_path):
    cache = load_method_cache(cache_record(tmp_path), workspace_root=tmp_path)
    times = supported_frame_times(cache['timestamps'])
    np.testing.assert_allclose(times, [0, .05, .1, .15, .2])
    acceleration, orientation = sample_imu_at_times(cache, times)
    np.testing.assert_allclose(acceleration[:, 0, 0], [0, .5, 1, 1.5, 2])
    np.testing.assert_allclose(orientation, np.broadcast_to(np.eye(3), orientation.shape))
    with pytest.raises(ValueError, match='outside source support'):
        sample_imu_at_times(cache, np.arange(6) / 20)


def test_cache_rejects_schema_and_quantity_before_model(tmp_path):
    record = cache_record(tmp_path)
    with pytest.raises(ValueError, match='manifest'):
        load_method_cache(dict(record, schema='unknown'), workspace_root=tmp_path)
    with pytest.raises(ValueError, match='quantity'):
        load_method_cache(dict(record, quantity='body_specific_force'), workspace_root=tmp_path)


@pytest.mark.parametrize('embedded_schema', ['standard_imu_prefix_diagnostic_v1', ['standard_imu_v1']])
def test_relabelled_or_malformed_embedded_schema_is_rejected(tmp_path, embedded_schema):
    record = cache_record(tmp_path)
    with np.load(record['imu_path']) as data:
        arrays = {key: data[key] for key in data.files}
    np.savez(record['imu_path'], **arrays, schema=embedded_schema)
    with pytest.raises(ValueError, match='Embedded cache schema'):
        load_method_cache(record, workspace_root=tmp_path)


def test_explicit_standard_embedded_schema_remains_readable(tmp_path):
    record = cache_record(tmp_path)
    with np.load(record['imu_path']) as data:
        arrays = {key: data[key] for key in data.files}
    np.savez(record['imu_path'], **arrays, schema='standard_imu_v1')
    assert load_method_cache(record, workspace_root=tmp_path)['acceleration'].shape == (3, 6, 3)


def test_sampling_static_uses_current_checkpoint_normalization(tmp_path):
    helper = script('sample_mdm_imu_control')
    record = cache_record(tmp_path)
    base = dict(motion_id='fixture', sensor_config='head', length=5)
    static = helper._load_case(dict(base, imu_mode='static'), {'fixture': record}, normalization())
    zero = helper._load_case(dict(base, imu_mode='zero'), {'fixture': record}, normalization())
    null = helper._load_case(dict(base, imu_mode='null'), {'fixture': record}, normalization())
    np.testing.assert_allclose(static['imu'][..., :3], -.5)
    np.testing.assert_allclose(static['imu'][..., 3:].reshape(5,6,3,3), np.broadcast_to(np.eye(3), (5,6,3,3)))
    assert not np.array_equal(static['imu'], zero['imu'])
    assert static['physical_static_fixture']['imu_condition_present'] is True
    assert null['imu_mode'] == 'null'
    assert static['imu_source_kind'] == 'declared_static_fixture'
    with pytest.raises(ValueError, match='outside target time support'):
        helper._load_case(dict(base, length=6), {'fixture': record}, normalization())


def test_training_loader_uses_explicit_grid_and_rejects_short_support(tmp_path):
    helper = script('train_mdm_imu_control')
    from itm.data.dataset import CaptionRecord
    dataset = helper._PairedDataset.__new__(helper._PairedDataset)
    dataset.workspace_root = tmp_path
    dataset.mean, dataset.std = np.zeros(263), np.ones(263)
    dataset.normalization = normalization()
    sample = dict(motion_id='fixture', motion=np.zeros((5,263)), caption='a person stands',
                  caption_record=CaptionRecord('a person stands', '', 0., 0.))
    dataset.items = [(sample, cache_record(tmp_path), 'head', (4,))]
    item = dataset[0]
    _, y = helper._collate([item])
    np.testing.assert_allclose(y['imu_timestamps'][0].numpy(), np.arange(5)/20)
    assert item['input_metadata']['alignment_status'] == 'uncertified_diagnostic'
    sample['motion'] = np.zeros((6,263))
    with pytest.raises(ValueError, match='outside source support'):
        dataset[0]


def test_witness_requires_exact_training_interval():
    helper = script('train_mdm_imu_control')
    metadata = [dict(effective_frames=196, time_interval_seconds=[0., 9.75])]
    row = dict(timestamps=dict(frames=199, fps=20., clock='HumanML_clip_local_seconds'))
    with pytest.raises(ValueError, match='no silent crop'):
        helper.validate_response_intervals([row], metadata)
    row['timestamps']['frames'] = 196
    row['_witness_timestamps_seconds'] = np.arange(199)/20
    with pytest.raises(ValueError, match='artifact time axis'):
        helper.validate_response_intervals([row], metadata)
    row['_witness_timestamps_seconds'] = np.arange(196)/20
    helper.validate_response_intervals([row], metadata)
    row['timestamps']['start_seconds'] = .05
    with pytest.raises(ValueError, match='no silent crop'):
        helper.validate_response_intervals([row], metadata)


def test_relabeling_uncertain_candidate_does_not_make_it_admissible(tmp_path):
    import json
    helper = script('train_mdm_imu_control')
    row = json.loads((ROOT/'docs/itm_rework/protocol/arm_spread_candidates_20260924_v1/candidates.jsonl').read_text().splitlines()[0])
    row['compatibility_label'] = 'compatible'
    path = tmp_path/'renamed.jsonl'; path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='compatible train'):
        helper.load_response_manifest(path)
