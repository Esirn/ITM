"""Integration against the fixed admitted B0 delivery; no model training/generation."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from itm.models.mdm_canonical_conditions import (CanonicalB0Package, CANONICAL_SCOPE, CANONICAL_ALIGNMENT,
                                                load_canonical_cache)
from itm.models.mdm_conditions import load_method_cache

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'docs/itm_rework/protocol/canonical_b0_20260925_v3/package.json'


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture
def package():
    if not (ROOT/'outputs/itm_rework/canonical_b0_20260925_v3/train/000506/imu.npz').exists():
        pytest.skip('Fixed canonical B0 artifacts are not available')
    return CanonicalB0Package(PACKAGE, workspace_root=ROOT)


def dataset(package, **changes):
    options = dict(caption_plans=package.plans['train'], canonical_package=package)
    options.update(changes)
    return script('train_mdm_imu_control')._PairedDataset(
        package.manifests['train_motion'], package.manifests['train_imu'], np.zeros(263), np.ones(263),
        {'head': (4,), 'wrists': (0, 1)}, None, ROOT, **options)


def independent_normalization(package):
    values = []
    for mid, plan in package.plans['train'].items():
        with np.load(package.imus['train'][mid]['imu_path']) as d:
            times = np.arange(plan['frames'])/20.
            acc = np.stack([np.interp(times, d['timestamps'], column) for column in
                            d['acceleration'].reshape(-1, 18).T], axis=1).reshape(-1, 6, 3)
            values.append(acc.astype(np.float32).astype(np.float64))
    values = np.concatenate(values)
    return values.mean(0).astype(np.float32), np.sqrt(np.maximum(values.var(0), 1e-8)).astype(np.float32)


def test_normalization_exact_20hz_unique_train_only(package):
    ds = dataset(package)
    norm = ds.compute_normalization()
    mean, std = independent_normalization(package)
    np.testing.assert_array_equal(norm['acceleration_mean'], mean)
    np.testing.assert_allclose(norm['acceleration_std'], std, rtol=1e-7, atol=0)
    assert norm['source']['unique_frames'] == 398
    assert norm['source']['unique_motion_ids'] == ['000027', '000506', '004728']
    ds.set_normalization(norm)
    sampler = script('sample_mdm_imu_control')
    for index in range(len(ds)):
        item = ds[index]
        sample = sampler._load_case({'motion_id': item['motion_id'], 'sensor_config': item['sensor_config']},
                                    {}, norm, canonical_package=package, canonical_split='train')
        np.testing.assert_array_equal(item['imu'], sample['imu'])
        assert item['caption'] == sample['text']
        assert item['input_metadata']['alignment_status'] == CANONICAL_ALIGNMENT
    assert package.prepare('train', '000027')['input_metadata']['caption_index'] == 1


def test_all_36_cases_have_direct_gt_and_explicit_conditions(package):
    norm = dataset(package).compute_normalization()
    sampler = script('sample_mdm_imu_control')
    cases = json.loads(package.sampling_plan_path.read_text())
    assert len(cases) == 36
    for case in cases:
        sample = sampler._load_case(case, {}, norm, canonical_package=package)
        np.testing.assert_array_equal(sample['gt'], np.load(case['gt_path']))
        assert sample['text_present'] == case['text_present']
        assert sample['imu_present'] == case['imu_present']
        assert sample['input_metadata']['world_alignment_scope'] == CANONICAL_SCOPE
        assert sample['input_metadata']['training_admissible'] is False
        assert sample['text'] == package.plans['val'][case['motion_id']]['caption']
        if case['condition'] == 'numeric_zero':
            assert not sample['imu'].any() and sample['imu_present']
        if case['condition'] == 'physical_static':
            observed = package.prepare('val', case['motion_id'])['orientation'][0]
            np.testing.assert_array_equal(sample['raw_orientation'], np.broadcast_to(observed, sample['raw_orientation'].shape))
            assert not sample['raw_acceleration'].any()
            np.testing.assert_allclose(sample['imu'][..., :3], np.broadcast_to(
                -np.asarray(norm['acceleration_mean'])/np.asarray(norm['acceleration_std']), sample['imu'][..., :3].shape), rtol=1e-6)


def test_canonical_is_opt_in_and_val_cannot_enter_training(package):
    with pytest.raises(ValueError, match='manifest'):
        load_method_cache(package.imus['train']['000506'], workspace_root=ROOT)
    with pytest.raises(ValueError, match='manifests together'):
        package.require_manifests('train', package.manifests['val_motion'], package.manifests['val_imu'],
                                  package.manifests['val_caption_plans'])
    with pytest.raises(ValueError, match='complete delivered'):
        dataset(package, caption_plans=package.plans['val'])
    with pytest.raises(ValueError, match='not admitted'):
        package.prepare('train', '000010')


@pytest.mark.parametrize('key,value', [('text', 'unreviewed edit'), ('text_present', False),
                                       ('length', 12), ('caption_index', 1), ('condition', 'shuffled')])
def test_case_overrides_rejected(package, key, value):
    norm = dataset(package).compute_normalization()
    with pytest.raises(ValueError):
        script('sample_mdm_imu_control')._load_case(dict(motion_id='000433', **{key: value}), {}, norm,
                                                   canonical_package=package)


def test_embedded_mapping_and_hash_mismatch_rejected(package, tmp_path):
    record = dict(package.imus['train']['000506'])
    with np.load(record['imu_path']) as data:
        arrays = {k: data[k] for k in data.files}
    arrays['mapping_version'] = np.array('wrong-mapping')
    path = tmp_path/'wrong.npz'; np.savez(path, **arrays)
    record['imu_path'] = str(path)
    with pytest.raises(ValueError, match='embedded mapping_version'):
        load_canonical_cache(record, workspace_root=ROOT)
    entry = dict(package.package['admission'], sha256='0'*64)
    with pytest.raises(ValueError, match='hash mismatch'):
        package.verify(entry)


def test_admission_scope_and_caption_cannot_be_relabelled(package):
    package.plans['train']['000506']['world_alignment_scope'] = 'real_device'
    with pytest.raises(ValueError, match='admission mismatch'):
        package.prepare('train', '000506')


def test_canonical_input_only_cli_exits_before_model_or_device_setup(package, tmp_path, monkeypatch):
    trainer = script('train_mdm_imu_control')
    mean, std = tmp_path/'Mean.npy', tmp_path/'Std.npy'
    np.save(mean, np.zeros(263)); np.save(std, np.ones(263))
    mdm, options = tmp_path/'mdm.pt', tmp_path/'args.json'
    mdm.write_bytes(b'identity-only fixture; never deserialized'); options.write_text('{}')
    report = tmp_path/'report.json'
    argv = ['train', '--canonical-package', str(package.path), '--contract-version', 'itm-rework-v0.3-draft',
            '--manifest', str(package.manifests['train_motion']), '--standard-imu-manifest', str(package.manifests['train_imu']),
            '--caption-plan', str(package.manifests['train_caption_plans']), '--mean', str(mean), '--std', str(std),
            '--mdm-checkpoint', str(mdm), '--mdm-args', str(options), '--output', str(tmp_path/'never.pt'),
            '--input-only-report', str(report)]
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(trainer, '_legacy_compatibility', lambda: None)
    monkeypatch.setattr(trainer.os, 'chdir', lambda _: pytest.fail('Unexpected MDM import'))
    monkeypatch.setattr(trainer, '_check_device', lambda *_: pytest.fail('Unexpected device setup'))
    assert trainer.main() == 0
    result = json.loads(report.read_text())
    assert result['imu_normalization']['source']['unique_frames'] == 398
    assert len(result['inputs']) == 6 and result['training_updates'] == 0
    assert result['training_input_provenance']['payload']['input_pipeline_version'] == 'itm-method-canonical-b0-v1'
    assert not (tmp_path/'never.pt').exists()
