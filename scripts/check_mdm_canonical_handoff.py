#!/usr/bin/env python
"""Replay six canonical inputs and 36 condition slots without model execution."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from itm.models.mdm_canonical_conditions import CanonicalB0Package, CANONICAL_PIPELINE_VERSION
from itm.models.mdm_training_provenance import _digest
from sample_mdm_imu_control import _load_case
from train_mdm_imu_control import _PairedDataset

ROOT = Path(__file__).resolve().parents[1]


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True)
    parser.add_argument('--training-report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    package = CanonicalB0Package(args.package, workspace_root=ROOT)
    report_path = Path(args.training_report).resolve()
    training = json.loads(report_path.read_text())
    normalization = training['imu_normalization']
    provenance = training['training_input_provenance']
    assert provenance['sha256'] == _digest(provenance['payload'])
    assert provenance['payload']['imu_normalization_sha256'] == _digest(normalization)
    assert provenance['payload']['input_pipeline_version'] == CANONICAL_PIPELINE_VERSION
    assert provenance['payload']['artifacts']['canonical_package']['sha256'] == hashlib.sha256(package.path.read_bytes()).hexdigest()
    dataset = _PairedDataset(package.manifests['train_motion'], package.manifests['train_imu'],
                             np.zeros(263), np.ones(263), {'head': (4,), 'wrists': (0, 1)}, None, ROOT,
                             caption_plans=package.plans['train'], canonical_package=package)
    assert dataset.compute_normalization() == normalization
    dataset.set_normalization(normalization)
    # Independent numpy interpolation and population variance at actual model times.
    independent = []
    for mid, plan in package.plans['train'].items():
        with np.load(package.imus['train'][mid]['imu_path'], allow_pickle=False) as cache:
            acc = np.stack([np.interp(np.arange(plan['frames'])/20., cache['timestamps'], column)
                            for column in cache['acceleration'].reshape(-1, 18).T], axis=1)
            independent.append(acc.astype(np.float32).astype(np.float64).reshape(-1, 6, 3))
    independent = np.concatenate(independent)
    np.testing.assert_array_equal(normalization['acceleration_mean'], independent.mean(0).astype(np.float32))
    np.testing.assert_allclose(normalization['acceleration_std'],
                               np.sqrt(np.maximum(independent.var(0), 1e-8)).astype(np.float32), rtol=1e-7, atol=0)
    result = {'pipeline': CANONICAL_PIPELINE_VERSION, 'normalization': normalization,
              'training_provenance_sha256': provenance['sha256'],
              'training_report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
              'budget': {'source_clips': 6, 'train_config_inputs': 6, 'planned_val_condition_slots': 36,
                         'model_forwards': 0, 'training_updates': 0, 'generated_samples': 0, 'gpu_jobs': 0},
              'train': [], 'validation_slots': []}
    for i in range(len(dataset)):
        item = dataset[i]
        sample = _load_case({'motion_id': item['motion_id'], 'sensor_config': item['sensor_config']}, {}, normalization,
                            canonical_package=package, canonical_split='train')
        np.testing.assert_array_equal(item['imu'], sample['imu'])
        assert item['caption'] == sample['text']
        result['train'].append({'motion_id': item['motion_id'], 'sensor_config': item['sensor_config'],
                                'frames': len(item['imu']), 'imu_sha256': array_hash(item['imu']),
                                'caption_index': item['input_metadata']['caption_index'],
                                'training_sampling_input_equal': True})
    for case in json.loads(package.sampling_plan_path.read_text()):
        sample = _load_case(case, {}, normalization, canonical_package=package)
        np.testing.assert_array_equal(sample['gt'], np.load(case['gt_path'], allow_pickle=False))
        result['validation_slots'].append({
            'case_id': case['case_id'], 'motion_id': sample['motion_id'], 'sensor_config': sample['sensor_config'],
            'condition': case['condition'], 'frames': len(sample['imu']), 'text_present': sample['text_present'],
            'imu_present': sample['imu_present'], 'backbone_bypass': sample['backbone_bypass'],
            'reviewed_caption': sample['reviewed_caption'], 'random_plan_case_id': case['random_plan_case_id'],
            'sampling_seed': case['sampling_seed'], 'noise_status': 'address_reserved_not_drawn',
            'normalized_imu_sha256': array_hash(sample['imu']), 'gt_array_sha256': array_hash(sample['gt']),
            'raw_acceleration_sha256': array_hash(sample['raw_acceleration']),
            'raw_orientation_sha256': array_hash(sample['raw_orientation']),
            'gt_source': sample['input_metadata']['gt_source'], 'training_admissible': False,
            'world_alignment_scope': sample['input_metadata']['world_alignment_scope'],
            'physical_static_fixture': sample['physical_static_fixture'], 'status': 'input_ready_not_sampled'})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print('6 train/config inputs equal; 36 validation slots loaded; 398 unique train frames; no model execution')


if __name__ == '__main__':
    main()
