#!/usr/bin/env python
"""Bounded CPU input-only check of owner-02's v0.2 handoff. No model forwards."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from itm.models.mdm_conditions import INPUT_CONTRACT_VERSION, INPUT_PIPELINE_VERSION
from train_mdm_imu_control import _PairedDataset, _collate, load_response_manifest
from sample_mdm_imu_control import _load_case

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    checkpoint = ROOT/'outputs/itm_rework/method/b0_cpu_smoke_20260924_v1/model.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    report = dict(kind='input_only_diagnostic', input_contract_version=INPUT_CONTRACT_VERSION,
                  input_pipeline_version=INPUT_PIPELINE_VERSION, checkpoint_sha256=sha(checkpoint),
                  checkpoint_contract_version=state['contract_version'], model_forwards=0, gpu_jobs=0,
                  normalization_recomputed=False, alignment_certified=False, train=[], val=[], static=[])
    dataset = _PairedDataset(ROOT/'outputs/manifests_full/train.jsonl',
                            ROOT/'outputs/manifests_full/train_standard_imu.jsonl',
                            np.load('/home/a200/0proj/datasets/all/Mean.npy'),
                            np.load('/home/a200/0proj/datasets/all/Std.npy'),
                            {'head': (4,), 'wrists': (0, 1)}, 3, ROOT)
    dataset.set_normalization(state['imu_normalization'])
    items = [dataset[index] for index in range(len(dataset))]
    _, y = _collate(items)
    assert torch.isfinite(y['imu']).all()
    for item in items:
        report['train'].append(dict(motion_id=item['motion_id'], sensor_config=item['sensor_config'],
                                    metadata=item['input_metadata'], finite=True))
    records = {r['motion_id']: r for r in rows(ROOT/'outputs/manifests_full/val_standard_imu.jsonl')}
    for row in rows(ROOT/'docs/itm_rework/protocol/validation_20260924_v1/panel.jsonl'):
        case = _load_case(dict(motion_id=row['motion_id'], length=row['length_frames_20fps'], sensor_config='head'),
                          records, state['imu_normalization'])
        assert np.isfinite(case['imu']).all()
        report['val'].append(dict(motion_id=row['motion_id'], frames=len(case['imu']),
                                  interval=case['time_interval_seconds'], finite=True))
    records = {r['motion_id']: r for r in rows(ROOT/'outputs/manifests_full/train_standard_imu.jsonl')}
    for config in ('head', 'wrists'):
        case = _load_case(dict(motion_id='000004', length=12, sensor_config=config, imu_mode='static'),
                          records, state['imu_normalization'])
        assert np.count_nonzero(case['raw_acceleration']) == 0
        assert not np.array_equal(case['imu'], np.zeros_like(case['imu']))
        report['static'].append(case['physical_static_fixture'])
    try:
        load_response_manifest(ROOT/'docs/itm_rework/protocol/arm_spread_candidates_20260924_v1/candidates.jsonl')
    except ValueError as exc:
        report['uncertain_candidate_rejection'] = str(exc)
    else:
        raise AssertionError('Uncertain witnesses must remain blocked')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(train=len(report['train']), val=len(report['val']), static=len(report['static']),
                         model_forwards=0, status='input_checks_passed_world_alignment_still_uncertified')))


if __name__ == '__main__':
    main()
