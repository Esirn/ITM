#!/usr/bin/env python
"""Owner02 F software acceptance: immutable CPU replay and independent raw inputs."""
import argparse
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import torch

from accept_b0_correction_implementation import ROOT, DATA, read, lines, sha, write
from accept_canonical_b0_handoff import reference
from itm.models.mdm_sampling_alignment import describe, input_arrays
from itm.models.mdm_feature_balance import FeaturePackage, PACKAGE_SHA
import run_mdm_feature_balance_bounded as runner

METHOD = ROOT/'outputs/itm_rework/method/feature_balance_implementation_20261003_v2/implementation.json'
MANIFEST_SHA = 'db323ef515ede3252898f322f7badda10913aea64d67cb4abb1f3e9fc4605e30'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    manifest = read(METHOD)
    assert sha(METHOD) == MANIFEST_SHA
    archive = ROOT/'docs/itm_rework/method/feature_balance_implementation_20261003_v1/implementation.json'
    assert archive.read_bytes() == METHOD.read_bytes()
    assert manifest['package']['sha256'] == PACKAGE_SHA
    assert len(manifest['expected_artifacts']) == 1145
    for entry in manifest['expected_artifacts']:
        assert sha(ROOT/entry['path']) == entry['sha256'], entry['path']
    forbidden = []

    def reject(*a, **kw):
        forbidden.append('model_construction_forward_GPU_or_process')
        raise AssertionError(forbidden[-1])

    with ExitStack() as stack:
        for name in ('__init__', '_call_impl'):
            stack.enter_context(patch.object(torch.nn.Module, name, reject))
        for name in ('init', '_lazy_init', 'is_available', 'device_count', 'get_device_properties',
                     'synchronize', 'current_device', 'set_device'):
            stack.enter_context(patch.object(torch.cuda, name, reject))
        for name in ('preflight_gpu', 'supervise'):
            stack.enter_context(patch.object(runner, name, reject))
        for name in ('Popen', 'run', 'call', 'check_call', 'check_output'):
            stack.enter_context(patch.object(subprocess, name, reject))
        with redirect_stdout(io.StringIO()):
            replay = runner.prepare(args.prepare_dir)
        assert replay == manifest
        assert (args.prepare_dir/'implementation.json').read_bytes() == METHOD.read_bytes()
        assert list(args.prepare_dir.iterdir()) == [args.prepare_dir/'implementation.json']
        package = FeaturePackage()
        bound = {str((ROOT/e['path']).resolve()): e['sha256'] for e in manifest['expected_artifacts']}
        sources = read(package.files['sources'])
        assert len(sources) == 1125
        for entry in sources:
            assert bound[str((ROOT/entry['path']).resolve())] == entry['sha256']
        initial = package.initial_state()  # Existing tensors only; no real model constructed.
        assert initial['tensor_hashes'] == manifest['initial_tensor_hashes']
        expected_parameters = [dict(name='seqTransEncoder.adapters.'+k, shape=list(v.shape), dtype=str(v.dtype))
                               for k, v in initial['adapters'].items() if k.endswith('.weight')]
        expected_parameters += [dict(name='imu_control_encoder.'+k, shape=list(v.shape), dtype=str(v.dtype))
                                for k, v in initial['imu_encoder'].items()]
        assert expected_parameters == manifest['parameter_contract']
        assert len(expected_parameters) == 36
        parameters = sum(int(np.prod(r['shape'])) for r in expected_parameters)
        assert parameters == 8705024
        refs, captions = {}, {}
        for split in ('train', 'val'):
            captions[split] = {r['motion_id']: r for r in lines(DATA/f'{split}_caption_plans.jsonl')}
            for row in lines(DATA/f'{split}_imu.jsonl'):
                mid = row['motion_id']
                refs[split, mid] = reference(row, captions[split][mid]['frames'])[:3]
        normal = package.common['imu_normalization']
        mean = np.asarray(normal['acceleration_mean'], dtype=np.float32)
        std = np.asarray(normal['acceleration_std'], dtype=np.float32)
        population = np.concatenate([refs['train', mid][0].astype(float) for mid in ('000027', '000506', '004728')])
        assert population.shape == (398, 6, 3)
        replay_mean = population.mean(0).astype(np.float32)
        replay_std = np.sqrt(np.maximum(population.var(0), 1e-8)).astype(np.float32)
        np.testing.assert_array_equal(replay_mean, mean)
        np.testing.assert_allclose(replay_std, std, atol=0, rtol=1e-7)
        checks = []
        for phase, rows, entries in (('train', package.schedule[:6], manifest['training_inputs']),
                                    ('sample', package.samples, manifest['inputs'])):
            assert len(rows) == len(entries)
            for row, entry in zip(rows, entries):
                key = 'exposure_id' if phase == 'train' else 'case_id'
                assert row[key] == entry[key]
                split, mid = row.get('data_split', 'train'), row['motion_id']
                a, o, times = refs[split, mid]
                n = len(times)
                static_o = np.broadcast_to(o[:1], o.shape)
                ref = np.concatenate([(np.zeros_like(a)-mean)/std, static_o.reshape(n, 6, 9)], -1)
                condition = row.get('condition', 'paired')
                ea, eo = a, o
                if condition == 'physical_static':
                    ea, eo = np.zeros_like(a), static_o
                elif condition == 'tail_reversed':
                    index = np.r_[0, np.arange(n-1, 0, -1)]
                    ea, eo = a[index], o[index]
                imu = np.concatenate([(ea-mean)/std, eo.reshape(n, 6, 9)], -1)
                if condition == 'numeric_zero':
                    imu = np.zeros_like(imu)
                item = package.inputs(row)
                np.testing.assert_array_equal(item['raw_acceleration'], ea)
                for name, expected in (('raw_orientation', eo), ('reference', ref),
                                       ('imu', imu), ('original_orientation0', o[0])):
                    np.testing.assert_allclose(item[name], expected, atol=2e-7, rtol=0)
                np.testing.assert_array_equal(item['timestamps'], times)
                mask = np.zeros(6, bool)
                mask[[4] if row['sensor_config'] == 'head' else [0, 1]] = True
                np.testing.assert_array_equal(item['sensor_mask'], mask)
                plan = captions[split][mid]
                assert item['text'] == plan['caption']
                assert row['caption_index'] == plan['caption_index']
                if phase == 'sample':
                    assert row['text'] == plan['caption']
                gt_path = ROOT/next(e['path'] for e in plan['artifacts'] if Path(e['path']).name == 'joints.npy')
                features = np.load(gt_path.with_name('motion.npy'), allow_pickle=False)
                np.testing.assert_array_equal(item['gt'], np.load(gt_path, allow_pickle=False))
                np.testing.assert_array_equal(item['motion'], ((features-package.mean)/package.std).astype(np.float32))
                arrays = input_arrays(item, dict(condition='paired', **{k:v for k,v in row.items() if k != 'condition'})
                                      if phase == 'train' else row)
                assert {k: describe(v) for k, v in arrays.items()} == entry['arrays']
                assert arrays['frame_mask'].all()
                checks.append(dict(phase=phase, slot=row[key], source=mid, split=split,
                    config=row['sensor_config'], condition=condition, frames=n, passed=True,
                    normalized_max_abs_error=float(np.max(np.abs(item['imu']-imu))),
                    reference_max_abs_error=float(np.max(np.abs(item['reference']-ref)))))
    assert len(checks) == 66 and not forbidden
    result = dict(read_commit=commit, manifest_sha256=MANIFEST_SHA, package_sha256=PACKAGE_SHA,
        bound_files_verified=1145, frozen_sources_included=1125, manifest_copy_byte_identical=True,
        rebuilt_manifest_byte_identical=True, preparation_path=str(args.prepare_dir),
        preparation_has_only_manifest=True, initial=manifest['initial'], initial_tensors_exact=True,
        trainable_parameter_tensors=36, trainable_parameters=parameters, unique_train_frames=398,
        train_mean_max_abs=float(np.max(np.abs(replay_mean-mean))),
        train_std_max_abs=float(np.max(np.abs(replay_std-std))), input_checks=checks, forbidden_calls=forbidden,
        current_model_constructions=0, current_real_model_forwards=0, current_GPU_queries=0,
        current_new_samples=0, current_training_updates=0, current_new_checkpoints=0,
        new_authorization_created=False, code=dict(path=str(Path(__file__)), sha256=sha(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, result)
    print(dict(bound_files=1145, frozen_sources=1125, input_sets=66, train_frames=398,
               parameter_tensors=36, parameters=parameters, manifest_byte_identical=True, forbidden_calls=0))


if __name__ == '__main__':
    main()
