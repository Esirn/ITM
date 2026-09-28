#!/usr/bin/env python
"""Owner02 V software acceptance: independent raw inputs, states and CPU manifest replay."""
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
from itm.models.mdm_state_diagnostic import StatePackage, PACKAGE_SHA, CHECKPOINT_SHA
import run_mdm_state_diagnostic_bounded as runner

METHOD = ROOT/'outputs/itm_rework/method/state_diagnostic_implementation_20260928_v1/implementation.json'
MANIFEST_SHA = '71cb4e2306559295e59ee5c179ce57e971a1a60dcb9f021cf1ddc33c4c45ee46'


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
    copy = ROOT/'docs/itm_rework/method/state_diagnostic_implementation_20260928_v1/implementation.json'
    assert METHOD.read_bytes() == copy.read_bytes()
    assert manifest['package']['sha256'] == PACKAGE_SHA
    assert manifest['checkpoint']['sha256'] == CHECKPOINT_SHA
    assert len(manifest['expected_artifacts']) == 1656
    for entry in manifest['expected_artifacts']:
        assert sha(ROOT/entry['path']) == entry['sha256'], entry['path']
    forbidden = []

    def reject(*a, **kw):
        forbidden.append('model_forward_GPU_or_process')
        raise AssertionError(forbidden[-1])

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.nn.Module, '_call_impl', reject))
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
        package = StatePackage()
        bound = {str((ROOT/e['path']).resolve()): e['sha256'] for e in manifest['expected_artifacts']}
        frozen_sources = read(package.files['sources'])
        assert len(frozen_sources) == 1640
        for entry in frozen_sources:
            assert bound[str((ROOT/entry['path']).resolve())] == entry['sha256']
        state = package.state()  # Existing checkpoint tensors on CPU, no model construction.
        refs, plans = {}, {}
        for split in ('train', 'val'):
            plans[split] = {r['motion_id']: r for r in lines(DATA/f'{split}_caption_plans.jsonl')}
            for row in lines(DATA/f'{split}_imu.jsonl'):
                mid = row['motion_id']
                refs[split, mid] = reference(row, plans[split][mid]['frames'])[:3]
        normal = package.common['imu_normalization']
        mean = np.array(normal['acceleration_mean'], dtype=np.float32)
        std = np.array(normal['acceleration_std'], dtype=np.float32)
        population = np.concatenate([refs['train', mid][0].astype(float) for mid in ('000027', '000506', '004728')])
        assert population.shape == (398, 6, 3)
        np.testing.assert_array_equal(population.mean(0).astype(np.float32), mean)
        np.testing.assert_allclose(np.sqrt(np.maximum(population.var(0), 1e-8)).astype(np.float32), std, atol=0, rtol=1e-7)
        inputs = []
        for case, prepared in zip(package.cases, manifest['inputs']):
            assert case['cache_id'] == prepared['cache_id']
            split, mid = case['data_split'], case['motion_id']
            a, o, times = refs[split, mid]
            n = len(times)
            static_o = np.broadcast_to(o[:1], o.shape)
            ref = np.concatenate([(np.zeros_like(a)-mean)/std, static_o.reshape(n, 6, 9)], -1)
            items, rows, descriptors = package.inputs(case)
            assert descriptors == prepared['branches']
            plan = plans[split][mid]
            assert case['caption'] == case['sample_spec']['text'] == plan['caption']
            for branch, item, row in zip(case['branches'], items, rows):
                b = branch['branch']
                idx = np.r_[0, np.arange(n-1, 0, -1)]
                ea = np.zeros_like(a) if b == 'S' else a[idx] if b == 'R' else a
                eo = static_o if b == 'S' else o[idx] if b == 'R' else o
                imu = np.concatenate([(ea-mean)/std, eo.reshape(n, 6, 9)], -1)
                np.testing.assert_array_equal(item['raw_acceleration'], ea)
                for name, expected in (('raw_orientation', eo), ('reference', ref), ('imu', imu), ('original_orientation0', o[0])):
                    np.testing.assert_allclose(item[name], expected, atol=2e-7, rtol=0)
                np.testing.assert_array_equal(item['timestamps'], times)
                mask = np.zeros(6, bool)
                mask[[4] if case['sensor_config'] == 'head' else [0, 1]] = True
                np.testing.assert_array_equal(item['sensor_mask'], mask)
                assert row['text_present'] and row['imu_present'] == (b != 'T')
                assert input_arrays(item, row)['frame_mask'].all()
                inputs.append(dict(cache_id=case['cache_id'], branch=b, source=mid, split=split,
                    config=case['sensor_config'], frames=n, passed=True,
                    normalized_max_abs_error=float(np.max(np.abs(item['imu']-imu))),
                    reference_max_abs_error=float(np.max(np.abs(item['reference']-ref)))))
        states = []
        for group, prepared in zip(package.groups, manifest['states']):
            assert group['group_id'] == prepared['group_id']
            arrays = package.saved(group)
            assert {k: describe(v) for k, v in arrays.items()} == prepared['arrays']
            with np.load(ROOT/group['saved_state']['path'], allow_pickle=False) as z:
                for k in ('x_t', 'f10', 'f11', 'frame_mask'):
                    np.testing.assert_array_equal(arrays[k], z[k])
            features = np.load(Path(group['sample_spec']['gt_path']).with_name('motion.npy'))
            target = ((features-package.mean)/package.std).astype(np.float32).T[:, None]
            np.testing.assert_array_equal(arrays['target'], target)
            assert arrays['x_t'].shape == (1, 263, 1, group['frames'])
            assert arrays['frame_mask'].all()
            states.append(dict(group_id=group['group_id'], trace_id=group['trace_id'], timestep=group['timestep'],
                source=group['motion_id'], config=group['sensor_config'], passed=True,
                saved_state_exact=True, target_exact=True))
    assert len(inputs) == len(states) == 48 and not forbidden
    result = dict(read_commit=commit, manifest_sha256=MANIFEST_SHA, package_sha256=PACKAGE_SHA,
        checkpoint_sha256=CHECKPOINT_SHA, bound_files_verified=1656, frozen_sources_included=1640,
        manifest_copy_byte_identical=True, rebuilt_manifest_byte_identical=True,
        preparation_path=str(args.prepare_dir), preparation_has_only_manifest=True,
        checkpoint_updates=state['execution']['updates_completed'], unique_train_frames=398,
        input_checks=inputs, state_checks=states, forbidden_calls=forbidden,
        current_real_model_forwards=0, current_GPU_queries=0, current_new_samples=0,
        current_training_updates=0, current_new_checkpoints=0, new_authorization_created=False,
        code=dict(path=str(Path(__file__)), sha256=sha(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, result)
    print(dict(bound_files=1656, frozen_sources=1640, input_sets=48, states=48,
               train_frames=398, manifest_byte_identical=True, forbidden_calls=0))


if __name__ == '__main__':
    main()
