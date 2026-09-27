#!/usr/bin/env python
"""Owner02 independent fixed-R preparation/raw-input audit; no model forward/GPU."""
import argparse
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import torch

from accept_b0_correction_implementation import read, sha, lines, write, ROOT, DATA
from accept_canonical_b0_handoff import reference
from itm.models.mdm_sampling_alignment import AlignmentPackage, describe, input_arrays
import run_mdm_sampling_alignment_bounded as runner

METHOD = ROOT/'outputs/itm_rework/method/sampling_alignment_implementation_20260927_v1/implementation.json'
MANIFEST_SHA = '793fdd9a95a102484d2dbc7579aa398f3af0120c66422ba6ec966cc01f8cb7c4'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    manifest = read(METHOD)
    assert sha(METHOD) == MANIFEST_SHA
    assert METHOD.read_bytes() == (ROOT/'docs/itm_rework/method/sampling_alignment_implementation_20260927_v1/implementation.json').read_bytes()
    for entry in manifest['expected_artifacts']:
        assert sha(ROOT/entry['path']) == entry['sha256'], entry['path']
    assert len(manifest['expected_artifacts']) == 743
    prohibited = []
    def reject(*a, **kw):
        prohibited.append('unexpected_model_forward_GPU_or_launch')
        raise AssertionError(prohibited[-1])
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.nn.Module, '_call_impl', reject))
        for name in ('preflight_gpu', 'supervise'):
            stack.enter_context(patch.object(runner, name, reject))
        for name in ('init', '_lazy_init', 'is_available', 'device_count', 'get_device_properties',
                     'synchronize', 'current_device', 'set_device'):
            stack.enter_context(patch.object(torch.cuda, name, reject))
        with redirect_stdout(io.StringIO()): replay = runner.prepare(args.prepare_dir)
        package = AlignmentPackage()
        state = package.state()
    assert not prohibited and replay == manifest
    assert sha(args.prepare_dir/'implementation.json') == MANIFEST_SHA
    assert list(args.prepare_dir.iterdir()) == [args.prepare_dir/'implementation.json']
    refs, captions = {}, {}
    for split in ('train', 'val'):
        captions[split] = {r['motion_id']: r for r in lines(DATA/f'{split}_caption_plans.jsonl')}
        for row in lines(DATA/f'{split}_imu.jsonl'):
            mid = row['motion_id']
            refs[split, mid] = reference(row, captions[split][mid]['frames'])[:3]
    norm = package.common['imu_normalization']
    mean = np.array(norm['acceleration_mean'], np.float32)
    std = np.array(norm['acceleration_std'], np.float32)
    train = np.concatenate([refs['train', mid][0].astype(float) for mid in ('000027', '000506', '004728')])
    assert len(train) == 398
    np.testing.assert_array_equal(train.mean(0).astype(np.float32), mean)
    np.testing.assert_allclose(np.sqrt(np.maximum(train.var(0), 1e-8)).astype(np.float32), std, atol=0, rtol=1e-7)
    checks, unique = [], set()
    for phase, rows, bound in (('Q', package.probes, manifest['Q_inputs']), ('S1', package.samples, manifest['inputs'])):
        assert len(rows) == len(bound)
        for row, entry in zip(rows, bound):
            key = 'probe_id' if phase == 'Q' else 'case_id'
            assert row[key] == entry[key]
            split, mid = row.get('data_split', row.get('split')), row['motion_id']
            a, o, t = refs[split, mid]; n = len(t)
            static_o = np.broadcast_to(o[:1], o.shape)
            ref = np.concatenate([(np.zeros_like(a)-mean)/std, static_o.reshape(n, 6, 9)], -1)
            condition = row.get('input_transform', row['condition'])
            expected_a, expected_o = a, o
            if condition == 'physical_static': expected_a, expected_o = np.zeros_like(a), static_o
            if condition == 'tail_reversed':
                index = np.r_[0, np.arange(n-1, 0, -1)]
                expected_a, expected_o = a[index], o[index]
            u = np.concatenate([(expected_a-mean)/std, expected_o.reshape(n, 6, 9)], -1)
            if condition == 'numeric_zero': u = np.zeros_like(u)
            item = package.inputs(row)
            np.testing.assert_array_equal(item['raw_acceleration'], expected_a)
            for name, expected in (('raw_orientation', expected_o), ('reference', ref),
                                   ('imu', u), ('original_orientation0', o[0])):
                np.testing.assert_allclose(item[name], expected, atol=2e-7, rtol=0)
            np.testing.assert_array_equal(item['timestamps'], t)
            mask = np.zeros(6, bool); mask[[4] if row['sensor_config'] == 'head' else [0, 1]] = True
            np.testing.assert_array_equal(item['sensor_mask'], mask)
            assert item['text'] == captions[split][mid]['caption'] == row['text']
            gt_path = ROOT/next(a['path'] for a in captions[split][mid]['artifacts']
                                if Path(a['path']).name == 'joints.npy')
            if phase == 'S1': assert Path(row['gt_path']).resolve() == gt_path
            else: assert Path(row['target_feature_path']).resolve() == gt_path.with_name('motion.npy')
            gt = np.load(gt_path, allow_pickle=False)
            features = np.load(gt_path.with_name('motion.npy'), allow_pickle=False)
            np.testing.assert_array_equal(item['gt'], gt)
            np.testing.assert_array_equal(item['motion'], ((features-package.mean)/package.std).astype(np.float32))
            arrays = input_arrays(item, row)
            assert {k: describe(v) for k, v in arrays.items()} == entry['arrays']
            assert arrays['frame_mask'].all()
            unique.add((split, mid, row['sensor_config'], condition))
            checks.append(dict(phase=phase, slot=row[key], passed=True,
                normalized_max_abs_error=float(np.max(np.abs(item['imu']-u))),
                reference_max_abs_error=float(np.max(np.abs(item['reference']-ref)))))
    result = dict(read_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        manifest_sha256=MANIFEST_SHA, bound_files_verified=743, method_manifest_copy_byte_identical=True,
        rebuilt_manifest_byte_identical=True, independent_preparation_path=str(args.prepare_dir),
        checkpoint=manifest['checkpoint'], checkpoint_final_updates=state['execution']['updates_completed'],
        unique_train_frames=398, distinct_transformed_inputs=len(unique), input_slot_checks=checks,
        prohibited_calls_observed=prohibited, preparation_has_only_manifest=True,
        real_model_forwards=0, training_updates=0, real_samples=0, GPU_queries=0,
        new_authorization_created=False, code=dict(path=str(Path(__file__)), sha256=sha(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True); write(args.output, result)
    print(dict(bound_files=743, input_slots=len(checks), distinct_inputs=len(unique),
               train_frames=398, manifest_byte_identical=True, prohibited_calls=0))


if __name__ == '__main__': main()
