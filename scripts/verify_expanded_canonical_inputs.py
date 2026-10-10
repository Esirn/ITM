#!/usr/bin/env python
"""Check the prepared expanded inputs against source pairs and train-only statistics."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from itm.data.manifest import read_jsonl
from itm.models.mdm_conditions import normalize_imu
from itm.models.mdm_correction import read, verify, artifact
from audit_caption_intervals import write_json


def verify_inputs(path, output):
    if output.exists(): raise FileExistsError(output)
    data = read(path)
    pairs = {(r['split'], r['motion_id']): r for r in read_jsonl(verify(data['pairs']))}
    mean, std = [np.load(verify(data[k])).astype(np.float32) for k in ('motion_mean', 'motion_std')]
    train_values, counts, groups = [], Counter(), {'train': set(), 'val': set()}
    checks = 0
    for r in data['records']:
        original = pairs[r['split'], r['motion_id']]
        with np.load(verify(r['input']), allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        assert arrays['motion'].shape == (r['frames'], 263)
        assert arrays['gt'].shape == (r['frames'], 22, 3)
        assert arrays['frame_mask'].dtype == np.bool_ and arrays['frame_mask'].all()
        np.testing.assert_array_equal(arrays['gt'], np.load(original['joints_path']))
        np.testing.assert_array_equal(arrays['motion'], (np.load(original['joint_vec_path']).astype(np.float32) - mean) / std)
        np.testing.assert_array_equal(arrays['imu'], normalize_imu(arrays['acceleration'], arrays['orientation'], data['normalization']))
        fixed = np.broadcast_to(arrays['orientation'][0], arrays['orientation'].shape)
        np.testing.assert_array_equal(arrays['reference'], normalize_imu(np.zeros_like(arrays['acceleration']), fixed, data['normalization']))
        np.testing.assert_array_equal(arrays['timestamps'], (np.arange(r['frames'], dtype=np.float64) / 20).astype(np.float32))
        assert r['training_admissible'] is (r['split'] == 'train')
        assert r['benchmark_admissible'] is False
        assert original['caption_index'] == r['caption_index'] and original['caption'] == r['text']
        counts[r['split']] += 1; groups[r['split']].add(r['source_group_id']); checks += 10
        if r['split'] == 'train': train_values.append(arrays['acceleration'].astype(np.float64))
    assert counts == {'train': 128, 'val': 32}
    assert len(groups['train']) == 128 and len(groups['val']) == 32 and not groups['train'] & groups['val']
    values = np.concatenate(train_values)
    np.testing.assert_array_equal(np.asarray(data['normalization']['acceleration_mean'], dtype=np.float32), values.mean(0).astype(np.float32))
    np.testing.assert_array_equal(np.asarray(data['normalization']['acceleration_std'], dtype=np.float32), np.maximum(values.std(0), 1e-6).astype(np.float32))
    assert len(values) == data['normalization']['unique_train_frames']
    report = dict(status='passed', dataset=artifact(path), clips=160, checks=checks + 3,
                  train_groups=128, val_groups=32, unique_train_frames=len(values),
                  exact_prepared_arrays=True, split_overlap=False, official_test_used=False,
                  new_GPU_queries=0, new_model_calls=0,
                  annotation_scope='Full-interval dataset text; no new semantic certification')
    write_json(output, report); print(report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify_inputs(a.dataset.resolve(), a.output.resolve())
