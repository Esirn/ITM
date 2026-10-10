#!/usr/bin/env python
"""Prepare training-only normalization and common inputs for expanded comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_caption_intervals import artifact, write_json
from itm.data.caption_time import parse_timed_caption, plan_caption_crop
from itm.data.manifest import read_jsonl
from itm.models.mdm_canonical_conditions import load_canonical_cache
from itm.models.mdm_conditions import sample_imu_at_times, normalize_imu
from itm.models.mdm_correction import verify
from prepare_expanded_canonical_pilot import ROOT, COUNTS, ANCHORS


def prepare(directory, verification, output):
    if output.exists():
        raise FileExistsError(output)
    pairs = list(read_jsonl(directory / 'built_pairs.jsonl'))
    check = json.loads(verification.read_text())
    if check['passed'] != sum(COUNTS.values()) or check['pairs'] != artifact(directory / 'built_pairs.jsonl'):
        raise ValueError('Every expanded pair must pass verification')
    group_sets = [{r['source_group_id'] for r in pairs if r['split'] == s} for s in COUNTS]
    if group_sets[0] & group_sets[1] or any(len(g) != COUNTS[s] for g, s in zip(group_sets, COUNTS)):
        raise ValueError('One source group per clip and disjoint splits required')
    rows, train_acceleration = [], []
    for r in pairs:
        for a in r['artifacts']:
            verify(a)
        cache = load_canonical_cache(r, workspace_root=ROOT)
        with np.load(directory / r['split'] / r['motion_id'] / 'source_20hz.npz', allow_pickle=False) as z:
            local_rotation = z['local_rotation'][:-1].copy()
        lines = [s for s in Path(r['text_path']).read_text().splitlines() if s.strip()]
        caption = parse_timed_caption(lines[r['caption_index']])
        cp = plan_caption_crop(caption, r['motion_frames'], cache['timestamps'])
        acceleration, orientation = sample_imu_at_times(cache, cp['source_times_seconds'])
        if r['split'] == 'train':
            train_acceleration.append(acceleration.astype(np.float64))
        rows.append((r, acceleration, orientation, local_rotation))
    values = np.concatenate(train_acceleration, axis=0)
    norm = dict(acceleration_mean=values.mean(axis=0).astype(np.float32).tolist(),
                acceleration_std=np.maximum(values.std(axis=0), 1e-6).astype(np.float32).tolist(),
                source='128 unique train clips at exact model 20Hz times; no sensor-config duplication or validation',
                unique_train_frames=len(values))
    common = json.loads((ROOT / 'docs/itm_rework/protocol/b0_correction_20260926_v1/design.json').read_text())['common']
    mean_path, std_path = [verify(a) for a in common['motion_normalization']]
    mean, std = np.load(mean_path).astype(np.float32), np.load(std_path).astype(np.float32)
    if mean.shape != (263,) or std.shape != (263,) or not (std > 0).all():
        raise ValueError('Pretrained motion normalization invalid')
    output.mkdir(parents=True)
    records = []
    for r, acceleration, orientation, rotations in rows:
        frames = r['motion_frames']
        reference_orientation = np.broadcast_to(orientation[0], orientation.shape).copy()
        dest = output / r['split'] / (r['motion_id'] + '.npz')
        dest.parent.mkdir(parents=True, exist_ok=True)
        motion = (np.load(r['joint_vec_path']).astype(np.float32) - mean) / std
        payload = dict(motion=motion, gt=np.load(r['joints_path']).astype(np.float32),
                       imu=normalize_imu(acceleration, orientation, norm),
                       reference=normalize_imu(np.zeros_like(acceleration), reference_orientation, norm),
                       acceleration=acceleration, orientation=orientation, local_rotation=rotations,
                       timestamps=(np.arange(frames, dtype=np.float64) / 20.).astype(np.float32),
                       frame_mask=np.ones(frames, dtype=bool))
        if any(not np.isfinite(a).all() for a in payload.values()):
            raise FloatingPointError(r['motion_id'])
        np.savez_compressed(dest, **payload)
        records.append(dict(motion_id=r['motion_id'], split=r['split'], source_group_id=r['source_group_id'],
                            frames=frames, text=r['caption'], caption_index=r['caption_index'],
                            semantic_review='original_B0_coarse_review' if r['motion_id'] in sum(ANCHORS.values(), []) else 'not_visually_certified_dataset_annotation',
                            benchmark_admissible=False, training_admissible=r['split'] == 'train',
                            input=artifact(dest), source_pair_artifacts=r['artifacts']))
    write_json(output / 'dataset.json', dict(version='expanded-canonical-development-inputs-v1',
        records=records, normalization=norm, motion_mean=artifact(mean_path), motion_std=artifact(std_path),
        pairs=artifact(directory / 'built_pairs.jsonl'), verification=artifact(verification),
        preparer=artifact(Path(__file__)), source_group_disjoint=True, test_inputs_used=False,
        sensor_configs={'head': [4], 'wrists': [0, 1]},
        physical_scope='original offline pre-canonicalized SMPL bridge; no real-device calibration',
        generated_IMU_closure_certified=False, manual_semantic_admission=False))
    print(dict(inputs=len(records), unique_train_frames=len(values), splits=COUNTS), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs-directory', type=Path, required=True)
    p.add_argument('--verification', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    prepare(a.pairs_directory, a.verification, a.output)
