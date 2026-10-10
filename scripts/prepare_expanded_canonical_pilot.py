#!/usr/bin/env python
"""Build a source-disjoint development dataset through the existing SMPL bridge."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from audit_caption_intervals import artifact, write_json
from build_canonical_b0_panel import build, ORIGINS, CANDIDATES, MODEL, MDM, IMUPOSER
from itm.data.caption_time import parse_timed_caption, plan_caption_crop
from itm.data.humanml_clock import PREFIX_TRIM_SECONDS
from itm.data.manifest import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261009
COUNTS = {'train': 128, 'val': 32}
ANCHORS = {'train': ['000027', '000506', '004728'], 'val': ['000139', '000382', '000433']}
KNOWN_UNCERTAIN = {'000010', '000123', '000015', '000133', '000510', '000134'}


def rank(value):
    return hashlib.sha256(f'{SEED}:{value}'.encode()).hexdigest()


def select(output):
    if output.exists():
        raise FileExistsError(output)
    origins = list(read_jsonl(ORIGINS))
    by_id = {r['motion_id']: r for r in origins}
    group_splits = defaultdict(set)
    for row in origins:
        group_splits[row['source_group_id']].add(row['split'])
    candidates = list(read_jsonl(CANDIDATES))
    grouped = defaultdict(list)
    for row in candidates:
        grouped[(row['split'], row['source_group_id'])].append(row)
    old_plan = json.loads((ROOT / 'docs/itm_rework/protocol/canonical_b0_20260925_v3/plan.json').read_text())
    old_cases = {r['motion_id']: r for r in old_plan['cases']}
    selected, considered, groups = [], [], set()

    def candidate(c):
        o = by_id[c['motion_id']]
        reason = None
        if c['motion_id'] in KNOWN_UNCERTAIN:
            reason = 'known_B0_semantic_uncertainty_or_conflict'
        elif group_splits[o['source_group_id']] != {c['split']}:
            reason = 'source_not_exclusive_to_original_split'
        elif c['frame_slice'][0] != 0 or not 40 <= c['frame_slice'][1] <= 196:
            reason = 'not_full_2_to_9_8_second_clip'
        elif o['source_group_id'] in groups:
            reason = 'duplicate_source_group'
        if reason:
            considered.append(dict(motion_id=c['motion_id'], split=c['split'], reason=reason))
            return None
        with np.load(o['resolved_source_path'], allow_pickle=False) as d:
            fps, frames = float(d['mocap_framerate']), len(d['poses'])
        prefix = sum(seconds for dataset, seconds in PREFIX_TRIM_SECONDS.items()
                     if dataset in o['source_group_id'].split('/'))
        if fps not in (60., 120.):
            considered.append(dict(motion_id=c['motion_id'], split=c['split'], reason='nonexact_60_120Hz', fps=fps))
            return None
        start = int((o['start_frame_20fps'] + prefix * 20) * fps / 20)
        stop = int((o['end_frame_20fps'] + prefix * 20) * fps / 20)
        endpoint = start + c['frame_slice'][1] * int(fps / 20)
        if (o['end_frame_20fps'] < 0 or endpoint >= frames
                or stop > frames + int(fps / 20) - 1
                or (min(stop, frames) - start) / fps * 30 > 300):
            considered.append(dict(motion_id=c['motion_id'], split=c['split'], reason='required_endpoint_or_budget_unavailable'))
            return None
        considered.append(dict(motion_id=c['motion_id'], split=c['split'], reason='selected'))
        groups.add(o['source_group_id'])
        return dict(c, source=o, source_fps=fps, source_frames=frames, prefix_seconds=prefix,
                    source_artifact=artifact(o['resolved_source_path']),
                    requested_source_slice=[start, stop], available_source_slice=[start, min(stop, frames)],
                    last_required_20hz_endpoint_source_frame=endpoint,
                    terminal_policy='Original endpoint-support policy; no extrapolation or time stretching.',
                    builder_end_frame_20fps=-1 if stop > frames else o['end_frame_20fps'] + int(prefix * 20))

    for split, count in COUNTS.items():
        current = []
        for mid in ANCHORS[split]:
            # Retain the original chosen caption, including index 1 for 000027.
            chosen = candidate(old_cases[mid])
            if chosen is None:
                raise ValueError('Original development anchor unavailable: ' + mid)
            current.append(chosen)
        split_groups = sorted((g for (s, g) in grouped if s == split), key=rank)
        for g in split_groups:
            if len(current) == count:
                break
            for c in sorted(grouped[(split, g)], key=lambda r: rank(r['motion_id'])):
                chosen = candidate(c)
                if chosen is not None:
                    current.append(chosen)
                    break
        if len(current) != count:
            raise ValueError(f'Insufficient eligible {split} groups: {len(current)}/{count}')
        selected.extend(current)
    output.parent.mkdir(parents=True, exist_ok=True)
    code = [Path(__file__).resolve(), ROOT / 'scripts/build_canonical_b0_panel.py',
            ROOT / 'scripts/build_standard_imu_cache.py', ROOT / 'scripts/check_canonical_b0_panel.py',
            ROOT / 'src/itm/data/canonical_pair.py', ROOT / 'src/itm/data/standard_imu.py',
            ROOT / 'src/itm/data/amass_bridge.py', ROOT / 'src/itm/data/humanml_clock.py',
            MDM / 'data_loaders/humanml/common/skeleton.py', MDM / 'data_loaders/humanml/common/quaternion.py',
            MDM / 'data_loaders/humanml/utils/paramUtil.py', IMUPOSER / 'src/imuposer/smpl/parametricModel.py']
    write_json(output, dict(version='canonical-smpl-pair-v1', experiment='expanded-canonical-development-v1',
        selection_seed=SEED, selection='Original six development anchors plus seeded source-group order; one full clip per group; no generated-result selection.',
        data_counts=COUNTS, scope='offline_synthetic_development_only_no_formal_test_or_real_device_claim',
        annotation_policy='Dataset text with verified full-clip time support; new captions are not visually certified. No promotion to original B0 semantic admission.',
        budget=dict(cases=len(selected), CPU_FK_calls=2 * len(selected), max_frames_per_call=300,
                    threads=2, absolute_seconds=1800, gpu_jobs=0, training_updates=0, model_generations=0),
        gates=dict(source_fps=[60, 120], max_feature_frames=196, RIC_roundtrip_max_abs_m=1e-5,
                   rotation_orthogonality_max_abs=2e-5),
        cases=selected, considered=considered,
        inputs=[artifact(ORIGINS), artifact(CANDIDATES), artifact(MODEL)], code=[artifact(p) for p in code]))
    print(dict(selected=Counter(r['split'] for r in selected), considered=len(considered), groups=len(groups)), flush=True)


def prepare(directory, verification, output):
    if output.exists():
        raise FileExistsError(output)
    pairs = list(read_jsonl(directory / 'built_pairs.jsonl'))
    check = json.loads(verification.read_text())
    if check['passed'] != sum(COUNTS.values()) or check['pairs'] != artifact(directory / 'built_pairs.jsonl'):
        raise ValueError('All expanded pairs must pass saved-array verification')
    from itm.models.mdm_canonical_conditions import load_canonical_cache
    from itm.models.mdm_conditions import sample_imu_at_times, normalize_imu
    rows, train_acceleration = [], []
    for r in pairs:
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
                source='128 unique train clips at exact model 20Hz times; no duplicate sensor configs or validation frames',
                unique_train_frames=len(values))
    mean_path = ROOT / 'outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/meta/Mean.npy'
    std_path = mean_path.with_name('Std.npy')
    if not mean_path.exists():
        common = json.loads((ROOT / 'docs/itm_rework/protocol/b0_correction_20260926_v1/common.json').read_text())
        mean_path = Path(common['training_input_provenance']['payload']['artifacts']['motion_mean']['path'])
        std_path = Path(common['training_input_provenance']['payload']['artifacts']['motion_std']['path'])
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
                       timestamps=np.arange(frames, dtype=np.float32) / 20., frame_mask=np.ones(frames, dtype=bool))
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
        source_group_disjoint=True, test_inputs_used=False, sensor_configs={'head': [4], 'wrists': [0, 1]},
        physical_scope='original offline pre-canonicalized SMPL bridge; no real-device calibration',
        generated_IMU_closure_certified=False, manual_semantic_admission=False))
    print(dict(inputs=len(records), unique_train_frames=len(values), splits=COUNTS), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('plan'); q.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('build'); q.add_argument('--plan', type=Path, required=True); q.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('inputs'); q.add_argument('--pairs-directory', type=Path, required=True); q.add_argument('--verification', type=Path, required=True); q.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.action == 'plan': select(a.output)
    elif a.action == 'build': build(a.plan, a.output)
    else: prepare(a.pairs_directory, a.verification, a.output)
