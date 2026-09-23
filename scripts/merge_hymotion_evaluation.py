#!/usr/bin/env python3
"""Audit and merge two predetermined HY-Motion evaluation index ranges."""

import argparse
import json
from pathlib import Path

import numpy as np

from train_hymotion_imu import digest, write_json
from evaluate_hymotion_imu import scores
from itm.data.hymotion import SENSOR_CONFIGS


MODES = ('text_only', 'paired', 'zero', 'shuffled')
METRICS = ('imu_score', 'acceleration_rmse_m_s2', 'orientation_mean_deg', 'jerk_mean_m_s3')


def paired_statistics(values, seed, samples=20_000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = values[rng.integers(0, len(values), (samples, len(values)))].mean(axis=1)
    signs = rng.choice((-1, 1), size=(samples, len(values)))
    randomized = (signs * values).mean(axis=1)
    return dict(mean=float(values.mean()),
                ci95=[float(value) for value in np.quantile(bootstrap, (0.025, 0.975))],
                sign_flip_p=float(np.mean(np.abs(randomized) >= abs(values.mean()))))


def aggregate(rows, seed=20260909, jerk_noninferiority_relative_margin=None):
    summary = {}
    for sensor_index, sensor in enumerate(('head', 'wrists')):
        grouped = {mode: {r['motion_id']: r for r in rows if r['sensor'] == sensor and r['mode'] == mode}
                   for mode in MODES}
        ids = set(grouped['paired'])
        if not ids or any(set(values) != ids for values in grouped.values()):
            raise ValueError('Missing matched evaluation groups')
        means = {mode: {metric: float(np.mean([r[metric] for r in values.values()])) for metric in METRICS}
                 for mode, values in grouped.items()}
        wins = sum(grouped['paired'][key]['imu_score'] < grouped['shuffled'][key]['imu_score'] for key in ids)
        zero_wins = sum(grouped['paired'][key]['imu_score'] < grouped['zero'][key]['imu_score'] for key in ids)
        ordered_ids = sorted(ids)
        jerk_difference = [grouped['paired'][key]['jerk_mean_m_s3']
                           - grouped['text_only'][key]['jerk_mean_m_s3'] for key in ordered_ids]
        score_difference = [grouped['paired'][key]['imu_score']
                            - grouped['shuffled'][key]['imu_score'] for key in ordered_ids]
        jerk_statistics = paired_statistics(jerk_difference, seed + sensor_index * 2)
        score_statistics = paired_statistics(score_difference, seed + sensor_index * 2 + 1)
        text_jerk = means['text_only']['jerk_mean_m_s3']
        noninferiority_margin = (None if jerk_noninferiority_relative_margin is None
                                 else jerk_noninferiority_relative_margin * text_jerk)
        summary[sensor] = dict(records=len(ids), paired_beats_shuffled_count=wins,
                               paired_beats_shuffled_fraction=wins / len(ids),
                               paired_beats_zero_fraction=zero_wins / len(ids),
                               paired_vs_shuffled_gate=wins / len(ids) >= 0.7,
                               paired_vs_shuffled_ci_gate=score_statistics['ci95'][1] < 0,
                               jerk_gate=means['paired']['jerk_mean_m_s3'] <= means['text_only']['jerk_mean_m_s3'],
                               jerk_noninferiority_relative_margin=jerk_noninferiority_relative_margin,
                               jerk_noninferiority_absolute_margin_m_s3=noninferiority_margin,
                               jerk_noninferiority_gate=(None if noninferiority_margin is None
                                                         else jerk_statistics['ci95'][1] <= noninferiority_margin),
                               paired_to_text_only_jerk_ratio=(means['paired']['jerk_mean_m_s3'] / means['text_only']['jerk_mean_m_s3']
                                                               if means['text_only']['jerk_mean_m_s3'] > 0 else None),
                               paired_minus_text_only_jerk=jerk_statistics,
                               paired_minus_shuffled_imu_score=score_statistics,
                               means=means)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--data-run', type=Path, required=True)
    parser.add_argument('--first-shard', type=Path, required=True)
    parser.add_argument('--second-shard', type=Path, required=True)
    parser.add_argument('--split-index', type=int, default=64)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--end-index', type=int)
    parser.add_argument('--seed', type=int, default=20260909)
    parser.add_argument('--jerk-noninferiority-relative-margin', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.data_run / 'run.json').read_text())['train_records']
    train = json.loads((args.training_run / 'run.json').read_text())
    adapter_sha256 = digest(args.training_run / 'adapter.pt')
    end_index = len(data) if args.end_index is None else args.end_index
    if (train.get('motion_padding_frames') != 360
            or not 0 <= args.start_index < args.split_index < end_index <= len(data)):
        raise ValueError('Invalid training padding or shard boundary')
    if (args.jerk_noninferiority_relative_margin is not None
            and (not np.isfinite(args.jerk_noninferiority_relative_margin)
                 or not 0 <= args.jerk_noninferiority_relative_margin < 1)):
        raise ValueError('Invalid jerk non-inferiority margin')
    if args.output.exists():
        raise FileExistsError(args.output)
    train_ids = {r['motion_id'] for r in train['train_records']}
    if any(r['motion_id'] in train_ids or r['split'] != 'val' for r in data):
        raise ValueError('Evaluation must be disjoint validation records')
    shards = []
    for folder in (args.first_shard, args.second_shard):
        integration = json.loads((folder / 'integration.json').read_text())
        if integration['zero_init_max_absolute_difference'] != 0:
            raise ValueError('Failed real-backbone zero-init check')
        if (integration.get('adapter_sha256') != adapter_sha256
                or integration.get('motion_padding_frames') != 360
                or integration.get('rope_table_placement_max_absolute_difference') != 0):
            raise ValueError('Shard checkpoint/padding/RoPE verification mismatch')
        values = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
        keyed = {(r['motion_id'], r['sensor'], r['mode']): r for r in values}
        if len(keyed) != len(values):
            raise ValueError('Duplicate evaluation records within a shard')
        shards.append(keyed)
    rows = []
    max_orthogonality_error = 0.0
    for index in range(args.start_index, end_index):
        record = data[index]
        with np.load(args.data_run / 'records' / f"{record['motion_id']}.npz") as target_file:
            target_imu = target_file['imu']
        shard_index = int(index >= args.split_index)
        folder = (args.first_shard, args.second_shard)[shard_index]
        baselines = []
        for sensor in ('head', 'wrists'):
            for mode in MODES:
                key = (record['motion_id'], sensor, mode)
                entry = dict(shards[shard_index][key])
                if (entry['seed'] != args.seed + index or entry['caption'] != record['caption']
                        or entry['frames'] != record['num_frames']
                        or entry['shuffled_motion_id'] != data[(index + 1) % len(data)]['motion_id']):
                    raise ValueError(f'Protocol mismatch: {key}')
                if not all(np.isfinite(entry[k]) for k in METRICS):
                    raise ValueError(f'Nonfinite metric: {key}')
                path = folder / f'{key[0]}_{sensor}_{mode}.npz'
                with np.load(path, allow_pickle=False) as motion:
                    if json.loads(motion['metadata'].item()) != entry:
                        raise ValueError(f'Output/metric metadata mismatch: {key}')
                    joints, rotations = motion['joints'], motion['rotations']
                    if joints.shape != (record['num_frames'], 22, 3) or rotations.shape != joints.shape[:-1] + (3, 3):
                        raise ValueError(f'Invalid shape: {key}')
                    if not np.isfinite(joints).all() or not np.isfinite(rotations).all():
                        raise ValueError(f'Nonfinite output: {key}')
                    error = float(np.abs(np.swapaxes(rotations, -1, -2) @ rotations - np.eye(3)).max())
                    max_orthogonality_error = max(max_orthogonality_error, error)
                    if error > 1e-4 or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
                        raise ValueError(f'Invalid rotations: {key}')
                    if mode == 'text_only':
                        baselines.append(joints.copy())
                    recomputed = scores(joints, rotations, target_imu, list(SENSOR_CONFIGS[sensor]), np.array(train['acc_std']))
                    if any(not np.isclose(recomputed[k], entry[k], rtol=1e-6, atol=1e-6) for k in METRICS):
                        raise ValueError(f'Recomputed metric mismatch: {key}')
                entry['output_file'] = str(path.resolve())
                rows.append(entry)
        if not np.array_equal(*baselines):
            raise ValueError(f'Text-only noise/control mismatch: {record["motion_id"]}')
    summary = dict(records=end_index - args.start_index, generated_outputs=len(rows),
                   start_index=args.start_index, end_index=end_index, split_index=args.split_index,
                   shards=[str(p.resolve()) for p in (args.first_shard, args.second_shard)],
                   adapter_sha256=adapter_sha256,
                   backbone_sha256=train['checkpoint_sha256'], max_rotation_orthogonality_error=max_orthogonality_error,
                   coverage_complete=True, train_val_disjoint=True, text_only_identical_between_sensor_configs=True,
                   semantic_gate='Not formally evaluated', full_training_allowed=False,
                   **aggregate(rows, seed=args.seed,
                               jerk_noninferiority_relative_margin=args.jerk_noninferiority_relative_margin))
    args.output.mkdir(parents=True)
    (args.output / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
