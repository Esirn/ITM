#!/usr/bin/env python3
"""Measure acceleration/orientation contributions to a trained HY-Motion adapter."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from itm.data.hymotion import SENSOR_CONFIGS
from itm.models.hymotion_imu_adapter import HYMotionIMUConfig, make_hymotion_imu_adapter
from train_hymotion_imu import write_json


VARIANTS = ('zero', 'acceleration', 'orientation', 'both')


def residual_statistics(value):
    return dict(
        amplitude_rms=float(value.square().mean().sqrt()),
        temporal_difference_rms=float(torch.diff(value, dim=-2).square().mean().sqrt()),
        temporal_curvature_rms=float(torch.diff(value, n=2, dim=-2).square().mean().sqrt()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--data-run', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-index', type=int, default=16)
    parser.add_argument('--end-index', type=int, default=80)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    training = json.loads((args.training_run / 'run.json').read_text())
    data = json.loads((args.data_run / 'run.json').read_text())['train_records']
    if not 0 <= args.start_index < args.end_index <= len(data):
        raise ValueError('invalid record interval')
    saved = torch.load(args.training_run / 'adapter.pt', map_location='cpu', weights_only=True)
    config = HYMotionIMUConfig(**saved['adapter_config'])
    adapter = make_hymotion_imu_adapter(config).eval()
    adapter.load_state_dict(saved['adapter'], strict=True)
    injection_end = (config.double_blocks + config.single_blocks
                     if config.control_injection_end is None else config.control_injection_end)
    layers = list(range(config.control_injection_start, injection_end))
    evaluated = [json.loads(line) for line in (args.evaluation / 'metrics.jsonl').read_text().splitlines()]
    metric_by_key = {(row['motion_id'], row['sensor'], row['mode']): row for row in evaluated}
    acc_mean = np.asarray(training['acc_mean'], dtype=np.float32)
    acc_std = np.asarray(training['acc_std'], dtype=np.float32)
    records = []
    with torch.no_grad():
        for row in data[args.start_index:args.end_index]:
            with np.load(args.data_run / 'records' / f"{row['motion_id']}.npz") as value:
                imu = value['imu'].copy()
            imu[..., :3] = (imu[..., :3] - acc_mean) / acc_std
            frames = len(imu)
            conditions, masks, labels = [], [], []
            for sensor, slots in SENSOR_CONFIGS.items():
                mask = np.zeros(6, dtype=bool)
                mask[list(slots)] = True
                for variant in VARIANTS:
                    condition = imu.copy()
                    if variant in ('zero', 'orientation'):
                        condition[..., :3] = 0
                    if variant in ('zero', 'acceleration'):
                        condition[..., 3:] = 0
                    padded = np.zeros((config.max_frames, 6, 12), dtype=np.float32)
                    padded[:frames] = condition
                    conditions.append(padded)
                    masks.append(mask)
                    labels.append((sensor, variant))
            condition = torch.from_numpy(np.stack(conditions))
            sensor_mask = torch.from_numpy(np.stack(masks))
            frame_mask = torch.arange(config.max_frames)[None] < frames
            frame_mask = frame_mask.expand(len(labels), -1)
            tokens, _ = adapter(condition, sensor_mask, frame_mask,
                                keep=torch.ones(len(labels), dtype=torch.bool))
            residuals = torch.stack([adapter.residuals[index](tokens) for index in layers])
            outputs = {label: residuals[:, index:index + 1, :frames]
                       for index, label in enumerate(labels)}
            for sensor in SENSOR_CONFIGS:
                comparisons = {
                    'acceleration_vs_zero': outputs[sensor, 'acceleration'] - outputs[sensor, 'zero'],
                    'orientation_vs_zero': outputs[sensor, 'orientation'] - outputs[sensor, 'zero'],
                    'both_vs_zero': outputs[sensor, 'both'] - outputs[sensor, 'zero'],
                    'acceleration_marginal': outputs[sensor, 'both'] - outputs[sensor, 'orientation'],
                    'orientation_marginal': outputs[sensor, 'both'] - outputs[sensor, 'acceleration'],
                }
                paired = metric_by_key[row['motion_id'], sensor, 'paired']
                text = metric_by_key[row['motion_id'], sensor, 'text_only']
                for comparison, value in comparisons.items():
                    records.append(dict(
                        motion_id=row['motion_id'], sensor=sensor, comparison=comparison,
                        frames=frames,
                        observed_jerk_delta=(paired['jerk_mean_m_s3']
                                             - text['jerk_mean_m_s3']),
                        **residual_statistics(value),
                    ))
    summary = dict(records=args.end_index - args.start_index, active_layers=layers,
                   start_index=args.start_index, end_index=args.end_index, sensors={})
    fields = ('amplitude_rms', 'temporal_difference_rms', 'temporal_curvature_rms')
    for sensor in SENSOR_CONFIGS:
        summary['sensors'][sensor] = {}
        for comparison in sorted({row['comparison'] for row in records}):
            selected = [row for row in records
                        if row['sensor'] == sensor and row['comparison'] == comparison]
            jerk = np.asarray([row['observed_jerk_delta'] for row in selected])
            summary['sensors'][sensor][comparison] = {
                field: dict(
                    mean=float(np.mean([row[field] for row in selected])),
                    correlation_with_observed_jerk_delta=float(np.corrcoef(
                        [row[field] for row in selected], jerk)[0, 1]),
                ) for field in fields
            }
    args.output.mkdir(parents=True)
    (args.output / 'records.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
