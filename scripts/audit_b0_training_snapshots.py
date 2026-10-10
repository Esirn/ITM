#!/usr/bin/env python
"""Read six existing checkpoint files on CPU; construct no model or new state."""
import argparse
from pathlib import Path
import json

import torch

from itm.models.mdm_correction import ROOT, artifact, tensor_hashes, environment
from itm.models.mdm_feature_balance import FeaturePackage, parameter_contract, ARM, SCHEMA
from itm.models.mdm_sampling_alignment import validate_state as validate_u

RUN = ROOT/'outputs/itm_rework/method/feature_balance_authorized_20261008_v1'
IMPL = 'db323ef515ede3252898f322f7badda10913aea64d67cb4abb1f3e9fc4605e30'
FINAL_SHA = 'd0e4f39f8c7ee5b980243baac48bc1bc02dc8f7c15c7a1f47f0adabb0b7e0ad3'
U_SHA = '91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642'
INITIAL_SHA = '3ad3c1bca7ff7398386662e4831eb29136d3c0f2e06e5abbea8e7c4cb13d0401'
DEFAULT = ROOT/'outputs/itm_rework/protocol/training_sufficiency_preparation_20261008_v2/checkpoint_audit.json'


def validate_tensors(state, initial):
    for group in ('imu_encoder', 'adapters'):
        if list(state[group]) != list(initial[group]):
            raise ValueError('Checkpoint tensor keys/order changed')
        for name, value in state[group].items():
            reference = initial[group][name]
            if (value.device.type != 'cpu' or value.shape != reference.shape or value.dtype != reference.dtype
                    or not torch.isfinite(value).all()):
                raise ValueError('Checkpoint tensor shape/dtype/finite/device mismatch')
            if group == 'adapters' and name.endswith('.bias') and not torch.equal(value, reference):
                raise ValueError('Frozen adapter bias changed')


def validate_f_snapshot(state, package, initial, updates, identity):
    """Read-only diagnostic acceptance; never usable as final200 sampling acceptance."""
    if updates not in (50, 100, 150, 200):
        raise ValueError('Only the four predeclared F snapshots permitted')
    if state.get('checkpoint_schema') != SCHEMA or state.get('arm') != ARM:
        raise ValueError('F snapshot schema/arm mismatch')
    for name, expected in identity.items():
        if state.get(name) != expected:
            raise ValueError('F snapshot provenance mismatch: '+name)
    expected = dict(initial['control_config'], reference_mode='static_delta')
    if state['control_config'] != expected:
        raise ValueError('F static-reference config mismatch')
    if state['execution'] != dict(updates_completed=updates, updates_attempted=updates,
                                  final_update_200=updates == 200, exposures=updates*3,
                                  mdm_examples=updates*3, encoder_examples=updates*6,
                                  caption_calls=updates*3):
        raise ValueError('Snapshot update/exposure identity mismatch')
    if Path(state['training_run_directory']).resolve() != RUN.resolve():
        raise ValueError('F snapshot belongs to another run')
    if state['training_audit']['trainable'] != parameter_contract(initial) or state['training_audit']['frozen_unchanged'] is not True:
        raise ValueError('F training parameter audit mismatch')
    defaults = dict(package.common['optimizer']); defaults.pop('name'); defaults['betas'] = tuple(defaults['betas'])
    if state['optimizer_defaults'] != defaults:
        raise ValueError('F optimizer defaults changed')
    hashes = tensor_hashes(state)
    if hashes != state['tensor_hashes'] or hashes == initial['tensor_hashes']:
        raise ValueError('F tensor identity mismatch or initialization relabelled')
    validate_tensors(state, initial)
    groups = state['optimizer']['param_groups']
    if len(groups) != 1 or len(groups[0]['params']) != len(parameter_contract(initial)):
        raise ValueError('F optimizer parameter population changed')
    if set(state['optimizer']['state']) != set(groups[0]['params']):
        raise ValueError('Incomplete F optimizer state')
    if any(float(v['step']) != updates for v in state['optimizer']['state'].values()):
        raise ValueError('F optimizer step disagrees with snapshot update')


def drift(state, initial):
    result = {}
    for group in ('imu_encoder', 'adapters'):
        squared, reference, maximum, count = 0., 0., 0., 0
        for name, value in state[group].items():
            a, b = value.double(), initial[group][name].double()
            squared += float((a-b).square().sum()); reference += float(b.square().sum())
            maximum = max(maximum, float((a-b).abs().max())); count += a.numel()
        result[group] = dict(count=count, rms_from_initial=(squared/count)**.5,
                             initial_rms=(reference/count)**.5, max_abs_from_initial=maximum)
    return result


def audit():
    package = FeaturePackage(); initial = package.initial_state()
    if artifact(package.initial)['sha256'] != INITIAL_SHA:
        raise ValueError('Wrong common initialization')
    if any(v.count_nonzero() for v in initial['adapters'].values()):
        raise ValueError('Initialization adapters must be exactly zero')
    identity = package.identity(IMPL)
    rows = [dict(snapshot_id='I', updates=0, artifact=artifact(package.initial),
                 raw_control_config=initial['control_config'],
                 runtime_control_config=dict(initial['control_config'], reference_mode='static_delta'),
                 runtime_override='Only reference_mode; weights unchanged, no initialization/RNG',
                 tensor_hashes=tensor_hashes(initial), drift_from_initial=drift(initial, initial))]
    for updates in (50, 100, 150, 200):
        path = RUN/'F'/('model.pt' if updates == 200 else f'update_{updates:03d}.pt')
        state = torch.load(path, map_location='cpu', weights_only=True)
        validate_f_snapshot(state, package, initial, updates, identity)
        entry = artifact(path)
        if updates == 200 and entry['sha256'] != FINAL_SHA:
            raise ValueError('Wrong completed F final200')
        rows.append(dict(snapshot_id=f'F{updates:03d}', updates=updates, artifact=entry,
                         raw_control_config=state['control_config'], runtime_control_config=state['control_config'],
                         execution=state['execution'], tensor_hashes=tensor_hashes(state),
                         drift_from_initial=drift(state, initial), role='diagnostic_only_no_sampling_or_selection'))
    path = package.alignment.checkpoint
    state = torch.load(path, map_location='cpu', weights_only=True)
    validate_u(state, package.alignment); validate_tensors(state, initial)
    if artifact(path)['sha256'] != U_SHA:
        raise ValueError('Wrong historical U final200')
    rows.append(dict(snapshot_id='U200', updates=200, artifact=artifact(path),
                     raw_control_config=state['control_config'], runtime_control_config=state['control_config'],
                     execution=state['execution'], tensor_hashes=tensor_hashes(state),
                     drift_from_initial=drift(state, initial), role='historical_uniform_control'))
    if len({json.dumps(r['tensor_hashes'], sort_keys=True) for r in rows}) != 6:
        raise ValueError('Snapshots have duplicate tensor identities')
    return dict(version='itm-training-snapshot-CPU-audit-v1', snapshots=rows, environment=environment(),
                feature_package=artifact(package.path), original_implementation_sha256=IMPL,
                unique_checkpoint_files=6, tensor_deserializations=7,
                deserialization_accounting='initial_state once + identity revalidates initial once + four F + one U = seven CPU loads; six distinct files',
                model_constructions=0, model_forwards=0, GPU_queries=0,
                new_training=0, new_samples=0, interpretation='Weight drift confirms distinct tensors, not sufficient training/convergence/behavior.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, default=DEFAULT)
    args = p.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    result = audit()
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(dict(snapshots=len(result['snapshots']), output=str(args.output), model_forwards=0, GPU_queries=0)))
