#!/usr/bin/env python
"""One predeclared 100-epoch development comparison, without a weight sweep."""
import argparse
from pathlib import Path

from itm.models.mdm_correction import read, artifact, verify
from run_expanded_canonical_training import schedule, write


def training(base, output):
    if output.exists(): raise FileExistsError(output)
    plan = read(base); dataset = read(verify(plan['dataset']))
    exposures = schedule(dataset['records'], 100, plan['batch_size'], plan['seed'])
    if exposures[:len(plan['exposures'])] != plan['exposures']: raise ValueError('Original training prefix changed')
    plan.update(epochs=100, exposures=exposures, snapshot_selection='final epoch100 only; no checkpoint selection',
                comparison='Extend per-clip exposure from20 to100; fresh same initialization and exact first20 schedule; both simple baselines also100epochs',
                base_twenty_epoch_plan=artifact(base))
    for key in ('updates', 'exposures', 'mdm_calls', 'encoder_calls', 'baseline_updates', 'baseline_train_examples'):
        plan['counts'][key] *= 5
    plan['expected_artifacts'] += [artifact(base), artifact(Path(__file__).resolve())]
    output.mkdir(parents=True); write(output / 'plan.json', plan); print(plan['counts'])


def sampling(base, training_plan, training_run, output):
    if output.exists(): raise FileExistsError(output)
    plan, train, status = read(base), read(training_plan), read(training_run / 'queue_status.json')
    if status['status'] != 'completed' or status['worker']['updates'] != 1600:
        raise ValueError('Complete epoch100 training required')
    plan.update(training_plan=artifact(training_plan), checkpoint=status['worker']['main_checkpoint'],
                method='expanded U final epoch100, uniform loss, factorized unit guidance',
                base_twenty_epoch_sampling_plan=artifact(base))
    verify(plan['checkpoint'])
    plan['expected_artifacts'] = train['expected_artifacts'] + [artifact(base), artifact(training_plan),
        artifact(training_run / 'queue_status.json'), artifact(Path(__file__).resolve()),
        artifact(Path(__file__).with_name('run_expanded_canonical_sampling.py'))]
    output.mkdir(parents=True); write(output / 'plan.json', plan); print(plan['counts'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('training'); q.add_argument('--base', type=Path, required=True); q.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('sampling'); q.add_argument('--base', type=Path, required=True); q.add_argument('--training-plan', type=Path, required=True); q.add_argument('--training-run', type=Path, required=True); q.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.action == 'training': training(a.base.resolve(), a.output.resolve())
    else: sampling(a.base.resolve(), a.training_plan.resolve(), a.training_run.resolve(), a.output.resolve())
