#!/usr/bin/env python
"""Recheck saved F trace arithmetic with frozen S1 tolerances; no new forwards."""
import argparse
from pathlib import Path

import numpy as np

from evaluate_b0_feature_balance import FROZEN, IMPLEMENTATION_SHA, PACKAGE_SHA, Sources, read, require, write


def array_descriptor(a):
    import hashlib
    return dict(shape=list(a.shape), dtype=str(a.dtype),
                sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())


def arithmetic(arrays, case):
    values = [arrays[k].astype(np.float64) for k in ('f00', 'f10', 'f01', 'f11')]
    f00, f10, f01, f11 = values
    actual = arrays['actual_combined_prediction'].astype(np.float64)
    require(all(v.shape == actual.shape and np.isfinite(v).all() for v in values) and
            np.isfinite(actual).all(), 'Finite matching trace branches required')
    full = f00+(f10-f00)+(f01-f00)+(f11-f10-f01+f00)
    scale = np.abs(f00)+(np.abs(f10)+np.abs(f00))+(np.abs(f01)+np.abs(f00)) + (
        np.abs(f11)+np.abs(f10)+np.abs(f01)+np.abs(f00))
    unit = 2.**-24
    bound = (16*unit/(1-16*unit))*np.maximum(1., scale)
    ratio = max(float(np.max(np.abs(actual-v)/bound)) for v in (full, f11))
    require(ratio <= 1, 'Frozen float32 CFG combination bound exceeded')
    pairs = []
    if not case['text_present']:
        pairs += [('f00', 'f10'), ('f01', 'f11')]
    if not case['imu_present'] or case['condition'] in ('physical_static', 'backbone_bypass'):
        pairs += [('f00', 'f01'), ('f10', 'f11')]
    for a, b in pairs:
        np.testing.assert_allclose(arrays[a], arrays[b], atol=1e-6, rtol=1e-6)
    return dict(combination_max_error_over_bound=ratio, absent_branch_checks=len(pairs),
                text_increment_rms=float(np.sqrt(np.mean((f10-f00)**2))),
                imu_increment_rms=float(np.sqrt(np.mean((f11-f10)**2))),
                actual_prediction_rms=float(np.sqrt(np.mean(actual**2))))


def audit(run, output):
    require(not output.exists(), 'Refusing to overwrite trace evidence')
    sources = Sources()
    sources.bind(Path(__file__))
    sources.bind(FROZEN/'package.json', PACKAGE_SHA)
    status = read(sources.bind(run/'runtime/status.json'))
    manifest = read(sources.verify(status['implementation']))
    require(status['implementation']['sha256'] == IMPLEMENTATION_SHA, 'Implementation identity')
    specs = read(sources.bind(FROZEN/'trace_slots.json'))
    cases = {r['case_id']: r for r in read(sources.bind(FROZEN/'sampling_slots.json'))}
    inputs = {r['case_id']: r['arrays'] for r in manifest['inputs']}
    rows = []
    for spec in specs:
        row = dict(trace_id=spec['trace_id'], case_id=spec['case_id'], timestep=spec['timestep'], status='missing')
        path = run/'traces'/(spec['trace_id']+'.npz')
        if path.exists():
            try:
                case = cases[spec['case_id']]
                meta = read(sources.bind(path.with_suffix('.json')))
                sources.bind(path)
                require(all(meta[k] == v for k, v in spec.items() if k != 'status') and meta['status'] == 'completed', 'Trace slot identity')
                require(meta['checkpoint_sha256'] == status['checkpoint']['sha256'] and
                        meta['package_sha256'] == PACKAGE_SHA and meta['implementation_sha256'] == IMPLEMENTATION_SHA, 'Trace checkpoint/input identity')
                require(meta['input_hashes'] == inputs[spec['case_id']] and meta['guidance'] == case['guidance'], 'Trace input/guidance')
                require(meta['random_plan_sha256'] == case['historical_random_plan_sha256'] and meta['scaled_t'] == [spec['timestep']], 'Trace noise/timestep')
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {k: archive[k] for k in archive.files}
                require({k: array_descriptor(v) for k, v in arrays.items()} == meta['arrays'], 'Trace array descriptors')
                for k in ('f00', 'f10', 'f01', 'f11', 'x_t', 'actual_combined_prediction'):
                    require(list(arrays[k].shape) == spec['shape'] and arrays[k].dtype == np.float32 and np.isfinite(arrays[k]).all(), 'Trace shape/dtype/finite')
                require(arrays['frame_mask'].shape == (1, case['frames']) and arrays['frame_mask'].dtype == np.bool_ and arrays['frame_mask'].all(), 'Trace valid frames')
                require(array_descriptor(arrays['actual_combined_prediction'])['sha256'] == meta['actual_return_sha256'], 'Trace returned prediction')
                row.update(status='valid', values=arithmetic(arrays, case))
            except (ValueError, KeyError, OSError, AssertionError) as error:
                row.update(status='invalid', reason=str(error))
        rows.append(row)
    output.mkdir(parents=True)
    result = dict(planned=240, valid=sum(r['status'] == 'valid' for r in rows), rows=rows,
        extra_model_calls=0, GPU_queries=0,
        interpretation='Local arithmetic at each saved F state only. F and historical U have different x_t; '
                       'no same-state cross-arm causal comparison or new Q/V probes. Trace checks do not establish motion quality.',
        sources=list(sources.rows.values()))
    write(output/'traces.json', result)
    print({k: result[k] for k in ('planned', 'valid', 'extra_model_calls', 'GPU_queries')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.run.resolve(), args.output.resolve())
