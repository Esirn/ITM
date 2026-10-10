#!/usr/bin/env python
"""CPU saved-array evaluation for frozen F/U. Never launches or resumes a run.

The default absent F directory is intentional: preparation replays all historical
U arrays and preserves every missing F slot. Synthetic fixtures belong in tests,
never in this evidence population. Review notes must precede decoding the key.
"""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from itm.metrics.feature_balance import (
    SCALARS, aggregate, contrast, curves, feature_views, gate, motion_metrics,
    primary_decision, subtract,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'docs/itm_rework/protocol/b0_feature_balance_20261003_v1'
REQUEST = ROOT / 'docs/itm_rework/method/B0_FEATURE_BALANCE_AUTHORIZATION_REQUEST_20261005_V2.json'
PACKAGE_SHA = 'a4189b89f1924af022c67dea71c02c2792e65f4a185775501b07d037fba51af9'
IMPLEMENTATION_SHA = 'db323ef515ede3252898f322f7badda10913aea64d67cb4abb1f3e9fc4605e30'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Sources:
    def __init__(self):
        self.rows = {}

    def bind(self, path, expected=None):
        path = (ROOT / path).resolve()
        key = str(path)
        if key not in self.rows:
            self.rows[key] = dict(path=key, sha256=sha(path))
        require(expected is None or self.rows[key]['sha256'] == expected, 'Source hash mismatch: ' + key)
        return path

    def verify(self, entry):
        return self.bind(Path(entry['path']), entry['sha256'])


def identity_audit(sources):
    snapshot = read(sources.bind(ROOT / 'docs/itm_rework/session_handoff_20261007_v1/snapshot.json'))
    snapshot_drift = []
    for entry in snapshot['anchors']:
        if entry['path'].endswith('/STATUS.md'):
            actual = sha(sources.bind(Path(entry['path'])))
            if actual != entry['sha256']:
                snapshot_drift.append(dict(path=entry['path'], current_sha256=actual, snapshot_sha256=entry['sha256']))
        else:
            sources.verify(entry)
    for entry in [r['status_file'] for r in snapshot['previous_attempts']]:
        sources.verify(entry)
    request = read(REQUEST)
    manifest = read(sources.bind(Path(request['implementation']), IMPLEMENTATION_SHA))
    sources.bind(FROZEN / 'package.json', PACKAGE_SHA)
    for entry in manifest['expected_artifacts']:
        sources.verify(entry)
    for entry in read(FROZEN / 'package.json')['artifacts']:
        sources.verify(entry)
    for path in [Path(__file__), ROOT / 'src/itm/metrics/feature_balance.py', ROOT / 'src/itm/metrics/canonical_b0.py']:
        sources.bind(path)
    return request, manifest, dict(manifest_bound_files=len(manifest['expected_artifacts']),
        snapshot_anchors_checked=len(snapshot['anchors']), mutable_status_doc_drift=snapshot_drift,
        previous_failures=snapshot['previous_attempts'],
        archive_byte_identical=Path(request['implementation']).read_bytes() ==
        (ROOT / 'docs/itm_rework/method/feature_balance_implementation_20261003_v1/implementation.json').read_bytes(),
        request_authorized=request['authorized'], new_GPU_queries=0, new_model_forwards=0,
        scope='Main-session self-check/replay; not independent acceptance or model effectiveness')


def runtime_audit(run, sources, slots):
    """Fail closed for support, while allowing diagnostic arrays to be reported."""
    path = run / 'runtime/status.json'
    if not path.exists():
        return None, dict(status='not_run', eligible=False, reason='No F runtime status; no real F result')
    status = read(sources.bind(path))
    errors = []
    try:
        require(status['implementation']['sha256'] == IMPLEMENTATION_SHA, 'Runtime implementation')
        sources.verify(status['implementation'])
        auth = status['authorization']
        require(auth['authorized'] is True and auth['approved_by'] == 'user' and bool(auth['user_instruction_reference']), 'Runtime authorization record')
        require(auth['implementation_sha256'] == IMPLEMENTATION_SHA and auth['package_sha256'] == PACKAGE_SHA, 'Runtime authorization identity')
        require(auth['run_directory'] == str(run), 'Runtime run directory')
        expected = dict(updates=('update', list(range(1, 201))),
            exposures=('exposure_id', [x['exposure_id'] for x in read(FROZEN / 'training_schedule.json')['rows']]),
            samples=('case_id', [x['case_id'] for x in slots]),
            traces=('trace_id', [x['trace_id'] for x in read(FROZEN / 'trace_slots.json')]),
            checks=('check_id', [x['check_id'] for x in read(FROZEN / 'software_checks.json')]))
        for key, (field, ids) in expected.items():
            require([x[field] for x in status[key]] == ids, 'Runtime denominator/order: ' + key)
        if status['status'] != 'completed':
            errors.append('Queue did not complete; saved arrays are diagnostic only')
        else:
            require(all(x['status'] == 'completed' and not x.get('diagnostic_only', False)
                        for key in expected for x in status[key]), 'Incomplete ledger')
            sources.verify(status['checkpoint'])
            require(Path(status['checkpoint']['path']).resolve() == run / 'F/model.pt', 'Only own F final200 checkpoint')
            require(len(status['completed_receipts']) == 61, 'Expected train plus 60 receipts')
            completed_slots = []
            completed_traces = []
            artifact_paths = set()
            for entry in status['completed_receipts']:
                receipt_path = sources.verify(entry)
                receipt = read(receipt_path)
                context = read(sources.bind(receipt_path.with_name(receipt_path.name.replace('.receipt.', '.context.'))))
                require(receipt['status'] == 'completed' and receipt['context'] == context, 'Receipt context/status')
                require(context['run_directory'] == str(run) and context['gpu_uuid'] == auth['gpu_uuid'], 'Receipt run/GPU identity')
                require(context['memory_bytes'] == 24 * 1024**3, 'Receipt memory budget')
                require(context['implementation_sha256'] == IMPLEMENTATION_SHA and context['package_sha256'] == PACKAGE_SHA, 'Receipt input identity')
                train = context['phase'] == 'train'
                require(context['phase'] in ('train', 'sample'), 'Unexpected phase')
                if not train:
                    require(context['checkpoint'] == status['checkpoint'], 'Receipt final200 identity')
                for kind, cap in dict(mdm=600 if train else 4000, encoder=1200 if train else 2, caption=600 if train else 1).items():
                    counts = receipt['counts'][kind]
                    calls = 1000 if kind == 'mdm' and not train else cap
                    require(all(counts[k] == cap for k in ('reserved', 'actual', 'completed')) and
                            counts['calls'] == counts['completed_calls'] == calls, 'Receipt counters')
                require(receipt['completed_slots'] == context['expected_slots'] and receipt['completed_traces'] == context['expected_traces'], 'Receipt slot/trace IDs')
                completed_slots.extend(receipt['completed_slots'])
                completed_traces.extend(receipt['completed_traces'])
                require((receipt['updates_attempted'], receipt['updates_completed'], receipt['exposures'],
                         receipt['denoising_calls'], receipt['branch_examples']) ==
                        ((200, 200, 600, 0, 0) if train else (0, 0, 0, 1000, 4000)), 'Receipt work counts')
                require(receipt['peak_allocated_bytes'] <= 24 * 1024**3, 'Receipt memory cap')
                require(receipt['last_monotonic'] <= context['deadline_monotonic'] <= status['total_deadline_monotonic'], 'Receipt deadline')
                require(context['deadline_monotonic'] == status['training_deadline_monotonic' if train else 'sampling_deadline_monotonic'], 'Phase deadline identity')
                if not train:
                    require(context['deadline_monotonic'] <= receipt['started_monotonic'] + 3000, 'Shared sampling cap')
                for artifact in receipt['artifacts']:
                    artifact_paths.add(str(sources.verify(artifact)))
            require(completed_slots == [f'U{i:03d}' for i in range(1, 201)] + [x['case_id'] for x in slots], 'Receipt population/order')
            require(completed_traces == [x['trace_id'] for x in read(FROZEN / 'trace_slots.json')], 'Receipt trace population/order')
            required = {str(run / 'F' / f'update_{i:03d}.json') for i in range(1, 201)}
            required.update(str(run / 'F/exposures' / (x['exposure_id'] + '.json')) for x in read(FROZEN / 'training_schedule.json')['rows'])
            required.update(str(run / 'F' / name) for name in ('model.pt', 'update_050.pt', 'update_100.pt', 'update_150.pt'))
            for slot in slots:
                required.update(str(run / 'samples' / (slot['case_id'] + ext)) for ext in ('.json', '.npz'))
                required.add(str(run / 'raw_samples' / (slot['case_id'] + '.npz')))
            for trace in read(FROZEN / 'trace_slots.json'):
                required.update(str(run / 'traces' / (trace['trace_id'] + ext)) for ext in ('.json', '.npz'))
            require(required.issubset(artifact_paths), 'Missing receipt-bound output/trace/training artifact')
            start = status['started_monotonic']
            require(status['total_deadline_monotonic'] == start + 3600 and status['training_deadline_monotonic'] == start + 600, 'Queue deadlines')
            require(status['finished_monotonic'] <= start + 3600, 'Queue elapsed cap')
    except (ValueError, KeyError, OSError, TypeError) as exc:
        errors.append(str(exc))
    return status, dict(status=status.get('status'), eligible=not errors, errors=errors,
                        ledger_counts={k: dict(Counter(r['status'] for r in status.get(k, [])))
                                       for k in ('updates', 'exposures', 'samples', 'traces', 'checks')})


def load_sample(path, slot, arm, sources, mean, std, manifest, runtime):
    meta = read(sources.bind(path.with_suffix('.json')))
    sources.bind(path)
    cid = slot['case_id'] if arm == 'F' else slot['historical_case_id']
    require(meta['case_id'] == cid, 'Case identity')
    for key in ('data_split', 'motion_id', 'sensor_config', 'condition', 'frames', 'text', 'caption_index', 'guidance'):
        require(meta[key] == slot[key], 'Metadata identity: ' + key)
    require(meta['sampler'] == 'ddpm' and meta['model_forward_calls'] == 1000 and meta['MDM_branch_examples'] == 4000, 'Sampling plan/counters')
    old_meta = read(Path(slot['historical_metadata']['path']))
    require(meta['random_plans'] == old_meta['random_plans'], 'Frozen historical noise plan')
    if arm == 'F':
        require(runtime is not None, 'Unledgered F output')
        require(meta['implementation_sha256'] == IMPLEMENTATION_SHA and meta['package_sha256'] == PACKAGE_SHA, 'F input identity')
        require(meta['control_checkpoint_sha256'] == runtime['checkpoint']['sha256'], 'F final200 identity')
        require(meta['checkpoint_schema'] == 'itm-feature-balance-control-v1', 'F checkpoint schema')
        require(meta['loss_version'] == manifest['loss']['loss_version'] and meta['loss_weights'] == manifest['loss']['weights'], 'F loss identity')
        require(meta['loss_weights_sha256'] == manifest['loss']['weights_sha256'], 'F loss weights hash')
        require(meta['initialization_sha256'] == manifest['initial']['sha256'] and meta['initial_tensor_hashes'] == manifest['initial_tensor_hashes'], 'F initialization identity')
        require(meta['training_schedule_sha256'] == manifest['training_schedule']['sha256'] and meta['environment'] == manifest['environment'], 'F training plan/environment')
        require(meta['training_execution']['updates_completed'] == 200 and meta['training_execution']['exposures'] == 600 and meta['training_execution']['final_update_200'], 'F training counts')
        require(meta['training_run_directory'] == runtime['authorization']['run_directory'], 'F source run')
    else:
        require(meta['control_checkpoint_sha256'] == read(FROZEN / 'design.json')['historical_checkpoint']['sha256'], 'U checkpoint identity')
    with np.load(path, allow_pickle=False) as arrays:
        require(json.loads(str(arrays['metadata'])) == meta, 'Embedded metadata differs')
        for key, spec in slot['historical_input_hashes'].items():
            a = arrays[key]
            descriptor = dict(shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
            require(descriptor == spec, 'Input array identity: ' + key)
        p, features = arrays['motion'].copy(), arrays['normalized_humanml263'].copy()
        require(p.shape == (1, slot['frames'], 22, 3) and features.shape == (1, 263, 1, slot['frames']), 'Full sequence shape')
        require(float(arrays['fps']) == 20 and np.isfinite(p).all() and np.isfinite(features).all(), 'Finite full20Hz output')
    # Existing CPU-only algebra; no checkpoint deserialization or neural forward.
    import torch
    from itm.data.humanml_torch import recover_from_ric_torch
    vectors = features[0, :, 0, :].T
    decoded = recover_from_ric_torch(torch.from_numpy(vectors * std + mean)).numpy()
    np.testing.assert_allclose(decoded, p[0], atol=2e-6, rtol=1e-6)
    return p[0], vectors, float(np.abs(decoded-p[0]).max())


def software_checks(rows, predictions):
    by_id = {r['case_id']: r for r in rows}
    result = []
    for spec in read(FROZEN / 'software_checks.json'):
        a, b = (by_id[spec[k]] for k in ('new_case_id', 'reference_case_id'))
        record = dict(check_id=spec['check_id'], kind=spec['kind'], status='not_comparable', passed=None)
        if a['status'] == b['status'] == 'valid':
            p, q = predictions[a['case_id']], predictions[b['case_id']]
            d = p.astype(float) - q
            maximum = float(np.abs(d).max())
            record.update(status='passed' if maximum <= spec['max_abs_m'] else 'failed',
                          passed=maximum <= spec['max_abs_m'], bitwise=bool(np.array_equal(p, q)),
                          max_abs_m=maximum, RMS_m=float(np.sqrt(np.mean(d*d))),
                          finite_count=int(np.isfinite(d).sum()), count=d.size)
        result.append(record)
    return result


def comparisons(rows, predictions, slots):
    by_id = {r['case_id']: r for r in rows}
    between, within, did = [], [], []
    for s in slots:
        group = {k: s[k] for k in ('data_split', 'motion_id', 'sensor_config', 'condition')}
        between.append(dict(group, **contrast(by_id[s['case_id']], by_id[s['historical_case_id']], predictions)))
    for spec in read(FROZEN / 'within_contrasts.json'):
        group = {k: spec[k] for k in ('data_split', 'motion_id', 'sensor_config', 'comparator')}
        pair = []
        for arm in ('F', 'U'):
            row = dict(group, arm=arm, **contrast(by_id[spec[arm+'_paired']], by_id[spec[arm+'_reference']], predictions))
            within.append(row)
            pair.append(row)
        a, b = pair
        row = dict(group, status='not_comparable', reasons=[a.get('reasons'), b.get('reasons')])
        if a['status'] == b['status'] == 'valid':
            row.update(status='valid', **{k: subtract(a[k], b[k]) for k in ('metrics', 'extent_delta', 'feature_delta')},
                       output_distance_difference_m=a['mean_joint_distance_m']-b['mean_joint_distance_m'])
        did.append(row)
    return dict(F_minus_U=between, within=within, difference_of_contrasts=did,
                between_aggregates=aggregate(between, ('data_split', 'sensor_config', 'condition')),
                within_aggregates=aggregate(within, ('arm', 'data_split', 'sensor_config', 'comparator')),
                difference_aggregates=aggregate(did, ('data_split', 'sensor_config', 'comparator')))


def decisions(rows, notes, runtime_ok):
    index = {(r['arm'], r['data_split'], r['motion_id'], r['sensor_config'], r['condition']): r for r in rows}
    primary, regressions = {}, []
    for config in ('head', 'wrists'):
        f, u, text, static = [index[a, 'train', '000506', config, c] for a, c in
                             [('F', 'paired'), ('U', 'paired'), ('F', 'text_only'), ('F', 'physical_static')]]
        result = primary_decision(f, u, text, static, notes.get(f['case_id'], {}).get('classification', 'missing'),
                                  notes.get(u['case_id'], {}).get('classification', 'missing'))
        if not runtime_ok:
            result['decision'] = 'inconclusive'
            result['eligibility_reason'] = 'No complete accepted F queue; diagnostic data excluded from support'
        primary[config] = result
    for key, f in index.items():
        arm, split, mid, config, condition = key
        if arm != 'F' or condition != 'paired' or (split, mid) == ('train', '000506'):
            continue
        u = index['U', split, mid, config, condition]
        check = gate(f, u)
        exceedances = [k for k, v in check['metrics'].items() if not v['nonworse']]
        fn, un = notes.get(f['case_id'], {}), notes.get(u['case_id'], {})
        regressions.append(dict(data_split=split, motion_id=mid, sensor_config=config,
            status='not_comparable' if check['status'] == 'inconclusive' else 'evaluated',
            numerical_exceedances=exceedances, metrics=check['metrics'],
            F_review=fn, U_review=un,
            explicitly_reported_F_behavior_losses=fn.get('behavior_losses'),
            behavior_loss=True if fn.get('behavior_losses') or (fn.get('classification') == 'incomplete' and un.get('classification') == 'complete')
            else None if fn.get('classification') not in ('complete', 'incomplete') or un.get('classification') not in ('complete', 'incomplete')
            else False))
    return dict(primary=primary, other_paired_regressions=regressions,
                interpretation='Per-config local evidence only; no broad panel certification. Review all mandatory negatives.',
                limitations=read(FROZEN / 'evaluation.json')['limitations'])


def review_notes(path, sources):
    order = read(FROZEN / 'review_order.json')['rows']
    if path is None:
        return {}, None
    sources.bind(path)
    notes = read(path)
    require([r['review_id'] for r in notes] == [r['review_id'] for r in order], 'All120 notes required in frozen order before key decoding')
    for note in notes:
        require(note['classification'] in ('complete', 'incomplete', 'uncertain', 'missing'), 'Review classification')
        require(all(isinstance(note.get(k), str) and note[k].strip() for k in ('evidence', 'side_order_return', 'reason')), 'Review evidence/reason required')
        require(isinstance(note.get('behavior_losses'), list) and all(isinstance(x, str) and x.strip() for x in note['behavior_losses']), 'Explicit behavior loss list required (empty if none observed)')
        require(not {'arm', 'case_id', 'condition', 'sensor_config'}.intersection(note), 'Notes must use hidden review IDs only')
    return {r['case_id']: n for r, n in zip(order, notes)}, notes


def training_losses(run, sources):
    rows = []
    for spec in read(FROZEN / 'training_schedule.json')['rows']:
        path = run / 'F/exposures' / (spec['exposure_id'] + '.json')
        row = dict(exposure_id=spec['exposure_id'], update=spec['update'], status='missing')
        if path.exists():
            try:
                record = read(sources.bind(path))
                require(all(record[k] == v for k, v in spec.items()), 'Exposure schedule identity')
                losses = record['loss_components']
                require(set(losses) == {'uniform', 'root_ric', 'remaining', 'balanced'} and
                        np.isfinite(list(losses.values())).all(), 'Finite four-view exposure losses')
                require(record['actual_shape'] == [1, 263, 1, spec['frames']] and
                        record['actual_noise_sha256'] == spec['noise']['noise_sha256'], 'Exposure shape/noise')
                # Float32 reduction replay only; this is not an effect tolerance.
                np.testing.assert_allclose([losses['balanced'], losses['uniform'], record['loss'], record['loss_divided_by_three']],
                    [.5*(losses['root_ric']+losses['remaining']), (67*losses['root_ric']+196*losses['remaining'])/263,
                     losses['balanced'], losses['balanced']/3], rtol=2e-6, atol=1e-7)
                row.update(status='recorded', record=record)
            except (ValueError, KeyError, OSError, AssertionError, TypeError, BadZipFile, EOFError) as exc:
                row.update(status='invalid', reason=str(exc))
        rows.append(row)
    return dict(planned=600, recorded=sum(r['status'] == 'recorded' for r in rows), rows=rows,
                interpretation='Same-forward saved losses; evolving noise/t/dropout/weights, not controlled convergence evidence')


def build(output, run=None, notes_path=None, render=True):
    output = Path(output).resolve()
    require(not output.exists(), 'Refusing to overwrite evaluation artifacts')
    sources = Sources()
    request, manifest, audit = identity_audit(sources)
    run = Path(run or request['run_directory']).resolve()
    # Evaluation never creates anything inside a runtime directory.
    require(not output.is_relative_to(run), 'Evaluation output cannot be inside the F run directory')
    slots = read(FROZEN / 'sampling_slots.json')
    status, runtime = runtime_audit(run, sources, slots)
    notes, submitted_notes = review_notes(notes_path, sources)
    common = read(FROZEN / 'design.json')['common']['training_input_provenance']['payload']['artifacts']
    mean, std = [np.load(sources.verify(common[k]), allow_pickle=False) for k in ('motion_mean', 'motion_std')]
    rows, predictions, targets = [], {}, {}
    for slot in slots:
        gtpath = sources.bind(Path(slot['gt_path']))
        gt = np.load(gtpath, allow_pickle=False)
        target = np.load(sources.bind(gtpath.with_name('motion.npy')), allow_pickle=False)
        targets[slot['case_id']] = targets[slot['historical_case_id']] = gt
        for arm in ('F', 'U'):
            cid = slot['case_id'] if arm == 'F' else slot['historical_case_id']
            row = dict(case_id=cid, arm=arm, **{k: slot[k] for k in ('data_split', 'motion_id', 'sensor_config', 'condition', 'frames')},
                       status='missing', reason='F not run' if arm == 'F' else 'Historical output missing')
            path = run / 'samples' / (cid + '.npz') if arm == 'F' else Path(slot['historical_output']['path'])
            try:
                if not path.exists():
                    if status and arm == 'F':
                        ledger_row = next((r for r in status.get('samples', []) if r['case_id'] == cid), {})
                        row['reason'] = 'F output absent; ledger=' + ledger_row.get('status', 'missing')
                    rows.append(row)
                    continue
                if arm == 'U':
                    sources.verify(slot['historical_output'])
                    sources.verify(slot['historical_metadata'])
                p, f, decode_error = load_sample(path, slot, arm, sources, mean, std, manifest, status)
                predictions[cid] = p
                row.update(status='valid' if arm == 'U' or runtime['eligible'] else 'diagnostic_incomplete',
                           reason=None if arm == 'U' or runtime['eligible'] else 'F queue incomplete/invalid',
                           metrics=motion_metrics(p, gt, slot['sensor_config'], slot['data_split'], slot['motion_id'], target),
                           feature_views=feature_views(f, (target-mean)/std), decode_max_abs_m=decode_error)
            except (ValueError, KeyError, OSError, AssertionError, TypeError, BadZipFile, EOFError) as exc:
                row.update(status='invalid', reason=str(exc))
            rows.append(row)
    losses = training_losses(run, sources)
    if status and runtime['eligible'] and losses['recorded'] != 600:
        runtime['eligible'] = False
        runtime.setdefault('errors', []).append('Missing/invalid training loss records')
        for r in rows:
            if r['arm'] == 'F' and r['status'] == 'valid':
                r.update(status='diagnostic_incomplete', reason='Training loss record failure')
    checks = software_checks(rows, predictions)
    if any(r['status'] == 'failed' for r in checks):
        runtime['eligible'] = False
        runtime.setdefault('errors', []).append('Recomputed software equality failed')
        for r in rows:
            if r['arm'] == 'F' and r['status'] == 'valid':
                r.update(status='diagnostic_incomplete', reason='Software equality failure')
    if sum(r['arm'] == 'F' and r['status'] == 'valid' for r in rows) != 60:
        runtime['eligible'] = False
    report = dict(version='itm-feature-balance-effect-evaluation-v1', mode='real_saved_F_U' if status else 'CPU_preparation_historical_U_only',
        synthetic_data=False, F_run_directory=str(run), denominators=read(FROZEN / 'evaluation.json')['denominators'],
        populations=[dict(arm=arm, planned=60, valid=sum(r['arm'] == arm and r['status'] == 'valid' for r in rows),
                          failure_or_missing_n=sum(r['arm'] == arm and r['status'] != 'valid' for r in rows),
                          failure_or_missing_fraction=sum(r['arm'] == arm and r['status'] != 'valid' for r in rows)/60,
                          statuses=dict(Counter(r['status'] for r in rows if r['arm'] == arm))) for arm in ('F', 'U')],
        rows=rows, aggregates=aggregate(rows, ('arm', 'data_split', 'sensor_config', 'condition')))
    output.mkdir(parents=True)
    write(output / 'identity_audit.json', audit)
    write(output / 'runtime_audit.json', runtime)
    write(output / 'samples.json', report)
    write(output / 'contrasts.json', comparisons(rows, predictions, slots))
    write(output / 'software_checks.json', checks)
    write(output / 'training_losses.json', losses)
    write(output / 'decision.json', decisions(rows, notes, runtime['eligible']))
    write(output / 'curves.json', {cid: dict(prediction=curves(p), GT=curves(targets[cid])) for cid, p in predictions.items()})
    write(output / 'review_notes_template.json', [dict(review_id=r['review_id'], classification='missing',
          evidence='', side_order_return='', reason='', behavior_losses=[]) for r in read(FROZEN / 'review_order.json')['rows']])
    if submitted_notes is not None:
        write(output / 'submitted_review_notes.json', submitted_notes)
        write(output / 'decoded_review.json', [dict(key, note=note) for key, note in zip(read(FROZEN / 'review_order.json')['rows'], submitted_notes)])
    if render:
        render_review(output, rows, predictions, targets, slots)
    write(output / 'sources.json', list(sources.rows.values()))
    write(output / 'artifact_index.json', [dict(path=str(p), sha256=sha(p)) for p in sorted(output.rglob('*')) if p.is_file()])
    return report


def render_review(output, rows, predictions, targets, slots):
    """Local review artifact: hidden labels, ten poses, all frames and curves."""
    folder = output / 'review'
    folder.mkdir()
    by_id = {r['case_id']: r for r in rows}
    captions = {cid: s['text'] for s in slots for cid in (s['case_id'], s['historical_case_id'])}
    links = []
    for key in read(FROZEN / 'review_order.json')['rows']:
        cid, rid = key['case_id'], key['review_id']
        p = predictions.get(cid)
        g = targets[cid]
        data = dict(prediction=None if p is None else p.tolist(), GT=g.tolist(),
                    curves=None if p is None else curves(p), GT_curves=curves(g))
        payload = json.dumps(data, allow_nan=False).replace('<', '\\u003c')
        document = REVIEW_HTML.replace('__TITLE__', rid).replace('__CAPTION__', html.escape(captions[cid])).replace('__STATUS__', by_id[cid]['status']).replace('__PAYLOAD__', payload)
        (folder / (rid + '.html')).write_text(document)
        links.append(f'<li><a href="{rid}.html">{rid}</a></li>')
    (folder / 'index.html').write_text('<meta charset="utf-8"><h1>Fixed 120-row review</h1><p>Historical U already known; not fully blind or independent human validation. Save all notes before decoding. Blue prediction, gray GT. No fitting.</p><ol>' + ''.join(links) + '</ol>')


REVIEW_HTML = '''<!doctype html><meta charset="utf-8"><title>__TITLE__</title>
<style>body{font:16px sans-serif;margin:24px}canvas{border:1px solid #ccc;max-width:100%}pre{white-space:pre-wrap}</style>
<h1>__TITLE__</h1><p>__CAPTION__</p><p>Availability: __STATUS__</p><p>Blue: prediction; gray: GT. Original XYZ, no fitting. Ten uniform poses, full sequence and full-frame curves. Missing prediction remains in the denominator. GT 000506 begins already flexed and ends descending; no off-clip start step/final stand is required.</p>
<canvas id="poses" width="1400" height="420"></canvas><p><button id="play">Play / pause (20 Hz)</button> <input id="frame" type="range" min="0" value="0"> <span id="number"></span> <select id="plane"><option value="0">XY</option><option value="2">ZY</option></select></p>
<canvas id="sequence" width="450" height="450"></canvas><p>Curve: <select id="curve"></select></p><canvas id="plot" width="1000" height="350"></canvas><pre id="na"></pre>
<script>const data=__PAYLOAD__;
const chains=[[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]];
const frame=document.getElementById('frame'),plane=document.getElementById('plane'),selector=document.getElementById('curve');frame.max=data.GT.length-1;
function skeleton(ctx,m,t,x,y,w,h){let axis=+plane.value;let all=data.GT.concat(data.prediction||[]).flat();let xs=all.map(v=>v[axis]),ys=all.map(v=>v[1]);let lo=Math.min(...xs),hi=Math.max(...xs),bottom=Math.min(...ys),top=Math.max(...ys);let scale=Math.min((w-12)/Math.max(hi-lo,.01),(h-12)/Math.max(top-bottom,.01));function point(v){return[x+w/2+(v[axis]-(hi+lo)/2)*scale,y+h-6-(v[1]-bottom)*scale]};for(let chain of chains){ctx.beginPath();chain.forEach((j,i)=>{let a=point(m[t][j]);i?ctx.lineTo(...a):ctx.moveTo(...a)});ctx.stroke()}}
function draw(){let t=+frame.value;document.getElementById('number').textContent=`frame ${t} / ${frame.max}`;for(let [id,count] of [['poses',10],['sequence',1]]){let canvas=document.getElementById(id),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);for(let i=0;i<count;i++){let k=count===1?t:Math.round(i*(data.GT.length-1)/9),w=canvas.width/count;ctx.strokeStyle='#999';skeleton(ctx,data.GT,k,i*w,0,w,canvas.height-20);if(data.prediction){ctx.strokeStyle='#1769bb';skeleton(ctx,data.prediction,k,i*w,0,w,canvas.height-20)}ctx.fillText(k,i*w+5,canvas.height-4)}}}
let series={};for(let [k,v] of Object.entries(data.GT_curves)){if(!Array.isArray(v)||k==='frames'||k==='seconds')continue;if(Array.isArray(v[0])){for(let a=0;a<3;a++)series[k+' '+['X','Y','Z'][a]]=[v.map(x=>x[a]),data.curves?data.curves[k].map(x=>x[a]):null]}else series[k]=[v,data.curves?data.curves[k]:null]};for(let k of Object.keys(series)){let option=document.createElement('option');option.textContent=k;selector.append(option)}
function plot(){let c=document.getElementById('plot'),ctx=c.getContext('2d'),ss=series[selector.value],v=ss.flat().filter(x=>x!==null);ctx.clearRect(0,0,c.width,c.height);let lo=Math.min(...v),hi=Math.max(...v);ss.forEach((s,i)=>{if(!s)return;ctx.strokeStyle=i?'#1769bb':'#999';ctx.beginPath();let pen=false;s.forEach((y,t)=>{if(y===null){pen=false;return}let x=40+t/(s.length-1)*940,py=310-(y-lo)/Math.max(hi-lo,1e-9)*280;pen?ctx.lineTo(x,py):ctx.moveTo(x,py);pen=true});ctx.stroke()});ctx.fillText(`min ${lo.toPrecision(5)} max ${hi.toPrecision(5)}; frame 0..${frame.max}`,20,335)}
document.getElementById('na').textContent=JSON.stringify({GT_knee_NA:[data.GT_curves.left_knee_NA_count,data.GT_curves.right_knee_NA_count],prediction_knee_NA:data.curves?[data.curves.left_knee_NA_count,data.curves.right_knee_NA_count]:null});frame.oninput=draw;plane.onchange=draw;selector.onchange=plot;let playing=false;document.getElementById('play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){frame.value=(+frame.value+1)%data.GT.length;draw()}},50);draw();plot();</script>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', type=Path, help='Read saved F run only; default is pending request directory')
    parser.add_argument('--review-notes', type=Path, help='All120 hidden-ID notes saved before key decoding')
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()
    # Guard all evaluator paths, including lazy CPU decoding, against model/GPU use.
    import torch
    import subprocess
    from contextlib import ExitStack
    from unittest.mock import patch
    def forbidden(*args, **kwargs):
        raise RuntimeError('CPU saved-array evaluator forbids model forward and CUDA')
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.nn.Module, '_call_impl', forbidden))
        stack.enter_context(patch.object(subprocess, 'Popen', forbidden))
        for name in ('_lazy_init', 'is_available', 'device_count', 'current_device', 'get_device_properties'):
            stack.enter_context(patch.object(torch.cuda, name, forbidden))
        report = build(args.output, args.run, args.review_notes, not args.no_render)
    print(json.dumps(dict(output=str(args.output), mode=report['mode'], populations=report['populations']), indent=2))


if __name__ == '__main__':
    main()
