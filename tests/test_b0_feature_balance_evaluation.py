"""CPU arithmetic fixtures and historical-U replay; NEVER real F evidence."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from itm.metrics import feature_balance as m

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('f_effect_eval', ROOT / 'scripts/evaluate_b0_feature_balance.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def synthetic_row(value=1., status='valid', case_id='synthetic'):
    return dict(case_id=case_id, status=status, metrics={k: value for k in m.SCALARS},
                feature_views=dict(uniform=value), arm='F', data_split='train', sensor_config='head', condition='paired')


def test_synthetic_partition_boundaries_and_weighted_reconstruction():
    p = np.zeros((4, 263))
    p[:, :67] = 2
    p[:, 67:] = 3
    result = m.feature_views(p, p * 0)
    assert result['root_ric67'] == 4 and result['remaining196'] == 9
    assert result['balanced'] == 6.5
    assert result['uniform'] == (67 * 4 + 196 * 9) / 263
    assert result['ric_legs'] == result['ric_arms'] == result['ric_trunk_head'] == 4
    p[:] = 0
    p[:, 66], p[:, 67] = 1, 2
    result = m.feature_views(p, p * 0)
    assert result['root_ric67'] == 1 / 67 and result['remaining196'] == 4 / 196


@pytest.mark.parametrize('shape', [(263, 4), (0, 263), (1, 263, 1, 4)])
def test_synthetic_feature_wrong_layout_rejected(shape):
    with pytest.raises(ValueError):
        m.feature_views(np.zeros(shape), np.zeros(shape))


def test_synthetic_nonfinite_rejected():
    p = np.zeros((4, 263)); p[0, 0] = np.nan
    with pytest.raises(ValueError): m.feature_views(p, np.zeros_like(p))


def test_synthetic_guardrail_exact_slack_and_strict_boundary():
    reference = synthetic_row(2.)
    new = synthetic_row(2.)
    epsilon = 3e-6
    new['metrics'][m.GUARDRAILS[0]] += epsilon
    new['metrics'][m.GUARDRAILS[1]] -= epsilon
    result = m.gate(new, reference)
    assert result['metrics'][m.GUARDRAILS[0]]['nonworse']
    assert not result['metrics'][m.GUARDRAILS[1]]['strictly_improved']
    new['metrics'][m.GUARDRAILS[1]] -= 1e-9
    assert m.gate(new, reference)['status'] == 'passed'
    new['metrics'][m.GUARDRAILS[0]] += 1e-9
    assert m.gate(new, reference)['status'] == 'failed'


@pytest.mark.parametrize('f_behavior,u_behavior,expected', [
    ('complete', 'incomplete', 'supported_local_only'),
    ('incomplete', 'incomplete', 'not_supported'),
    ('uncertain', 'incomplete', 'inconclusive'),
    ('complete', 'missing', 'inconclusive'),
    ('complete', 'complete', 'not_supported'),
])
def test_synthetic_behavior_cannot_be_replaced_by_lower_error(f_behavior, u_behavior, expected):
    f, u = synthetic_row(1), synthetic_row(2)
    assert m.primary_decision(f, u, u, u, f_behavior, u_behavior)['decision'] == expected


def test_synthetic_own_null_and_missing_gates():
    f, u = synthetic_row(1), synthetic_row(2)
    assert m.primary_decision(f, u, synthetic_row(.5), u, 'complete', 'incomplete')['decision'] == 'not_supported'
    assert m.primary_decision(f, u, synthetic_row(status='missing'), u, 'complete', 'incomplete')['decision'] == 'inconclusive'
    assert m.gate(synthetic_row(status='diagnostic_incomplete'), u)['status'] == 'inconclusive'


def test_synthetic_equal_source_aggregation_retains_missing_and_NA():
    rows = [synthetic_row(1), synthetic_row(3), synthetic_row(status='missing')]
    rows[0]['frames'], rows[1]['frames'] = 4, 400
    for r in rows: r['metrics']['foot_contact_speed_proxy_m_s'] = None
    a = m.aggregate(rows, ('arm',))[0]
    assert (a['planned_n'], a['valid_n']) == (3, 2)
    assert a['metrics']['metrics.global_MPJPE_m']['mean'] == 2
    assert a['metrics']['metrics.foot_contact_speed_proxy_m_s']['valid_n'] == 0
    assert a['metrics']['metrics.foot_contact_speed_proxy_m_s']['mean'] is None
    missing = m.aggregate([rows[-1]], ('arm',))[0]
    assert missing['metrics']['metrics.global_MPJPE_m']['mean'] is None


def test_synthetic_curves_axes_and_degenerate_knee_NA():
    p = np.zeros((4, 22, 3))
    p[:, 1, 1], p[:, 7, 1] = 1, -1
    p[:, 0, 2] = np.arange(4)
    result = m.curves(p)
    assert result['left_knee_degrees'] == [180.] * 4
    assert result['right_knee_degrees'] == [None] * 4 and result['right_knee_NA_count'] == 4
    assert result['pelvis_path_m'] == [0., 1., 2., 3.]
    assert result['left_ankle_relative_xyz_m'][-1] == [0., -1., -3.]


@pytest.mark.parametrize('split,mid,applicable', [('train', '000506', False), ('val', '000139', False), ('val', '000433', True), ('val', '000382', True)])
def test_synthetic_contact_admission_never_expands(split, mid, applicable):
    p = np.zeros((4, 22, 3)); features = np.zeros((4, 263)); features[:, 259:] = 1
    metrics = m.motion_metrics(p, p, 'head', split, mid, features)
    assert (metrics['foot_contact_speed_proxy_m_s'] is not None) == applicable


def test_synthetic_signed_contrast_and_NA():
    a, b = synthetic_row(2, case_id='a'), synthetic_row(5, case_id='b')
    extent = dict(range_xyz=[1, 2, 3], pelvis_path_length_m=2.)
    for r in (a, b): r['metrics']['motion_extent_m'] = extent
    b['metrics']['foot_contact_speed_proxy_m_s'] = None
    result = m.contrast(a, b, {'a': np.zeros((4, 22, 3)), 'b': np.ones((4, 22, 3))})
    assert result['metrics']['global_MPJPE_m'] == -3
    assert result['metrics']['foot_contact_speed_proxy_m_s'] is None
    assert result['max_joint_distance_m'] == pytest.approx(np.sqrt(3))
    assert m.subtract({'x': [-1., 2.]}, {'x': [2., 1.]}) == {'x': [-3., 1.]}


@pytest.fixture(scope='module')
def replay(tmp_path_factory):
    base = tmp_path_factory.mktemp('historical_U_only')
    out = base / 'evaluation'
    report = e.build(out, run=base / 'absent_F', render=False)
    return out, report


def test_historical_U_replay_exact_metrics_and_complete_failure_denominators(replay):
    out, report = replay
    assert [(x['planned'], x['valid']) for x in report['populations']] == [(60, 0), (60, 60)]
    old = {r['case_id']: r for r in e.read(ROOT / 'docs/itm_rework/protocol/b0_sampling_alignment_evaluation_20260928_v1/samples.json')['rows']}
    for row in report['rows']:
        if row['arm'] == 'U': assert row['metrics'] == old[row['case_id']]['metrics']
    contrasts = e.read(out / 'contrasts.json')
    assert [len(contrasts[k]) for k in ('F_minus_U', 'within', 'difference_of_contrasts')] == [60, 72, 36]
    assert sum(r['status'] == 'valid' for r in contrasts['within']) == 36
    assert all(r['status'] == 'not_comparable' for r in contrasts['F_minus_U'])
    assert len(e.read(out / 'training_losses.json')['rows']) == 600
    assert all(r['decision'] == 'inconclusive' for r in e.read(out / 'decision.json')['primary'].values())
    assert len(e.read(out / 'decision.json')['other_paired_regressions']) == 10
    assert all(r['passed'] is None for r in e.read(out / 'software_checks.json'))


def test_synthetic_complete_population_pairings_and_DID(replay):
    _, report = replay
    rows = copy.deepcopy(report['rows'])
    predictions = {}
    for row in rows:
        row.update(status='valid', reason=None, metrics={k: 1. if row['arm'] == 'F' else 3. for k in m.SCALARS}, feature_views={'uniform': 2.})
        if row['condition'] != 'paired': row['metrics'] = {k: 4. if row['arm'] == 'F' else 5. for k in m.SCALARS}
        row['metrics']['motion_extent_m'] = dict(range_xyz=[1., 2., 3.], pelvis_path_length_m=1.)
        predictions[row['case_id']] = np.zeros((row['frames'], 22, 3))
    result = e.comparisons(rows, predictions, e.read(e.FROZEN / 'sampling_slots.json'))
    assert all(r['metrics']['global_MPJPE_m'] == -1 for r in result['difference_of_contrasts'])
    assert len(e.software_checks(rows, predictions)) == 48
    assert all(r['passed'] for r in e.software_checks(rows, predictions))
    # F/U IMU-only are effect comparisons, never software identity checks.
    for row in rows:
        if row['arm'] == 'F' and row['condition'] == 'imu_only': predictions[row['case_id']] += 1
    assert all(r['passed'] for r in e.software_checks(rows, predictions))


@pytest.mark.parametrize('run_name', ['feature_balance_authorized_20261004_v1', 'feature_balance_authorized_20261005_v1'])
def test_old_failed_queues_remain_ineligible_without_worker(run_name):
    status, audit = e.runtime_audit(ROOT / 'outputs/itm_rework/method' / run_name, e.Sources(), e.read(e.FROZEN / 'sampling_slots.json'))
    assert audit['eligible'] is False
    assert audit['ledger_counts']['samples'] == {'not_run_after_stop': 60}
    assert status['completed_receipts'] == []


def test_review_notes_require_all120_evidence_before_decoding(tmp_path):
    order = e.read(e.FROZEN / 'review_order.json')['rows']
    notes = [dict(review_id=r['review_id'], classification='uncertain', evidence='synthetic fixture',
                  side_order_return='synthetic unknown', reason='test only', behavior_losses=[]) for r in order]
    path = tmp_path / 'synthetic_notes.json'
    e.write(path, notes)
    decoded, _ = e.review_notes(path, e.Sources())
    assert len(decoded) == 120
    notes[-1]['evidence'] = ''
    e.write(path, notes)
    with pytest.raises(ValueError, match='evidence'): e.review_notes(path, e.Sources())
    e.write(path, notes[:-1])
    with pytest.raises(ValueError, match='All120'): e.review_notes(path, e.Sources())


def test_synthetic_F_metadata_parser_rejects_relabel_and_wrong_loss(tmp_path):
    """Copied U arrays are parser fixtures only, not generated F or effect evidence."""
    slot = e.read(e.FROZEN / 'sampling_slots.json')[0]
    manifest = e.read(e.read(e.REQUEST)['implementation'])
    with np.load(slot['historical_output']['path'], allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = e.read(slot['historical_metadata']['path'])
    meta.update(case_id=slot['case_id'], implementation_sha256=e.IMPLEMENTATION_SHA,
        package_sha256=e.PACKAGE_SHA, control_checkpoint_sha256='SYNTHETIC_NOT_A_REAL_CHECKPOINT',
        checkpoint_schema='itm-feature-balance-control-v1', loss_version=manifest['loss']['loss_version'],
        loss_weights=manifest['loss']['weights'], loss_weights_sha256=manifest['loss']['weights_sha256'],
        initialization_sha256=manifest['initial']['sha256'], initial_tensor_hashes=manifest['initial_tensor_hashes'],
        training_schedule_sha256=manifest['training_schedule']['sha256'], environment=manifest['environment'],
        training_execution=dict(updates_completed=200, exposures=600, final_update_200=True), training_run_directory=str(tmp_path))
    runtime = dict(checkpoint={'sha256': 'SYNTHETIC_NOT_A_REAL_CHECKPOINT'}, authorization={'run_directory': str(tmp_path)})
    artifacts = e.read(e.FROZEN / 'design.json')['common']['training_input_provenance']['payload']['artifacts']
    mean, std = [np.load(artifacts[k]['path']) for k in ('motion_mean', 'motion_std')]
    path = tmp_path / 'synthetic_parser_fixture.npz'
    def save():
        arrays['metadata'] = np.array(json.dumps(meta))
        np.savez(path, **arrays)
        e.write(path.with_suffix('.json'), meta)
    save()
    prediction, _, _ = e.load_sample(path, slot, 'F', e.Sources(), mean, std, manifest, runtime)
    assert prediction.shape == (41, 22, 3)
    meta['loss_version'] = 'wrong'
    save()
    with pytest.raises(ValueError, match='loss identity'): e.load_sample(path, slot, 'F', e.Sources(), mean, std, manifest, runtime)
    meta['case_id'] = slot['historical_case_id']
    save()
    with pytest.raises(ValueError, match='Case identity'): e.load_sample(path, slot, 'F', e.Sources(), mean, std, manifest, runtime)


def test_render_hidden_labels_and_all_frame_payload(replay, tmp_path):
    _, report = replay
    slots = e.read(e.FROZEN / 'sampling_slots.json')
    targets, predictions = {}, {}
    for s in slots:
        target = np.load(s['gt_path'])
        for cid in (s['case_id'], s['historical_case_id']): targets[cid] = target
        with np.load(s['historical_output']['path']) as z: predictions[s['historical_case_id']] = z['motion'][0]
    e.render_review(tmp_path, report['rows'], predictions, targets, slots)
    pages = list((tmp_path / 'review').glob('FREV*.html'))
    assert len(pages) == 120
    text = pages[0].read_text()
    assert 'F_root_ric_balanced--' not in text and 'S1_R_unit_guidance--' not in text
    assert '__PAYLOAD__' not in text and 'setInterval' in text and 'GT_curves' in text


def test_existing_evidence_cannot_be_overwritten(replay):
    with pytest.raises(ValueError, match='overwrite'): e.build(replay[0])


@pytest.mark.parametrize('f_class,u_class,f_losses,u_losses,expected', [
    ('incomplete', 'incomplete', ['missing_kick_return'], ['missing_kick_return'], False),
    ('incomplete', 'incomplete', ['missing_kick_return', 'new_leg_deformation'], ['missing_kick_return'], True),
    ('incomplete', 'complete', [], [], True),
    ('uncertain', 'incomplete', ['missing_kick_return'], [], None),
])
def test_behavior_regression_is_new_vs_U_not_shared_GT_deficit(replay, f_class, u_class, f_losses, u_losses, expected):
    _, report = replay
    rows = copy.deepcopy(report['rows'])
    pair = [r for r in rows if r['data_split'] == 'val' and r['motion_id'] == '000433'
            and r['sensor_config'] == 'head' and r['condition'] == 'paired']
    notes = {r['case_id']: dict(classification=f_class if r['arm'] == 'F' else u_class,
                               behavior_losses=f_losses if r['arm'] == 'F' else u_losses) for r in pair}
    result = e.decisions(rows, notes, runtime_ok=True)['other_paired_regressions']
    matched = next(r for r in result if r['data_split'] == 'val' and r['motion_id'] == '000433' and r['sensor_config'] == 'head')
    assert matched['behavior_loss'] is expected


def test_corrupt_F_archive_stays_in_failure_denominator(tmp_path, monkeypatch):
    run = tmp_path / 'synthetic_corrupt_run'
    folder = run / 'samples'; folder.mkdir(parents=True)
    slot = e.read(e.FROZEN / 'sampling_slots.json')[0]
    (folder / (slot['case_id'] + '.npz')).write_text('SYNTHETIC CORRUPT FIXTURE')
    original = e.load_sample
    def load(path, slot, arm, *args):
        if arm == 'F': raise e.BadZipFile('synthetic corrupt archive')
        return original(path, slot, arm, *args)
    monkeypatch.setattr(e, 'load_sample', load)
    report = e.build(tmp_path / 'result', run=run, render=False)
    assert report['populations'][0]['statuses'] == {'invalid': 1, 'missing': 59}
    assert report['populations'][0]['failure_or_missing_n'] == 60


def test_synthetic_training_loss_records_keep_all600_and_reject_wrong_divisor(tmp_path):
    row = e.read(e.FROZEN / 'training_schedule.json')['rows'][0]
    path = tmp_path / 'F/exposures' / (row['exposure_id'] + '.json')
    path.parent.mkdir(parents=True)
    record = dict(row, actual_shape=[1, 263, 1, row['frames']], actual_noise_sha256=row['noise']['noise_sha256'],
                  loss=6.5, loss_divided_by_three=6.5/3,
                  loss_components=dict(uniform=(67*4+196*9)/263, root_ric=4., remaining=9., balanced=6.5))
    e.write(path, record)
    result = e.training_losses(tmp_path, e.Sources())
    assert result['recorded'] == 1 and len(result['rows']) == 600
    record['loss_divided_by_three'] = 6.5/2
    e.write(path, record)
    result = e.training_losses(tmp_path, e.Sources())
    assert result['recorded'] == 0 and result['rows'][0]['status'] == 'invalid'


def test_synthetic_successful_runtime_receipts_and_counter_rejection(tmp_path):
    """All files here contain SYNTHETIC markers, no tensors or real approval."""
    slots = e.read(e.FROZEN / 'sampling_slots.json')
    schedule = e.read(e.FROZEN / 'training_schedule.json')['rows']
    traces = e.read(e.FROZEN / 'trace_slots.json')
    checks = e.read(e.FROZEN / 'software_checks.json')
    runtime = tmp_path / 'runtime'; runtime.mkdir()
    def artifact(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists(): path.write_text('SYNTHETIC TEST ONLY - NOT MODEL OUTPUT OR AUTHORIZATION')
        return dict(path=str(path), sha256=e.sha(path))
    checkpoint = artifact(tmp_path / 'F/model.pt')
    request = e.read(e.REQUEST)
    status = dict(status='completed', implementation={'path': request['implementation'], 'sha256': e.IMPLEMENTATION_SHA},
        authorization=dict(authorized=True, approved_by='user', user_instruction_reference='SYNTHETIC TEST ONLY',
                           implementation_sha256=e.IMPLEMENTATION_SHA, package_sha256=e.PACKAGE_SHA,
                           run_directory=str(tmp_path), gpu_uuid='SYNTHETIC_NO_GPU'),
        updates=[dict(update=i, status='completed') for i in range(1, 201)],
        exposures=[dict(exposure_id=r['exposure_id'], status='completed') for r in schedule],
        samples=[dict(case_id=r['case_id'], status='completed') for r in slots],
        traces=[dict(trace_id=r['trace_id'], status='completed') for r in traces],
        checks=[dict(check_id=r['check_id'], status='completed') for r in checks],
        checkpoint=checkpoint, started_monotonic=0, training_deadline_monotonic=600,
        total_deadline_monotonic=3600, sampling_deadline_monotonic=3200, finished_monotonic=3000,
        completed_receipts=[])
    for i in range(61):
        train = i == 0
        ids = [f'U{u:03d}' for u in range(1, 201)] if train else [slots[i-1]['case_id']]
        tids = [] if train else [t['trace_id'] for t in traces if t['case_id'] == ids[0]]
        context = dict(phase='train' if train else 'sample', run_directory=str(tmp_path), gpu_uuid='SYNTHETIC_NO_GPU',
                       implementation_sha256=e.IMPLEMENTATION_SHA, package_sha256=e.PACKAGE_SHA, memory_bytes=24*1024**3,
                       expected_slots=ids, expected_traces=tids, deadline_monotonic=600 if train else 3200)
        if not train: context['checkpoint'] = checkpoint
        paths = ([tmp_path / 'F' / f'update_{u:03d}.json' for u in range(1, 201)] +
                 [tmp_path / 'F/exposures' / (r['exposure_id'] + '.json') for r in schedule] +
                 [tmp_path / 'F' / n for n in ('model.pt', 'update_050.pt', 'update_100.pt', 'update_150.pt')]) if train else (
                 [tmp_path / 'samples' / (ids[0]+ext) for ext in ('.npz', '.json')] +
                 [tmp_path / 'raw_samples' / (ids[0]+'.npz')] +
                 [tmp_path / 'traces' / (tid+ext) for tid in tids for ext in ('.npz', '.json')])
        receipt = dict(status='completed', context=context, completed_slots=ids, completed_traces=tids,
                       updates_attempted=200 if train else 0, updates_completed=200 if train else 0,
                       exposures=600 if train else 0, denoising_calls=0 if train else 1000,
                       branch_examples=0 if train else 4000, peak_allocated_bytes=1,
                       started_monotonic=1 if train else 201, last_monotonic=200 if train else 2900,
                       artifacts=[artifact(p) for p in paths], counts={})
        for kind, cap in dict(mdm=600 if train else 4000, encoder=1200 if train else 2, caption=600 if train else 1).items():
            calls = 1000 if kind == 'mdm' and not train else cap
            receipt['counts'][kind] = dict(reserved=cap, actual=cap, completed=cap, calls=calls, completed_calls=calls)
        rp = runtime / (str(i)+'.receipt.json')
        e.write(rp, receipt); e.write(runtime / (str(i)+'.context.json'), context)
        status['completed_receipts'].append(artifact(rp))
    e.write(runtime / 'status.json', status)
    assert e.runtime_audit(tmp_path, e.Sources(), slots)[1]['eligible']
    bad_path = runtime / '1.receipt.json'
    bad = e.read(bad_path); bad['counts']['mdm']['actual'] -= 4
    e.write(bad_path, bad); status['completed_receipts'][1] = artifact(bad_path)
    e.write(runtime / 'status.json', status)
    result = e.runtime_audit(tmp_path, e.Sources(), slots)[1]
    assert not result['eligible'] and result['errors'] == ['Receipt counters']
