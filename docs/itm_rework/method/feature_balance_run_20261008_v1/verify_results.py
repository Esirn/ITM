"""Main-session saved-array/hash replay; no model imports, GPU or subprocesses."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT/'outputs/itm_rework/method/feature_balance_authorized_20261008_v1'
OUT = Path(__file__).resolve().parent
EVAL = ROOT/'outputs/itm_rework/protocol/feature_balance_evaluation_20261008_v2'
FROZEN = ROOT/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1'
KEYS = ['global_MPJPE_m', 'root_relative_MPJPE_m', 'root_path_RMSE_m',
        'head_or_wrists_position_joint_proxy_m', 'velocity_joint_proxy_RMSE_m_s']


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    status = read(RUN/'runtime/status.json')
    assert status['status'] == 'completed'
    assert (RUN/'runtime/status.json').read_bytes() == (OUT/'runtime/status.json').read_bytes()
    auth = read(ROOT/'docs/itm_rework/method/B0_FEATURE_BALANCE_AUTHORIZATION_20261008.json')
    assert auth == status['authorization'] and auth['authorized'] is True and auth['approved_by'] == 'user'
    assert sha(auth['GPU_delegation']['path']) == auth['GPU_delegation']['sha256']
    assert auth['run_directory'] == str(RUN) and auth['gpu_uuid'] == status['gpu_preflight']['gpu_uuid']
    assert status['gpu_preflight']['compute_processes'] == 0 and status['gpu_preflight']['free_MiB'] >= 4096
    manifest = read(auth['implementation'])
    assert sha(auth['implementation']) == auth['implementation_sha256']
    for a in manifest['expected_artifacts']:
        assert sha(ROOT/a['path']) == a['sha256'], a['path']
    for old in [auth['previous_attempt'], *auth['earlier_attempts']]:
        assert sha(Path(old['run_directory'])/'runtime/status.json') == old['status_sha256']
    receipts = [read(a['path']) for a in status['completed_receipts']]
    assert len(receipts) == 61
    assert status['counts'] == {
        kind: dict(reserved=n, actual=n, completed=n, calls=calls, completed_calls=calls)
        for kind, n, calls in [('mdm', 240600, 60600), ('encoder', 1320, 1320), ('caption', 660, 660)]}
    assert max(a['peak_allocated_bytes'] for a in receipts) <= 24*1024**3
    assert status['finished_monotonic'] <= status['total_deadline_monotonic']
    assert receipts[0]['last_monotonic'] <= status['training_deadline_monotonic']
    assert all(a['last_monotonic'] <= status['sampling_deadline_monotonic'] for a in receipts[1:])
    report = read(EVAL/'samples.json')
    assert [(r['planned'], r['valid']) for r in report['populations']] == [(60, 60), (60, 60)]
    assert read(EVAL/'runtime_audit.json')['eligible'] is True
    assert read(EVAL/'training_losses.json')['recorded'] == 600
    assert all(r['passed'] for r in read(EVAL/'software_checks.json'))
    trace = read(ROOT/'outputs/itm_rework/protocol/feature_balance_trace_audit_20261008_v1/traces.json')
    assert trace['planned'] == trace['valid'] == 240 and trace['extra_model_calls'] == 0
    slots = read(FROZEN/'sampling_slots.json')
    plans = {cid: s for s in slots for cid in (s['case_id'], s['historical_case_id'])}
    recomputed = {}
    for row in report['rows']:
        s = plans[row['case_id']]
        path = RUN/'samples'/(row['case_id']+'.npz') if row['arm'] == 'F' else Path(s['historical_output']['path'])
        with np.load(path, allow_pickle=False) as z:
            pred = z['motion'][0].astype(np.float64)
        gt = np.load(s['gt_path'], allow_pickle=False).astype(np.float64)
        error = pred-gt
        sensor = [15] if row['sensor_config'] == 'head' else [20, 21]
        values = [np.linalg.norm(error, axis=-1).mean(),
                  np.linalg.norm(error-error[:, :1], axis=-1).mean(),
                  np.sqrt(np.square(error[:, 0]).sum(-1).mean()),
                  np.linalg.norm(error[:, sensor], axis=-1).mean(),
                  np.sqrt(np.square(np.diff(error, axis=0)*20).sum(-1).mean())]
        np.testing.assert_allclose(values, [row['metrics'][k] for k in KEYS], atol=1e-12, rtol=1e-12)
        recomputed[row['case_id']] = dict(zip(KEYS, values))
        admitted = row['data_split'] == 'val' and row['motion_id'] in ('000433', '000382')
        if not admitted:
            assert row['metrics']['foot_contact_speed_proxy_m_s'] is None
    contrasts = read(EVAL/'contrasts.json')
    assert [len(contrasts[k]) for k in ('F_minus_U', 'within', 'difference_of_contrasts')] == [60, 72, 36]
    for key in ('F_minus_U', 'within'):
        for c in contrasts[key]:
            assert c['status'] == 'valid'
            a, b = recomputed[c['case_id']], recomputed[c['reference_case_id']]
            np.testing.assert_allclose([c['metrics'][k] for k in KEYS], [a[k]-b[k] for k in KEYS], atol=1e-12, rtol=1e-12)
    wi = {(r['arm'], r['data_split'], r['motion_id'], r['sensor_config'], r['comparator']): r for r in contrasts['within']}
    for c in contrasts['difference_of_contrasts']:
        key = (c['data_split'], c['motion_id'], c['sensor_config'], c['comparator'])
        a, b = wi[('F', *key)], wi[('U', *key)]
        np.testing.assert_allclose([c['metrics'][k] for k in KEYS], [a['metrics'][k]-b['metrics'][k] for k in KEYS], atol=1e-12, rtol=1e-12)
    frozen = read(OUT/'review_freeze_before_decode.json')
    assert sha(OUT/'review_notes_hidden_final.json') == frozen['notes_sha256']
    notes = read(OUT/'review_notes_hidden_final.json')
    assert len(notes) == 120 and all('case_id' not in r and 'arm' not in r for r in notes)
    assert read(EVAL/'submitted_review_notes.json') == notes
    decisions = read(EVAL/'decision.json')
    assert all(d['decision'] == 'not_supported' and d['F_behavior'] == 'incomplete' and d['U_behavior'] == 'incomplete'
               for d in decisions['primary'].values())
    for d in decisions['primary'].values():
        for g in d['numerical'].values():
            assert g['status'] == 'passed'
            for key, v in g['metrics'].items():
                epsilon = 1e-6+1e-6*abs(v['reference'])
                assert v['epsilon'] == epsilon and v['new'] <= v['reference']+epsilon
                if key == 'root_relative_MPJPE_m':
                    assert v['new'] < v['reference']-epsilon
    numerical_regressions = sum(bool(r['numerical_exceedances']) for r in decisions['other_paired_regressions'])
    assert numerical_regressions == 9
    for r in decisions['other_paired_regressions']:
        expected = [k for k, v in r['metrics'].items() if v['new'] > v['reference']+1e-6+1e-6*abs(v['reference'])]
        assert r['numerical_exceedances'] == expected
    for directory in [EVAL, ROOT/'outputs/itm_rework/protocol/feature_balance_evaluation_20261008_v1']:
        for entry in read(directory/'artifact_index.json') + read(directory/'sources.json'):
            assert sha(entry['path']) == entry['sha256'], entry['path']
    for entry in trace['sources']:
        assert sha(entry['path']) == entry['sha256']
    images = read(ROOT/'outputs/itm_rework/protocol/feature_balance_review_images_20261008_v1/rendering.json')
    for entry in images['images']+images['sources']:
        assert sha(entry['path']) == entry['sha256']
    return dict(status='passed', scope='main-session self-check/replay, not independent 02 acceptance',
        real_F=60, historical_U=60, primary_metrics_recomputed=120*5, contrasts_recomputed=60+72+36,
        review_rows=120, software_checks_passed=48, traces_valid=240, training_losses_valid=600,
        primary_head='not_supported', primary_wrists='not_supported', primary_numeric_gates_all_passed=True,
        other_paired_numerical_regressions=numerical_regressions, other_paired_denominator=10,
        bound_runtime_files_verified=len(manifest['expected_artifacts']), old_failures_unchanged=True,
        GPU_queries=0, model_imports=0, model_calls=0, standing_GPU_delegation_active=True)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
