"""File/hash-only self-check of the 2026-10-07 failed F attempt. No GPU calls."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'outputs/itm_rework/method/feature_balance_authorized_20261005_v2'
FROZEN = ROOT / 'docs/itm_rework/protocol/b0_feature_balance_20261003_v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    status = read(RUN / 'runtime/status.json')
    request_path = ROOT / 'docs/itm_rework/method/B0_FEATURE_BALANCE_AUTHORIZATION_REQUEST_20261005_V2.json'
    request = read(request_path)
    approval = read(request['execution_authorization_record'])
    assert approval['authorized'] is True and approval['approved_by'] == 'user'
    assert approval['authorization_request_sha256'] == sha(request_path)
    assert status['authorization'] == approval
    for k in ('implementation', 'implementation_sha256', 'package', 'package_sha256', 'initial',
              'loss_version', 'weights_sha256', 'run_directory', 'gpu_uuid', 'limits', 'stop_policy', 'deadlines'):
        assert approval[k] == request[k], k
    assert sha(approval['implementation']) == approval['implementation_sha256']
    manifest = read(approval['implementation'])
    for entry in manifest['expected_artifacts']:
        assert sha(ROOT / entry['path']) == entry['sha256'], entry['path']
    for previous in [request['previous_attempt'], *request['earlier_attempts']]:
        assert sha(Path(previous['run_directory']) / 'runtime/status.json') == previous['status_sha256']
    assert status['status'] == 'failed'
    assert status['failure'] == dict(type='RuntimeError', message='GPU identity/free-memory preflight failed')
    assert status['completed_receipts'] == [] and status['job_counts'] == {}
    assert all(k not in status for k in ('active_job', 'active_receipt', 'active_result', 'counts', 'checkpoint', 'gpu_preflight'))
    assert sorted(str(p.relative_to(RUN)) for p in RUN.rglob('*') if p.is_file()) == ['runtime/status.json']
    assert (RUN / 'runtime/status.json').read_bytes() == (OUT / 'runtime/status.json').read_bytes()
    expected = dict(updates=('update', list(range(1, 201))),
        exposures=('exposure_id', [r['exposure_id'] for r in read(FROZEN / 'training_schedule.json')['rows']]),
        samples=('case_id', [r['case_id'] for r in read(FROZEN / 'sampling_slots.json')]),
        traces=('trace_id', [r['trace_id'] for r in read(FROZEN / 'trace_slots.json')]),
        checks=('check_id', [r['check_id'] for r in read(FROZEN / 'software_checks.json')]))
    for k, (field, ids) in expected.items():
        assert [r[field] for r in status[k]] == ids
        assert all(r['status'] == 'not_run_after_stop' for r in status[k])
    summary = read(OUT / 'execution_summary.json')
    assert summary['elapsed_seconds'] == status['finished_monotonic'] - status['started_monotonic']
    assert status['training_deadline_monotonic'] == status['started_monotonic'] + 600
    assert status['total_deadline_monotonic'] == status['started_monotonic'] + 3600
    assert summary['parent_exit_code'] == 1 and summary['worker_launched'] is False
    log = (OUT / 'execute.log').read_text()
    assert "raise RuntimeError('GPU identity/free-memory preflight failed')" in log
    assert log.rstrip().endswith('RuntimeError: GPU identity/free-memory preflight failed')
    evaluation = ROOT / 'outputs/itm_rework/protocol/feature_balance_failed_evaluation_20261007_v1'
    samples = read(evaluation / 'samples.json')
    assert [(r['planned'], r['valid']) for r in samples['populations']] == [(60, 0), (60, 60)]
    assert all(r['decision'] == 'inconclusive' for r in read(evaluation / 'decision.json')['primary'].values())
    assert read(evaluation / 'runtime_audit.json')['eligible'] is False
    for entry in read(evaluation / 'artifact_index.json') + read(evaluation / 'sources.json'):
        assert sha(entry['path']) == entry['sha256'], entry['path']
    return dict(status='passed', scope='main-session self-check, not independent acceptance',
        status_sha256=sha(RUN / 'runtime/status.json'), archive_byte_identical=True,
        bound_files_verified=len(manifest['expected_artifacts']), previous_failures_unchanged=True,
        planned_denominators={k: len(status[k]) for k in expected}, all_not_run_after_stop=True,
        elapsed_seconds=summary['elapsed_seconds'], worker_launched=False, allocated_peak_bytes=None,
        model_calls_inferred_from_no_worker=0, raw_worker_counters_present=False, authorization_consumed=True,
        F_valid=0, U_valid=60, evaluation_decision='inconclusive',
        current_verifier_GPU_queries=0, current_verifier_model_imports=0)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
