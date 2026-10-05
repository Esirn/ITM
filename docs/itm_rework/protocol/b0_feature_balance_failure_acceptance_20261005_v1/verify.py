"""Owner02 file-only acceptance of the failed F preflight; no model/GPU execution."""
from collections import Counter
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch
import builtins
import hashlib
import io
import json
import runpy
import subprocess

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
METHOD = ROOT/'docs/itm_rework/method'
DELIVERY = METHOD/'feature_balance_run_20261005_v1'
FROZEN = ROOT/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1'
RUN = ROOT/'outputs/itm_rework/method/feature_balance_authorized_20261005_v1'
SOURCES = {}


def bind(path):
    path = ROOT/Path(path)
    key = str(path)
    if key not in SOURCES:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block)
        SOURCES[key] = dict(path=key, sha256=h.hexdigest())
    return SOURCES[key]['sha256']


def read(path):
    bind(path)
    return json.loads((ROOT/Path(path)).read_text())


def main():
    target = OUT/'acceptance.json'
    if target.exists():
        raise FileExistsError(target)
    status = read(RUN/'runtime/status.json')
    summary = read(DELIVERY/'execution_summary.json')
    approval = read(METHOD/'B0_FEATURE_BALANCE_AUTHORIZATION_20261005.json')
    request_path = METHOD/'B0_FEATURE_BALANCE_AUTHORIZATION_REQUEST_20261005.json'
    request = read(request_path)
    assert bind(request_path) == approval['authorization_request_sha256']
    for key in ('implementation', 'implementation_sha256', 'package', 'package_sha256',
                'initial', 'loss_version', 'weights_sha256', 'run_directory', 'gpu_uuid', 'limits',
                'accepted_implementation_commit', 'independent_acceptance_commit', 'stop_policy', 'deadlines'):
        assert request[key] == approval[key], key
    assert approval['authorized'] is True and approval['approved_by'] == 'user'
    assert approval['user_instruction_reference'] and status['authorization'] == approval
    assert approval['run_directory'] == str(RUN) == summary['run_directory']
    implementation = 'db323ef515ede3252898f322f7badda10913aea64d67cb4abb1f3e9fc4605e30'
    package = 'a4189b89f1924af022c67dea71c02c2792e65f4a185775501b07d037fba51af9'
    assert bind(approval['implementation']) == approval['implementation_sha256'] == implementation
    assert bind(approval['package']['path']) == approval['package_sha256'] == package
    assert status['implementation'] == dict(path=approval['implementation'], sha256=implementation)
    manifest = read(approval['implementation'])
    assert len(manifest['expected_artifacts']) == 1145
    for row in manifest['expected_artifacts']:
        assert bind(row['path']) == row['sha256'], row['path']
    index = read(DELIVERY/'artifact_index.json')
    assert len(index) == 6
    for row in index:
        assert bind(row['path']) == row['sha256'], row['path']
    assert status['status'] == 'failed' and status['failure'] == summary['failure']
    assert status['failure'] == dict(type='RuntimeError', message='Selected GPU already has a compute process')
    assert summary['parent_exit_code'] == 1
    previous = approval['previous_attempt']
    previous_status = Path(previous['run_directory'])/'runtime/status.json'
    assert bind(previous_status) == previous['status_sha256'] == '1c77c0f0ca4d2cad2b0501857d1326e7263fbb41a696ccc8a34f427d88228845'
    assert previous_status.read_bytes() == (METHOD/'feature_balance_run_20261004_v1/runtime/status.json').read_bytes()
    assert summary['previous_failure_status_unchanged'] is True
    elapsed = status['finished_monotonic']-status['started_monotonic']
    assert elapsed == summary['elapsed_seconds'] and 0 <= elapsed < 600
    assert status['training_deadline_monotonic']-status['started_monotonic'] == 600
    assert status['total_deadline_monotonic']-status['started_monotonic'] == 3600
    assert status['completed_receipts'] == [] and status['job_counts'] == {}
    for key in ('active_job', 'active_receipt', 'active_result', 'gpu_preflight', 'counts', 'checkpoint',
                'sampling_deadline_monotonic'):
        assert key not in status, key
    files = sorted(str(p.relative_to(RUN)) for p in RUN.rglob('*') if p.is_file())
    assert files == ['runtime/status.json']
    assert (RUN/'runtime/status.json').read_bytes() == (DELIVERY/'runtime/status.json').read_bytes()
    expected = dict(updates=('update', list(range(1,201))),
        exposures=('exposure_id', [r['exposure_id'] for r in read(FROZEN/'training_schedule.json')['rows']]),
        samples=('case_id', [r['case_id'] for r in read(FROZEN/'sampling_slots.json')]),
        traces=('trace_id', [r['trace_id'] for r in read(FROZEN/'trace_slots.json')]),
        checks=('check_id', [r['check_id'] for r in read(FROZEN/'software_checks.json')]))
    ledger = {}
    for key, (field, ids) in expected.items():
        assert [r[field] for r in status[key]] == ids
        assert all(r['status'] == 'not_run_after_stop' for r in status[key])
        assert summary['completed_'+key] == 0
        ledger[key] = dict(Counter(r['status'] for r in status[key]))
    assert [len(status[k]) for k in expected] == [200,600,60,240,48]
    assert summary['worker_launched'] is False and summary['model_calls'] == 0
    assert summary['allocated_peak_bytes'] is None and summary['authorization_consumed'] is True
    assert summary['gpu_query_attempts'] == 2 and summary['additional_gpu_queries'] == summary['retries'] == 0
    bind(DELIVERY/'logs/parent.log')
    log = (DELIVERY/'logs/parent.log').read_text()
    assert "status['gpu_preflight']=preflight_gpu(gpu)" in log
    assert "raise RuntimeError('Selected GPU already has a compute process')" in log
    assert log.rstrip().endswith('RuntimeError: Selected GPU already has a compute process')
    # Secondary replay of01's read-only audit, with imports/processes prohibited.
    capture = io.StringIO()
    with redirect_stdout(capture):
        runpy.run_path(str(DELIVERY/'verify_delivery.py'), run_name='__main__')
    assert json.loads(capture.getvalue()) == read(DELIVERY/'delivery_integrity.json')
    result = dict(status='failure_delivery_accepted_no_model_results', read_method_commit='ac10cd6',
        implementation_sha256=implementation, package_sha256=package, elapsed_seconds=elapsed,
        bound_files_verified=1145, delivery_index_files_verified=6, archived_status_byte_identical=True,
        owner01_verifier_replayed_equal=True, ledger=ledger, run_files=files,
        worker_launched=False, model_calls_inferred_from_no_worker=0, raw_worker_counters_present=False,
        model_outputs=0, allocated_peak_bytes=None, preflight_queries_inferred_from_control_flow=2,
        preflight_cause='Selected GPU matched a compute-process row; PID/owner/memory unknown without raw stdout.',
        cause_basis='Bound source and saved traceback; first query UUID/free-memory checks passed, second query matched selected UUID.',
        second_compute_process_query_reached=True, previous_failure_status_unchanged=True, authorization_consumed=True,
        current_audit_GPU_queries=0, current_audit_model_imports=0,
        new_authorization_created=False, retry_performed=False,
        audit_code=dict(path=str(Path(__file__)),sha256=bind(__file__)),
        sources=[SOURCES[k] for k in sorted(SOURCES)])
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'sources'}, ensure_ascii=False))


if __name__ == '__main__':
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'tensorflow', 'jax'):
            raise AssertionError('Model imports forbidden')
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('Subprocess/GPU query forbidden')

    with ExitStack() as stack:
        stack.enter_context(patch.object(builtins, '__import__', guarded_import))
        for name in ('Popen', 'run', 'call', 'check_call', 'check_output'):
            stack.enter_context(patch.object(subprocess, name, forbidden))
        main()
