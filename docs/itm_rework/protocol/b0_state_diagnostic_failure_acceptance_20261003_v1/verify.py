"""Owner02 independent file-only audit of failed V preflight, no model/GPU imports."""
from contextlib import ExitStack, redirect_stdout
from collections import Counter
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
DELIVERY = ROOT/'docs/itm_rework/method/state_diagnostic_run_20260928_v1'
FROZEN = ROOT/'docs/itm_rework/protocol/b0_state_diagnostic_20260928_v1'
RUN = ROOT/'outputs/itm_rework/method/state_diagnostic_authorized_20260928_v1'
SOURCES = {}


def bind(path):
    path = ROOT/Path(path)
    key = str(path)
    if key not in SOURCES:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024*1024), b''):
                h.update(block)
        SOURCES[key] = dict(path=key, sha256=h.hexdigest())
    return SOURCES[key]['sha256']


def read(path):
    bind(path)
    return json.loads((ROOT/Path(path)).read_text())


def main():
    status = read(RUN/'runtime/status.json')
    summary = read(DELIVERY/'execution_summary.json')
    approval = read(ROOT/'docs/itm_rework/method/B0_STATE_DIAGNOSTIC_AUTHORIZATION_20260928.json')
    assert status['authorization'] == approval
    assert approval['authorized'] is True and approval['approved_by'] == 'user'
    assert approval['run_directory'] == str(RUN)
    assert status['status'] == 'failed'
    assert status['failure'] == summary['failure'] == dict(type='RuntimeError', message='GPU identity/free-memory preflight failed')
    elapsed = status['finished_monotonic'] - status['started_monotonic']
    assert elapsed == summary['elapsed_seconds'] and 0 <= elapsed < 300
    assert status['deadline_monotonic'] - status['started_monotonic'] == 300
    for key in ('gpu_preflight', 'receipt', 'worker_result'):
        assert key not in status
    assert [str(p.relative_to(RUN)) for p in RUN.rglob('*') if p.is_file()] == ['runtime/status.json']
    assert (RUN/'runtime/status.json').read_bytes() == (DELIVERY/'runtime/status.json').read_bytes()
    ledger = {}
    for key, filename, idkey, count in [('groups','state_groups','group_id',48), ('branches','branch_slots','slot_id',192), ('checks','software_checks','check_id',144)]:
        frozen = read(FROZEN/(filename+'.json'))
        assert len(status[key]) == len(frozen) == count
        assert [r[idkey] for r in status[key]] == [r[idkey] for r in frozen]
        assert all(r['status'] == 'not_run_after_stop' for r in status[key])
        if key != 'checks':
            assert all(r['executed'] is False for r in status[key])
        ledger[key] = dict(Counter(r['status'] for r in status[key]))
    assert status['counts'] == summary['counts']
    assert set(status['counts']) == {'mdm', 'encoder', 'caption'}
    for values in status['counts'].values():
        assert set(values) == {'reserved', 'actual', 'completed', 'reserved_calls', 'actual_calls', 'completed_calls'}
        assert all(v == 0 for v in values.values())
    expected = dict(implementation='71cb4e2306559295e59ee5c179ce57e971a1a60dcb9f021cf1ddc33c4c45ee46',
        package='6f050dbfbe57c508e86d83a30b3cdfb505f7700b6781cb52b43d85b6ffc0b012',
        checkpoint='91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642')
    for key, value in expected.items():
        assert summary[key]['sha256'] == value == bind(summary[key]['path'])
    assert status['implementation'] == summary['implementation']
    assert approval['implementation_sha256'] == expected['implementation']
    assert approval['package_sha256'] == expected['package']
    request = read(approval['request']['path'])
    assert bind(approval['request']['path']) == approval['request']['sha256']
    for key in ('implementation_sha256','package_sha256','checkpoint','run_directory','gpu_uuid','limits'):
        assert approval[key] == request[key]
    manifest = read(summary['implementation']['path'])
    assert len(manifest['expected_artifacts']) == 1656
    for e in manifest['expected_artifacts']:
        assert bind(e['path']) == e['sha256'], e['path']
    for e in read(DELIVERY/'artifact_index.json'):
        assert bind(e['path']) == e['sha256'], e['path']
    for key in ('authorization','original_status','delivery_integrity'):
        assert bind(summary[key]['path']) == summary[key]['sha256']
    bind(DELIVERY/'logs/parent.log')
    log = (DELIVERY/'logs/parent.log').read_text()
    assert "status['gpu_preflight']=preflight_gpu(gpu)" in log
    assert log.rstrip().endswith('RuntimeError: GPU identity/free-memory preflight failed')
    # Run01's read-only verifier as a secondary check; independent checks above
    # derive IDs from the frozen package rather than reconstructing its ledger.
    capture = io.StringIO()
    with redirect_stdout(capture):
        runpy.run_path(str(DELIVERY/'verify_delivery.py'), run_name='__main__')
    assert json.loads(capture.getvalue()) == read(DELIVERY/'delivery_integrity.json')
    result = dict(status='failure_delivery_accepted_no_model_results',
        read_method_commit='ef659e0', implementation_sha256=expected['implementation'], package_sha256=expected['package'],
        elapsed_seconds=elapsed, bound_files_verified=1656, archived_status_byte_identical=True,
        owner01_verifier_replayed_equal=True, ledger=ledger, counts=status['counts'],
        run_files=['runtime/status.json'], worker_launched=False, real_model_calls=0, model_results=0,
        software_anchors_executed=0, allocated_peak_bytes=None,
        preflight_cause='Unresolved: row count !=1 OR UUID mismatch OR free MiB <4096; no original stdout saved.',
        first_query_reached=True, second_compute_process_query_not_reached=True,
        cause_basis='Bound source code + recorded traceback; query stdout and command return code not independently preserved.',
        authorization_consumed=True, current_audit_GPU_queries=0, current_audit_model_imports=0,
        new_authorization_created=False, retry_performed=False,
        sources=[SOURCES[k] for k in sorted(SOURCES)], audit_code=dict(path=str(Path(__file__)),sha256=bind(__file__)))
    target = OUT/'acceptance.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','counts')}, ensure_ascii=False))


if __name__ == '__main__':
    original_import = builtins.__import__
    def guarded_import(name, *a, **kw):
        if name.split('.')[0] in ('torch','tensorflow','jax'):
            raise AssertionError('Model imports forbidden')
        return original_import(name, *a, **kw)
    def forbidden(*a, **kw):
        raise AssertionError('Subprocess/GPU query forbidden')
    with ExitStack() as stack:
        stack.enter_context(patch.object(builtins, '__import__', guarded_import))
        for name in ('Popen','run','call','check_call','check_output'):
            stack.enter_context(patch.object(subprocess, name, forbidden))
        main()
