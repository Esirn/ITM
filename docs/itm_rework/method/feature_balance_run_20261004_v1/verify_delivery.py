"""Read-only source/ledger verification; no model imports or GPU queries."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/a200/0proj/ITM')
DOC = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


auth_path = DOC.parent / 'B0_FEATURE_BALANCE_AUTHORIZATION_20261004.json'
auth = read(auth_path)
manifest = read(auth['implementation'])
assert sha(auth['implementation']) == auth['implementation_sha256']
assert sha(auth['package']['path']) == auth['package_sha256']
for entry in manifest['expected_artifacts']:
    assert sha(entry['path']) == entry['sha256'], entry['path']
run = Path(auth['run_directory'])
status_path = run / 'runtime/status.json'
assert status_path.read_bytes() == (DOC / 'runtime/status.json').read_bytes()
status = read(status_path)
assert status['authorization'] == auth and auth['authorized'] is True
assert auth['approved_by'] == 'user' and auth['user_instruction_reference']
assert status['implementation']['sha256'] == auth['implementation_sha256']
assert status['status'] == 'failed'
assert status['failure']['type'] == 'CalledProcessError'
assert 'exit status 9' in status['failure']['message']
assert status['completed_receipts'] == [] and status['job_counts'] == {}
assert 'active_job' not in status and 'gpu_preflight' not in status
assert 'counts' not in status  # No worker launched; do not invent raw counters.
assert sorted(p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file()) == ['runtime/status.json']
package_dir = Path(auth['package']['path']).parent
expected = {
    'updates': ('update', list(range(1, 201))),
    'exposures': ('exposure_id', [r['exposure_id'] for r in read(package_dir / 'training_schedule.json')['rows']]),
    'samples': ('case_id', [r['case_id'] for r in read(package_dir / 'sampling_slots.json')]),
    'traces': ('trace_id', [r['trace_id'] for r in read(package_dir / 'trace_slots.json')]),
    'checks': ('check_id', [r['check_id'] for r in read(package_dir / 'software_checks.json')]),
}
for key, (field, ids) in expected.items():
    assert [r[field] for r in status[key]] == ids, key
    assert all(r['status'] == 'not_run_after_stop' for r in status[key]), key
assert [len(status[k]) for k in expected] == [200, 600, 60, 240, 48]
elapsed = status['finished_monotonic'] - status['started_monotonic']
assert 0 <= elapsed < 600
result = dict(
    passed=True, verification='CPU file/hash/ledger audit only',
    elapsed_seconds=elapsed, verified_bound_files=len(manifest['expected_artifacts']),
    ledger_status={key: dict(Counter(r['status'] for r in status[key])) for key in expected},
    worker_launched=False, model_outputs=0,
    model_calls_inferred_from_no_worker=0, raw_worker_counters_present=False,
    status_sha256=sha(status_path), authorization_sha256=sha(auth_path),
    implementation_sha256=sha(auth['implementation']), gpu_queries_during_audit=0,
)
print(json.dumps(result, ensure_ascii=False, indent=2))
