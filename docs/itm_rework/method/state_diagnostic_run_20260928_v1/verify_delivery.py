"""Read-only CPU verification of the stopped V run; never queries a GPU."""
from collections import Counter
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parent
RUN = ROOT / 'outputs/itm_rework/method/state_diagnostic_authorized_20260928_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    status_path = RUN / 'runtime/status.json'
    status = read(status_path)
    approval = read(EVIDENCE.parent / 'B0_STATE_DIAGNOSTIC_AUTHORIZATION_20260928.json')
    assert status['authorization'] == approval
    assert approval['authorized'] and approval['approved_by'] == 'user'
    assert status['status'] == 'failed'
    assert status['failure'] == dict(type='RuntimeError', message='GPU identity/free-memory preflight failed')
    assert 'worker_result' not in status and 'receipt' not in status and 'gpu_preflight' not in status
    assert sorted(str(p.relative_to(RUN)) for p in RUN.rglob('*') if p.is_file()) == ['runtime/status.json']
    elapsed = status['finished_monotonic'] - status['started_monotonic']
    assert 0 <= elapsed < 300
    assert status['deadline_monotonic'] - status['started_monotonic'] == 300
    expected = {
        'groups': [f'V{i:03d}' for i in range(1, 49)],
        'branches': [f'V{i:03d}-{b}' for i in range(1, 49) for b in ('T', 'P', 'S', 'R')],
        'checks': [f'V{i:03d}-{c}' for i in range(1, 49)
                   for c in ('P_vs_trace_f11', 'T_vs_trace_f10', 'S_vs_T')],
    }
    for key, id_key in (('groups', 'group_id'), ('branches', 'slot_id'), ('checks', 'check_id')):
        assert [row[id_key] for row in status[key]] == expected[key]
        assert all(row['status'] == 'not_run_after_stop' for row in status[key])
        if key != 'checks':
            assert all(row['executed'] is False for row in status[key])
    assert set(status['counts']) == {'mdm', 'encoder', 'caption'}
    for count in status['counts'].values():
        assert set(count) == {'reserved', 'actual', 'completed', 'reserved_calls', 'actual_calls', 'completed_calls'}
        assert all(value == 0 for value in count.values())
    manifest_path = Path(approval['implementation'])
    assert sha(manifest_path) == approval['implementation_sha256'] == status['implementation']['sha256']
    manifest = read(manifest_path)
    for entry in manifest['expected_artifacts']:
        assert sha(entry['path']) == entry['sha256'], entry['path']
    for key in ('package', 'checkpoint', 'request'):
        assert sha(approval[key]['path']) == approval[key]['sha256']
    assert (EVIDENCE / 'runtime/status.json').read_bytes() == status_path.read_bytes()
    log = (EVIDENCE / 'logs/parent.log').read_text()
    assert "status['gpu_preflight']=preflight_gpu(gpu)" in log
    assert log.rstrip().endswith('RuntimeError: GPU identity/free-memory preflight failed')
    print(json.dumps(dict(
        passed=True, verification='CPU file/hash/ledger audit only',
        elapsed_seconds=elapsed, verified_bound_files=len(manifest['expected_artifacts']),
        ledger_status={key: dict(Counter(row['status'] for row in status[key])) for key in expected},
        counts=status['counts'], worker_launched=False, model_outputs=0,
        status_sha256=sha(status_path), authorization_sha256=sha(EVIDENCE.parent / 'B0_STATE_DIAGNOSTIC_AUTHORIZATION_20260928.json'),
        gpu_queries_during_audit=0,
    ), indent=2))


if __name__ == '__main__':
    main()
