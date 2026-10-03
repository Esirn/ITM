"""CPU-only V delivery audit; no torch imports or GPU queries."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parent
RUN = ROOT / 'outputs/itm_rework/method/state_diagnostic_authorized_20261003_v1'
FREEZE = ROOT / 'docs/itm_rework/protocol/b0_state_diagnostic_20260928_v1'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def verify(a):
    assert sha(a['path']) == a['sha256'], a['path']


def desc(a):
    return dict(shape=list(a.shape), dtype=str(a.dtype),
                sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())


def main():
    s = read(RUN / 'runtime/status.json')
    c = read(RUN / 'runtime/V.receipt.json')
    a = read(EVIDENCE.parent / 'B0_STATE_DIAGNOSTIC_AUTHORIZATION_20261003.json')
    assert s['status'] == c['status'] == 'completed'
    assert s['authorization'] == a and a['authorized'] and a['approved_by'] == 'user'
    assert a['run_directory'] == str(RUN)
    assert s['worker_result']['returncode'] == 0 and s['worker_result']['launched']
    elapsed = s['finished_monotonic'] - s['started_monotonic']
    assert 0 < elapsed < 300
    assert s['deadline_monotonic'] - s['started_monotonic'] == 300
    assert 0 < c['peak_allocated_bytes'] <= c['allocator_cap_bytes'] <= 24 * 1024**3
    assert s['gpu_preflight']['gpu_uuid'] == c['gpu_uuid'] == a['gpu_uuid']
    assert s['gpu_preflight']['compute_processes'] == 0
    assert c['context'] == read(RUN / 'runtime/V.context.json')
    verify(s['receipt']); verify(s['implementation'])
    assert s['implementation']['sha256'] == a['implementation_sha256']
    for k in ('package', 'checkpoint', 'request'):
        verify(a[k])
    m = read(a['implementation'])
    for entry in m['expected_artifacts']:
        verify(entry)
    assert c['updates_attempted'] == c['updates_completed'] == c['denoising_calls'] == 0
    for k, (examples, calls) in dict(mdm=(192, 48), encoder=(72, 72), caption=(12, 12)).items():
        assert s['counts'][k] == c['counts'][k]
        for stage in ('reserved', 'actual', 'completed'):
            assert c['counts'][k][stage] == examples
            assert c['counts'][k][stage + '_calls'] == calls
    for k, ids, id_key in (
        ('groups', [f'V{i:03d}' for i in range(1, 49)], 'group_id'),
        ('branches', [f'V{i:03d}-{b}' for i in range(1, 49) for b in ('T', 'P', 'S', 'R')], 'slot_id'),
        ('checks', [f'V{i:03d}-{b}' for i in range(1, 49)
                    for b in ('P_vs_trace_f11', 'T_vs_trace_f10', 'S_vs_T')], 'check_id'),
    ):
        assert s[k] == c[k] and [r[id_key] for r in c[k]] == ids
        assert all(r['status'] == 'completed' for r in c[k])
        if k != 'checks':
            assert all(r['executed'] and not r['diagnostic_only'] for r in c[k])
    assert len(c['artifacts']) == 684
    paths = {entry['path'] for entry in c['artifacts']}
    assert len(paths) == 684
    assert paths == {str(p) for p in (RUN / 'V').rglob('*') if p.is_file()}
    for entry in c['artifacts']:
        verify(entry)
    arrays = elements = npzs = 0
    for p in (RUN / 'V').rglob('*.npz'):
        npzs += 1
        with np.load(p, allow_pickle=False) as z:
            for key in z.files:
                v = z[key]
                if v.dtype.kind in 'biufc':
                    assert np.isfinite(v).all(), (p, key)
                    arrays += 1; elements += v.size
    for case in m['inputs']:
        for branch, expected in zip(('T', 'P', 'S', 'R'), case['branches']):
            with np.load(RUN / 'V/inputs' / (case['cache_id'] + '-' + branch + '.npz')) as z:
                assert {k: desc(z[k]) for k in expected} == expected
    checks = []
    by_id = {r['check_id']: r for r in c['checks']}
    for g in read(FREEZE / 'state_groups.json'):
        gid = g['group_id']
        with np.load(RUN / 'V/states' / (gid + '.npz')) as z, np.load(g['saved_state']['path']) as old:
            assert np.array_equal(z['x_t'], old['x_t'])
            assert np.array_equal(z['frame_mask'], old['frame_mask'])
            with np.load(g['target_artifact']['path']) as target:
                assert np.array_equal(z['target'], target[g['target_array']])
            predictions = {}
            for b in ('T', 'P', 'S', 'R'):
                p = RUN / 'V/branches' / (gid + '-' + b + '.npz')
                with np.load(p) as branch:
                    pred = branch['predicted_x0'].copy()
                assert pred.shape == z['target'].shape and pred.dtype == np.float32
                meta = read(p.with_suffix('.json'))
                assert meta['arrays']['predicted_x0'] == desc(pred)
                assert meta['checkpoint_sha256'] == a['checkpoint']['sha256']
                assert meta['implementation_sha256'] == a['implementation_sha256']
                assert meta['status'] == 'raw_saved_pending_anchors'
                assert read(meta['final_group_metadata'])['status'] == 'completed'
                predictions[b] = pred[None]
            for suffix, x, y in (
                ('P_vs_trace_f11', predictions['P'], old['f11']),
                ('T_vs_trace_f10', predictions['T'], old['f10']),
                ('S_vs_T', predictions['S'], predictions['T']),
            ):
                report = read(RUN / 'V/checks' / (gid + '-' + suffix + '.json'))
                assert report['atol'] == report['rtol'] == 1e-6
                d = x.astype(np.float64) - y.astype(np.float64)
                result = dict(max_abs=float(np.abs(d).max()), rms=float(np.sqrt(np.mean(d*d))),
                              bitwise=desc(x) == desc(y), count=d.size, finite_count=d.size,
                              passed=bool(np.all(np.abs(d) <= 1e-6 + 1e-6*np.abs(y.astype(np.float64)))))
                assert result['passed']
                for key, value in result.items():
                    assert report[key] == by_id[report['check_id']][key] == value
                checks.append(result)
    for p in (RUN / 'runtime').iterdir():
        assert p.read_bytes() == (EVIDENCE / 'runtime' / p.name).read_bytes()
    old = ROOT / 'outputs/itm_rework/method/state_diagnostic_authorized_20260928_v1/runtime/status.json'
    assert sha(old) == a['previous_attempt']['status_sha256']
    print(json.dumps(dict(passed=True, elapsed_seconds=elapsed, verified_bound_files=len(m['expected_artifacts']),
        receipt_artifacts=len(paths), run_files=sum(p.is_file() for p in RUN.rglob('*')),
        numeric_npz_files=npzs, finite_arrays=arrays, finite_elements=elements,
        groups=48, branches=192, inputs_checked=48, states_checked=48,
        software_checks=len(checks), checks_bitwise=sum(v['bitwise'] for v in checks),
        checks_max_abs=max(v['max_abs'] for v in checks), checks_max_rms=max(v['rms'] for v in checks),
        peak_allocated_bytes=c['peak_allocated_bytes'], counts=c['counts'],
        old_failure_unchanged=True, gpu_queries_during_audit=0, model_calls_during_audit=0), indent=2))


if __name__ == '__main__':
    main()
