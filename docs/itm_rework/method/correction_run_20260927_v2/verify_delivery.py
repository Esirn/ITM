"""Read-only CPU verification of the single authorized v2 queue; no model calls."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / 'outputs/itm_rework/method/correction_authorized_20260927_v2'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify():
    status = read(RUN / 'runtime/status.json')
    assert status['status'] in ('completed', 'failed'), 'Queue must have stopped'
    assert len(status['probes']) == 480 and len(status['samples']) == 120
    assert status['authorization']['implementation_sha256'] == (
        'a802ef3b03aea7bab7fff3fda5704cf56a2c5cf6c5ed36c693a892fc35d6de13')
    verified = {}

    def verify_artifact(entry):
        path = entry['path']
        if path not in verified:
            assert sha(path) == entry['sha256'], path
            verified[path] = entry['sha256']
        else:
            assert verified[path] == entry['sha256'], path

    verify_artifact(status['implementation'])
    receipts = []
    for path in sorted((RUN / 'runtime').glob('*.receipt.json')):
        receipt = read(path)
        assert receipt['status'] in ('completed', 'failed')
        for entry in receipt['artifacts']:
            verify_artifact(entry)
        receipts.append(dict(path=str(path), sha256=sha(path), **{
            key: receipt[key] for key in ('phase', 'status', 'started_monotonic',
                'last_monotonic', 'updates_attempted', 'updates_completed',
                'denoising_calls', 'branch_examples', 'peak_allocated_bytes',
                'mdm_examples', 'encoder_examples', 'exposures')},
            completed_slots=len(receipt['completed_slots']),
            failure=receipt.get('failure')))
    for row in status['stages'] + status['samples']:
        if 'receipt' in row:
            verify_artifact(row['receipt'])
        if 'checkpoint' in row:
            verify_artifact(row['checkpoint'])

    arrays_checked = 0
    sample_records = []
    for row in status['samples']:
        path = RUN / row['output']
        if not path.exists():
            assert row['status'] != 'completed'
            continue
        meta = read(path.with_suffix('.json'))
        with np.load(path, allow_pickle=False) as arrays:
            assert json.loads(str(arrays['metadata'].item())) == meta
            assert meta['case_id'] == row['case_id']
            assert meta['frame_count'] == row['spec']['frames']
            assert arrays['motion'].shape == (1, meta['frame_count'], 22, 3)
            assert arrays['normalized_humanml263'].shape == (1, 263, 1, meta['frame_count'])
            for key in arrays.files:
                array = arrays[key]
                if np.issubdtype(array.dtype, np.number):
                    assert np.isfinite(array).all(), (path, key)
                    arrays_checked += 1
        arm = row['spec']['arm']
        assert meta['control_checkpoint_sha256'] == sha(RUN / arm / 'model.pt')
        assert meta['model_forward_calls'] == 1000 and meta['MDM_branch_examples'] == 4000
        sample_records.append(dict(case_id=row['case_id'], status=row['status'],
            path=str(path), sha256=sha(path), checkpoint_sha256=meta['control_checkpoint_sha256'],
            frames=meta['frame_count'], sampling_seconds=meta['sampling_seconds'],
            consistency=row.get('consistency')))

    probe_records = []
    for row in status['probes']:
        path = RUN / 'D' / (row['probe_id'] + '.npz')
        if not path.exists():
            assert row['status'] != 'completed'
            continue
        meta = read(path.with_suffix('.json'))
        assert meta['probe_id'] == row['probe_id'] and meta['status'] == 'completed'
        with np.load(path, allow_pickle=False) as arrays:
            for key in arrays.files:
                array = arrays[key]
                if np.issubdtype(array.dtype, np.number):
                    assert np.isfinite(array).all(), (path, key)
                    arrays_checked += 1
        probe_records.append(dict(probe_id=row['probe_id'], status=row['status'],
                                  path=str(path), sha256=sha(path)))

    training = []
    schedule = read(ROOT / 'docs/itm_rework/protocol/b0_correction_20260926_v1/training_schedule.json')['rows']
    for arm in ('L_exact_length', 'R_static_reference_delta'):
        updates = [read(p) for p in sorted((RUN / arm).glob('update_*.json'))]
        for index, update in enumerate(updates, 1):
            assert update['update'] == index and update['optimizer_step_completed']
            assert len(update['examples']) == 3
            assert np.isfinite(update['losses']).all()
            for offset, example in enumerate(update['examples']):
                planned = schedule[(index - 1) * 3 + offset]
                assert all(example[key] == value for key, value in planned.items())
                assert example['actual_noise_sha256'] == planned['noise']['noise_sha256']
                assert example['actual_shape'] == [1, 263, 1, planned['frames']]
        training.append(dict(arm=arm, completed_update_records=len(updates),
            exposures=sum(len(update['examples']) for update in updates),
            final_checkpoint_sha256=sha(RUN / arm / 'model.pt')
                if (RUN / arm / 'model.pt').exists() else None))
    if status['status'] == 'completed':
        assert all(row['status'] == 'completed' for row in status['stages'])
        assert len(probe_records) == 480 and len(sample_records) == 120
        assert all(row['status'] == 'completed' for row in status['probes'] + status['samples'])
        assert all(row['completed_update_records'] == 200 and row['exposures'] == 600
                   for row in training)
        assert sum(row['denoising_calls'] for row in receipts) == 120000
    return dict(version='itm-correction-v2-delivery-integrity-v1', queue_status=status['status'],
        run_directory=str(RUN), elapsed_seconds=status['finished_monotonic'] - status['total_started_monotonic'],
        peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in receipts),
        verified_artifact_count=len(verified), finite_numeric_arrays_checked=arrays_checked,
        probes_by_status=dict(Counter(row['status'] for row in status['probes'])),
        samples_by_status=dict(Counter(row['status'] for row in status['samples'])),
        training=training, receipts=receipts, probes=probe_records, samples=sample_records,
        runtime_manifest=[dict(path=str(path), sha256=sha(path))
                          for path in sorted((RUN / 'runtime').iterdir()) if path.is_file()],
        failure=status.get('failure'))


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
