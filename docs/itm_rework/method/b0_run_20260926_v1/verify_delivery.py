"""CPU-only integrity audit of the single authorized B0 delivery; no generation."""
import collections
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / 'outputs/itm_rework/method/canonical_b0_authorized_20260926_v1'
OUT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    status = read(RUN / 'runtime/status.json')
    assert status['status'] in ('completed', 'failed'), 'Run is still active'
    authorization = read(ROOT / 'docs/itm_rework/method/B0_AUTHORIZATION_20260926.json')
    assert status['authorization'] == authorization
    assert str(RUN) == authorization['run_directory']
    receipts = [read(p) for p in sorted((RUN / 'runtime').glob('*.receipt.json'))]
    rows, noise_by_group = [], {}
    final = RUN / 'model.pt'
    final_sha = sha(final) if final.exists() else None
    checkpoint_provenance_verified = False
    if final.exists():
        import torch
        state = torch.load(final, map_location='cpu', weights_only=True)
        expected = read(ROOT / 'docs/itm_rework/method/canonical_training_input_20260925_v1.json')
        assert state['training_input_provenance'] == expected['training_input_provenance']
        assert state['execution']['final_update_200'] is True
        checkpoint_provenance_verified = True
    for slot in status['slots']:
        row = {k: slot[k] for k in ('case_id', 'status', 'launched', 'reason') if k in slot}
        if slot.get('receipt'):
            assert sha(slot['receipt']['path']) == slot['receipt']['sha256']
        if slot['status'] == 'completed':
            receipt = read(slot['receipt']['path'])
            assert receipt['status'] == 'completed'
            assert receipt['denoising_calls'] == 1000 and receipt['branch_examples'] == 4000
            assert receipt['gpu_uuid'] == authorization['gpu_uuid']
            for artifact in slot['artifacts']:
                assert sha(artifact['path']) == artifact['sha256']
            meta = read(Path(slot['output']).with_suffix('.json'))
            assert meta['control_checkpoint_sha256'] == final_sha
            assert meta['steps'] == 1000
            case = meta['cases'][0]
            assert case['case_id'] == slot['case_id']
            plan = meta['random_plans'][0]
            group = case['random_plan_case_id']
            if group in noise_by_group:
                assert plan == noise_by_group[group], group
            else:
                noise_by_group[group] = plan
            with np.load(slot['output'], allow_pickle=False) as arrays:
                assert json.loads(str(arrays['metadata'])) == meta
                for key in arrays.files:
                    value = arrays[key]
                    if np.issubdtype(value.dtype, np.number):
                        assert np.isfinite(value).all(), (slot['case_id'], key)
                assert arrays['motion'].shape == (1, case['length'], 22, 3)
                row['array_shapes'] = {k: list(arrays[k].shape) for k in ('motion', 'normalized_humanml263')}
            row.update(frame_count=case['length'], condition=case['condition'], sensor_config=case['sensor_config'],
                       random_plan_case_id=group, random_plan_id=plan['random_plan_id'],
                       sampling_seconds=meta['sampling_seconds'], peak_allocated_bytes=receipt['peak_allocated_bytes'],
                       artifacts=slot['artifacts'])
        rows.append(row)
    train_path = RUN / 'runtime/train.receipt.json'
    train = read(train_path) if train_path.exists() else None
    if status['status'] == 'completed':
        assert train['updates_completed'] == train['updates_attempted'] == 200
        assert len(rows) == 36 and all(r['status'] == 'completed' for r in rows)
        assert status['denoising_calls'] == 36000 and status['branch_examples'] == 144000
        assert len(noise_by_group) == 6
        assert next(a['sha256'] for a in train['artifacts'] if a['path'] == str(final)) == final_sha
    assert all(r['peak_allocated_bytes'] <= 24 * 1024 ** 3 for r in receipts)
    assert all(r.get('gpu_uuid', authorization['gpu_uuid']) == authorization['gpu_uuid'] for r in receipts)
    end = max((r['last_monotonic'] for r in receipts), default=None)
    start = status.get('total_deadline_monotonic', 14400) - 14400
    sample_start = status.get('sampling_deadline_monotonic', 10800) - 10800
    report = dict(version='method-B0-delivery-integrity-v1', scope='Integrity only; independent effectiveness evaluation belongs to 02.',
                  run_directory=str(RUN), status=status['status'], gpu_preflight=status.get('gpu_preflight'),
                  slot_counts=dict(collections.Counter(r['status'] for r in rows)),
                  updates_completed=train['updates_completed'] if train else 0,
                  denoising_calls=status['denoising_calls'], branch_examples=status['branch_examples'],
                  peak_allocated_bytes=max((r['peak_allocated_bytes'] for r in receipts), default=0),
                  train_start_to_last_receipt_seconds=train['last_monotonic']-start if train else None,
                  sampling_phase_to_last_receipt_seconds=end-sample_start if 'sampling_deadline_monotonic' in status else None,
                  total_start_to_last_receipt_seconds=end-start if end else None,
                  timing_note='Last persisted child receipt; parent verification/exit may add small overhead. Absolute deadlines enforced by accepted runner.',
                  checkpoint=dict(path=str(final), sha256=final_sha),
                  checkpoint_provenance_matches_accepted_input=checkpoint_provenance_verified,
                  six_condition_noise_plans_equal_within_groups=list(noise_by_group), slots=rows)
    (OUT / 'delivery_integrity.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    snapshot = OUT / 'runtime'
    snapshot.mkdir(exist_ok=True)
    manifest=[]
    for p in sorted((RUN / 'runtime').iterdir()):
        if p.is_file():
            target = snapshot / p.name
            if target.exists():
                assert sha(target) == sha(p), target
            else:
                shutil.copy2(p, target)
            manifest.append(dict(source=str(p), snapshot=str((snapshot/p.name).relative_to(ROOT)), sha256=sha(p)))
    (OUT / 'runtime_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'slots'}, indent=2))


if __name__ == '__main__':
    main()
