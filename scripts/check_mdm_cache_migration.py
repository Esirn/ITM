#!/usr/bin/env python
"""Bounded input-only check of owner-02's five diagnostic cache artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

from itm.models.mdm_conditions import load_method_cache

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    output = Path(parser.parse_args().output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = ROOT/'docs/itm_rework/protocol/prefix_rebuild_20260925_v1/diagnostic_manifest.jsonl'
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if [row['motion_id'] for row in records] != ['000123', '000015', '000506', '004728', '000027']:
        raise ValueError('Expected unchanged five-cache diagnostic plan')
    report = {'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(), 'cases': [],
              'budget': {'cache_files': 5, 'attempts_per_cache': 2, 'model_forwards': 0,
                         'training_updates': 0, 'generated_samples': 0, 'gpu_jobs': 0}}
    for record in records:
        path = ROOT/record['imu_path']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record['sha256']:
            raise ValueError(f'Changed diagnostic artifact: {path}')
        result = {'motion_id': record['motion_id'], 'path': str(path), 'sha256': digest}
        for name, candidate in [('original_manifest', record),
                                ('relabelled_manifest', dict(record, schema='standard_imu_v1', training_admissible=True))]:
            try:
                load_method_cache(candidate, workspace_root=ROOT)
            except ValueError as exc:
                result[name] = {'status': 'rejected', 'reason': str(exc)}
            else:
                raise AssertionError(f'Diagnostic cache unexpectedly loaded: {name} {path}')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        report['cases'].append(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print('5 original diagnostic manifests and 5 relabelled manifests rejected; source hashes unchanged')


if __name__ == '__main__':
    main()
