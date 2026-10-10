"""Read-only main-session replay of saved CPU preparation evidence; no models."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / 'outputs/itm_rework/protocol/feature_balance_evaluation_preparation_20261007_v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def verify(directory):
    rows = read(directory / 'samples.json')
    old = read(ROOT / 'docs/itm_rework/protocol/b0_sampling_alignment_evaluation_20260928_v1/samples.json')
    old = {r['case_id']: r for r in old['rows']}
    assert rows['synthetic_data'] is False
    assert rows['mode'] == 'CPU_preparation_historical_U_only'
    assert len(rows['rows']) == 120
    assert [(r['planned'], r['valid']) for r in rows['populations']] == [(60, 0), (60, 60)]
    assert all(r['status'] == 'missing' for r in rows['rows'] if r['arm'] == 'F')
    historical = [r for r in rows['rows'] if r['arm'] == 'U']
    assert all(r['metrics'] == old[r['case_id']]['metrics'] for r in historical)
    contrasts = read(directory / 'contrasts.json')
    assert [len(contrasts[k]) for k in ('F_minus_U', 'within', 'difference_of_contrasts')] == [60, 72, 36]
    assert sum(r['status'] == 'valid' for r in contrasts['within']) == 36
    checks = read(directory / 'software_checks.json')
    assert len(checks) == 48 and all(r['passed'] is None for r in checks)
    losses = read(directory / 'training_losses.json')
    assert losses['planned'] == 600 and losses['recorded'] == 0
    assert len(read(directory / 'review_notes_template.json')) == 120
    decision = read(directory / 'decision.json')
    assert len(decision['other_paired_regressions']) == 10
    assert all(r['decision'] == 'inconclusive' for r in decision['primary'].values())
    index = read(directory / 'artifact_index.json')
    assert {str(p) for p in directory.rglob('*') if p.is_file() and p.name != 'artifact_index.json'} == {r['path'] for r in index}
    sources = read(directory / 'sources.json')
    for r in index + sources:
        assert sha(r['path']) == r['sha256'], r['path']
    request = read(ROOT / 'docs/itm_rework/method/B0_FEATURE_BALANCE_AUTHORIZATION_REQUEST_20261005_V2.json')
    assert request['authorized'] is False
    assert not Path(request['run_directory']).exists()
    assert not Path(request['execution_authorization_record']).exists()
    return dict(status='passed', scope='main-session self-check/replay, not independent acceptance',
        historical_U_exact_metric_rows=60, planned_F=60, real_F=0, rows=120,
        between=60, within=72, valid_historical_within=36, difference_of_contrasts=36,
        software_checks=48, passed_software_checks=0, training_losses_planned=600, training_losses_recorded=0,
        review_rows=120, primary_configs=['head', 'wrists'], decision='inconclusive',
        artifact_hashes_verified=len(index), source_hashes_verified=len(sources),
        run_directory_absent=True, actual_authorization_absent=True,
        GPU_queries=0, real_model_forwards=0, checkpoint_tensor_loads=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.directory.resolve())
    content = json.dumps(result, indent=2) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(content)
    print(content)
