"""Single explicitly authorized read-only GPU query, durable output, no retry."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
REQUEST = ROOT / 'docs/itm_rework/method/B0_FEATURE_BALANCE_GPU_DIAGNOSTIC_REQUEST_20261007.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    request = json.loads(REQUEST.read_text())
    auth_path = Path(request['execution_authorization_record'])
    auth = json.loads(auth_path.read_text())
    assert auth['authorized'] is True and auth['approved_by'] == 'user'
    assert auth['authorization_request_sha256'] == sha(REQUEST)
    for key in ('command', 'gpu_uuid', 'limits', 'output_directory', 'policy', 'previous_failure_status'):
        assert auth[key] == request[key]
    assert request['command'] == ['nvidia-smi', '--id=' + request['gpu_uuid'],
        '--query-gpu=uuid,memory.total,memory.free', '--format=csv,noheader,nounits']
    assert request['limits'] == dict(query_commands=1, command_timeout_seconds=10,
        retries=0, polling=0, model_calls=0, training_updates=0, samples=0)
    assert sha(request['previous_failure_status']['path']) == request['previous_failure_status']['sha256']
    output = Path(request['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status='query_started', started_at=timestamp(), argv=request['command'],
        authorization_sha256=sha(auth_path), authorization_request_sha256=sha(REQUEST),
        executor_sha256=sha(__file__), timeout_seconds=10, query_attempts=1,
        additional_queries=0, retries=0, model_calls=0, authorization_consumed=True)
    (output / 'status.json').write_text(json.dumps(record, indent=2) + '\n')
    start = time.monotonic()
    stdout, stderr = b'', b''
    try:
        result = subprocess.run(request['command'], capture_output=True, timeout=10, check=False)
        stdout, stderr = result.stdout, result.stderr
        record.update(status='completed' if result.returncode == 0 else 'query_failed', return_code=result.returncode)
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or b'', error.stderr or b''
        record.update(status='timeout', return_code=None, error=str(error))
    except OSError as error:
        record.update(status='launch_failed', return_code=None, error=str(error))
    record.update(finished_at=timestamp(), elapsed_seconds=time.monotonic()-start)
    # Exact raw bytes preserved, without relying on the terminal rendering.
    (output / 'stdout.txt').write_bytes(stdout)
    (output / 'stderr.txt').write_bytes(stderr)
    rows = [line.split(',') for line in stdout.decode('utf-8', errors='replace').strip().splitlines()]
    parsed = []
    for row in rows:
        item = dict(raw_fields=row, uuid=row[0].strip() if row else None)
        try:
            item.update(total_MiB=int(row[1]), free_MiB=int(row[2]))
        except (ValueError, IndexError) as error:
            item['parse_error'] = str(error)
        parsed.append(item)
    predicates = dict(exactly_one_row=len(rows) == 1,
        uuid_matches=parsed[0]['uuid'] == request['gpu_uuid'] if len(rows) == 1 else None,
        free_MiB_at_least_4096=parsed[0]['free_MiB'] >= 4096 if len(rows) == 1 and 'free_MiB' in parsed[0] else None)
    record.update(row_count=len(rows), parsed_rows=parsed, original_first_query_predicates=predicates,
        first_query_gate_passed=record.get('return_code') == 0 and all(v is True for v in predicates.values()),
        compute_process_check_performed=False, full_GPU_readiness_established=False,
        temporal_scope='This query time only; cannot reconstruct earlier preflight stdout or establish future availability.',
        stdout_sha256=sha(output / 'stdout.txt'), stderr_sha256=sha(output / 'stderr.txt'))
    (output / 'status.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
