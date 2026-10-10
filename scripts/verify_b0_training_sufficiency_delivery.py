#!/usr/bin/env python
"""Standard-library verification of the occupancy-stopped first queue."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PACKAGE_SHA='471376c4799b80975c09faee4a80e2adfa220a6119a73a325761ed2a89029593'


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()
def verify(a):
    path=ROOT/a['path'];assert sha(path)==a['sha256'],path
    return path


def validate(manifest_path,run,evaluation):
    manifest_path=Path(manifest_path).resolve();run=Path(run).resolve();evaluation=Path(evaluation).resolve()
    manifest=read(manifest_path)
    assert manifest['package']['sha256']==PACKAGE_SHA
    for a in manifest['expected_artifacts']:verify(a)
    package_path=verify(manifest['package']);package=read(package_path)
    files={Path(a['path']).stem:verify(a) for a in package['artifacts']}
    jobs=read(files['jobs']);states=read(files['states']);checks=read(files['software_checks']);rules=read(files['evaluation'])
    status=read(run/'runtime/status.json')
    assert status['status']=='failed' and status['failure']==dict(type='RuntimeError',message='Selected GPU already has a compute process')
    assert status['deadline_monotonic']==status['started_monotonic']+600
    assert status['implementation']['sha256']==sha(manifest_path)
    execution=status['execution_record'];assert execution['implementation_sha256']==sha(manifest_path)
    acceptance=read(verify(execution['CPU_acceptance']))
    assert acceptance['status']=='passed' and acceptance['incremental_tests_passed']==26
    assert acceptance['implementation_sha256']==sha(manifest_path)
    for a in acceptance['artifacts']:verify(a)
    for a in (execution['policy'],execution['implementation']):verify(a)
    for key,denominator in (('states',192),('jobs',1152),('branches',4608),('checks',2880)):
        assert len(status[key])==denominator
        assert all(r['status']=='not_run_after_stop' for r in status[key])
    assert [r['job_id'] for r in status['jobs']]==[r['job_id'] for r in jobs]
    assert [r['state_id'] for r in status['states']]==[r['state_id'] for r in states]
    assert [r['check_id'] for r in status['checks']]==[r['check_id'] for r in checks]
    assert all(value==0 for count in status['counts'].values() for value in count.values())
    assert 'worker_result' not in status and 'gpu_preflight' not in status
    assert not any((run/'runtime'/name).exists() for name in ('diagnostic.context.json','diagnostic.receipt.json','diagnostic.log'))
    assert not (run/'diagnostic').exists()
    for a in read(evaluation/'sources.json'):verify(a)
    for a in read(evaluation/'artifact_index.json'):verify(a)
    rows=read(evaluation/'branches.json');within=read(evaluation/'within_contrasts.json');progress=read(evaluation/'checkpoint_contrasts.json')
    assert [r['row_id'] for r in rows]==[b for j in jobs for b in j['branch_ids']]
    assert all(r['status']=='missing' and 'temporal' not in r for r in rows)
    expected=[j['job_id']+'-'+a+'--'+j['job_id']+'-'+b for j in jobs for a,b in rules['branch_contrasts']]
    assert [r['row_id'] for r in within]==expected
    expected=[a+'-'+s['state_id']+'-'+branch+'--'+b+'-'+s['state_id']+'-'+branch
              for s in states for a,b in rules['checkpoint_contrasts'] for branch in ('T','P','S','R')]
    assert [r['row_id'] for r in progress]==expected
    assert all(r['status']=='not_comparable' and 'delta' not in r for r in within+progress)
    assert len(rows)==len(within)==len(progress)==4608
    aggregate=read(evaluation/'aggregates.json')
    assert all(r['planned_sources']==3 and r['valid_sources']==0 for items in aggregate.values() for r in items)
    summary=read(evaluation/'summary.json')
    assert summary['decision']=='inconclusive_incomplete_or_failed_software'
    assert summary['branch_statuses']=={'missing':4608}
    return dict(version='itm-sufficiency-occupancy-stop-verification-v1',package_sha256=PACKAGE_SHA,
                implementation_sha256=sha(manifest_path),bound_files_verified=len(manifest['expected_artifacts']),
                CPU_tests_passed=26,planned_states=192,planned_jobs=1152,planned_branches=4608,planned_checks=2880,
                missing_branches_verified=4608,within_not_comparable_verified=4608,progress_not_comparable_verified=4608,
                elapsed_seconds=status['finished_monotonic']-status['started_monotonic'],worker_launched=False,
                real_model_calls_inferred_from_no_worker=0,raw_worker_counters_present=False,allocated_peak_bytes=None,
                preflight_query_attempts_inferred_from_bound_control_flow=2,first_identity_free_memory_gate_passed=True,
                second_compute_process_gate_rejected=True,raw_preflight_outputs_archived=False,
                retries=0,new_training=0,new_samples=0,verification_GPU_queries=0,verification_model_imports=0,
                limitations='Current bound preflight stopped on registered compute process; raw stdout/PID/owner/memory unknown. No model diagnosis or convergence claim. Main-session self-check, not independent review.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('implementation','run-dir','evaluation','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=validate(a.implementation,a.run_dir,a.evaluation)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))
