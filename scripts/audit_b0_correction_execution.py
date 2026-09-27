#!/usr/bin/env python
"""02 read-only receipt/cap audit; never queries hardware or invokes a model."""
import json
from pathlib import Path
from evaluate_b0_correction import ROOT,RUN,OUT,FROZEN,ARMS,read,write,sha


def main():
    status=read(RUN/'runtime/status.json');auth=read(ROOT/'docs/itm_rework/method/B0_CORRECTION_AUTHORIZATION_20260927_V2.json')
    assert status['authorization']==auth and auth['authorized'] and auth['run_directory']==str(RUN)
    assert status['status']=='completed' and all(r['status']=='completed' for r in status['stages']+status['probes']+status['samples'])
    assert sha(status['implementation']['path'])==auth['implementation_sha256'] and sha(FROZEN/'package.json')==auth['package_sha256']
    assert status['total_deadline_monotonic']-status['total_started_monotonic']==7200
    assert status['finished_monotonic']<=status['total_deadline_monotonic']
    receipt_paths=sorted((RUN/'runtime').glob('*.receipt.json'));assert len(receipt_paths)==123
    receipts=[];checked={}
    for path in receipt_paths:
        r=read(path);c=r['context'];assert c==read(path.with_name(path.name.replace('.receipt.json','.context.json')))
        assert r['status']=='completed' and r['last_monotonic']<=c['deadline_monotonic']<=status['total_deadline_monotonic']
        assert r['gpu_uuid']==c['gpu_uuid']==auth['gpu_uuid']
        assert r['peak_allocated_bytes']<=c['memory_bytes']==auth['limits']['allocated_memory_bytes']
        assert r['peak_allocated_bytes']<=r['allocator_cap_bytes']<=c['memory_bytes']
        for a in r['artifacts']:
            if a['path'] not in checked:checked[a['path']]=sha(a['path'])
            assert checked[a['path']]==a['sha256']
        for k,limit in c['counters'].items():assert r[k]<=limit
        receipts.append(dict(path=str(path),sha256=sha(path),context_sha256=sha(path.with_name(path.name.replace('.receipt.json','.context.json'))),**{k:r[k] for k in ['phase','status','updates_attempted','updates_completed','denoising_calls','branch_examples','mdm_examples','encoder_examples','exposures','peak_allocated_bytes','started_monotonic','last_monotonic']},arm=c['arm'],completed_slots=len(r['completed_slots'])))
    probes=[r for r in receipts if r['phase']=='probe'];trains=[r for r in receipts if r['phase']=='train'];samples=[r for r in receipts if r['phase']=='sample']
    assert len(probes)==1 and len(trains)==2 and len(samples)==120
    d=probes[0];assert d['mdm_examples']==d['encoder_examples']==d['completed_slots']==480 and d['updates_completed']==0
    for r in trains:
        assert r['updates_attempted']==r['updates_completed']==200 and r['mdm_examples']==r['exposures']==600
        assert r['encoder_examples']==(600 if r['arm']==ARMS[0] else 1200)
        assert r['last_monotonic']<=status['training_deadline_monotonic']
    for r in samples:
        assert r['denoising_calls']==1000 and r['mdm_examples']==r['branch_examples']==4000 and r['completed_slots']==1
        assert r['last_monotonic']<=status['sampling_deadline_monotonic']
    assert len(status['probes'])==480 and len(status['samples'])==120
    assert sum(r['encoder_examples'] for r in samples)==180
    assert all(sum(r['arm']==a for r in samples)==60 for a in ARMS)
    stages={r['stage']:r for r in status['stages']}
    for st in status['stages']:
        if 'receipt' in st:assert sha(st['receipt']['path'])==st['receipt']['sha256']
    summary=dict(read_commit='43af316',runtime_status_sha256=sha(RUN/'runtime/status.json'),authorization_sha256=sha(ROOT/'docs/itm_rework/method/B0_CORRECTION_AUTHORIZATION_20260927_V2.json'),
        package_sha256=auth['package_sha256'],implementation_sha256=auth['implementation_sha256'],
        planned_probes=480,completed_probes=480,planned_samples=120,valid_samples=120,failures=0,
        receipt_count=123,receipts=receipts,elapsed_seconds=status['finished_monotonic']-status['total_started_monotonic'],
        peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in receipts),
        stage_deadlines={k:v['deadline_monotonic'] for k,v in stages.items()},
        all_recorded_caps_pass=True,source='historical receipts only; no fresh GPU query',
        audited_artifact_hashes=[dict(path=p,sha256=h) for p,h in sorted(checked.items())],
        code_sha256=sha(__file__),current_new_model_forwards=0,current_new_updates=0,current_new_samples=0,current_gpu_queries=0)
    assert not (OUT/'execution_audit.json').exists()
    write(OUT/'execution_audit.json',summary)
    print(json.dumps({k:summary[k] for k in ['receipt_count','elapsed_seconds','peak_allocated_bytes','all_recorded_caps_pass']}))

if __name__=='__main__':main()
