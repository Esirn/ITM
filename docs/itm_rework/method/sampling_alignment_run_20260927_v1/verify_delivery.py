"""CPU-only identity/finite audit of the single completed-or-stopped queue; no model calls."""
from pathlib import Path
import json
from collections import Counter
import numpy as np
from itm.models.mdm_correction import artifact,verify,array_sha
from itm.models.mdm_sampling_alignment import describe,PACKAGE_SHA,CHECKPOINT_SHA,GUIDANCE

ROOT=Path(__file__).resolve().parents[4]
RUN=ROOT/'outputs/itm_rework/method/sampling_alignment_authorized_20260927_v1'
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
s=read(RUN/'runtime/status.json')
assert s['status'] in ('completed','failed')
assert [len(s[k]) for k in ('probes','samples','traces')]==[480,60,240]
assert len(s['consistency_checks'])==24
receipts=[read(p) for p in sorted((RUN/'runtime').glob('*.receipt.json'))]
verified={}
for receipt in receipts:
    assert receipt['checkpoint_sha256']==CHECKPOINT_SHA
    assert receipt['updates_attempted']==receipt['updates_completed']==0
    assert receipt['peak_allocated_bytes']<=24*1024**3
    for entry in receipt['artifacts']:
        verify(entry);verified[entry['path']]=entry
counts=dict(array_files=0,numeric_arrays=0,numeric_elements=0)
for p in sorted(RUN.rglob('*.npz')):
    with np.load(p,allow_pickle=False) as a:
        counts['array_files']+=1
        for key in a.files:
            v=a[key]
            if np.issubdtype(v.dtype,np.number):
                assert np.isfinite(v).all(),(str(p),key)
                counts['numeric_arrays']+=1;counts['numeric_elements']+=v.size
        sidecar=p.with_suffix('.json')
        if not sidecar.exists():continue
        m=read(sidecar)
        for key,value in m.get('arrays',{}).items():assert describe(a[key])==value,(str(p),key)
        if 'actual_return_sha256' in m:
            assert array_sha(a['actual_combined_prediction'])==m['actual_return_sha256']
            for key in ('x_t','f00','f10','f01','f11','actual_combined_prediction'):
                assert a[key].dtype==np.float32 and list(a[key].shape)==m['shape']
        if 'metadata' in a:assert json.loads(str(a['metadata']))==m
completed={k:[v for v in s[k] if v['status']=='completed'] for k in ('probes','samples','traces')}
for row in completed['probes']:
    meta=read(RUN/'Q'/(row['probe_id']+'.json'))
    assert meta['package_sha256']==PACKAGE_SHA and meta['checkpoint_sha256']==CHECKPOINT_SHA
    assert meta['status']=='completed' and meta['probe_id']==row['probe_id']
for row in completed['samples']:
    spec=row['spec'];meta=read((RUN/row['output']).with_suffix('.json'))
    assert meta['package_sha256']==PACKAGE_SHA and meta['control_checkpoint_sha256']==CHECKPOINT_SHA
    assert meta['case_id']==row['case_id'] and meta['guidance']==GUIDANCE
    assert meta['comparison_case_id']==spec['comparison_case_id']
    assert meta['model_forward_calls']==1000 and meta['MDM_branch_examples']==4000
    assert len(meta['trace_ids'])==4 and row['diagnostic_complete']
    historical=read(verify(spec['historical_metadata']))
    assert meta['random_plans']==historical['random_plans']
    assert meta['decode_check']['passed']
for row in completed['traces']:
    meta=read(RUN/'traces'/(row['trace_id']+'.json'))
    assert meta['status']=='completed' and meta['trace_id']==row['trace_id'] and meta['case_id']==row['case_id']
    assert all(r['passed'] for r in meta['checks']['combination'].values())
    assert all(r['passed'] for k,r in meta['checks'].items() if k!='combination')
if s['status']=='completed':
    assert [len(completed[k]) for k in ('probes','samples','traces')]==[480,60,240]
    assert len(receipts)==61 and all(v['status']=='completed' for v in receipts)
    assert sum(v['mdm_examples'] for v in receipts)==240480
    assert sum(v['encoder_examples'] for v in receipts)<=696
    assert all(v['status']=='passed' for v in s['consistency_checks'])
    d=read(RUN/'Q/derived.json')
    assert [len(d[k]) for k in ('conditions','within','difference_of_differences')]==[288,384,192]
elapsed=s['finished_monotonic']-s['total_started_monotonic']
result=dict(run_status=s['status'],run_directory=str(RUN),runtime_status=artifact(RUN/'runtime/status.json'),
    counts={k:dict(Counter(v['status'] for v in s[k])) for k in ('probes','samples','traces')},
    elapsed_seconds=elapsed,MDM_branch_examples=sum(v['mdm_examples'] for v in receipts),
    encoder_examples=sum(v['encoder_examples'] for v in receipts),
    denoising_calls=sum(v['denoising_calls'] for v in receipts),
    peak_allocated_bytes=max((v['peak_allocated_bytes'] for v in receipts),default=0),
    verified_receipt_artifacts=len(verified),finite_arrays=counts,
    full_sample_checks=dict(Counter(v['status'] for v in s['consistency_checks'])),
    full_sample_max_abs_m=max((v.get('max_abs_m',0) for v in s['consistency_checks']),default=0),
    full_sample_bitwise_count=sum(v.get('bitwise',False) for v in s['consistency_checks']),
    failure=s.get('failure'),CPU_only=True,model_effectiveness='Not evaluated here; pending owner02 frozen evaluation')
(OUT/'delivery_integrity.json').write_text(json.dumps(result,indent=2)+'\n')
index=[artifact(p) for p in sorted(RUN.rglob('*')) if p.is_file()]
(OUT/'artifact_index.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps(result,indent=2))
