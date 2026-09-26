#!/usr/bin/env python
"""Owner02 read-only failed-run and replacement-manifest audit; no model/GPU."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
METHOD=ROOT/'docs/itm_rework/method/correction_run_20260927_v1'
RUN=ROOT/'outputs/itm_rework/method/correction_authorized_20260927_v1'
OLD=ROOT/'outputs/itm_rework/method/correction_implementation_20260926_v3/implementation.json'
NEW=ROOT/'outputs/itm_rework/method/correction_implementation_20260927_v1/implementation.json'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def artifact(p):return {'path':str(p),'sha256':sha(p)}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    old,new=read(OLD),read(NEW)
    assert sha(OLD)=='449b8d278a5a4d1ee31c6dad8a9eedbe59c2990f1efceedc45ad4ae70c805676'
    assert sha(NEW)=='a802ef3b03aea7bab7fff3fda5704cf56a2c5cf6c5ed36c693a892fc35d6de13'
    assert NEW.read_bytes()==(METHOD/'implementation.json').read_bytes()
    before={a['path']:a['sha256'] for a in old['expected_artifacts']};after={a['path']:a['sha256'] for a in new['expected_artifacts']}
    assert set(before)==set(after) and len(after)==126
    changed=[p for p in after if before[p]!=after[p]]
    assert changed==[str(ROOT/'scripts/run_mdm_correction_worker.py')]
    old_code=subprocess.check_output(['git','show','8e4df1e:scripts/run_mdm_correction_worker.py'],cwd=ROOT)
    assert hashlib.sha256(old_code).hexdigest()==before[changed[0]]
    for a in new['expected_artifacts']+[new['initialization']['artifact']]:assert sha(ROOT/a['path'])==a['sha256'],a['path']
    for k in set(new)-{'expected_artifacts','initialization'}:assert new[k]==old[k],k
    for k in set(new['initialization'])-{'artifact'}:assert new['initialization'][k]==old['initialization'][k],k
    initial_old,initial_new=old['initialization']['artifact'],new['initialization']['artifact']
    assert sha(initial_old['path'])==sha(initial_new['path'])==initial_new['sha256']==initial_old['sha256']
    summary=read(METHOD/'run_summary.json');copies=[]
    for a in summary['preserved_runtime']:
        original=Path(a['original_path']);copy=ROOT/a['path']
        assert original.read_bytes()==copy.read_bytes() and sha(original)==a['sha256']
        copies.append({'original':artifact(original),'copy':artifact(copy),'byte_identical':True})
    assert {p.name for p in (RUN/'runtime').iterdir()}=={'status.json','D_probe.context.json','D_probe.receipt.json','D_probe.log'}
    s=read(RUN/'runtime/status.json');r=read(RUN/'runtime/D_probe.receipt.json');context=read(RUN/'runtime/D_probe.context.json')
    auth=read(ROOT/'docs/itm_rework/method/B0_CORRECTION_AUTHORIZATION_20260927.json')
    assert s['authorization']==auth and auth['implementation_sha256']==sha(OLD)
    assert auth['run_directory']==str(RUN) and auth['authorized'] and auth['gpu_uuid']==r['gpu_uuid']
    assert s['implementation']['sha256']==sha(OLD)
    assert r['context']==context and context['phase']=='probe' and context['arm'] is None
    assert s['status']==r['status']=='failed' and s['stages'][0]['status']=='failed'
    assert all(x['status']=='not_run_after_stop' for x in s['stages'][1:])
    assert len(s['probes'])==480 and Counter(x['status'] for x in s['probes'])=={'not_run_after_stop':480}
    assert len(s['samples'])==120 and Counter(x['status'] for x in s['samples'])=={'not_run_after_stop':120}
    counters={k:r[k] for k in ['mdm_examples','encoder_examples','updates_attempted','updates_completed','denoising_calls','branch_examples','exposures']}
    assert all(v==0 for v in counters.values()) and r['completed_slots']==[] and r['artifacts']==[]
    assert r['failure']=={'kind':'exception','type':'AttributeError','message':"'NoneType' object has no attribute 'eval'"}
    assert s['active_result']['returncode']==1 and s['active_result']['launched'] is True
    assert not list(RUN.rglob('*.pt')) and not list(RUN.rglob('*.npz'))
    elapsed=s['finished_monotonic']-s['total_started_monotonic']
    assert abs(elapsed-summary['elapsed_seconds'])<1e-9
    verification=read(METHOD/'cpu_verification.json')
    for a in verification['changed_code']+verification['evidence']:assert sha(ROOT/a['path'])==a['sha256'],a['path']
    suite=next(ET.parse(METHOD/'logs/regression.xml').getroot().iter('testsuite'))
    assert {k:int(suite.get(k)) for k in ['tests','failures','errors','skipped']}=={'tests':204,'failures':0,'errors':0,'skipped':0}
    report={'read_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'old_manifest':artifact(OLD),'accepted_replacement_manifest':artifact(NEW),'changed_bound_artifacts':changed,
        'replacement_bound_files_verified':127,'unchanged_initialization':initial_new,'unchanged_environment':new['environment'],
        'unchanged_input_metadata_count':len(new['inputs']),'unchanged_package':new['package'],
        'runtime_copies':copies,'failure':r['failure'],'counters':counters,'elapsed_seconds_from_record':elapsed,
        'peak_allocated_bytes_from_receipt':r['peak_allocated_bytes'],'probes_not_run':480,'samples_not_run':120,
        'observed_run_runtime_children':['D_probe'],'new_model_or_sample_artifacts_in_failed_run':0,
        'method_joint_204_evidence_verified_not_rerun':True,'current_audit_model_forwards':0,'current_audit_GPU_queries':0,
        'replacement_authorization_created':False,'old_authorization_consumed':True,'code':artifact(Path(__file__))}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'runtime_copies':4,'changed_bound_files':changed,'verified_new_files':127,'elapsed_seconds':elapsed,'probe_forwards':0,'training_updates':0,'samples':0}))

if __name__=='__main__':main()
