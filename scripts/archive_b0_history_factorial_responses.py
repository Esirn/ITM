#!/usr/bin/env python
"""Archive completed CPU response delivery; preserve raw/inherited sources in place."""
from pathlib import Path
import shutil
from evaluate_b0_native_training_sufficiency import read,verify,artifact,write

ROOT=Path(__file__).resolve().parents[1]

def run():
    dest=ROOT/'docs/itm_rework/method/history_factorial_response_run_20261009_v1'
    if dest.exists():raise FileExistsError('Fresh CPU archive required')
    cpu=ROOT/'outputs/itm_rework/method/history_factorial_response_implementation_20261009_v1'
    results=ROOT/'outputs/itm_rework/protocol/history_factorial_response_evaluation_20261009_v1'
    delivery=read(cpu/'delivery_verification.json')
    if delivery['status']!='passed':raise ValueError('Completed CPU delivery required')
    for k in ('implementation','results_index','panel_index','component_index','next_package','verifier'):verify(delivery[k])
    directories=[cpu,results,ROOT/'outputs/itm_rework/protocol/history_factorial_response_panel_20261009_v1',ROOT/'outputs/itm_rework/protocol/imu_response_component_algebra_20261009_v1',ROOT/'docs/itm_rework/protocol/b0_imu_main_knockout_20261009_v1']
    paths=[p for directory in directories for p in directory.iterdir() if p.is_file()]
    paths.extend(ROOT/'scripts'/n for n in ['analyze_b0_history_factorial_responses.py','verify_b0_history_factorial_responses.py','summarize_b0_history_factorial_responses.py','summarize_b0_imu_response_components.py','verify_b0_imu_component_algebra.py','verify_b0_history_response_delivery.py','freeze_b0_imu_main_knockout.py','archive_b0_history_factorial_responses.py','analyze_b0_suffix_responses.py','evaluate_b0_native_training_sufficiency.py'])
    paths.extend(ROOT/'tests'/n for n in ['test_history_factorial_response_adapter.py','test_imu_main_knockout_freeze.py','test_suffix_response_adapter.py'])
    paths.extend(ROOT/'src/itm/metrics'/n for n in ['training_sufficiency.py','trace_response.py'])
    paths.extend(ROOT/'docs/itm_rework'/n for n in ['GPU_DEVICE_PREFERENCE_20261008.json','GPU_CONTENDED_EXECUTION_POLICY_20261008.json','GPU_EXECUTION_POLICY_20261008.json','protocol/B0_HISTORY_FACTORIAL_RESPONSE_EVALUATION_20261009.md','protocol/b0_history_factorial_response_CPU_20261009_v1/plan.json'])
    paths.append(ROOT/'outputs/itm_rework/method/history_factorial_response_freeze_tests_20261009_v1/pytest.xml')
    inherited=read(results/'inherited_F.json')
    paths.extend([verify(inherited['implementation']),verify(inherited['proof']),verify(inherited['artifact_index'])])
    paths=sorted(set(paths));dest.mkdir(parents=True);copies=[]
    for p in paths:
        source=artifact(p);out=dest/'snapshot'/p.relative_to(ROOT);out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
        copied=artifact(out)
        if source['sha256']!=copied['sha256']:raise ValueError('Archive copy changed')
        copies.append(dict(source=source,archive=copied))
    write(dest/'source_copy_index.json',copies)
    summary=dict(status='CPU_phase_completed',new_T_rows=3600,inherited_F_rows=3600,effective_rows=7200,path_contrasts=3600,factorial_interactions=1800,independent_scalar_checks=4640400,posthoc_raw_scalar_checks=28800,panel_scalar_checks=70200,adapter_tests=7,next_freeze_tests=3,new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,original_F_behavior='not_supported',new_history_behavior='not_supported',paper_claims_changed=False,independent_review=False,next_package=delivery['next_package'],next_intervention_implemented=False,next_intervention_executed=False)
    write(dest/'execution_summary.json',summary)
    write(dest/'artifact_index.json',[artifact(p) for p in sorted(dest.rglob('*')) if p.is_file()])
    for a in read(dest/'artifact_index.json'):verify(a)
    print(dict(status='passed',copies=len(copies),archive_artifacts=len(read(dest/'artifact_index.json'))),flush=True)

if __name__=='__main__':run()
