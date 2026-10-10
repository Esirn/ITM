#!/usr/bin/env python
"""Archive unchanged CPU results, syntax-failure provenance, compatible proof and next plan."""
from pathlib import Path
import shutil
from evaluate_b0_native_training_sufficiency import read,verify,artifact,write

ROOT=Path(__file__).resolve().parents[1]

def run():
    dest=ROOT/'docs/itm_rework/method/cross_source_component_run_20261009_v1'
    if dest.exists():raise FileExistsError('Fresh immutable CPU archive required')
    cpu=ROOT/'outputs/itm_rework/method/cross_source_component_implementation_20261009_v2';delivery=read(cpu/'delivery_verification.json')
    if delivery['status']!='passed':raise ValueError('Complete CPU delivery first')
    for k in ('implementation','results_index','panel_index','next_probe_plan','verifier','report'):verify(delivery[k])
    dirs=[cpu,ROOT/'outputs/itm_rework/method/cross_source_component_implementation_20261009_v1',ROOT/'outputs/itm_rework/protocol/cross_source_component_evaluation_20261009_v1',ROOT/'outputs/itm_rework/protocol/cross_source_component_panel_20261009_v1']
    paths=[p for d in dirs for p in d.iterdir() if p.is_file()]
    paths.extend(ROOT/'scripts'/n for n in ['analyze_b0_cross_source_components.py','verify_b0_cross_source_components.py','verify_b0_cross_source_components_v2.py','prepare_b0_cross_source_verification_v2.py','summarize_b0_cross_source_components.py','freeze_b0_control_mean_probe.py','verify_b0_cross_source_delivery.py','archive_b0_cross_source_components.py','evaluate_b0_native_training_sufficiency.py'])
    paths.extend(ROOT/'tests'/n for n in ['test_cross_source_component_adapter.py','test_control_mean_probe_freeze.py'])
    paths.extend(ROOT/'src/itm/metrics'/n for n in ['training_sufficiency.py','trace_response.py','feature_balance.py'])
    paths.extend(ROOT/'docs/itm_rework'/n for n in ['protocol/B0_CROSS_SOURCE_COMPONENT_EVALUATION_20261009.md','protocol/b0_cross_source_component_CPU_20261009_v1/plan.json','protocol/b0_control_mean_probe_20261009_v1/plan.json','GPU_DEVICE_PREFERENCE_20261008.json','GPU_CONTENDED_EXECUTION_POLICY_20261008.json','GPU_EXECUTION_POLICY_20261008.json'])
    for name in ['cross_source_component_CPU_tests_20261009_v1','control_mean_probe_freeze_tests_20261009_v1']:paths.append(ROOT/'outputs/itm_rework/method'/name/'pytest.xml')
    dest.mkdir(parents=True);copies=[]
    for p in sorted(set(paths)):
        source=artifact(p);target=dest/'snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);copy=artifact(target)
        if source['sha256']!=copy['sha256']:raise ValueError('Archive copy mismatch')
        copies.append(dict(source=source,archive=copy))
    write(dest/'source_copy_index.json',copies)
    write(dest/'execution_summary.json',dict(status='complete_CPU_audit_only',rows=480,F_U_pairs=240,paired_rows=96,sources=6,condition_count=7,old_final_references=120,adapter_tests=9,formula_scalar_checks=602400,null_scalar_checks=1008,summary_fields=delivery['summary_scalar_or_count_fields'],evaluation_sources=17911,verification_sources=17922,new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,original_F_behavior='not_supported',paper_claims_changed=False,independent_review=False,original_result_index=delivery['results_index'],verification_only_v2=True,original_invalid_v1_verifier_preserved=True,next_probe_plan=delivery['next_probe_plan'],next_probe_states=48,next_probe_tests=3,next_probe_implemented=False,next_probe_executed=False))
    write(dest/'artifact_index.json',[artifact(p) for p in sorted(dest.rglob('*')) if p.is_file()])
    for a in read(dest/'artifact_index.json'):verify(a)
    print(dict(status='passed',copies=len(copies),archive_artifacts=len(read(dest/'artifact_index.json'))),flush=True)

if __name__=='__main__':run()
