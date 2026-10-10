#!/usr/bin/env python
"""Archive complete two-path run/evaluation/proofs and link every retained raw artifact."""
from pathlib import Path
import shutil
from evaluate_b0_native_training_sufficiency import read,verify,artifact,write

ROOT=Path(__file__).resolve().parents[1]

def run():
    dest=ROOT/'docs/itm_rework/method/imu_main_knockout_run_20261009_v1'
    if dest.exists():raise FileExistsError('Fresh immutable archive required')
    cpu=ROOT/'outputs/itm_rework/method/imu_main_knockout_implementation_20261009_v1';proof=read(cpu/'delivery_verification.json')
    if proof['status']!='passed':raise ValueError('Completed delivery proof required')
    for k in ('implementation','report','verifier','next_CPU_plan'):verify(proof[k])
    run_dir=ROOT/'outputs/itm_rework/method/imu_main_knockout_authorized_20261009_v1';status=read(run_dir/'runtime/status.json')
    if status['status']!='completed':raise ValueError('Complete queue required')
    paths=list(cpu.iterdir())
    for name in ['imu_main_knockout_evaluation_20261009_v1','imu_main_knockout_responses_20261009_v1','imu_main_knockout_response_panel_20261009_v1','imu_main_knockout_review_20261009_v1','imu_main_knockout_behavior_20261009_v1']:
        paths.extend((ROOT/'outputs/itm_rework/protocol'/name).iterdir())
    paths.extend((ROOT/'docs/itm_rework/protocol/b0_imu_main_knockout_20261009_v1').iterdir())
    paths.append(verify(proof['next_CPU_plan']))
    paths.extend(ROOT/'docs/itm_rework'/n for n in ['GPU_DEVICE_PREFERENCE_20261008.json','GPU_CONTENDED_EXECUTION_POLICY_20261008.json','GPU_EXECUTION_POLICY_20261008.json','method/B0_IMU_MAIN_KNOCKOUT_EXECUTION_20261009.json','method/B0_IMU_MAIN_KNOCKOUT_RUN_20261009.md','protocol/B0_IMU_MAIN_KNOCKOUT_EVALUATION_20261009.md'])
    paths.extend(ROOT/'scripts'/n for n in ['run_mdm_imu_main_knockout_worker.py','run_mdm_imu_main_knockout_bounded.py','evaluate_b0_imu_main_knockout.py','verify_b0_imu_main_knockout_results.py','verify_b0_imu_main_knockout_contrasts.py','render_b0_imu_main_knockout.py','analyze_b0_imu_main_knockout_responses.py','verify_b0_imu_main_knockout_responses.py','summarize_b0_imu_main_knockout_responses.py','decide_b0_imu_main_knockout.py','freeze_b0_cross_source_component_CPU.py','verify_b0_imu_main_knockout_delivery.py','archive_b0_imu_main_knockout.py'])
    paths.append(ROOT/'src/itm/models/mdm_imu_main_knockout.py')
    paths.extend(ROOT/'tests'/n for n in ['test_mdm_imu_main_knockout.py','test_cross_source_component_CPU_freeze.py'])
    for name in ['imu_main_knockout_CPU_tests_20261009_v1','cross_source_component_CPU_freeze_tests_20261009_v1']:paths.append(ROOT/'outputs/itm_rework/method'/name/'pytest.xml')
    paths.extend(run_dir/'runtime'/n for n in ['status.json','suffix.receipt.json','suffix.receipt.journal.jsonl','suffix.context.json','suffix.log'])
    paths.extend((run_dir/'runtime/preflight').rglob('*'))
    dest.mkdir(parents=True);copies=[]
    for p in sorted(set(p for p in paths if p.is_file())):
        source=artifact(p);out=dest/'snapshot'/p.relative_to(ROOT);out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out);copy=artifact(out)
        if source['sha256']!=copy['sha256']:raise ValueError('Archive copy mismatch')
        copies.append(dict(source=source,archive=copy))
    run_files=[artifact(p) for p in sorted(run_dir.rglob('*')) if p.is_file()]
    write(dest/'complete_run_file_index.json',run_files);write(dest/'source_copy_index.json',copies)
    write(dest/'execution_summary.json',dict(status='complete_GPU_queue_and_CPU_evaluation_behavior_not_supported',physical_GPU='cuda:1',GPU_uuid=status['execution_record']['gpu_uuid'],implementation=status['implementation'],MDM_calls=1800,MDM_examples=7200,software_checks=1812,worker_artifacts=3626,run_files_indexed=len(run_files),wall_seconds=status['finished_monotonic']-status['started_monotonic'],peak_allocated_bytes=status['peak_allocated_bytes'],new_final_rows=2,new_trace_rows=6,combined_cells=10,comparisons=8,saved_response_rows=1800,saved_response_formula_checks=2413800,original_F_behavior='not_supported',new_knockout_behavior='not_supported',CPU_tests=7,next_CPU_freeze_tests=3,paper_claims_changed=False,independent_review=False,next_CPU_plan=proof['next_CPU_plan'],next_CPU_executed=False,no_retry_resume=True))
    write(dest/'artifact_index.json',[artifact(p) for p in sorted(dest.rglob('*')) if p.is_file()])
    for a in read(dest/'artifact_index.json'):verify(a)
    print(dict(status='passed',copies=len(copies),archive_artifacts=len(read(dest/'artifact_index.json')),run_files=len(run_files)),flush=True)

if __name__=='__main__':run()
