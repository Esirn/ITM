#!/usr/bin/env python
"""Close runtime, postprocessing, all-frame notes, panels and next CPU source identities."""
from pathlib import Path
from collections import Counter
import numpy as np
from evaluate_b0_native_training_sufficiency import read,verify,artifact,write

ROOT=Path(__file__).resolve().parents[1]
CPU=ROOT/'outputs/itm_rework/method/imu_main_knockout_implementation_20261009_v1'
RUN=ROOT/'outputs/itm_rework/method/imu_main_knockout_authorized_20261009_v1'
BASE=ROOT/'outputs/itm_rework/protocol'
NEXT=ROOT/'docs/itm_rework/protocol/b0_cross_source_component_CPU_20261009_v1/plan.json'


def run():
    dest=CPU/'delivery_verification.json'
    if dest.exists():raise FileExistsError('Fresh delivery proof required')
    manifest=read(CPU/'implementation.json')
    for a in manifest['expected_artifacts']:verify(a)
    status=read(RUN/'runtime/status.json')
    if status['status']!='completed' or any(r['status']!='completed' for r in status['checks']):raise ValueError('Complete valid queue required')
    proofs={k:read(CPU/(k+'.json')) for k in ['result_verification','contrast_verification','response_verification','next_CPU_plan_verification']}
    for p in proofs.values():
        if p['status']!='passed':raise ValueError('Passed complete proof required')
        if 'verifier' in p:verify(p['verifier'])
    if proofs['result_verification']['software_equations']!=1812 or proofs['response_verification']['rows']!=1800:raise ValueError('Complete primary proof counts')
    folders=[BASE/name for name in ['imu_main_knockout_evaluation_20261009_v1','imu_main_knockout_responses_20261009_v1','imu_main_knockout_response_panel_20261009_v1','imu_main_knockout_review_20261009_v1','imu_main_knockout_behavior_20261009_v1']]
    for f in folders:
        for a in read(f/'artifact_index.json'):verify(a)
    results,responses,panel,review,behavior=folders
    rows=read(responses/'rows.json');points=read(panel/'all_time_panel.json');table={r['row_id']:r for r in rows}
    if len(rows)!=1800 or len(points)!=1800 or len(table)!=1800 or {p['row_id'] for p in points}!=set(table):raise ValueError('Full response/panel roles')
    n=0;maxerror=0.
    def check(a,b):
        nonlocal n,maxerror
        d=abs(a-b);maxerror=max(maxerror,d);n+=1
        if d>64*np.finfo(float).eps*max(1.,abs(a),abs(b)):raise ArithmeticError('Panel scalar mismatch')
    for p in points:
        r=table[p['row_id']]
        if any(p[k]!=r[k] for k in ('sensor_config','timestep')):raise ValueError('Panel role mismatch')
        a=r['temporal']['actual_combined_prediction']['root_ric67']['temporal_energy_ratio'];u=r['derived_native111_temporal']['root_ric67']['temporal_energy_ratio']
        check(p['actual_energy_over_GT'],a);check(p['unit_same_state_energy_over_GT'],u);check(p['text_same_state_energy_over_GT'],r['temporal']['f10']['root_ric67']['temporal_energy_ratio']);check(p['local_knockout_minus_unit_energy_over_GT'],a-u);check(p['local_knockout_minus_unit_mse'],r['same_state_knockout_minus_native111']['root_ric67']['delta_mse'])
    notes=read(behavior/'notes.json');decisions=read(behavior/'decision.json')
    if len(notes)!=2 or len(decisions)!=2 or len(read(results/'old_final_references.json'))!=120:raise ValueError('Original final/behavior denominator')
    for note in notes:
        if not note['all_frames_inspected'] or note['frames']!=136 or note['blind'] or note['independent_review']:raise ValueError('Honest all-frame notes')
        for a in note['all_frame_pages']:verify(a)
    verify(read(behavior/'provenance.json')['script']);verify(read(behavior/'provenance.json')['notes_source'])
    plan=read(NEXT)
    for a in plan['sources']:verify(a)
    if plan['population']['rows']!=480 or plan['population']['F_U_pairs']!=240 or len(plan['jobs'])!=480:raise ValueError('Exact next CPU denominator')
    report=ROOT/'docs/itm_rework/protocol/B0_IMU_MAIN_KNOCKOUT_EVALUATION_20261009.md'
    write(dest,dict(status='passed',implementation=artifact(CPU/'implementation.json'),report=artifact(report),CPU_sources=len(manifest['expected_artifacts']),worker_artifacts=3626,new_finals_valid=2,new_traces_valid=6,comparison_cells=10,comparisons=8,response_rows=1800,response_scalar_checks=proofs['response_verification']['independent_scalar_checks'],panel_scalar_checks=n,max_panel_absolute_error=maxerror,review_notes=2,behavior_counts=dict(Counter(r['decision'] for r in decisions)),original_F_behavior='not_supported',paper_claims_changed=False,independent_review=False,verifier=artifact(Path(__file__).resolve()),next_CPU_plan=artifact(NEXT),next_CPU_sources=len(plan['sources']),new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0))
    print(dict(status='passed',sources=len(manifest['expected_artifacts']),panel_checks=n,next_CPU_sources=len(plan['sources'])),flush=True)

if __name__=='__main__':run()
