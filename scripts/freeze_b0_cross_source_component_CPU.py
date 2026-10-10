#!/usr/bin/env python
"""Freeze next saved-array CPU component/error audit over all original F/U sources."""
import argparse
from collections import Counter
from pathlib import Path
from evaluate_b0_native_training_sufficiency import read,verify,artifact,write

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'outputs/itm_rework/protocol/saved_trace_response_evaluation_20261008_v1'
CPU=ROOT/'outputs/itm_rework/method/imu_main_knockout_implementation_20261009_v1'


def validate_population(rows):
    conditions=Counter(r['condition'] for r in rows)
    expected=dict(paired=96,text_only=96,imu_only=48,backbone_bypass=48,numeric_zero=48,physical_static=96,tail_reversed=48)
    if len(rows)!=480 or conditions!=expected or any(r['status']!='valid' for r in rows):raise ValueError('All480 valid original seven-condition traces required')
    keys={(r['pair_case_id'],r['timestep'],r['arm']) for r in rows}
    pairs={(r['pair_case_id'],r['timestep']) for r in rows}
    if len(keys)!=480 or len(pairs)!=240 or keys!={(c,t,a) for c,t in pairs for a in ('F','U')}:raise ValueError('Complete unique F/U pairs')
    sources={(r['data_split'],r['motion_id']) for r in rows}
    if sources!={('train',m) for m in ('000027','000506','004728')}|{('val',m) for m in ('000139','000382','000433')}:raise ValueError('Original admitted six sources')
    if {r['timestep'] for r in rows}!={900,500,250,50} or {r['sensor_config'] for r in rows}!={'head','wrists'}:raise ValueError('Original fixed times/configs')
    return conditions


def freeze(output):
    if output.exists():raise FileExistsError('Fresh CPU plan required')
    manifest=read(CPU/'implementation.json');proof=read(CPU/'result_verification.json');response=read(CPU/'response_verification.json')
    if proof['status']!='passed' or proof['software_equations']!=1812 or response['status']!='passed' or response['rows']!=1800:raise ValueError('Completed main-knockout proofs first')
    oldproof=ROOT/'outputs/itm_rework/method/saved_trace_response_verification_20261008_v1/verification.json'
    verification=read(oldproof)
    if verification['status']!='passed' or verification['trace_points']!=480 or verification['final_samples']!=120:raise ValueError('Full original CPU proof')
    code=artifact(ROOT/'scripts/verify_b0_saved_trace_responses.py')
    if code['sha256']!=verification['verifier_sha256']:raise ValueError('Original second-formula verifier identity')
    for a in read(OLD/'artifact_index.json'):verify(a)
    rows=read(OLD/'traces.json');conditions=validate_population(rows)
    original_plan=ROOT/'docs/itm_rework/protocol/b0_saved_trace_response_20261008_v1/plan.json';plan=read(original_plan)
    if len(plan['traces'])!=480 or len(read(OLD/'samples.json'))!=120:raise ValueError('Original trace/final denominators')
    jobs=[]
    for job,row in zip(plan['traces'],rows):
        if any(job[k]!=row[k] for k in job):raise ValueError('Original role/time/path identity')
        jobs.append(dict(job,raw=artifact(Path(job['path'])),metadata=artifact(Path(job['path']).with_suffix('.json'))))
    sources={a['path']:a for a in manifest['expected_artifacts']}
    paths=[original_plan,oldproof,ROOT/'scripts/verify_b0_saved_trace_responses.py',Path(__file__).resolve(),ROOT/'tests/test_cross_source_component_CPU_freeze.py',*OLD.iterdir(),*[CPU/(n+'.json') for n in ['implementation','CPU_acceptance','implementation_delivery','result_verification','contrast_verification','response_verification']]]
    paths.extend(ROOT/'scripts'/n for n in ['analyze_b0_imu_main_knockout_responses.py','verify_b0_imu_main_knockout_responses.py','summarize_b0_imu_main_knockout_responses.py','decide_b0_imu_main_knockout.py'])
    for name in ['imu_main_knockout_evaluation_20261009_v1','imu_main_knockout_responses_20261009_v1','imu_main_knockout_response_panel_20261009_v1','imu_main_knockout_behavior_20261009_v1','imu_main_knockout_review_20261009_v1']:
        paths.extend((ROOT/'outputs/itm_rework/protocol'/name).iterdir())
    for p in paths:
        if p.is_file():a=artifact(p);sources[a['path']]=a
    for a in sources.values():verify(a)
    output.mkdir(parents=True)
    value=dict(version='b0-cross-source-component-error-CPU-v1',date='2026-10-09',status='CPU_frozen_not_implemented_not_executed',
        objective='Determine how the saved IMU main/interaction energy and error patterns vary across all original admitted sources and controls before choosing another method/GPU intervention.',
        rationale='Main knockout from T history restores wrists energy near text baseline but fails full behavior and all five U guards. At every saved knockout state, removal also increases root+RIC MSE. Need source/condition-specific error-versus-dynamic audit rather than assume a globally harmful main term from one clip.',
        population=dict(rows=480,F_U_pairs=240,paired_condition_rows=96,paired_condition_F_U_pairs=48,sources=6,configs=12,timesteps=[900,500,250,50],condition_counts=dict(conditions),original_final_references=120),
        inherited_results=dict(directory=str(OLD),artifact_index=artifact(OLD/'artifact_index.json'),proof=artifact(oldproof)),normalization=plan['normalization'],jobs=jobs,
        metrics='All original11 views and existing balanced fields. On each saved state set a=f10,u=f01-f00,j=f11-f10-f01+f00 in float64. Compute temporal decomposition and unchanged same_state_delta against a/GT for a+u and a+j; preserve 2Cov(u,j), signed energy changes and signed target-error cross terms. These hypothetical arrays are CPU algebra, not new native returns/rollouts. Undefined GT energy ratios remain null with full denominator.',
        aggregation='Source/config/condition/arm/time retained. F/U pairs compare matched original cases/t but differing states, not fixed-state checkpoint causation. Equal-source descriptive summaries only; no frame/time replication, significance, source selection or causal allocation percentages. Keep all seven conditions and failures/missing rows.',
        references='Retain all120 original finals, behavior labels/negative cases and original 480 raw traces; static/strict absence/backbone bypass controls remain visible. New main-knockout results are motivating bound references, not additions to the original480 population.',
        verification='Independent raw formulas for every planned row and every F/U contrast, hashes/roles/full masks/original dimensions/finite validation; immutable original metrics helpers and64eps context scale. No scientific tolerance or behavior rule change.',
        interpretation='Posthoc CPU audit after exposed main-case intervention; not blind preregistration, independent02/human review, behavior repair or generalization claim. The original F and new knockout remain not_supported.',
        new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,prohibited=['GPU query','model construction/forward','noise generation','sampling','decode','training','new source/seed/checkpoint/caption/time boundary','guidance sweep','paper claim changes'],sources=[sources[k] for k in sorted(sources)],
        next_task='Implement/complete this saved-array CPU audit, then decide a minimal next method intervention from all-source evidence; no further GPU intervention is selected in this plan.')
    write(output/'plan.json',value);print(dict(plan=artifact(output/'plan.json'),sources=len(sources),rows=480,pairs=240),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();freeze(a.output.resolve())
