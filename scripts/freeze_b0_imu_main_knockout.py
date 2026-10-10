#!/usr/bin/env python
"""Freeze two prospective IMU-main knockout suffix paths; no GPU/model/noise generation."""
import argparse
from copy import deepcopy
from pathlib import Path
from evaluate_b0_native_training_sufficiency import read,verify,artifact,write,arrays,descriptor

ROOT=Path(__file__).resolve().parents[1]
CPU=ROOT/'outputs/itm_rework/method/history_factorial_response_implementation_20261009_v1'
EVAL=ROOT/'outputs/itm_rework/protocol/history_factorial_response_evaluation_20261009_v1'
PANEL=ROOT/'outputs/itm_rework/protocol/history_factorial_response_panel_20261009_v1'
COMPONENT=ROOT/'outputs/itm_rework/protocol/imu_response_component_algebra_20261009_v1'
PREVIOUS=ROOT/'docs/itm_rework/protocol/b0_suffix_history_factorial_20261008_v2'


def select_on(rows):
    picked=[r for r in rows if r['condition']=='switch_imu_on']
    if len(picked)!=2 or {r['sensor_config'] for r in picked}!={'head','wrists'}:raise ValueError('Unique two original T-prefix paired paths')
    if any(not r['imu_present'] or not r['text_present'] or r['frames']!=136 or r['checkpoint']!='F200' or r['motion_id']!='000506' or r['data_split']!='train' or r['diffusion_steps']!=list(range(899,-1,-1)) for r in picked):raise ValueError('Original role/source/weight/schedule')
    return sorted(picked,key=lambda r:r['sensor_config'])


def first_anchor(row,job):
    if job['trajectory_id']!=row['trajectory_id'] or job['timestep']!=899 or job['sensor_config']!=row['sensor_config'] or job['condition']!='switch_imu_on' or job['prefix_history']!='T_text_only' or job['suffix_condition']!='paired' or not job['imu_present']:raise ValueError('Exact original paired start branch anchor')
    return dict(trajectory_id=row['trajectory_id'],timestep=899,condition='switch_imu_on',raw=job['raw'],metadata=job['metadata'],arrays=['x_t','f00','f10','f01','f11'])


def build():
    impl=read(CPU/'implementation.json');proof=read(CPU/'result_verification.json')
    if proof['status']!='passed' or proof['total_effective_rows']!=7200 or proof['factorial_interactions']!=1800:raise ValueError('Complete history CPU proof first')
    verify(proof['results_index']);verify(proof['verifier'])
    plan=read(verify(impl['plan']));old=select_on(read(PREVIOUS/'trajectories.json'));jobs={(j['trajectory_id'],j['timestep']):j for j in plan['jobs']}
    starts=read(PREVIOUS/'starts.json');paths=[];anchors=[];checks=[]
    for i,r in enumerate(old,1):
        anchor=first_anchor(r,jobs[r['trajectory_id'],899]);meta=read(verify(anchor['metadata']));z=arrays(verify(anchor['raw']))
        if meta['trajectory_id']!=r['trajectory_id'] or meta['timestep']!=899 or meta['arrays']!={k:descriptor(v) for k,v in z.items()}:raise ValueError('Original anchor raw descriptors')
        start=arrays(verify(r['start']))
        if descriptor(start['x_899'])!=descriptor(z['x_t']) or descriptor(start['frame_mask'])!=descriptor(z['frame_mask']):raise ValueError('Paired anchor exact T-prefix start')
        p=deepcopy(r);p.update(trajectory_id=f'MK{i:03d}',condition='remove_IMU_main_retain_joint',prefix_history='T_text_only',suffix_condition='text_plus_joint_interaction',imu_present=True,text_present=True)
        p['guidance']['imu_scale']=0.;p['guidance']['text_scale']=1.;p['guidance']['joint_scale']=1.
        paths.append(p);anchor.update(trajectory_id=p['trajectory_id'],reference_trajectory_id=r['trajectory_id']);anchors.append(anchor)
        checks.append(dict(check_id='START-'+p['panel_case_id'],kind='original_start',trajectory_id=p['trajectory_id']))
        checks.extend(dict(check_id='CFG-'+p['trajectory_id']+'-'+str(t),kind='factorized_(1,0,1)',trajectory_id=p['trajectory_id'],timestep=t) for t in p['diffusion_steps'])
        checks.extend(dict(check_id='ANCH-'+p['trajectory_id']+'-899-'+a,kind='original_paired_first_branch',trajectory_id=p['trajectory_id'],timestep=899,array=a) for a in anchor['arrays'])
    sources={a['path']:a for a in impl['expected_artifacts']}
    more=[*[CPU/(n+'.json') for n in ('implementation','CPU_test_results','result_verification')],*EVAL.glob('*.json'),*PANEL.iterdir(),*COMPONENT.iterdir(),*PREVIOUS.glob('*.json'),ROOT/'scripts/summarize_b0_imu_response_components.py',Path(__file__).resolve(),ROOT/'tests/test_imu_main_knockout_freeze.py',ROOT/'docs/itm_rework/GPU_DEVICE_PREFERENCE_20261008.json',ROOT/'docs/itm_rework/GPU_CONTENDED_EXECUTION_POLICY_20261008.json']
    for path in more:
        if path.is_file():a=artifact(path);sources[a['path']]=a
    for a in sources.values():verify(a)
    design=dict(version='b0-IMU-main-knockout-suffix-v1',date='2026-10-09',status='CPU_frozen_not_implemented_not_executed',
        question='Does deleting the IMU main term while retaining the text-IMU interaction change the contracted paired suffix trajectory from the original T-prefix x899?',
        rationale='Full history CPU and posthoc saved-state algebra show the no-text IMU main term has a negative energy change at wrists t899/500/250 while interaction-only changes at those states are positive. Local signs do not determine later states or behavior. One coefficient knockout is the smallest prospective rollout test; no coefficient search.',
        intervention='Original factorized native combination order with (text_scale,imu_scale,joint_scale)=(1,0,1), both text/IMU presence true and original paired static-delta control retained. f00+1*(f10-f00)+0*(f01-f00)+1*(f11-f10-f01+f00). Equivalent real-number expression f11-f01+f00 is explanatory only, not a replacement floating-point execution order.',
        scope='train000506/wrists primary, head companion; original F200 only, T-prefix original x899, fixed t899..0, 136 real frames. New2 paths; prior T text/paired and F paired/off stay bound references, not new contemporaneous replications.',
        schedule='Original unrespaced1000 DDPM, clip_denoised=false, native float32 posterior, caption batch1/MDM factorized batch4. Original noise t899..1 descriptor/address/bytes before use; no noise t0/initial/new seed.',
        software_gate='Raw before check, 2 exact source starts,10 first-t899 original paired x_t/f00/f10/f01/f11 anchors,1800 native (1,0,1) CFG equations at original 1e-6+1e-6*abs(reference); masks/dtypes/finite/hash/posterior/noise/count/deadline required. First fault stops whole queue, preserve all slots. No later old trajectory anchor is imposed after intervention states diverge.',
        metrics='Same original11 views/balanced temporal/response and final five motion guardrails plus original full observed alternating-lunge recovery review; all8 factorial finals/120 original finals/480 traces/seven-condition negatives retained. No fitting, changed success threshold, independent t/frame samples or claimed generalization.',
        decision='Report each configuration and all differences against its T text/paired and old F references. Dynamic increase alone cannot support behavior repair. Changed rollout shows dependence on removal of this computational term in this case, not unique encoder/adapter causation, allocation percentages or independence from evolving state.',
        exposure='Chosen after exposed saved-state/full-trajectory analyses; prospective knockout only, not blind preregistration, independent02 or human review. No new model result exists in this package.',
        prohibitions=['training','new source/seed/caption/checkpoint','guidance sweep','new or adaptive boundary','automatic retry/resume','interference with unrelated tasks','paper claim changes'])
    budgets=dict(total_wall_seconds=1200,peak_cuda_allocated_cap_bytes=24*1024**3,trajectories=2,suffix_steps_per_trajectory=900,total_MDM_calls=1800,total_MDM_examples=7200,encoder_calls=4,encoder_examples=4,caption_calls=2,caption_examples=2,old_noise_regenerations=1798,distinct_old_noise_addresses=1798,new_initial_noise_draws=0,new_noise_seeds=0,final_motion_decodes=2,updates=0,automatic_retry=False,resume=False,preferred_physical_GPU='cuda:1',GPU_uuid='GPU-43c2f925-db05-c655-bb46-25196d0627d6',admission='Standing user authorization; project UUID lock,3 samples2sec apart,free>=4096MiB; utilization recorded without cutoff. Native allocator, fixed whole-attempt absolute1200sec deadline.')
    return dict(design=design,trajectories=paths,starts=starts,noise=read(PREVIOUS/'noise.json'),checkpoints=read(PREVIOUS/'checkpoints.json'),historical_anchors=anchors,software_checks=checks,budgets=budgets,sources=[sources[k] for k in sorted(sources)])


def freeze(output):
    if output.exists():raise FileExistsError('Fresh scientific package required')
    payload=build();output.mkdir(parents=True)
    for k,v in payload.items():write(output/(k+'.json'),v)
    package=dict(version='b0-IMU-main-knockout-package-v1',date='2026-10-09',status='CPU_frozen_not_implemented_not_executed',artifacts=[artifact(output/(k+'.json')) for k in sorted(payload)],new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0)
    write(output/'package.json',package);print(dict(package=artifact(output/'package.json'),sources=len(payload['sources']),payloads=len(payload),trajectories=2,checks=len(payload['software_checks'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();freeze(a.output.resolve())
