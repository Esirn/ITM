#!/usr/bin/env python
"""CPU freeze of a prospective same-history IMU-off suffix intervention."""
import argparse
from pathlib import Path

import numpy as np

from evaluate_b0_native_training_sufficiency import read,verify,artifact,arrays,descriptor,write

ROOT=Path(__file__).resolve().parents[1]
PREFIX=ROOT/'outputs/itm_rework/method/prefix_mechanism_authorized_20261008_v3'
PREFIX_MANIFEST_SHA='be04358cbeca4d35b4aeed2d3222f4f1013105cf6ab930e011eb8c16a113194f'
PREFIX_PACKAGE=ROOT/'docs/itm_rework/protocol/b0_prefix_mechanism_20261008_v1'
F=ROOT/'outputs/itm_rework/method/feature_balance_authorized_20261008_v1'
LATE=ROOT/'outputs/itm_rework/protocol/late_attribution_evaluation_20261008_v1'
LATE_IMPL=ROOT/'outputs/itm_rework/method/late_attribution_implementation_20261008_v1'
OUT=ROOT/'docs/itm_rework/protocol/b0_suffix_switch_20261008_v2'


def suffix_noise(records):
    if len(records)!=1000 or [r['step'] for r in records]!=[-1]+list(range(999,0,-1)):
        raise ValueError('Complete historical noise plan required')
    suffix=records[101:]
    if [r['step'] for r in suffix]!=list(range(899,0,-1)) or len(suffix)!=899:
        raise ValueError('Suffix noise addresses changed')
    return suffix


def validate_terminal(z,frames):
    if set(z)!=set(['x_899','frame_mask']):raise ValueError('Exact completed prefix terminal required')
    x=z['x_899'];mask=z['frame_mask']
    if x.dtype!=np.float32 or x.shape!=(1,263,1,frames) or not np.isfinite(x).all():
        raise ValueError('Native finite full-frame x899 required')
    if mask.dtype!=np.bool_ or mask.shape!=(1,frames) or not mask.all():raise ValueError('Original full mask required')


def freeze(output):
    if output.exists():raise FileExistsError('Fresh prospective package required')
    sources={}
    def bind(path,expected=None):
        p=(ROOT/path).resolve();a=artifact(p)
        if expected is not None and a['sha256']!=expected:raise ValueError('Previous source identity changed: '+str(p))
        sources[a['path']]=a;return p
    status=read(bind(PREFIX/'runtime/status.json'))
    if status['status']!='completed' or any(r['status']!='completed' for r in status['checks']):raise ValueError('Complete prefix gate required')
    if status['implementation']['sha256']!=PREFIX_MANIFEST_SHA:raise ValueError('Exact completed prefix identity')
    manifest=read(bind(Path(status['implementation']['path']),PREFIX_MANIFEST_SHA))
    for a in manifest['expected_artifacts']:bind(Path(a['path']),a['sha256'])
    receipt=read(bind(Path(status['receipt']['path']),status['receipt']['sha256']))
    bind(Path(status['receipt_journal']['path']),status['receipt_journal']['sha256'])
    if receipt['status']!='completed' or receipt['counts']['mdm']['completed_calls']!=3960:raise ValueError('Complete prefix receipt required')
    received={r['path']:r for r in receipt['artifacts']}
    late_status=read(bind(LATE/'summary.json'));proof=read(bind(LATE_IMPL/'result_verification_v2.json'))
    if late_status['status']!='completed' or late_status['counts']!={'valid':96} or proof['status']!='passed' or proof['rows']!=96:
        raise ValueError('Complete late CPU gate required')
    bind(LATE_IMPL/'implementation.json')
    for a in read(bind(LATE/'artifact_index.json')):bind(Path(a['path']),a['sha256'])
    for a in read(LATE/'sources.json'):bind(Path(a['path']),a['sha256'])
    for p in [Path(__file__),ROOT/'tests/test_suffix_switch_freeze.py',ROOT/'scripts/verify_b0_late_attribution.py',
              ROOT/'scripts/summarize_b0_late_attribution.py',ROOT/'docs/itm_rework/GPU_CONTENDED_EXECUTION_POLICY_20261008.json',
              ROOT/'docs/itm_rework/protocol/B0_FEATURE_BALANCE_EVALUATION_20261008.md']:
        bind(p)
    cases=read(bind(ROOT/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/sampling_slots.json'))
    oldtraces=read(bind(ROOT/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/trace_slots.json'))
    prefix_paths=read(bind(PREFIX_PACKAGE/'trajectories.json'))
    original=[c for c in cases if c['data_split']=='train' and c['motion_id']=='000506' and c['condition']=='paired']
    if len(original)!=2 or {c['sensor_config'] for c in original}!={'head','wrists'}:raise ValueError('Original primary + companion required')
    paths=[];noises=[];anchors=[];starts=[]
    for c in original:
        px=next(r for r in prefix_paths if r['panel_case_id']==c['case_id'] and r['path_arm']=='F')
        terminal=PREFIX/'prefix/terminal'/(px['trajectory_id']+'.npz')
        if str(terminal) not in received:raise ValueError('Unreceipted terminal')
        bind(terminal,received[str(terminal)]['sha256']);z=arrays(terminal);validate_terminal(z,c['frames'])
        tm=read(bind(PREFIX/'prefix/trajectories'/(px['trajectory_id']+'.json')))
        if tm['status']!='completed' or tm['terminal']!=descriptor(z['x_899']):raise ValueError('Terminal descriptor/status')
        inp=PREFIX/'prefix/inputs'/(c['case_id']+'.npz');bind(inp,received[str(inp)]['sha256'])
        metadata=read(bind(F/'samples'/(c['case_id']+'.json')))
        if len(metadata['random_plans'])!=1:raise ValueError('Single historical plan required')
        records=metadata['random_plans'][0]['noise_draws'];noise=suffix_noise(records)
        if tm['noise_draws']!=records[:101]:raise ValueError('Prefix and old suffix do not share the same plan')
        nid='SN'+str(len(noises)+1).zfill(3)
        noises.append(dict(noise_id=nid,case_id=c['random_plan_case_id'],draws=noise))
        starts.append(dict(panel_case_id=c['case_id'],prefix_trajectory_id=px['trajectory_id'],terminal=artifact(terminal),
            x_899=descriptor(z['x_899']),inputs=artifact(inp),frame_mask=descriptor(z['frame_mask'])))
        for condition in ['keep_paired','switch_imu_off']:
            sid='SX'+str(len(paths)+1).zfill(3)
            paths.append(dict(trajectory_id=sid,panel_case_id=c['case_id'],input_slot_id=c['slot_id'],
                data_split=c['data_split'],motion_id=c['motion_id'],sensor_config=c['sensor_config'],frames=c['frames'],
                checkpoint='F200',caption=c['text'],guidance=c['guidance'],condition=condition,
                text_present=True,imu_present=condition=='keep_paired',start=artifact(terminal),start_array='x_899',noise_id=nid,
                diffusion_steps=list(range(899,-1,-1)),trace_timesteps=[500,250,50],status='not_run'))
            if condition=='keep_paired':
                for t in [500,250,50]:
                    tr=next(r for r in oldtraces if r['case_id']==c['case_id'] and r['timestep']==t)
                    path=bind(F/'traces'/(tr['trace_id']+'.npz'));meta=bind(path.with_suffix('.json'))
                    anchors.append(dict(trajectory_id=sid,timestep=t,raw=artifact(path),metadata=artifact(meta),
                        arrays=['x_t','f00','f10','f01','f11','actual_combined_prediction']))
                path=bind(F/'samples'/(c['case_id']+'.npz'))
                anchors.append(dict(trajectory_id=sid,kind='final',raw=artifact(path),metadata=artifact(bind(path.with_suffix('.json'))),
                    arrays=['normalized_humanml263','joints','frame_mask'],shape_mapping='Use original sample layout. No new alignment.'))
    design=dict(version='b0-suffix-switch-design-v1',date='2026-10-08',status='CPU_frozen_not_implemented_not_executed',
        question='After the identical completed F paired prefix ending at x899, does removing IMU conditioning for the remaining 900 steps change the later contraction and complete motion?',
        rationale='Late CPU algebra shows small same-state checkpoint changes versus growing history terms in both orderings. Old text-only recovers more temporal energy but has a different history. Isolate suffix conditioning under a shared F history.',
        scope='Original train000506/wrists primary with head companion; exploratory mechanism intervention, not a new primary model selection or efficacy rule. All old six-source/12-config/seven-condition negative references retained in CPU evidence.',
        switch='One preselected boundary: load completed native float32 x899 after t900 posterior transition. Both arms load exactly the same bytes. Off arm sets imu_condition_present strictly false and imu_control zero from t899 onward; text stays true. No second switch, guidance sweep or selected checkpoint.',
        weights='F200 final frozen checkpoint only. F/U and original text-only final references remain descriptive; no U suffix or new cross-state probe added.',
        schedule='Original unrespaced 1000-step DDPM, iterate t899 through t0 with original float32 posterior order and clip_denoised=false. No initial/q noise. Old t899 through t1 noise draws must match address and bytes before use. No t0 noise draw.',
        native_paths='Caption batch1, factorized MDM batch4 f00/f10/f01/f11, original static-delta encoder; cache paired observed/reference separately per case, shared by both suffix arms.',
        software_gate='Keep-paired arm must reproduce three old trace groups and original final normalized features/joints/mask at inherited tolerances. First failure stops the entire queue. Failed or missing slots retained; no primary intervention interpretation on a failed baseline replay.',
        metrics='All existing 11 normalized feature views plus balanced, original temporal decomposition and final motion numerical/behavioral criteria. Report paired differences within config and time, no pooling frames/timesteps or percentages of mechanism.',
        outcomes='Recovery of temporal energy alone means dynamic recovery only; original complete alternating-lunge and return/order requirements determine behavior support. A failure to recover after switching cannot establish that earlier F history is irreversible or that other boundaries would fail.',
        review='Exposed primary/companion chosen from existing main cases after completed analyses; prospective intervention only, not blind preregistration or independent 02/human review. Prior F not_supported and paper claims remain unchanged.',
        forbidden=['training','new source/seed/caption/checkpoint','noise regeneration before worker','new threshold','adaptive boundary','automatic retry/resume','interference with other GPU processes','paper claim changes'])
    budget=dict(total_wall_seconds=1200,peak_cuda_allocated_cap_bytes=24*1024**3,
        trajectories=4,suffix_steps_per_trajectory=900,total_MDM_calls=3600,total_MDM_examples=14400,
        encoder_calls=4,encoder_examples=4,caption_calls=2,caption_examples=2,
        old_noise_regenerations=3596,distinct_old_noise_addresses=1798,new_initial_noise_draws=0,new_noise_seeds=0,
        final_motion_decodes=4,updates=0,automatic_retry=False,resume=False,
        admission='Standing user authorization, fixed UUID/project lock, 3 samples at 2 seconds, free>=4096MiB; utilization recorded without cutoff.')
    checks=dict(shared_start_bitwise=2,per_step_CFG=3600,strict_absent_branch_pairs=3600,
        historical_trace_array_checks=36,historical_final_numeric_array_checks=4,historical_final_mask_checks=2,
        planned_equations=7244,
        tolerances='Original bitwise mask/start/noise checks; old numerical replay abs(new-ref)<=1e-6+1e-6*abs(ref); existing float32 CFG bounds unchanged.',
        storage='Save native raw branch/state arrays before each CFG/absence/anchor check. Save posterior metadata and final features/joints/mask. Precreate all4 paths/3600 steps/12 saved traces/4 finals/7244 checks. Durable reserved/actual/completed call ledger.',
        CPU_acceptance_required=['Suffix tiny-fixture vs suffix of full1000 schedule with common x899/noise',
            'Strict absence/branch order/native batch and exact expected call/noise budgets',
            'All complete/missing slot denominators and raw-before-first-failure',
            'Old baseline replay/shape/tolerance and original decode layout',
            'Durable reservation ledger, project lock and absolute1200-second parent deadline'])
    payloads=dict(design=design,budgets=budget,software_checks=checks,trajectories=paths,starts=starts,noise=noises,historical_anchors=anchors,
        checkpoints=[dict(snapshot_id='F200',artifact=manifest['snapshots']['F200'])],sources=[sources[k] for k in sorted(sources)])
    output.mkdir(parents=True)
    for name,value in payloads.items():write(output/(name+'.json'),value)
    package=dict(version='b0-suffix-switch-package-v1',artifacts=[artifact(output/(n+'.json')) for n in payloads],
        source_count=len(sources),counts=dict(trajectories=4,steps=3600,saved_traces=12,finals=4,software_checks=7244),
        new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,
        next_task='Implement bounded suffix worker/supervisor and CPU acceptance before one fresh queue; standing GPU authorization applies.')
    write(output/'package.json',package);print(dict(package=artifact(output/'package.json'),sources=len(sources),counts=package['counts']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();freeze(a.output.resolve())
