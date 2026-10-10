#!/usr/bin/env python
"""Freeze a minimal same-state control-mean-centering probe; no GPU/models/noise."""
import argparse
from collections import Counter
from pathlib import Path
from evaluate_b0_native_training_sufficiency import read,verify,artifact,arrays,descriptor,write

ROOT=Path(__file__).resolve().parents[1]
CPU=ROOT/'outputs/itm_rework/method/cross_source_component_implementation_20261009_v2'
RESULTS=ROOT/'outputs/itm_rework/protocol/cross_source_component_evaluation_20261009_v1'
PREFIX=ROOT/'outputs/itm_rework/method/prefix_mechanism_authorized_20261008_v3/prefix'
F200='d0e4f39f8c7ee5b980243baac48bc1bc02dc8f7c15c7a1f47f0adabb0b7e0ad3'


def select_states(rows):
    selected=[r for r in rows if r['arm']=='F' and r['condition']=='paired']
    if len(selected)!=48 or any(r['status']!='valid' for r in selected):raise ValueError('Full48 F paired states required')
    keys={(r['pair_case_id'],r['timestep']) for r in selected};cases={r['pair_case_id'] for r in selected}
    if len(cases)!=12 or len(keys)!=48 or keys!={(c,t) for c in cases for t in (900,500,250,50)}:raise ValueError('All12 configs/four original times required')
    return selected


def build():
    manifest=read(CPU/'implementation.json');proof=read(CPU/'result_verification.json')
    if proof['status']!='passed' or proof['rows']!=480 or proof['F_U_pairs']!=240:raise ValueError('Complete all-source CPU proof required')
    verify(proof['results_index']);verify(proof['verifier'])
    for a in read(RESULTS/'artifact_index.json'):verify(a)
    rows=select_states(read(RESULTS/'rows.json'))
    starts=read(ROOT/'outputs/itm_rework/method/prefix_mechanism_authorized_20261008_v3/runtime/status.json')
    if starts['status']!='completed':raise ValueError('Complete prefix cache provenance required')
    prefix_proof=read(ROOT/'outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v3/result_verification.json')
    if prefix_proof['status']!='passed':raise ValueError('Complete prefix numerical proof required')
    slots={r['case_id']:r for r in read(ROOT/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/sampling_slots.json')}
    cases=[];jobs=[]
    for i,cid in enumerate(sorted({r['pair_case_id'] for r in rows}),1):
        rs=[r for r in rows if r['pair_case_id']==cid];r=rs[0];slot=slots[cid]
        cache=PREFIX/'cache'/('F200-'+cid+'.npz');metadata=cache.with_suffix('.json');meta=read(metadata);z=arrays(cache)
        if meta['case_id']!=cid or meta['checkpoint']['sha256']!=F200 or meta['control']!=descriptor(z['control']) or z['control'].shape!=(1,r['frames'],512) or str(z['control'].dtype)!='float32':raise ValueError('Exact original F200/control cache required')
        case=dict(case_id=cid,data_split=r['data_split'],motion_id=r['motion_id'],sensor_config=r['sensor_config'],frames=r['frames'],input_slot_id=slot['slot_id'],caption=slot['text'],checkpoint='F200',text_present=True,imu_present=True,control_before=artifact(cache),control_metadata=artifact(metadata),original_caption_cache=artifact(PREFIX/'cache'/('text-'+cid+'.npz')),guidance=dict(text_scale=1.,imu_scale=1.,joint_scale=1.,mode='factorized',branch_execution='batched'))
        cases.append(case)
        for k,t in enumerate((900,500,250,50),1):
            source=next(x for x in rs if x['timestep']==t);jobs.append(dict(row_id=f'CM{i:02d}-{t}',case_id=cid,timestep=t,frames=r['frames'],source_trace_id=source['trace_id'],source_raw=source['raw'],source_metadata=source['metadata'],source_array='x_t',source_arm='F',source_condition='paired'))
    sources={a['path']:a for a in manifest['expected_artifacts']}
    paths=[CPU/'implementation.json',CPU/'result_verification.json',Path(__file__).resolve(),ROOT/'tests/test_control_mean_probe_freeze.py',ROOT/'docs/itm_rework/GPU_DEVICE_PREFERENCE_20261008.json',ROOT/'docs/itm_rework/GPU_CONTENDED_EXECUTION_POLICY_20261008.json',ROOT/'outputs/itm_rework/method/prefix_mechanism_authorized_20261008_v3/runtime/status.json',ROOT/'outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v3/result_verification.json',*RESULTS.iterdir()]
    paths.extend((ROOT/'outputs/itm_rework/protocol/cross_source_component_panel_20261009_v1').iterdir())
    for case in cases:
        paths.extend(verify(case[k]) for k in ('control_before','control_metadata','original_caption_cache'))
    for job in jobs:
        paths.extend(verify(job[k]) for k in ('source_raw','source_metadata'))
    for p in paths:
        if p.is_file():a=artifact(p);sources[a['path']]=a
    for a in sources.values():verify(a)
    return dict(version='b0-control-temporal-mean-fixed-state-probe-v1',date='2026-10-09',status='CPU_frozen_not_implemented_not_executed',
        question='Does removing only the temporal-constant component of the original F200 static-delta latent control alter source-specific mean-pose bias while retaining temporally varying control at the same saved states?',
        rationale='All-source audit finds F main-term MSE improvement at all24 train paired states but worsening at17/24 val states; mean-pose error increases at20/24 val states, with temporal-error signs varying by source. Whole-main knockout sacrifices accuracy and behavior. A48-forward same-state computational control test is smaller than another rollout/training run; it does not establish that latent mean is nuisance or uniquely causes output mean bias.',
        intervention='F200/eval/original encoder/static-delta control and raw IMU normalization unchanged. First compute original float32 C=encoder(observed)-encoder(physical_static_reference) with original real-frame mask. Then native float32 temporal mean mu_C=(C*mask).sum(time)/mask.sum(time), shape(1,1,512); replace only valid-frame control with C_centered=C-mu_C, zero originally invalid slots. All frozen masks are full real frames. Do not center raw IMU, GT, predicted motions or fit values. Retain text/IMU presence true and factorized(1,1,1), caption batch1/MDMbatch4.',
        population=dict(cases=12,sources=6,states=48,timesteps=[900,500,250,50],checkpoint='F200',source_arm='F',source_condition='paired'),cases=cases,jobs=jobs,
        checkpoint_sha256=F200,checkpoint_artifact=meta['checkpoint'],normalization=read(verify(manifest['plan']))['normalization'],
        software_checks=dict(total=240,original_state_copy=48,original_f00_f10_anchors=96,native_CFG=48,original_unmodified_cache=12,centered_valid_mean=12,control_reconstruction=12,zero_control_preservation=12),
        software_gate='Before modifying cache, original control/caption/input identities match bound completed prefix sources at inherited1e-6+1e-6*abs(reference), original x_t copied bitwise. Every new row f00/f10 equals original native raw at unchanged1e-6 abs/relative; raw arrays saved before first failed check stops entire queue. Centered valid control mean near zero and original=mean+centered at same original tolerance; exact-zero null control stays bitwise zero. Original float32 CFG rounding bound retained. No widening, retries, resume, added controls/model calls or budget resets.',
        metrics='All original11 views/balanced temporal/same_state_delta against each source original returned prediction and GT, separate mean-pose and temporal errors, signed feature/energy/error changes and response covariances. Source/config/time retained; compare sameF200/samex_t with one control transformation. Hold/mean information can be useful: retain all six sources and negative/zero controls. No decode, final motion metric or semantic success is measured by these prediction probes.',
        references='Original480 F/U traces/120 finals/seven conditions and original F/not_supported plus completed main-knockout negatives stay bound references, not new same-state U probes or new population entries.',
        budget=dict(MDM_calls=48,MDM_examples=192,encoder_calls=24,encoder_examples=24,caption_calls=12,caption_examples=12,new_noise_draws=0,new_sampling_steps=0,new_decodes=0,new_updates=0,total_wall_seconds=600,peak_cuda_allocated_cap_bytes=24*1024**3,automatic_retry=False,resume=False),
        GPU=dict(preferred_physical_device='cuda:1',uuid='GPU-43c2f925-db05-c655-bb46-25196d0627d6',admission='Standing user delegation;3 samples2sec apart/free>=4096MiB, utilization recorded without cutoff; UUID project lock/native allocator; fixed whole-attempt deadline/first-fault stop; no interference.'),
        interpretation='Prospective fixed-state computational test chosen after exposed audit; no independent/human/blind validation, unique mechanism allocation, pose-fitting, model efficacy or rollout recovery claim. Latent-control DC and decoded pose mean need not correspond linearly.',
        prohibited=['training','sampling/noise/decoding','new source/seed/caption/checkpoint/time selection','guidance/centering-factor sweep','retry/resume','interference with other GPU jobs','paper claims changed'],sources=[sources[k] for k in sorted(sources)],new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,
        next_task='Implement/CPU-accept this48-state bounded probe, then one new cuda:1-preferred600s queue using standing authorization and complete CPU evaluation. No GPU/model run exists yet in this plan.')


def freeze(output):
    if output.exists():raise FileExistsError('Fresh scientific probe plan required')
    value=build();output.mkdir(parents=True);write(output/'plan.json',value);print(dict(plan=artifact(output/'plan.json'),sources=len(value['sources']),cases=12,states=48,MDM_calls=48,checks=240),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();freeze(a.output.resolve())
