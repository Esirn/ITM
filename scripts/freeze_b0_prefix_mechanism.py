#!/usr/bin/env python
"""CPU freeze of the smallest matched-noise early-trajectory mechanism panel."""
import argparse
from pathlib import Path

from evaluate_b0_feature_balance import FROZEN, ROOT, read, require, sha, write

OUT=ROOT/'docs/itm_rework/protocol/b0_prefix_mechanism_20261008_v1'
NATIVE=ROOT/'outputs/itm_rework/method/training_sufficiency_native_implementation_20261008_v1/implementation.json'
NATIVE_SHA='cb30fb6d035e96c129ea67c5924ee8ef4c35648a97cb2077c7385227b7a41f28'
TIMES=[999,975,950,925,900]
RUNS=dict(F=ROOT/'outputs/itm_rework/method/feature_balance_authorized_20261008_v1',
          U=ROOT/'outputs/itm_rework/method/sampling_alignment_authorized_20260927_v1')


def prefix_noise(records):
    """Initial draw plus every frozen posterior draw for t999 through t900."""
    require(len(records)==1000 and [r['step'] for r in records]==[-1]+list(range(999,0,-1)),
            'Complete old addressed-noise order required')
    selected=records[:101]
    require([r['step'] for r in selected]==[-1]+list(range(999,899,-1)),'100-step prefix')
    return selected


def freeze(output):
    require(not output.exists(),'Refusing to overwrite mechanism package')
    require(sha(NATIVE)==NATIVE_SHA,'Completed native implementation identity')
    native=read(NATIVE);bound={}
    def bind(p,expected=None):
        p=(ROOT/p).resolve();h=sha(p)
        require(expected is None or h==expected,'Dependency changed: '+str(p))
        bound[str(p)]=dict(path=str(p),sha256=h);return p
    for e in native['expected_artifacts']:bind(Path(e['path']),e['sha256'])
    for p in [NATIVE,Path(__file__),ROOT/'tests/test_prefix_mechanism_freeze.py',
              ROOT/'scripts/analyze_b0_saved_trace_responses.py',ROOT/'src/itm/metrics/trace_response.py',
              ROOT/'scripts/verify_b0_saved_trace_responses.py',ROOT/'scripts/summarize_b0_saved_trace_responses.py',
              ROOT/'docs/itm_rework/protocol/b0_saved_trace_response_20261008_v1/plan.json',
              ROOT/'outputs/itm_rework/method/saved_trace_response_verification_20261008_v1/verification.json']:
        bind(p)
    for folder in ['saved_trace_response_evaluation_20261008_v1','saved_trace_response_panel_20261008_v1']:
        index=bind(ROOT/'outputs/itm_rework/protocol'/folder/'artifact_index.json')
        for e in read(index):bind(Path(e['path']),e['sha256'])
    verification=read(ROOT/'outputs/itm_rework/method/saved_trace_response_verification_20261008_v1/verification.json')
    require(verification['status']=='passed' and verification['trace_points']==480,'Complete CPU analysis gate')
    cases=read(bind(FROZEN/'sampling_slots.json'));old_traces=read(bind(FROZEN/'trace_slots.json'))
    paired=[c for c in cases if c['condition']=='paired'];require(len(paired)==12,'All paired configurations')
    trajectories=[];states=[];probes=[];noise=[];anchors=[]
    for ci,c in enumerate(paired):
        text=next(x for x in cases if x['data_split']==c['data_split'] and x['motion_id']==c['motion_id']
                  and x['sensor_config']==c['sensor_config'] and x['condition']=='text_only')
        records={}
        for arm,slot in [('F',c),('U',c),('F',text),('U',text)]:
            cid=slot['case_id'] if arm=='F' else slot['historical_case_id']
            p=bind(RUNS[arm]/'samples'/(cid+'.json'));meta=read(p)
            require(len(meta['random_plans'])==1,'Single historical addressed plan')
            records[(arm,slot['condition'])]=prefix_noise(meta['random_plans'][0]['noise_draws'])
        original=records[('F','paired')]
        require(all(v==original for v in records.values()),'Cross-arm/condition noise addresses and bytes differ')
        noise_id=f'PN{ci+1:03d}'
        noise.append(dict(noise_id=noise_id,case_id=c['random_plan_case_id'],draws=original,
            regeneration='Same 101 historical addresses/hashes, not a new seed/noise plan'))
        for arm in ['F','U','T']:
            tid=f'PX{len(trajectories)+1:03d}'
            row=dict(trajectory_id=tid,panel_case_id=c['case_id'],path_arm=arm,
                checkpoint='F200' if arm in ('F','T') else 'U200',
                condition='text_only' if arm=='T' else 'paired',text_present=True,imu_present=arm!='T',
                noise_id=noise_id,frames=c['frames'],sensor_config=c['sensor_config'],
                data_split=c['data_split'],motion_id=c['motion_id'],
                caption=c['text'],guidance=c['guidance'],input_slot_id=(text if arm=='T' else c)['slot_id'],
                diffusion_steps=list(range(999,899,-1)),trace_timesteps=TIMES,
                terminal='x_899 after the t900 posterior transition; not a generated motion',status='not_run')
            trajectories.append(row)
            for t in TIMES:
                sid=f'PS{len(states)+1:03d}'
                states.append(dict(state_id=sid,trajectory_id=tid,timestep=t,
                    x_t='Native float32 pre-denoise state, before the corresponding posterior update',
                    shape=[1,263,1,c['frames']],status='not_run'))
                for weight in ['F200','U200']:
                    probes.append(dict(probe_id=f'PC{len(probes)+1:03d}',state_id=sid,
                        checkpoint=weight,MDM_batch=4,branch_order=['f00','f10','f01','f11'],
                        delivered_condition='Original real caption and paired IMU for this panel case',status='not_run'))
            historical_case=(text if arm=='T' else c)['case_id']
            trace=next(x for x in old_traces if x['case_id']==historical_case and x['timestep']==900)
            for old_arm in (['F','U'] if arm=='T' else [arm]):
                old_id=trace['trace_id'] if old_arm=='F' else trace['historical_trace_id']
                npz=bind(RUNS[old_arm]/'traces'/(old_id+'.npz'));meta=bind(npz.with_suffix('.json'))
                anchors.append(dict(trajectory_id=tid,timestep=900,historical_arm=old_arm,
                    npz=dict(path=str(npz),sha256=sha(npz)),metadata=dict(path=str(meta),sha256=sha(meta)),
                    arrays=['x_t','f00','f10','f01','f11','actual_combined_prediction'],
                    tolerance='Inherited abs(new-ref) <= 1e-6 + 1e-6*abs(ref); record bitwise separately'))
    require((len(trajectories),len(states),len(probes),len(anchors))==(36,180,360,48),'Frozen denominator')
    design=dict(version='b0-prefix-mechanism-v1',date='2026-10-08',
        rationale='Main train000506/wrists F already contracts by first old trace t900. '
            'At t50 the local IMU energy change is positive; final paired vs text-only paths differ. '
            'Observe the previously missing first 100 steps under identical stochastic inputs.',
        question='Does the F paired prefix reproducibly diverge from U and text-only before t900, '
            'and do same-state factorized responses versus accumulated state changes describe that divergence?',
        interpretation='Matched-noise controlled prefix, conditional on this fixed caption/seed/panel. '
            'No suffix intervention, unique causal mechanism, behavior repair, convergence or generalization claim.',
        sources='Six original sources x head/wrists; original paired cases and common text-only baseline. '
            'All prior seven-condition/negative panels remain in CPU evidence; this queue has no new numeric-zero/reverse/IMU-only trajectories.',
        paths='F200 paired, U200 paired, common text-only via F200 with strict IMU false; '
            'same backbone and inactive-adapter full outputs already bitwise equal across F/U in 30 original slots.',
        identity='Original final200 weights only; native caption batch1 and MDM factorized batch4. '
            'Full1000 DDPM schedule, original betas/timestep map; truncate loop only after t900, clip_denoised=false. '
            'No rescaling, respacing, precision switch, adaptive endpoint or extra diffusion pass.',
        matched_noise='Verify every regenerated draw against old address/noise hash before using it. '
            'Use same initial and per-step noise bytes across all three paths per case. '
            'Native posterior mean/log-variance/update order exactly matches old sampler.',
        same_state='180 saved prefix x_t states x both F200/U200 weights x four factorized branches. '
            'Caption/IMU/t/mask/precision identical within each weight contrast; no q_sample.',
        metrics='Reuse saved trace temporal-energy/response decompositions and original normalized feature MSE '
            'with fixed-time GT reference. Report all11 views. Equal source means/min/max/n within '
            'split/config/path/t; do not pool times. No new behavioral threshold or checkpoint selector.',
        reporting='All planned slots precreated as not_run; first fault stops entire queue. '
            'Preserve raw predictions before checks, reserve/actual/completed counters and missing denominators. '
            'Any failed historical replay prevents primary mechanistic interpretation.',
        forbidden=['training','new checkpoint','loss/guidance sweep','new seed/source/caption',
            'sampling below t900','full motion decode','paper claim changes','interference with other GPU processes'],
        prior_completed_implementation=dict(path=str(NATIVE),sha256=NATIVE_SHA),
        checkpoint_sources={k:next(x for x in native['snapshots'] if x['snapshot_id']==k) for k in ['F200','U200']},
        scientific_result_status='not_run; package freeze is not implementation acceptance or execution')
    budgets=dict(total_wall_seconds=600,peak_cuda_allocated_cap_bytes=24*1024**3,
        prefix_MDM_calls=3600,prefix_MDM_examples=14400,
        fixed_state_MDM_calls=360,fixed_state_MDM_examples=1440,
        total_MDM_calls=3960,total_MDM_examples=15840,
        encoder_calls=48,encoder_examples=48,
        encoder_cache='2 weights x12 cases x (observed, reference) separately batch1; reusable across paths/state probes',
        caption_calls=12,caption_examples=12,caption_cache='Native batch1, share per case across fixed weights and paths',
        regenerated_noise_draws=3636,distinct_old_noise_addresses=1212,
        checkpoint_loads='Only F200 and U200; CPU identity then exact frozen reconstruction',
        GPU_policy='Standing user discretion; 3 shared-resource samples free>=4096MiB, GPU/controller<=50%, '
            'UUID isolation and project lock; no wait for completely idle GPU',
        resume=False,automatic_retry=False)
    checks=dict(shared_initial_states=12,historical_state_anchor_groups=48,historical_array_checks=288,
        same_state_own_prediction_array_checks=720,
        same_state_inactive_adapter_cross_weight_checks=360,
        core_equations=1380,
        per_prefix_step_CFG_arithmetic=3600,text_only_absent_branch_pairs=2400,
        tolerances='Initial noise identity bitwise; remaining software references inherit 1e-6 absolute +1e-6 relative. '
            'CFG float32 and temporal float64 bounds unchanged.',
        CPU_acceptance_required=['100-step tiny-fixture posterior replay vs full-loop prefix',
            'shared addressed-noise hashes and native call budgets',
            't900 anchor stop-on-mismatch and preserved unrun denominators',
            'cache flags/branch order and no full motion decode',
            'durable reservation journal, hard parent deadline, exclusive lock and artifact completeness'])
    output.mkdir(parents=True)
    files={'design':design,'budgets':budgets,'software_checks':checks,'trajectories':trajectories,
           'states':states,'probes':probes,'noise':noise,'historical_anchors':anchors,
           'sources':[bound[k] for k in sorted(bound)]}
    for name,v in files.items():write(output/(name+'.json'),v)
    write(output/'package.json',dict(version='b0-prefix-mechanism-package-v1',
        artifacts=[dict(path=str((output/(name+'.json')).resolve()),sha256=sha(output/(name+'.json'))) for name in files],
        counts=dict(panel_cases=12,trajectories=36,saved_states=180,same_state_jobs=360,software_core_checks=1380),
        source_count=len(bound),model_constructions=0,new_GPU_queries=0,new_model_calls=0,new_noise=0,new_motion_decodes=0))
    print(dict(package=str(output/'package.json'),sha256=sha(output/'package.json'),source_count=len(bound),
               trajectories=36,saved_states=180,same_state_jobs=360,new_model_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();freeze(a.output.resolve())
