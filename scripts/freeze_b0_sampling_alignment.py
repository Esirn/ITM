#!/usr/bin/env python
"""02: freeze fixed-R guidance experiment using only files and NumPy; no model calls."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REL='docs/itm_rework/protocol/b0_sampling_alignment_20260927_v1'
OLD=ROOT/'docs/itm_rework/protocol/b0_correction_20260926_v1'
EVAL=ROOT/'docs/itm_rework/protocol/b0_correction_evaluation_20260927_v1'
PROPOSAL=ROOT/'docs/itm_rework/method/b0_sampling_alignment_20260927_v1'
RUN=ROOT/'outputs/itm_rework/method/correction_authorized_20260927_v2'
R='R_static_reference_delta';S='S1_R_unit_guidance'
CHECKPOINT_SHA='91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642'
BRANCHES=['f00','f10','f11_paired','f11_static','f11_reversed']
TRACE_T=[900,500,250,50]

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def artifact(p):
    p=Path(p).resolve();return dict(path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),sha256=sha(p))

def reduced_analysis(f00,f10,f11,target):
    """Float64 conceptual guidance comparison; Q has no f01, so no float32 CFG claim."""
    f00,f10,f11,target=[np.asarray(v,dtype=np.float64) for v in (f00,f10,f11,target)]
    if len({v.shape for v in (f00,f10,f11,target)})!=1 or f11.size==0:raise ValueError('Equal nonempty shapes required')
    if not all(np.isfinite(v).all() for v in (f00,f10,f11,target)):raise ValueError('Nonfinite branch')
    a=f11-target;b=1.5*(f10-f00)
    unit=float(np.square(a).mean());legacy=float(np.square(a+b).mean())
    cross=float(2*np.mean(a*b));energy=float(np.square(b).mean());delta=legacy-unit
    bound=64*np.finfo(np.float64).eps*max(1.,unit,legacy,abs(cross),energy)
    return dict(unit_mse=unit,legacy_reduced_mse=legacy,legacy_minus_unit=delta,
                twice_error_text_inner_product=cross,text_energy=energy,
                decomposition_error=abs(delta-cross-energy),decomposition_bound=bound)

def combination_bound(branches,wt):
    """Frozen absolute forward-error allowance for existing float32 four-branch expression."""
    a,b,c,d=[np.asarray(v,dtype=np.float64) for v in branches]
    u=2.**-24;gamma16=16*u/(1-16*u)
    # Sum of weighted absolute input contributions to all four terms, before cancellation.
    scale=np.abs(a)+abs(wt)*(np.abs(b)+np.abs(a))+(np.abs(c)+np.abs(a))+(np.abs(d)+np.abs(b)+np.abs(c)+np.abs(a))
    return gamma16*np.maximum(1.,scale)

def build(output):
    if output.exists():raise FileExistsError('Choose a new directory; frozen artifacts are immutable')
    sources={}
    def bind(p,expected=None):
        p=(ROOT/Path(p)).resolve()
        if str(p) not in sources:sources[str(p)]=artifact(p)
        result=sources[str(p)]
        if expected is not None:assert result['sha256']==expected,str(p)
        return result
    def verify(entry):return bind(entry['path'],entry['sha256'])
    for a in read(PROPOSAL/'sources.json'):verify(a)
    for p in sorted(PROPOSAL.iterdir()):
        if p.is_file():bind(p)
    for p in [Path(__file__),ROOT/'tests/test_b0_sampling_alignment_protocol.py',ROOT/'docs/itm_rework/method/B0_SAMPLING_ALIGNMENT_PROPOSAL_20260927.md',ROOT/'docs/itm_rework/protocol/B0_CORRECTION_EVALUATION_20260927.md',EVAL/'samples.json',EVAL/'contrasts.json',EVAL/'review_notes_hidden.txt',EVAL/'review_unmasked.json',ROOT/'src/itm/metrics/canonical_b0.py',ROOT/'src/itm/models/mdm_sampling.py',ROOT/'src/itm/models/mdm_imu_control.py',RUN/'runtime/status.json']:
        bind(p)
    package=read(OLD/'package.json')
    for a in package['artifacts']:verify(a)
    status=read(RUN/'runtime/status.json');assert status['status']=='completed'
    implementation=read(status['implementation']['path']);verify(status['implementation'])
    for a in implementation['expected_artifacts']:verify(a)
    fixed=bind(RUN/R/'model.pt',CHECKPOINT_SHA)
    old_design=read(OLD/'design.json');budget=read(PROPOSAL/'proposed_budget.json')
    assert budget['existing_checkpoint']['sha256']==CHECKPOINT_SHA and not budget['authorized']
    old_samples=[r for r in read(OLD/'sampling_slots.json') if r['arm']==R]
    proposed=read(PROPOSAL/'proposed_sampling_slots.json');assert len(old_samples)==len(proposed)==60
    old_states={r['case_id']:r for r in status['samples']}
    samples=[];history=[];noise_groups={};input_keys=['acceleration','orientation','normalized_imu','normalized_reference_imu','original_orientation0','sensor_mask','timestamps_seconds','frame_mask','gt','text_condition_present','imu_condition_present']
    for i,(p,o) in enumerate(zip(proposed,old_samples)):
        assert p['comparison_case_id']==o['case_id'] and p['arm']==S
        changed={'arm','case_id','slot_index','status','checkpoint_selection'}
        for k,v in o.items():
            if k not in changed:assert p[k]==v,(o['case_id'],k)
        oldpath=RUN/'samples'/f"{o['case_id']}.npz";meta=read(oldpath.with_suffix('.json'))
        assert meta['control_checkpoint_sha256']==CHECKPOINT_SHA
        assert meta['guidance']==dict(text_scale=2.5,imu_scale=1.,joint_scale=1.,mode='factorized',branch_execution='batched')
        assert old_states[o['case_id']]['status']=='completed'
        for a in old_states[o['case_id']]['artifacts']+[old_states[o['case_id']]['receipt']]:verify(a)
        for k,v in o.items():
            if k!='status':assert meta[k]==v,k
        assert meta['training_input_provenance']==old_design['common']['training_input_provenance']
        plan=meta['random_plans'][0];group=o['random_plan_case_id']
        if group in noise_groups:assert noise_groups[group]['plan']==plan
        else:
            assert len(plan['noise_draws'])==1000 and [d['step'] for d in plan['noise_draws']]==[-1]+list(range(999,0,-1))
            assert plan['timestep_map']==list(range(1000)) and plan['clip_denoised'] is False
            for d in plan['noise_draws']:
                address={k:d[k] for k in ('version','backbone_id','case_id','sample_seed','step','shape')};h=digest(address)
                assert address['case_id']==group and address['sample_seed']==20260926 and address['shape']==[1,263,1,o['frames']]
                assert address['backbone_id']==old_design['common']['backbone']['sha256'] and h==d['address_sha256']
                noise=np.random.Generator(np.random.PCG64(int(h,16))).standard_normal(address['shape']).astype(np.float32)
                assert hashlib.sha256(noise.tobytes()).hexdigest()==d['noise_sha256']
            assert digest(plan['noise_draws'])==plan['random_plan_id']
            noise_groups[group]=dict(plan=plan,metadata=bind(oldpath.with_suffix('.json')),frames=o['frames'])
        with np.load(oldpath,allow_pickle=False) as z:
            assert json.loads(str(z['metadata']))==meta
            assert z['motion'].shape==(1,o['frames'],22,3)
            input_hashes={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype),sha256=hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest()) for k in input_keys}
        row=dict(p,status='frozen_not_executed',slot_index=i,checkpoint=fixed,
                 historical_output=bind(oldpath),historical_metadata=bind(oldpath.with_suffix('.json')),
                 historical_random_plan_sha256=digest(plan),historical_input_hashes=input_hashes,
                 trace_timesteps=TRACE_T,trace_policy='same actual forward, before posterior update; no extra calls')
        samples.append(row)
        history.append(dict(comparison_case_id=o['case_id'],new_case_id=row['case_id'],spec=o,
                            output=row['historical_output'],metadata=row['historical_metadata'],guidance=meta['guidance'],
                            random_plan_id=plan['random_plan_id'],checkpoint=fixed))
    groups=[r for r in read(OLD/'probe_slots.json') if r['context']=='exact_length' and r['condition']=='paired']
    assert len(groups)==96
    proposed_q=read(PROPOSAL/'proposed_probe_slots.json');assert len(proposed_q)==480
    old_noise=read(OLD/'probe_noise.json');noise={};probes=[];qgroups=[]
    for gi,g in enumerate(groups):
        draw=old_noise[g['noise_key']]['valid_noise'];h=digest({k:draw[k] for k in ('version','backbone_id','case_id','sample_seed','step','shape')})
        assert h==draw['address_sha256'];value=np.random.Generator(np.random.PCG64(int(h,16))).standard_normal(draw['shape']).astype(np.float32)
        assert hashlib.sha256(value.tobytes()).hexdigest()==draw['noise_sha256']
        noise[g['noise_key']]=dict(valid_noise=draw,padding_noise=None)
        ids=[]
        for j,branch in enumerate(BRANCHES):
            q=dict(proposed_q[gi*5+j]);assert q['branch']==branch and q['noise_key']==g['noise_key']
            for k in ['split','motion_id','sensor_config','frames','timestep','noise_seed','text','caption_index','target_feature_path','context','context_frames']:
                assert q[k]==g[k],k
            assert q['context']=='exact_length' and q['context_frames']==q['frames']
            q.update(status='frozen_not_executed',group_index=gi,branch_index=j,checkpoint=fixed,
                     input_transform=q['condition'] if j>=2 else 'paired',expected_MDM_batch_shape=[5,263,1,q['frames']])
            probes.append(q);ids.append(q['probe_id'])
        qgroups.append(dict(group_index=gi,group_id=g['noise_key'],probe_ids=ids,branch_order=BRANCHES,
                            data_split=g['split'],motion_id=g['motion_id'],sensor_config=g['sensor_config'],timestep=g['timestep'],
                            noise_seed=g['noise_seed'],frames=g['frames'],MDM_branch_examples=5,encoder_examples_max=6))
    trace=[dict(trace_id=f'T{i*4+j+1:04d}',case_id=s['case_id'],slot_index=i,timestep=t,
                branch_order=['f00','f10','f01','f11'],shape=[1,263,1,s['frames']],status='frozen_not_executed',
                checkpoint=fixed,additional_MDM_branch_examples=0)
           for i,s in enumerate(samples) for j,t in enumerate(TRACE_T)]
    # Both historical and new outputs re-reviewed in one frozen order, without replacing old notes.
    review_population=[dict(case_id=r['case_id'],arm=S,data_split=r['data_split'],motion_id=r['motion_id'],sensor_config=r['sensor_config'],condition=r['condition']) for r in samples]+[dict(case_id=r['case_id'],arm=R,data_split=r['data_split'],motion_id=r['motion_id'],sensor_config=r['sensor_config'],condition=r['condition']) for r in old_samples]
    order=np.random.Generator(np.random.PCG64(20260930)).permutation(120)
    review=dict(seed=20260930,algorithm='NumPy PCG64 permutation of [new60 in slot order, historical60 in slot order]',
                rows=[dict(review_id=f'S{i+1:03d}',**review_population[int(j)]) for i,j in enumerate(order)],
                limits='Hide arm/config/condition in figures; preserve all notes/uncertain before unmasking. Old results already known; do not claim fully blinded or independent human assessment.')
    tolerances=dict(purpose='Software arithmetic and identity only; no effectiveness cutoff',
        hashes_masks_ids_counts='exact; no tolerance',
        inputs=dict(acceleration='exact',orientation_normalized_imu_reference=dict(atol=2e-7,rtol=0),
                    original_orientation0='from original observation; no GT calibration'),
        RIC_decode=dict(atol_m=2e-6,rtol=1e-6),
        Q_static_vs_text=dict(atol=1e-6,rtol=1e-6),
        trace_absent_branch_equalities=dict(atol=1e-6,rtol=1e-6,description='Absent text: f00=f10 and f01=f11; absent IMU, bypass or R-static: f00=f01 and f10=f11. Save bitwise/max/RMS.'),
        trace_return='Observer attaches no replacement tensor; trace enabled/disabled CPU fixtures bitwise identical outputs AND RNG AND cache/call counters. Save returned prediction and same-call combined tensor with identical hashes.',
        float32_combination=dict(unit_roundoff=2**-24,gamma_ops=16,
            absolute_bound='gamma16*max(1,abs(f00)+abs(wt)*(abs(f10)+abs(f00))+(abs(f01)+abs(f00))+(abs(f11)+abs(f10)+abs(f01)+abs(f00))) elementwise',
            reference='Convert stored float32 branches to float64. Compare actual combined prediction to factorized float64 expression and reduced f11+(wt-1)*(f10-f00). No requirement that unit==f11 bitwise.',
            reporting='All max/RMS differences plus maximum absolute-error/bound; finite bounds required; frozen scale-dependent rounding bound, never widened from observations.'),
        float64_MSE_identity=dict(absolute_bound='64*eps_float64*max(1,MSE_legacy_reduced,MSE_unit,abs(2mean(a*b)),mean(b*b))',
            reference='a=float64(f11)-float64(target), b=1.5*(float64(f10)-float64(f00)); all reductions float64 over valid entries'),
        full_sample=dict(max_abs_m=1e-5,comparisons=['new text_only vs new backbone_bypass: val6','new R-static vs new text_only: all12','new IMU-only vs historical IMU-only: val6'],
            report='All24 checks, bitwise and max/RMS; same exact seeds/inputs, no coordinate fit. Exceedance is replay/software discrepancy, not quality; halt later slots and retain complete ledger.'),
        nonfinite='Any nonfinite saved/returned tensor or required reduction stops the queue; finite poor effects do not stop or adapt the candidate')
    design=dict(version='canonical-B0-sampling-alignment-design-v1',status='frozen_for_implementation_not_authorized',read_method_commit='d2d28fb',
        public_contract='itm-rework-v0.3-draft unchanged; no new data admission',scope='Fixed-R sampling intervention on exposed development panel',
        checkpoint=fixed,checkpoint_arm=R,new_checkpoints=0,training_updates=0,
        old_package=bind(OLD/'package.json'),old_implementation=verify(status['implementation']),
        common=dict(backbone=old_design['common']['backbone'],control_reference_mode='static_delta',
            imu_normalization=old_design['common']['imu_normalization'],training_input_provenance=old_design['common']['training_input_provenance'],
            motion_normalization=old_design['common']['motion_normalization'],environment=implementation['environment'],
            execution='all modules eval, float32, no autograd/AMP/optimizer/dropout; same backend as historical R; same input, reference, cache and motion decoding',
            input_transforms=old_design['input_transforms']),
        Q=dict(groups=96,branches_per_group=5,MDM_branch_examples=480,encoder_examples_max=576,
            batch='Exactly one same-length MDM batch of5 per group, in probe_groups.json order; branch examples count5, not1. No mixing groups, padding or sequential branch fallback.',
            conditions='f00/f10 use exact null cached control and false IMU presence, zero encoder calls; f00 text embedding zero. For paired/static/reversed compute observed and original-frame0 reference embeddings in eval, subtract with shared R encoder; max2 encodings per condition. Precache then form batch5.',
            x_t='Same GT normalization, original diffusion betas/timestep_map/scaled t and valid noise as old exact D. No padding noise consumed. Replicate one x_t five times, save/hash target/noise/x_t/mask.',
            output='Per480 slot save float32 f prediction [263,1,T], target, x_t, frame_mask, branch/source/config/t/noise/checkpoint/input/reference identities and actual scaled t; group5 ordering persisted. Float64 count/finite_count/MSE/RMS/max. Raw inputs/reference can use checked shared hash references.',
            composition='Per96 group and3 IMU conditions: conceptual float64 unit=f11, legacy_reduced=f11+1.5*(f10-f00), a,b, cross and b-energy. Q intentionally lacks f01: do not report actual float32 four-branch replay or add f01 calls. Static f11 vs f10 checked on same batch.',
            decision='Finite Q effect never changes S1 parameters or selects a model; software/identity/nonfinite/budget discrepancy stops remaining queue.'),
        S1=dict(guidance=dict(text_scale=1.,imu_scale=1.,joint_scale=1.,mode='factorized',branch_execution='batched'),
            historical_guidance=dict(text_scale=2.5,imu_scale=1.,joint_scale=1.,mode='factorized',branch_execution='batched'),
            only_intervention='Guidance scalar wt:2.5->1; no reduced-expression fast path or altered null/bypass execution. New versioned entrypoint, old D/L/R semantics unchanged.',
            sampling='60 fixed full samples, K1,1000-step DDPM,float32,clip_denoised=false, timestep_map=0..999; old12 noise groups preserved. Strict IMU-only gates text off despite audit caption.',
            reference='numeric_zero uses normalized zeros but reference retains original observed O[0]; static equals reference; tail-reversal AFTER 20Hz interpolation [0,T-1,...,1], not a physical donor.',
            output='Same120-independent-evaluation compatible arrays and identities as oldR, with new case/arm/package/guidance and explicit comparison_case_id + old artifacts; no metadata relabeling of old results.'),
        trace=dict(slots=240,timesteps_in_visit_order=TRACE_T,additional_MDM_calls=0,additional_encoder_calls=0,
            attach='Observe the existing four-branch batch and its actual combined return once at unscaled integer step before posterior update/noise addition. Detached independent snapshots only; no mutation, RNG use, callback model call, alternate branch execution or cache invalidation.',
            output='Pertrace float32 x_t,f00,f10,f01,f11,actual_combined_prediction and same-call returned-prediction hash; frame mask,slot/t/scaled_t,branch flags,checkpoint/package/guidance/input/random plan IDs. Retain dtype/shape/hash and all finite counts; missing trace is incomplete diagnostic output, never silently ignored.',
            offline='Recompose both weights on these same S1 states and report text-increment RMS/max plus combined differences. Optional direct GT feature MSE must be labeled S1-state diagnostic; no GT alignment, new model call or old-trajectory claim.',
            limits='Old2.5 trajectory branch values were not recorded. Counterfactual recombination here is local on S1 trajectory only, not a full counterfactual rollout or old-state comparison.'),
        software_tolerances=tolerances,
        CPU_acceptance=['Real new loader/entrypoint binds exact R final200, eval/static_delta, initial provenance, backbone/stats/environment; reject wrong checkpoint even if schema-compatible.',
            'Q batch5 branch order and gates including caption-present audit vs text-off; identical x_t/noise across all5; off branches skip encoder; no padding or hidden sixth f01.',
            'Actual four-branch function at wt1 and2.5 checked on analytic CPU fixtures; reduced float64 algebra tests do not substitute for method implementation tests.',
            'Trace on/off yields bitwise-identical fixture outputs, random stream and cache/counters including off/static/bypass; exactly four predeclared timesteps; never extra real verification forwards.',
            'Replay frozen60 input/reference/time/caption/noise identities incl train-reversed Q input; same numeric_zero/static semantics.',
            'Q480/S1_60/trace240 startup ledger, pre-forward counter reservation, total/stage caps, first failure retention (during probe, sampling or trace write), exact outputs/failed/not-run slots; no retry/fallback.',
            'All24 sample identity checks same-device tolerance frozen above; report checks only when counterparts completed, otherwise not_comparable, no fabricated pass.'],
        execution='01 implements/CPU validates;02 accepts concrete software;01 then obtains new authorization bound to final implementation+this package+unique run directory+GPU UUID. No actual GPU preflight before authorization.')
    evaluation=dict(version='canonical-B0-sampling-alignment-evaluation-v1',base_metrics=bind(ROOT/'docs/itm_rework/protocol/canonical_b0_20260925_v3/metrics.json'),
        previous_evaluation=bind(OLD/'evaluation.json'),alignment='Original20Hz GT without yaw/translation/temporal fitting; same root conventions.',
        sample_denominators=dict(new=60,historical=60,new_val=42,new_train=18,conditions=dict(Counter(r['condition'] for r in samples)),trace_records=240,Q_branches=480,Q_groups=96,Q_derived_state_condition_pairs=288),
        metrics='All frozen scalar metrics and per-axis extent/pelvis path, GT jerk/contact alongside; head/wrist joint proxy only, no generated IMU residual or new success threshold.',
        contact='Only val000433/000382 n=2 legal contact sources; val000139 nonflat N/A and ALL train floor-unadmitted N/A. No reclassification or zero fill.',
        contrasts=['New−historical for all60 matched source/config/condition cases, not just paired; same checkpoint/noise/input identities required.',
            'Within EACH guidance: paired−text_only,paired−static; val additionally paired−reversed,paired−numeric_zero.36 contrasts per guidance,72 combined.',
            '(S1 paired−S1 comparator)−(oldR paired−oldR comparator): all36 matching contrasts; retain raw scalar differences, extent differences and mean joint output distances. Nonzero distance is sensitivity, not control.',
            'Q: raw branch MSE plus unit/legacy_reduced errors and decomposition per96group/3condition; paired−static/reversed in each scale (384 rows), difference-of-contrasts (192 rows). Shared b cancels in raw paired−static/reversed prediction difference; squared-error contrast can still change via cross term.',
            'Q vs free output: report source/config/t separately, no replacement of complete free-sample evaluation by teacher-forced scores. No comparing Qbatch5 bitwise against tracebatch4 or old D of another checkpoint.'],
        aggregation='Train/val and config/condition separate. Mean two Q replicates per source first, then3 sources equally with min/max at each t; no pooled t headline. Full sample cells n=3, label partial n/missing reasons; failure denominator remains60 new,42val/18train plus240trace/480Q. No frame-weighted or cherry-picked means.',
        failure='Missing/not-run/timeout/OOM/nonfinite/bad identity/shape/mask/forbidden fit or fixed software tolerance violation remains in planned denominator. Missing required trace prevents complete delivery for its sample even if motion saved; preserve motion metrics labeled diagnostic_incomplete. Quality negatives remain finite valid outputs. First software/runtime fault stops later queue; no rerun.',
        review='All120 new+old cases in review_order.json. Hide arm/config/condition, show original coarse target. Ten fixed uniform poses plus all-frame hand/ankle/root traces; notes before key, uncertain retained. Old outputs known: no fully blind claim or semantic pass rate.',
        mandatory_negatives=['train000506 alternating forward lunges with return/order, not sustained crouch','train004728 RIGHT back leg direction/height/hold/return, not left-leg substitution','val000382 wrists clap without unrelated lower-body deformation','val000139 both configs walking preservation vs text-only/static/reversed','val000433 full kick-return and all numeric_zero/IMU-only rows'],
        interpretation=dict(support='If extra text increases Q MSE AND new free generation improves target motions, report limited panel support for guidance mismatch; dynamic benefit still requires improvement relative to own-scale text/static/reversed, not merely new text prior.',
            failure='If Q joint remains poor or S1 cannot freely generate training actions, this candidate has not resolved competence. No extra updates/data/guidance sweep automatically follows.',
            mixed='Q good/free bad indicates unresolved state/rollout mismatch, not unique causality; Q bad/free better or different signs bysource/t stay mixed. Do not turn root-motion tradeoffs into success.',
            IMU_only='Text absent means algebraic invariance to wt; cross-scale drift beyond1e-5m is software/replay discrepancy. Invariant bad IMU-only remains a model negative, cannot be repaired by this guidance intervention.',
            static='R static=text-only and null=bypass are architectural correctness, not learned IMU advantage.',
            limitations='Three exposed dev sources per split, fixed training seed,K1; no significance/generalization/diversity/strict IMU closure/compatible editing/real-device/test certification. Single AI coarse review not human validation.'))
    budget.update(version='canonical-B0-sampling-alignment-budget-v1',status='frozen_not_authorized',read_commit='d2d28fb',existing_checkpoint=fixed)
    budget['clock_rules']=dict(total='3600s from launch BEFORE preflight/load; deadline never extended',Q='min(total_deadline,Q_start+600)',S1='min(total_deadline,sampling_start+2400), shared by all60; no per-case reset',overhead='All load/save/trace validation overhead included. Sum stage caps3000 does not authorize600s extra model work; remaining total time is only preflight/orchestration, no transfer of unused caps.')
    budget['counter_rules']=dict(updates=0,new_checkpoints=0,Q_MDM=480,Q_encoder=576,S1_denoising=60000,S1_MDM=240000,S1_encoder=120,total_MDM=240480,total_encoder=696,trace_records=240,trace_extra_forwards=0,reservation='Reserve before forward; observed and reference encodings counted individually; f00/f10 Q zero encoder; no unlisted real warmup/verification/retry.')
    budget['required_ledger']='Predeclare all480 Q,60 sample and240 trace slots; terminal completed/failed/not_run_after_stop, attempted/reserved/completed counters, timings/GPU/peakallocated/artifact hashes; preserve partial output.'
    assert len(samples)==60 and len(probes)==480 and len(trace)==240 and len(history)==60 and len(noise_groups)==12
    validation=dict(scope='CPU files/NumPy/noise/algebra only; no pretrained model import/load/forward,training,sampling or GPU query',read_method_commit='d2d28fb',
        new_sample_slots=60,historical_samples_bound=60,train_samples=18,val_samples=42,Q_groups=96,Q_branches=480,trace_slots=240,
        Q_valid_noise_arrays_replayed=96,sampling_noise_groups=12,sampling_noise_draws_replayed=12000,
        expected_identity_checks=24,review_rows=120,source_files_verified=len(sources),training_updates=0,new_checkpoint_count=0,current_model_forwards=0,current_GPU_queries=0)
    output.mkdir(parents=True)
    files={'design.json':design,'evaluation.json':evaluation,'budget.json':budget,'probe_slots.json':probes,'probe_groups.json':qgroups,'probe_noise.json':noise,'sampling_slots.json':samples,'historical_samples.json':history,
           'sampling_noise_groups.json':{g:dict(metadata=v['metadata'],json_pointer='/random_plans/0',random_plan_id=v['plan']['random_plan_id'],random_plan_sha256=digest(v['plan']),frames=v['frames'],draws=1000) for g,v in noise_groups.items()},
           'trace_slots.json':trace,'review_order.json':review,'sources.json':[sources[k] for k in sorted(sources)],'validation.json':validation}
    for name,value in files.items():write(output/name,value)
    write(output/'package.json',dict(version='canonical-B0-sampling-alignment-package-v1',status='frozen_for_implementation_not_authorized',source_commit='d2d28fb',
        artifacts=[dict(path=REL+'/'+p.name,sha256=sha(p)) for p in sorted(output.iterdir())],execution_authorization=None,run_directory=None,GPU_UUID=None))
    print(json.dumps(validation))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/REL);args=parser.parse_args();build(args.output)
