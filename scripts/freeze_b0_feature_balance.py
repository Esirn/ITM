#!/usr/bin/env python
"""Owner02 F loss-balance protocol; saved files/NumPy only, no run authorization."""
import argparse
import builtins
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch
import numpy as np
from freeze_b0_state_diagnostic import ROOT, read, write, sha, describe

REL='docs/itm_rework/protocol/b0_feature_balance_20261003_v1'
PROPOSAL=ROOT/'docs/itm_rework/method/b0_feature_balance_proposal_20261003_v1'
OLD=ROOT/'docs/itm_rework/protocol/b0_sampling_alignment_20260927_v1'
CORRECTION=ROOT/'docs/itm_rework/protocol/b0_correction_20260926_v1'
S1=ROOT/'outputs/itm_rework/method/sampling_alignment_authorized_20260927_v1'
ARM='F_root_ric_balanced'
LOSS_VERSION='normalized-x0-root-ric-half-v1'
INITIAL_SHA='3ad3c1bca7ff7398386662e4831eb29136d3c0f2e06e5abbea8e7c4cb13d0401'
R_SHA='91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642'


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def loss_reference(error,mask=None):
    e=np.asarray(error,dtype=np.float64)
    if e.ndim!=2 or e.shape[0]!=263:raise ValueError('Expected feature/time [263,T]')
    m=np.ones(e.shape[1],bool) if mask is None else np.asarray(mask)
    if m.shape!=(e.shape[1],) or m.dtype!=bool or not m.any():raise ValueError('Nonempty boolean valid mask required')
    valid=e[:,m]
    if not np.isfinite(valid).all():raise ValueError('Nonfinite valid error')
    a=float(np.mean(valid[:67]**2));b=float(np.mean(valid[67:]**2))
    grad=np.zeros_like(e);grad[:67,m]=valid[:67]/(67*int(m.sum()));grad[67:,m]=valid[67:]/(196*int(m.sum()))
    return dict(uniform=float(np.mean(valid**2)),root_ric=a,remaining=b,proposed_balanced=.5*(a+b)),grad

def metric_gate(new,old):
    # Conservative development-panel guardrail; not statistical noninferiority.
    if not np.isfinite([new,old]).all():return dict(status='not_comparable',nonworse=False,improved=False)
    eps=1e-6+1e-6*abs(old)
    return dict(status='valid',slack=eps,nonworse=bool(new<=old+eps),improved=bool(new<old-eps))


def build(output):
    output=Path(output)
    if output.exists():raise FileExistsError('Freeze outputs cannot be overwritten')
    sources={}
    def bind(p,expected=None):
        p=(ROOT/Path(p)).resolve();key=str(p)
        if key not in sources:sources[key]=dict(path=key,sha256=sha(p))
        if expected is not None:assert sources[key]['sha256']==expected,key
        return sources[key]
    def verify(a):return bind(a['path'],a['sha256'])
    def get(p):bind(p);return read(p)
    def arrays(p):
        bind(p)
        with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files if k!='metadata'}
    for a in get(PROPOSAL/'package/sources.json'):verify(a)
    for p in sorted(PROPOSAL.rglob('*')):
        if p.is_file():bind(p)
    proposal=get(PROPOSAL/'package/design.json');budget=get(PROPOSAL/'package/budget.json')
    initial=verify(proposal['initial']);assert initial['sha256']==INITIAL_SHA
    historical_checkpoint=verify(proposal['historical_checkpoint']);assert historical_checkpoint['sha256']==R_SHA
    old_package=bind(OLD/'package.json','537895c45695a48cb4b5114941e4bf8cab3962c40be893fcc79b4c9eeb311d60')
    for a in get(OLD/'package.json')['artifacts']:verify(a)
    old_design=get(OLD/'design.json');old_evaluation=get(OLD/'evaluation.json')
    state=get(S1/'runtime/status.json');assert state['status']=='completed'
    old_implementation=verify(state['implementation'])
    for a in get(Path(old_implementation['path']))['expected_artifacts']:verify(a)
    common=deepcopy(proposal['common']);assert common==get(CORRECTION/'design.json')['common']
    # Independently re-evaluate old saved loss views; no tensor/model construction.
    saved_rows=get(PROPOSAL/'package/saved_branch_loss_views.json');views={}
    V=ROOT/'outputs/itm_rework/method/state_diagnostic_authorized_20261003_v1/V'
    targets={};max_error=0.
    for r in saved_rows:
        gid=r['group_id'];target=targets.setdefault(gid,arrays(V/'states'/(gid+'.npz'))['target'])
        pred=arrays(V/'branches'/(gid+'-'+r['branch']+'.npz'))['predicted_x0']
        v,_=loss_reference((pred.astype(float)-target.astype(float)).reshape(263,-1))
        for k,value in v.items():max_error=max(max_error,abs(value-r[k]));assert abs(value-r[k])<=1e-12
        views[gid,r['branch']]=v
    comparisons=get(PROPOSAL/'package/saved_contrast_loss_views.json')
    paradox=0
    for r in comparisons:
        a,b=r['contrast'].split('-');delta={k:views[r['group_id'],a][k]-views[r['group_id'],b][k] for k in views[r['group_id'],a]}
        for k,value in delta.items():assert abs(value-r['delta'][k])<=1e-12
        paradox+=bool(r['data_split']=='train' and r['contrast'] in ('P-S','P-R') and delta['uniform']<0 and delta['root_ric']>0)
    assert len(saved_rows)==len(comparisons)==192 and paradox==5
    schedule=get(CORRECTION/'training_schedule.json');assert len(schedule['rows'])==600
    train_dir=ROOT/'outputs/itm_rework/method/correction_authorized_20260927_v2/R_static_reference_delta'
    for u in range(1,201):
        record=get(train_dir/f'update_{u:03d}.json');rows=schedule['rows'][(u-1)*3:u*3]
        assert record['optimizer_step_completed'] and record['update']==u and len(record['examples'])==3
        for r,a in zip(rows,record['examples']):
            assert all(a[k]==v for k,v in r.items()) and a['actual_noise_sha256']==r['noise']['noise_sha256']
            draw=np.random.Generator(np.random.PCG64(int(r['noise']['address_sha256'],16))).standard_normal(r['noise']['shape']).astype(np.float32)
            assert describe(draw)['sha256']==r['noise']['noise_sha256']
    # Rebind the S1 unit-guidance control; do not inherit S1's own older2.5 comparator.
    oldslots=get(OLD/'sampling_slots.json');proposed=get(PROPOSAL/'package/sampling_pairs.json')
    population_fields=['split','data_split','condition','motion_id','sensor_config','frames','length','caption_index','reviewed_caption','text','text_present','imu_present','sensor_slots','gt_path','reference_source','random_plan_case_id','sampling_seed','steps','K','guidance']
    inputs=['acceleration','orientation','normalized_imu','normalized_reference_imu','original_orientation0','sensor_mask','timestamps_seconds','frame_mask','gt','text_condition_present','imu_condition_present']
    samples=[];history=[];noise_groups={};traces=[]
    for i,(o,p) in enumerate(zip(oldslots,proposed)):
        assert p['old_case_id']==o['case_id'] and p['slot_id']==f'F{i+1:03d}'
        path=S1/'samples'/(o['case_id']+'.npz');z=arrays(path);meta=get(path.with_suffix('.json'))
        assert meta['control_checkpoint_sha256']==R_SHA and meta['initialization_sha256']==INITIAL_SHA
        assert sha(path)==p['baseline_output']['sha256'] and sha(path.with_suffix('.json'))==p['baseline_metadata']['sha256']
        for k in ('data_split','motion_id','sensor_config','condition','frames','text','text_present','imu_present','random_plan_case_id','sampling_seed','guidance'):
            assert p[k]==o[k] and meta[k]==o[k]
        assert meta['guidance']==dict(text_scale=1.,imu_scale=1.,joint_scale=1.,mode='factorized',branch_execution='batched')
        plan=meta['random_plans'][0];key=o['random_plan_case_id']
        if key not in noise_groups:
            assert plan['steps']==1000 and plan['timestep_map']==list(range(1000)) and plan['clip_denoised'] is False
            assert [d['step'] for d in plan['noise_draws']]==[-1]+list(range(999,0,-1))
            for d in plan['noise_draws']:
                address={k:d[k] for k in ('version','backbone_id','case_id','sample_seed','step','shape')}
                h=digest(address);assert h==d['address_sha256']
                draw=np.random.Generator(np.random.PCG64(int(h,16))).standard_normal(d['shape']).astype(np.float32)
                assert describe(draw)['sha256']==d['noise_sha256']
            assert digest(plan['noise_draws'])==plan['random_plan_id']
            noise_groups[key]=dict(plan=plan,reference_metadata=bind(path.with_suffix('.json')))
        else:assert noise_groups[key]['plan']==plan
        assert p['random_plan_id']==plan['random_plan_id']
        row={k:o[k] for k in population_fields}
        row.update(slot_id=p['slot_id'],case_id=ARM+'--'+o['case_id'].split('--',1)[1],arm=ARM,slot_index=i,status='frozen_not_executed',
            checkpoint_selection='this_F_run_final200_only',loss_version=LOSS_VERSION,checkpoint_sha256=None,
            historical_case_id=o['case_id'],historical_output=bind(path),historical_metadata=bind(path.with_suffix('.json')),
            historical_input_hashes={k:describe(z[k]) for k in inputs},historical_random_plan_sha256=digest(plan),random_plan_id=plan['random_plan_id'],
            maximum_branch_example_forwards=4000,trace_timesteps=[900,500,250,50],trace_additional_model_forwards=0)
        samples.append(row);history.append(dict(case_id=o['case_id'],new_case_id=row['case_id'],arm='U_historical_R_unit',spec=o,output=bind(path),metadata=bind(path.with_suffix('.json')),checkpoint=historical_checkpoint))
        for j,t in enumerate((900,500,250,50)):
            traces.append(dict(trace_id=f'FT{i*4+j+1:04d}',case_id=row['case_id'],slot_id=row['slot_id'],timestep=t,
                historical_trace_id=f'T{i*4+j+1:04d}',branch_order=['f00','f10','f01','f11'],shape=[1,263,1,o['frames']],
                additional_MDM_calls=0,status='frozen_not_executed'))
    assert len(samples)==60 and len(noise_groups)==12 and len(traces)==240
    lookup={(r['data_split'],r['motion_id'],r['sensor_config'],r['condition']):r for r in samples}
    checks=[];within=[]
    for s in samples:
        key=(s['data_split'],s['motion_id'],s['sensor_config'])
        if s['condition']=='physical_static':checks.append(dict(kind='F_static_vs_F_text',new_case_id=s['case_id'],reference_case_id=lookup[key+('text_only',)]['case_id'],reference_arm='F'))
        if s['condition']=='backbone_bypass':checks.append(dict(kind='F_bypass_vs_F_text',new_case_id=s['case_id'],reference_case_id=lookup[key+('text_only',)]['case_id'],reference_arm='F'))
        if s['condition'] in ('text_only','physical_static','backbone_bypass'):
            checks.append(dict(kind='F_null_vs_U_same',new_case_id=s['case_id'],reference_case_id=s['historical_case_id'],reference_arm='U',reference_output=s['historical_output']))
        if s['condition']=='paired':
            for c in ['text_only','physical_static']+(['tail_reversed','numeric_zero'] if s['data_split']=='val' else []):
                r=lookup[key+(c,)];within.append(dict(data_split=key[0],motion_id=key[1],sensor_config=key[2],comparator=c,
                    F_paired=s['case_id'],F_reference=r['case_id'],U_paired=s['historical_case_id'],U_reference=r['historical_case_id']))
    assert len(checks)==48 and len(within)==36
    for i,c in enumerate(checks):c.update(check_id=f'FI{i+1:03d}',max_abs_m=1e-5,status='frozen_not_executed',fit='none',report=['bitwise','max_abs','RMS','finite_count','count','passed'])
    review_population=[dict(case_id=s['case_id'],arm='F',data_split=s['data_split'],motion_id=s['motion_id'],sensor_config=s['sensor_config'],condition=s['condition']) for s in samples]+[dict(case_id=h['case_id'],arm='U',**{k:h['spec'][k] for k in ('data_split','motion_id','sensor_config','condition')}) for h in history]
    order=np.random.Generator(np.random.PCG64(20261003)).permutation(120)
    review=dict(seed=20261003,algorithm='NumPy PCG64 permutation of [F60 slot order,U60 slot order]',rows=[dict(review_id=f'FREV{i+1:03d}',**review_population[int(j)]) for i,j in enumerate(order)],
        instructions='Hide arm/config/condition; show original coarse target. Save all120 notes before key. Old results known, singleAI review: not fully blind/human validation. Missing still keeps row.')
    weights=np.r_[np.full(67,263/134,dtype=np.float64),np.full(196,263/392,dtype=np.float64)]
    loss=dict(loss_version=LOSS_VERSION,feature_dim=263,groups=[dict(name='root_ric',half_open=[0,67],mass='1/2',per_coordinate='263/134'),dict(name='remaining',half_open=[67,263],mass='1/2',per_coordinate='263/392')],
        weights=weights.tolist(),weights_sha256=describe(weights)['sha256'],weights_dtype='float64 reference; cast to prediction float32 for metadata/implementation',
        formula='L_F=0.5*mean(e[0:67,valid]^2)+0.5*mean(e[67:263,valid]^2); uniform L_U=mean(e[0:263,valid]^2)',
        implementation='Compute two group mean squares in float32 and0.5-sum, no AMP. All263 retained. Valid frames only, three true-length per-example losses summed/3; no frame/source/condition weighting.',
        gradient_reference='dL/de=error/(67*Nvalid) inD and error/(196*Nvalid) inA, each divided by3 in update; zero on invalid frames.',
        logging='Per600 exposure record uniform,root_ric,remaining,balanced detached from same prediction; loss=balanced, loss/3 accumulation, has_gradient_graph; no extra forward and no comparable learning-curve claim.',
        fixtures=dict(float64_atol=1e-12,float64_rtol=1e-12,float32_atol=1e-7,float32_rtol=1e-6),
        uniform_fixture='All coordinate weights=1 must reproduce original scalar loss, gradients and optimizer update within fixture tolerance; production loss weights immutable, no runtime weight sweep.')
    common['initialization']='Load exact bound original initial.pt; verify tensor hashes/environment; new empty AdamW state. No new random initialization, no finalR resume.'
    common['loss']=loss['formula']+'; three microbatch losses/3; same null denominator.'
    design=dict(version='canonical-B0-feature-balance-design-v1',status='frozen_for_implementation_not_authorized',proposal_commit='2b8a842',arm=ARM,
        public_contract='itm-rework-v0.3-draft unchanged; restricted synthetic data unchanged',initial=initial,historical_checkpoint=historical_checkpoint,
        common=common,loss=loss,reference_mode='static_delta',historical_training_design=verify(proposal['common_training']),historical_schedule=verify(proposal['training_schedule']),
        old_sampling_package=old_package,old_sampling_implementation=old_implementation,environment=old_design['common']['environment'],
        only_change='Fixed loss partition mass. Preserve R parameter names/order/trainability, frozen bias, shared reference gradients/dropout, backbone eval/nativeCLIP, AdamW defaults, matched600 schedule and complete sampling path. Historical control, not a simultaneous rerun.',
        checkpoint_contract=dict(schema='itm-feature-balance-control-v1',arm=ARM,selection='final200 only',intermediate_updates=[50,100,150],
            required=['new package+implementation','loss_version+full263weights+hash','initial file/tensor/source package identities','training schedule SHA','same trained parameter names/order and frozen-state audit','optimizer defaults/state','environment','normalization/input provenance','final200 execution counts'],
            reject='OldR relabel, wrong/missing loss provenance, changed source/initial/schedule/environment/parameter set, intermediate checkpoint sampling or run resume. Incomplete200 stops; no replacement checkpoint.'),
        sampling=dict(slots=60,K=1,steps=1000,DDPM=True,clip_denoised=False,guidance=samples[0]['guidance'],
            noise='Preserve all12 complete historical random plans, same initial/each-step noises; case_id changes but random_plan_case_id/address remains historical.',
            inputs='Same20Hz true lengths/captions/presence/originalO[0] references; numeric_zero,tail_reversal and null/bypass definitions unchanged.',
            trace='240 same-call detached snapshots as oldS1; no mutation/RNG/cache changes/additionalforward. Read x_t,f00,f10,f01,f11,combined returned tensor,mask and all identities before posterior update.',
            historical_trace='Can compare losses/branches descriptively per source/config/t, but F and U x_t differ. Never assert same-state causal comparison or cross-model trace identity.'),
        software_tolerances=deepcopy(old_design['software_tolerances']),
        CPU_acceptance=['Independent analytical masked scalar/gradient andfinite-difference fixtures for67/196 split; uniform-weight reduction; true-length104/136/158 andequal-example loss/3, allpresence cases andnull no-grad retained.',
            'Actual training entry changes onlyloss; verify initial tensor hashes/parameter order/frozen params/optimizer defaults/dropout/address600; rejectwrongloss/checkpoint/schedule. No pretrainedforward orGPU inCPUpreparation.',
            'Actual sampler uses Ffinal200 provenance andsame old noise/CFG unit expression/native precision; IMU-only may change acrossF/U, null must satisfy frozen48 checks.',
            'Trace on/off exact CPU predictions/RNG/counters,240 fixedtimes zeroextra calls; save completearrays/provenance.',
            'Preflight-before deadline, global/train/sharedsample caps;200updates/600exposures/60samples/240traces/48checks ledgers beforepreflight; faultinjection andfullretention, no retry/resume/fallback.'])
    design['software_tolerances'].pop('Q_static_vs_text',None)
    design['software_tolerances']['full_sample']=dict(max_abs_m=1e-5,checks=48,plan='software_checks.json',note='No F/U IMU-only equality: learned weights change. Cross-arm null checks only text/static/bypass.')
    design['software_tolerances']['purpose']='Software arithmetic/identity; separate exploratory zero-tradeoff quality guardrails in evaluation, never relax numerical anchors after results.'
    quality_metrics=['global_MPJPE_m','root_relative_MPJPE_m','root_path_RMSE_m','head_or_wrists_position_joint_proxy_m','velocity_joint_proxy_RMSE_m_s']
    evaluation=dict(version='canonical-B0-feature-balance-evaluation-v1',base_metrics=old_evaluation['base_metrics'],historical_evaluation=bind(OLD/'evaluation.json'),
        denominators=dict(new=60,historical=60,new_train=18,new_val=42,conditions=dict(Counter(r['condition'] for r in samples)),traces=240,training_updates=200,exposures=600,software_checks=48,review_rows=120),
        complete_metrics=old_evaluation['metrics'],alignment=old_evaluation['alignment'],contact=old_evaluation['contact'],
        cross_model='All60 F-U matched cases, fulloriginalmetrics+extent+mean/max jointdistance; no new success rate. HistoricalU not contemporaneous rerun.',
        within='36 contrasts per arm: paired-text/static and valpaired-reversed/numeric_zero;72 total;36 difference-of-contrasts Fwithin-Uwithin, preserveallconditions.',
        same_prediction_feature_views='AllF/U complete normalized263 output vsGT: originaluniform,rootRIC67,remaining196,newbalanced andold5+RIC3 blocks; do not evaluate only newloss. Trace same-state arithmetic only within each arm; noQ/Vnewprobes.',
        loss_curve='600 exposures include same-forward L_U,L_D,L_A,L_F; evolving noise/t/dropout/weights mean first-last loss is not controlled convergence evidence.',
        primary=dict(source='train000506',configs=['head','wrists'],unit='each config separately, no pooledconfig success',
            semantic_rule='Visible alternating front-step squat/lunge sequence within this clippedGT support: initialleft-leading flexedpose recovers, thenright-forward step+lowering andmiddle recovery; retainterminaldescent. No demand for anunobserved initialleftstep orfinalstand beyondclip. Inspectfullsequence andtarget_review.json; sustainedcrouch,shallowercrouch,no sidechange ormissing observedmiddlerecovery fails; ambiguity uncertain, notpass.',
            classification=['complete','incomplete','uncertain','missing'],
            support='Only limited panel support perconfig if Fpaired complete andUpaired incomplete, all5 pairedF-vs-Upaired metrics nonworse, rootrelative strictly improves, andFpaired-vs-owntext/static all5 nonworse withrootrelative strictlyimproved. Anyuncertain/missing component => inconclusive, anyobservedfailure => notsupported. No B0certification.',
            guardrail=dict(metrics=quality_metrics,degradation_margin=0,slack='epsilon=1e-6+1e-6*abs(reference metric), in each metric native unit',
                nonworse='new <= reference+epsilon',strict_improvement='new < reference-epsilon',
                interpretation='Conservative operational zero-tradeoff rule beforeFoutputs; floating comparison slack only, not calibrated perceptual/physical/statistical noninferiority or general success threshold. Do not sweep margins.'),
            dynamic='Paired!=null alone insufficient; require full behavior andtarget-linked metrics beyondslack asabove. Fnull=Unull architectural identity isnotlearnedbenefit.'),
        regressions='Applysame5metrics FpairedvsUpaired perallother10 source/config cases, flageveryexceedance; fullcoarsereview flags newbehavior loss even whenmetricsimprove. Anyregression prevents claiming broadpanelrepair; keepprimarylocal andtradeoffs explicit.',
        movement=dict(fps=20,axes='OriginalcanonicalXYZ; +Yheight, signedX/Zrelativepelvis, +Zforwardproxy; no yawfit or perprediction directioncalibration',joints=dict(pelvis=0,left_hip=1,right_hip=2,left_knee=4,right_knee=5,left_ankle=7,right_ankle=8,left_toe=10,right_toe=11),curves='Allframe pelvisXYZ/height,path; bilateralankle/toe absoluteY andpelvisrelativeXYZ; kneeangle acos(clipped dot of hip-knee andankle-knee normalized); degeneratebones => NA withcount, notzero. ShowGT/U/F independentcurves, no resampling/timefit orfloorcontact inferredfortrain.',interpretation='Fullbehavior/side/order/return review plusallmetric deltas; no motion-shrink reward. Jerk/contact vsGT descriptive withpriorNA, not newpasscriteria.'),
        review='All120 randomorder inreview_order.json;10uniformposes+allframecurves/fullsequence. Notes beforekey, no arm/config/condition labels. ExistingUknown andsingleAI; notblindhuman validation. Storecycle side/order/return evidence andfailure/uncertain reasons. No temporal/spatialfit.',
        mandatory=old_evaluation['mandatory_negatives']+['train000027 punch then turn','train004728wrists limited oldcoarsepositive direction/amplitude uncertainty'],
        aggregation='Arms/split/config/condition separate,3source equalmeans/min/max withvalidn/plannedn andmissing reasons. No frameweighting orpooltrain/val; all60new+60old rows alwaysretained.',
        failure='Runtime/identity/software failure stopsqueue; incompleteoutput staysplanned denominator, diagnostic_incomplete metrics maybelisted butexcludedprimarysupport; missingcomparison not_comparable, notzero/pass. Finitequalitybad retainsvalidstatus andallfutureplannedslots.',
        interpretation=['Onlytrain improves => trainingfit evidence, notvalcontrol.',
            'Lowerbalanced/uniformfeatureloss withoutcompletealternatingbehavior => thisfixedweight intervention insufficient; no furtherweightsearch authorized.',
            'Primaryconfig improves butotherconfig/source regresses => localtradeoff, notoverallrepair.',
            'Paired collapses towardnull withouttarget-metric+semantic incrementalbenefit => dynamicIMU use notdemonstrated.',
            'Noeffect oruncertain isreported; notautomaticadditionalupdates/data/ablation.'],
        limitations='Historicalpairedcontrol,3exposedtrain+3devval,1trainingseed,K1. No significance/generalization/diversity/strictIMUclosure/compatibleedit/realdevice certification.')
    budget.update(version='canonical-B0-feature-balance-budget-v1',status='frozen_not_authorized',authorized=False,execution_authorization=None,run_directory=None,GPU_UUID=None,
        MDM_calls_total=60600,MDM_examples_total=240600,encoder_examples_total_max=1320,training_caption_calls_max=600,sampling_caption_calls_max=60,software_checks=48,
        deadlines='t0 beforepreflight; total=t0+3600,training=t0+600; sampling=min(samplingstart+3000,total). No stage reset,time transfer,warmup,extra probes orretry.',
        prereqs=['01implementation+CPUtests','02incrementalacceptance','newexplicituserauthorization boundmanifest/package/uniquedir/GPU','preflight'])
    for p in [Path(__file__),ROOT/'tests/test_b0_feature_balance_protocol.py',ROOT/'scripts/freeze_b0_state_diagnostic.py',ROOT/'docs/itm_rework/method/B0_FEATURE_BALANCE_PROPOSAL_20261003.md',
              ROOT/'docs/itm_rework/protocol/B0_STATE_DIAGNOSTIC_EVALUATION_20261003.md',ROOT/'docs/itm_rework/protocol/B0_SAMPLING_ALIGNMENT_EVALUATION_20260928.md',ROOT/'src/itm/metrics/canonical_b0.py']:
        bind(p)
    gt_path=next(s['gt_path'] for s in samples if s['motion_id']=='000506')
    gt=np.load(gt_path,allow_pickle=False).astype(np.float64);bind(gt_path)
    assert gt.shape==(136,22,3) and np.isfinite(gt).all()
    target_review=dict(source='train000506',frames=136,fps=20,artifact=bind(gt_path),
        review_status='GT-only descriptive staging, prior to Foutputs; no automatic semantic threshold',
        boundaries='Clipstarts inleft-leading flexion, notneutral; clipends indescendingflexion, notstanding. Do notdemandunobserved startstep/finalreturn.',
        windows=[dict(frames_half_open=[a,b],note=note) for a,b,note in [(0,22,'initialleft-leading poseandrecovery'),(22,36,'uprighttransition'),(36,67,'rightforwardstep andlowering'),(67,106,'middlerecoverytowardupright'),(106,136,'terminaldescent; outcomeafterclipunobserved')]],
        curve_frames=list(range(136)),pelvis_y_m=gt[:,0,1].tolist(),
        left_ankle_relative_z_m=(gt[:,7,2]-gt[:,0,2]).tolist(),right_ankle_relative_z_m=(gt[:,8,2]-gt[:,0,2]).tolist(),
        left_ankle_y_m=gt[:,7,1].tolist(),right_ankle_y_m=gt[:,8,1].tolist(),
        instruction='GTwindows guideorderedreview only, notgeneratedframeclassification ornew temporalfit. Fullframecurvesandviews retained; noautomaticpositive ifonlymetricdeclines.')
    validation=dict(proposal_commit='2b8a842',proposal_sources_verified=573,saved_branch_views=192,saved_contrasts=192,loss_view_max_abs=max_error,
        train_primary_uniform_better_rootRIC_worse=5,training_exposures=600,training_update_records=200,reconstructed_training_noise=600,
        samples=60,historical_controls=60,traces=240,sampling_noise_groups=12,reconstructed_sampling_noise=12000,software_checks=48,
        within_contrasts_per_arm=36,difference_of_contrasts=36,review_population=120,source_files=len(sources),
        current_model_imports=0,current_checkpoint_tensor_loads=0,current_model_forwards=0,current_GPU_queries=0,current_training_updates=0,current_new_samples=0)
    output.mkdir(parents=True)
    products={'design.json':design,'loss.json':loss,'evaluation.json':evaluation,'budget.json':budget,'training_schedule.json':schedule,
        'sampling_slots.json':samples,'historical_samples.json':history,'sampling_noise_groups.json':noise_groups,'trace_slots.json':traces,
        'target_review.json':target_review,'software_checks.json':checks,'within_contrasts.json':within,'review_order.json':review,'validation.json':validation,'sources.json':[sources[k] for k in sorted(sources)]}
    for name,value in products.items():write(output/name,value)
    write(output/'package.json',dict(version='canonical-B0-feature-balance-package-v1',status='frozen_for_implementation_not_authorized',source_commit='2b8a842',
        artifacts=[dict(path=REL+'/'+p.name,sha256=sha(p)) for p in sorted(output.iterdir())],execution_authorization=None,run_directory=None,GPU_UUID=None))
    print(json.dumps(validation))


def guarded_build(output):
    original=builtins.__import__
    def restrict(name,*a,**kw):
        if name.split('.')[0] in ('torch','tensorflow','jax'):raise AssertionError('Neural imports forbidden')
        return original(name,*a,**kw)
    def reject(*a,**kw):raise AssertionError('Process/GPU operations forbidden')
    with ExitStack() as stack:
        stack.enter_context(patch.object(builtins,'__import__',restrict))
        for name in ('Popen','run','call','check_call','check_output'):stack.enter_context(patch.object(subprocess,name,reject))
        build(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/REL);args=p.parse_args();guarded_build(args.output)
