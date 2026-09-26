#!/usr/bin/env python
"""Owner02: freeze CPU-only D/L/R schedules; never launch models or authorize work."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from itm.data.random_plan import sampling_noise

ROOT=Path(__file__).resolve().parents[1]
REL='docs/itm_rework/protocol/b0_correction_20260926_v1'
DATA=ROOT/'docs/itm_rework/protocol/canonical_b0_20260925_v3'
PROPOSAL=ROOT/'docs/itm_rework/method/b0_diagnosis_20260926_v1'
ARMS=['L_exact_length','R_static_reference_delta']
VERSION='itm-B0-correction-address-v1'

def read(p):return json.loads(Path(p).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def records(name):return [json.loads(x) for x in (DATA/name).read_text().splitlines() if x.strip()]
def artifact(p):
    p=Path(p);return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'sha256':sha(p)}
def rng(namespace,**keys):
    address={'version':VERSION,'seed':20260925,'namespace':namespace,**keys}
    h=digest(address)
    return np.random.Generator(np.random.PCG64(int(h,16))),h

def make_schedule(plans):
    population=[(mid,c) for mid in sorted(plans) for c in ['head','wrists']]
    rows=[]
    for epoch in range(1,101):
        order,_=rng('epoch_order',epoch=epoch)
        for position,index in enumerate(order.permutation(len(population))):
            mid,config=population[int(index)];n=plans[mid]['frames'];ordinal=len(rows)
            identity={'epoch':epoch,'position':position,'motion_id':mid,'sensor_config':config}
            t,_=rng('diffusion_timestep',**identity);td,th=rng('text_presence',**identity);ud,uh=rng('imu_presence',**identity)
            _,encoder_hash=rng('encoder_dropout',**identity);_,nh=rng('training_noise',**identity)
            noise=np.random.Generator(np.random.PCG64(int(nh,16))).standard_normal((1,263,1,n)).astype(np.float32)
            rows.append({'exposure_id':f'E{ordinal+1:04d}','epoch':epoch,'position_in_epoch':position,
                'update':ordinal//3+1,'microbatch_index':ordinal%3,'motion_id':mid,'sensor_config':config,
                'frames':n,'caption_index':plans[mid]['caption_index'],'timestep':int(t.integers(0,1000)),
                'text_present':bool(td.random()>=.1),'imu_present':bool(ud.random()>=.1),
                'text_presence_address_sha256':th,'imu_presence_address_sha256':uh,
                'encoder_dropout_seed':int(encoder_hash[:16],16) % (2**63),
                'noise':{'address_sha256':nh,'shape':[1,263,1,n],
                    'noise_sha256':hashlib.sha256(noise.tobytes(order='C')).hexdigest(),
                    'generator':'numpy PCG64(full address SHA256 integer); standard_normal float64 then cast float32; C order'}})
    return rows


def build(output):
    if output.exists():raise FileExistsError('Frozen packages are immutable: choose a fresh directory')
    budget=read(PROPOSAL/'proposed_budget.json');plans={s:{r['motion_id']:r for r in records(f'{s}_caption_plans.jsonl')} for s in ['train','val']}
    motions={s:{r['motion_id']:r for r in records(f'{s}_motion.jsonl')} for s in plans}
    accepted=read(ROOT/'docs/itm_rework/method/canonical_training_input_20260925_v1.json')
    backbone=accepted['training_args']['mdm_checkpoint'];backbone_hash=sha(backbone)
    sources=[ROOT/'docs/itm_rework/method/B0_CORRECTION_PROPOSAL_20260926.md',PROPOSAL/'proposed_budget.json',PROPOSAL/'proposed_probe_slots.json',PROPOSAL/'proposed_sampling_slots.json',PROPOSAL/'existing_artifact_diagnosis.json',PROPOSAL/'sources.json',ROOT/'docs/itm_rework/protocol/B0_EVALUATION_20260926.md',ROOT/'docs/itm_rework/method/canonical_training_input_20260925_v1.json',Path(__file__),ROOT/'src/itm/data/random_plan.py']
    sources += [DATA/name for name in ['package.json','metrics.json','semantic_review.json','sampling_cases.json','admission.jsonl','train_motion.jsonl','val_motion.jsonl','train_imu.jsonl','val_imu.jsonl','train_caption_plans.jsonl','val_caption_plans.jsonl']]
    for a in budget['dependencies']+[budget['diagnostic']['checkpoint']]:assert sha(ROOT/a['path'])==a['sha256']
    sources.append(ROOT/budget['diagnostic']['checkpoint']['path'])
    sources += [Path(backbone),Path(accepted['training_args']['mean']),Path(accepted['training_args']['std'])]
    probes=[];noise={}
    for i,row in enumerate(read(PROPOSAL/'proposed_probe_slots.json')):
        row=dict(row);s,mid,config=row['split'],row['motion_id'],row['sensor_config'];plan=plans[s][mid]
        assert plan['frames']==row['frames'];n=row['frames'];group=f'canonical-B0-D-{s}-{mid}-{config}'
        noise_key=f"{group}-t{row['timestep']}-s{row['noise_seed']}"
        if noise_key not in noise:
            _,draw=sampling_noise([1,263,1,n],backbone_id=backbone_hash,case_id=group,sample_seed=row['noise_seed'],step=row['timestep'])
            pad=None
            if s=='train' and n<158:
                _,pad=sampling_noise([1,263,1,158-n],backbone_id=backbone_hash,case_id=group+'-padding',sample_seed=row['noise_seed'],step=row['timestep'])
            noise[noise_key]={'valid_noise':draw,'padding_noise':pad}
        row.update(probe_id=f'D{i+1:04d}',status='frozen_not_executed',noise_key=noise_key,
            caption_index=plan['caption_index'],text=plan['caption'],text_present=True,imu_present=row['condition']!='text_only',
            target_feature_path=motions[s][mid]['joint_vec_path'],target_feature_storage='unnormalized_HumanML263',target_normalization='pretrained_Mean_Std',
            context_frames=n if row['context']=='exact_length' else 158,
            padded_batch_members=['000027','000506','004728'] if row['context']!='exact_length' else None)
        probes.append(row)
    cases=[];old={c['case_id']:c for c in read(DATA/'sampling_cases.json')}
    for slot,row in enumerate(read(PROPOSAL/'proposed_sampling_slots.json')):
        row=dict(row);split='train' if row['split']=='train_diagnostic' else 'val';mid=row['motion_id'];plan=plans[split][mid]
        assert row['frames']==plan['frames'];c=row['condition']
        if 'original_case_id' in row:
            before=old[row['original_case_id']]
            assert (row['random_plan_case_id'],row['sampling_seed'],row['frames'])==(before['random_plan_case_id'],before['sampling_seed'],before['length'])
        row.update(slot_index=slot,status='frozen_not_executed',data_split=split,length=row['frames'],
            caption_index=plan['caption_index'],reviewed_caption=plan['caption'],text=plan['caption'],
            text_present=c!='imu_only',imu_present=c not in ['text_only','backbone_bypass'],
            sensor_slots=[4] if row['sensor_config']=='head' else [0,1],gt_path=motions[split][mid]['joints_path'],
            reference_source='original delivered observed frame0 orientation + zero raw acceleration; never target GT or reversed/zero matrix',
            baseline_case_id=row.get('original_case_id'),
            checkpoint_selection=f"{row['arm']}:final_update_200",K=1)
        if c=='tail_reversed':row['comparison_gt']='original paired target; mismatched-input sensitivity only, not valid reversed-motion reconstruction'
        cases.append(row)
    schedule=make_schedule(plans['train'])
    condition_counts=Counter(f"{int(r['text_present'])}{int(r['imu_present'])}" for r in schedule)
    design={
        'version':'canonical-B0-correction-design-v1','status':'frozen_for_implementation_not_authorized',
        'read_method_commit':'cd1079e','data_contract':'itm-rework-v0.3-draft; original canonical package unchanged',
        'scope':'development mechanism diagnosis; no research test or new data admission',
        'arms':{
            ARMS[0]:{'definition':'original E and additive W E + b; exact-length microbatches', 'adapter_bias':'trainable, zero initialized'},
            ARMS[1]:{'definition':'W_l (E(u)-E(s(u0))); exact-length microbatches','adapter_bias':'unused AND frozen; zero initial; old schema may retain tensor',
                'reference_gradient':'no detach; same shared encoder weights',
                'reference_dropout':'same per-layer elementwise dropout masks as observed encoder, by replaying per-exposure torch RNG inside fork_rng. No joint concatenated 2B dropout. Resume caller CPU/CUDA RNG states on exit.',
                'limitation':'Removes pure static signal as a control residual; dynamic response can still depend on initial orientation through nonlinear E. This is an ablation, not full IMU design.'}},
        'common':{
            'fresh_seed':20260925,'initialization':'Create a single fresh original encoder/additive-adapter state after seed=20260925; clone E and W (and zero b) to both arms. Bind initial tensor hashes and implementation/env before authorization; do not resume old B0.',
            'backbone':{'path':str(backbone),'sha256':backbone_hash,'mode':'frozen_eval','mask_frames':False},
            'control_config':{'sensor_feature_dim':12,'sensor_count':6,'latent_dim':512,'encoder_layers':2,'encoder_heads':8,'dropout':.1,'encoder_variant':'ordered_concat','null_mode':'strict','time_encoding':'physical_sinusoidal','model_fps':20.,'adapter_variant':'additive'},
            'imu_normalization':accepted['imu_normalization'],'training_input_provenance':accepted['training_input_provenance'],
            'motion_normalization':[artifact(Path(accepted['training_args'][k])) for k in ['mean','std']],
            'loss':'Per-example mean squared normalized-x0 error over exactly 263*T valid entries; sum three example losses /3. No auxiliary losses, sample/frame weighting, or response loss.',
            'optimizer':{'name':'AdamW','lr':5e-5,'betas':[.9,.999],'eps':1e-8,'weight_decay':.01,'amsgrad':False,'foreach':None,'maximize':False,'capturable':False,'differentiable':False,'fused':None},
            'optimizer_note':'Retain old serialized defaults, including foreach=null and fused=null. Bind same torch/device/backend for both arms and serialize resolved settings; no independent optimizer change.',
            'precision':'float32; no AMP, gradient clipping, scheduler or early stopping',
            'accumulation':'zero_grad once per update; three true-length examples, backward(loss/3) each, then optimizer.step exactly once. Even null examples stay in denominator and MDM counts. If a null loss has no trainable graph, skip backward and record zero control-gradient contribution; do not renormalize other losses.',
            'training_rng':'Use training_schedule.json literally; do not redraw condition/t/noise in hooks or backbone. Equal schedule across arms; all listed examples forwarded once including null. Noise generation is PCG64 keyed by full address SHA256, not torch.randn.',
            'checkpoints':[50,100,150,200],'selection':'final200 only; never choose by D, losses, train or val results'},
        'input_transforms':{
            'base':'Existing independently accepted 30Hz->20Hz input, train-only normalization, caption index, coordinates and root reference unchanged.',
            'physical_static':'At model 20Hz: a=0, O[t]=original O[0]; same mask/timestamps; normalize with old accepted train-only stats.',
            'tail_reversed':'At model 20Hz after interpolation: raw a and O jointly indexed [0,T-1,T-2,...,1] on all slots, then normalize; preserve frame timestamps/masks and original O[0] reference. No velocity sign flip or rotation inverse. Deliberately nonphysical sensitivity input.',
            'numeric_zero':'All normalized input entries zero; presence remains true. R reference still uses ORIGINAL source O[0] and normalized raw a=0. Never interpret zero rotation as valid orientation.',
            'condition_off':'strict presence gate; original raw backbone bypass remains available unchanged.',
            'R_cache':'Build observed and reference embeddings once per case in eval; subtract before adapters, then cache. No new MDM branch forwards per diffusion step.'},
        'D':{
            'checkpoint':budget['diagnostic']['checkpoint'],'mode':'all modules eval; no autograd, optimizer, CFG or reverse diffusion loop',
            'x_t':'Use original normalized GT x0 and the frozen old backbone diffusion q_sample at indicated integer t; exactly same valid x_t across four conditions and context comparisons. Save q_sample config and effective scaled t.',
            'padding':'Original normalized target zero-padded to158; padded IMU/timestamps zero, encoder/loss mask false there. Concatenate SAME valid noise with addressed padding noise; q_sample entire padded tensor. Backbone mask_frames=false; do not repair it during D.',
            'padded_batch':'Three train sources in order 000027,000506,004728 at fixed config,t,noise seed,condition; 32 batch forwards x3=96 branch examples. Length158 member is batch-shape numerical control, not padding-treatment case.',
            'exact_batch':'384 batch-size1 forwards, one MDM branch each. Text always present; text_only is f10, other conditions f11. No four-way CFG.',
            'instrumentation':'Read hooks in those same forwards; no extra model calls to obtain hidden/reference tensors. Cache/reuse exact static probes for E(u)-E(static).',
            'stop':'A deterministic implementation/identity/nonfinite error stops entire queue before training. Poor finite diagnostic metrics do NOT adapt or stop arms.'},
        'software_tolerances':{
            'purpose':'Numerical identity checks only, not quality/effect thresholds; log all actual max/RMS values.',
            'input_acceleration':'exact equality to accepted reference; orientation/normalized_imu atol=2e-7 rtol=0',
            'RIC_decode':{'atol_m':2e-6,'rtol':1e-6},
            'static_residual_forward':'identical input/reference with shared dropout: encoded difference and injected residual exactly zero (float32); adapter bias never reintroduced',
            'static_gradient':'CPU fixture max absolute encoder/adapter gradient <=1e-7 for static loss contribution before optimizer/weight-decay; retain continuous max and finite counts',
            'null_and_R_static_x0':{'atol':1e-6,'rtol':1e-6,'description':'same-device matched single step; target is exact equality; tolerance accommodates operation ordering only'},
            'full_sample_null_or_R_static':{'max_abs_m':1e-5,'description':'continuous decoded max plus bitwise status; do not align. Exceedance is implementation/replay discrepancy, not model-quality failure.'},
            'accumulated_gradient_CPU':{'atol':1e-6,'rtol':1e-5,'description':'explicit sum of per-example objectives vs microbatch accumulation, same frozen RNG; optimizer steps exactly200; no tolerating count mismatch'},
            'hashes_masks_seeds_counts':'exact; no tolerance'},
        'required_outputs':{
            'D':'Each of480 slots: IDs/context/t/noise-address and actual x_t hashes, normalized predicted x0 float32, target/hash/mask, encoder pre-temporal + each2 layer + final output, each8 pre-add motion hidden and gated residual on valid frames, and scalar counts/sums_sq/RMS/max. Loss/probe reductions independently replayable by02. Identical tensors may be stored once by SHA256 with references; never add forwards to obtain hooks.',
            'training':'Common initial state hashes and immutable initialization artifact;600 per-exposure IDs/t/noise/condition/encoder-dropout addresses, shape/loss and200 step receipts per arm;actual optimizer/env/code/input/protocol hashes;selected final200 and periodic50/100/150 checkpoints;R reference construction and counters. No unplanned checkpoint selection.',
            'samples':'Original full motion + normalized263 + GT + input/raw/normalized/reference tensors + original frame0 source/hash + masks/times + condition flags + random plan, now also arm/split/final checkpoint/mode/package hashes. Preserve all120 planned statuses including failures/not-run. Float32/eval/condition_cache=true, no AMP.',
            'runtime':'One complete queue ledger, stage deadlines, reserved/attempted/completed counters, actual encoder calls,all receipts/errors and artifact hashes; preserve partially completed stages without promoting partial model to final.'},
        'execution':'01 implements and CPU validates; 02 independently accepts; then obtain new explicit authorization bound to package, final code, run directory, GPU UUID; preflight precedes single execution. This package is not an executable authorization.'}
    evaluation={
        'version':'canonical-B0-correction-evaluation-v1','base_metrics':artifact(DATA/'metrics.json'),
        'alignment':'direct original GT, 20Hz, no spatial/time fits; retain all original metric definitions',
        'split_rules':{'development_val':'42 per arm = original36 + tail-reversed6; six original conditions separately n=3 per config; reversed separately n=3',
                       'train_diagnostic':'18 per arm, separate n=3 per config/condition; never merge train and val or call memorization generalization',
                       'foot_contact':'Only original val000433/000382 use GT contacts; val000139 nonflat N/A; all train N/A (flat-floor admission not frozen for train). Other metrics apply all cases.'},
        'contrasts':[
            'Within each split/config/source: paired-text_only, paired-physical_static; val also paired-tail_reversed and paired-numeric_zero.',
            'Matched R-L per original source/config/condition, same noise; never mix checkpoints inside arm.',
            'Difference of contrasts (R paired - R comparator) - (L paired - L comparator), comparator text_only/static; val also tail_reversed. Raw metrics, deltas and output-to-output mean joint distance.',
            'Historical B0 vs L/R only for original36 identical-noise val cases; descriptive, not pure causal padding ablation.'],
        'aggregation':'Show all120 planned slots, separate arm/split/config/condition denominators; equal source means n=3 only with all3 valid. On missing, label partial mean and completed n, include missing denominators; paired contrasts only matched available groups with n and missing reasons. Never pool frames, conditions, arms, train/val.',
        'failure':'Any missing/not-run/timeout/OOM/nonfinite/bad length-mask/checkpoint-input-random identity is failure over planned120, and arm60, val42/train18 denominators. Quality negatives remain valid outputs. R-static architecture mismatches reported separately; stop subsequent queue as correctness discrepancy.',
        'D_rows':'Store per slot valid count, finite count, x0 masked MSE; RMS over valid feature/time entries and max_abs for encoder fused pre-temporal, each of2 temporal layers, final E; each of8 pre-add motion hidden h_l, gated residual r_l; ratio RMS(r_l)/RMS(h_l) with zero denominator=null and explicit status. Exclude prepended text/timestep token and padding.',
        'D_deltas':'For matched source/config/t/replicate: encoder and residual differences paired-static and paired-reversed; normalized x0 differences each condition-text_only, paired-static, paired-reversed; all absolute RMS/max and x0 MSE changes. padded-exact for text_only/paired, valid frames only. Never divide unlike physical units or claim direction/causation from norm alone.',
        'D_aggregation':'Retain480 rows and all96 matched padding rows. Average 2 replicas within source/config/t first; split train/val; source-equal means across3 plus min/max; keep each t/layer separately. No pooling t as a headline. Length158 padding comparison marked no_added_padding, others104/136 actual treatment.',
        'semantic_review':'All120 outputs in predeclared random order, hide arm/config/condition; show source coarse target; save notes before key. Pose snapshots plus every-frame traces; uncertain retained. No new automatic threshold, semantic pass rate or JSR. Train actions use existing reviewed original captions/order/side.',
        'interpretation':{
            'static_restoration':'R static==text-only is built in, not empirical IMU advantage.',
            'input_sensitivity':'paired != reversed/zero/static proves sensitivity only, not useful control; reversed is invalid time-correspondence, not a physical donor or ground-truth counterfactual.',
            'no_dynamic_evidence':'If R approaches text-only and neither preserved-target proxy nor coarse semantics benefits from real temporal input, report no demonstrated dynamic IMU use. Do not declare success based on static identity.',
            'possible_panel_evidence':'Report the sign/magnitude of all joint-position, velocity, root-relative and semantic contrasts; qualitative improvements only as developer-panel observations when real paired input is useful relative to both static and reversed and no renewed walk->kick negative. No binary certification or effect-size cutoff.',
            'train_val':'Train failure means learned free-generation competence unestablished even if loss falls; train better but val fails motivates separately reviewed coverage expansion, not reuse of uncertain data.',
            'causality':'D is local teacher-forced evidence. L-history changes training/RNG too. R-L tests whole reference-difference residual definition including removal of bias; cannot separately attribute to bias or static embedding.'},
        'not_estimable':['generalization/significance/CI from3 sources and1 seed','conditional diversity K=1','same-class shuffled donor advantage','generated strict IMU closure/JSR_IMU','compatible editing','real-device calibration']}
    budget['version']='canonical-B0-correction-budget-v1';budget['status']='frozen_not_authorized';budget['owner']='02 protocol / 01 implementation and execution'
    budget['read_commit']='cd1079e';budget['diagnostic']['slots']={'path':REL+'/probe_slots.json'};budget['sampling']['slots']={'path':REL+'/sampling_slots.json'}
    budget['execution_dependencies']=['01 implements exact frozen design and CPU tests','02 software acceptance of concrete code/inputs/counters','new explicit user authorization bound to package/code/run_directory/GPU_UUID','GPU preflight']
    budget['clock_rules']={'total':'7200s starts at queue launch BEFORE GPU preflight/load; never extends',
        'D':'min(total_deadline,D_start+600)',
        'training':'shared1200 starts immediately before L_train; each arm min(total,train_shared_deadline,arm_start+600)',
        'sampling':'shared5400 starts immediately before L samples; R shares same deadline; no reset per arm or slot',
        'stages':'All load/save/verification overhead counts; no transferring unused stage limits into another cap'}
    budget['counter_rules']={'D_MDM':480,'training_MDM_total':1200,'training_encoder_L_max':600,'training_encoder_R_max':1200,
        'sampling_denoising':120000,'sampling_MDM_branch_examples':480000,
        'sampling_encoder_max':{'L_observed':60,'R_observed':60,'R_reference':60},
        'D_encoder':'at most one observed encoding per branch example (<=480); eval caches may reduce this; static comparisons use already budgeted static probes. Hook extraction adds zero MDM calls.',
        'count':'Reserve counters before forward/update. Unused null-branch encoder calls may be skipped but not shifted into extra MDM work. Record actual encoder calls separately. All new real MDM probes/counts, including verification, must fit these caps; no unlisted GPU warmup probes.'}
    # Preserve the 120 slots, but explicitly bind every scalar/input missing in the proposal.
    assert len(probes)==480 and Counter(x['context']=='exact_length' for x in probes)=={True:384,False:96}
    assert len(cases)==120 and Counter(x['arm'] for x in cases)==dict.fromkeys(ARMS,60)
    assert len(schedule)==600 and set(Counter(x['update'] for x in schedule).values())=={3}
    assert len({x['case_id'] for x in cases})==120 and len({x['probe_id'] for x in probes})==480
    for arm in ARMS:
        assert Counter(x['split'] for x in cases if x['arm']==arm)=={'development_val':42,'train_diagnostic':18}
    order=np.random.Generator(np.random.PCG64(20260929)).permutation(120)
    review={'seed':20260929,'algorithm':'numpy PCG64 permutation of frozen120 slot order','rows':[{'review_id':f'C{i+1:03d}','case_id':cases[int(j)]['case_id']} for i,j in enumerate(order)]}
    output.mkdir(parents=True)
    files={'design.json':design,'evaluation.json':evaluation,'probe_slots.json':probes,'probe_noise.json':noise,'sampling_slots.json':cases,'training_schedule.json':{'version':VERSION,'seed':20260925,'numpy_version':np.__version__,'population_order':[[m,c] for m in sorted(plans['train']) for c in ['head','wrists']],'condition_counts':dict(condition_counts),'rows':schedule},'review_order.json':review,'sources.json':[artifact(p) for p in sources]}
    for name,x in files.items():write(output/name,x)
    for section,name in [('diagnostic','probe_slots.json'),('sampling','sampling_slots.json')]:budget[section]['slots']['sha256']=sha(output/name)
    write(output/'budget.json',budget)
    validation={'scope':'CPU plan construction and source/hash checks only; no model forward or new authorization',
        'D_slots':480,'D_exact':384,'D_padded':96,'D_unique_valid_noise_arrays':len(noise),
        'sampling_slots':120,'slots_per_arm':60,'val_per_arm':42,'train_per_arm':18,'training_updates_per_arm':200,
        'training_exposures_per_arm':600,'training_condition_counts_per_arm':dict(condition_counts),
        'review_rows':120,'old_val_cases_preserved_per_arm':36,'planned_GPU_jobs_executed':0}
    write(output/'validation.json',validation)
    package={'version':'canonical-B0-correction-package-v1','status':'frozen_for_implementation_not_authorized',
        'source_commit':'cd1079e','schema_scope':'new experiment package; does not revise old data contract or old authorization',
        'artifacts':[{'path':REL+'/'+p.name,'sha256':sha(p)} for p in sorted(output.iterdir())],
        'execution_authorization':None,'GPU_UUID':None,'run_directory':None}
    write(output/'package.json',package)
    print(json.dumps(validation))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/REL);a=p.parse_args();build(a.output)
