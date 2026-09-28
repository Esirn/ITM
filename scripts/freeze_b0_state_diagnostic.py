#!/usr/bin/env python
"""Owner02 immutable V protocol and independent saved-residual audit, NumPy/files only."""
import argparse
from collections import Counter
from contextlib import ExitStack
import builtins
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REL='docs/itm_rework/protocol/b0_state_diagnostic_20260928_v1'
PROPOSAL=ROOT/'docs/itm_rework/method/b0_state_diagnostic_20260928_v1'
OLD=ROOT/'docs/itm_rework/protocol/b0_sampling_alignment_20260927_v1'
RUN=ROOT/'outputs/itm_rework/method/sampling_alignment_authorized_20260927_v1'
CHECKPOINT_SHA='91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642'
BRANCHES=['T','P','S','R']
CONDITIONS=['text_only','paired','physical_static','tail_reversed']
TIMESTEPS=[900,500,250,50]
PRIMARY={'root':list(range(4)),'ric':list(range(4,67)),'rotation6d':list(range(67,193)),
         'local_velocity':list(range(193,259)),'contact':list(range(259,263))}
BLOCKS={'all':list(range(263)),**PRIMARY}
for name,joints in {'ric_legs':[1,2,4,5,7,8,10,11],'ric_trunk_head':[3,6,9,12,15],'ric_arms':[13,14,16,17,18,19,20,21]}.items():
    BLOCKS[name]=[4+3*(j-1)+k for j in joints for k in range(3)]

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def describe(a):return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=ah(a))
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files if k!='metadata'}
def decomposition(base,pred,target):
    raw=[np.asarray(x) for x in (base,pred,target)]
    if any(x.size==0 or x.size%263 or not np.isfinite(x).all() for x in raw):raise ValueError('Finite nonempty263 features required')
    x,y,g=[a.astype(np.float64).reshape(263,-1) for a in raw]
    if x.shape!=y.shape or y.shape!=g.shape:raise ValueError('Equal feature/frame shapes required')
    residual=y-x;error=x-g;result={}
    for name,ids in BLOCKS.items():
        baseline=float(np.mean(error[ids]**2));conditioned=float(np.mean((y[ids]-g[ids])**2))
        cross=float(2*np.mean(error[ids]*residual[ids]));energy=float(np.mean(residual[ids]**2));delta=conditioned-baseline
        bound=64*np.finfo(float).eps*max(1.,baseline,conditioned,abs(cross),energy)
        identity_error=abs(delta-cross-energy)
        if identity_error>bound:raise ValueError('Float64 identity bound exceeded')
        result[name]=dict(dimensions=len(ids),baseline_mse=baseline,conditioned_mse=conditioned,delta_mse=delta,
            cross=cross,residual_energy=energy,weighted_delta=delta*len(ids)/263,
            identity_error=identity_error,identity_bound=bound)
    weighted=sum(result[k]['weighted_delta'] for k in PRIMARY)
    bound=64*np.finfo(float).eps*max(1.,abs(result['all']['delta_mse']),sum(abs(result[k]['weighted_delta']) for k in PRIMARY))
    if abs(weighted-result['all']['delta_mse'])>bound:raise ValueError('Feature partition reconstruction failed')
    return result


def build(output):
    output=Path(output)
    if output.exists():raise FileExistsError('Frozen outputs cannot be overwritten')
    sources={}
    def bind(path,expected=None):
        path=(ROOT/Path(path)).resolve();key=str(path)
        if key not in sources:
            stored=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            sources[key]=dict(path=stored,sha256=sha(path))
        entry=sources[key]
        if expected is not None and entry['sha256']!=expected:raise ValueError('Source hash changed: '+key)
        return entry
    def verify(a):return bind(a['path'],a['sha256'])
    for a in read(PROPOSAL/'sources.json'):verify(a)
    for p in sorted(PROPOSAL.iterdir()):
        if p.is_file():bind(p)
    for p in [Path(__file__),ROOT/'tests/test_b0_state_diagnostic_protocol.py',ROOT/'docs/itm_rework/method/B0_STATE_DIAGNOSTIC_PROPOSAL_20260928.md',ROOT/'docs/itm_rework/protocol/B0_SAMPLING_ALIGNMENT_EVALUATION_20260928.md',ROOT/'docs/itm_rework/protocol/b0_sampling_alignment_evaluation_20260928_v1/probe_contrasts.json']:
        bind(p)
    oldpackage=read(OLD/'package.json');bind(OLD/'package.json','537895c45695a48cb4b5114941e4bf8cab3962c40be893fcc79b4c9eeb311d60')
    for a in oldpackage['artifacts']:verify(a)
    status=read(RUN/'runtime/status.json');bind(RUN/'runtime/status.json');assert status['status']=='completed'
    manifest=read(status['implementation']['path']);verify(status['implementation'])
    for a in manifest['expected_artifacts']:verify(a)
    common=read(OLD/'design.json')['common'];fixed=verify(manifest['checkpoint']);assert fixed['sha256']==CHECKPOINT_SHA
    partition=read(PROPOSAL/'feature_partition.json');assert partition['blocks']==BLOCKS and partition['primary_partition']==list(PRIMARY)
    # Independent recomputation, no proposal arithmetic function reused.
    residuals=read(PROPOSAL/'residual_decomposition.json');oldgroups=read(OLD/'probe_groups.json')
    qmap={pid:(g,i) for g in oldgroups for i,pid in enumerate(g['probe_ids'])}
    trace_map={t['trace_id']:t for t in read(OLD/'trace_slots.json')}
    sample_map={s['case_id']:s for s in read(OLD/'sampling_slots.json')}
    targets={};checks=[];loaded={}
    def load(p):
        p=Path(p)
        if str(p) not in loaded:loaded[str(p)]=arrays(p)
        return loaded[str(p)]
    for g in oldgroups:
        targets[g['data_split'],g['motion_id'],g['sensor_config']]=load(RUN/'Q'/(g['probe_ids'][1]+'.npz'))['target']
    for r in residuals['rows']:
        if r['state_family']=='GT_noised_Q':
            group,_=qmap[r['identity']];pred=load(RUN/'Q'/(r['identity']+'.npz'))
            base=load(RUN/'Q'/(group['probe_ids'][1]+'.npz'))['predicted_x0'];y=pred['predicted_x0'];target=pred['target']
        else:
            assert r['state_family']=='S1_actual_trace';z=load(RUN/'traces'/(r['identity']+'.npz'))
            base,y=z['f10'],z['f11'];target=targets[r['data_split'],r['motion_id'],r['sensor_config']]
        computed=decomposition(base,y,target)
        assert computed==r['blocks'],r['identity']
        checks.append(dict(state_family=r['state_family'],identity=r['identity'],condition=r['condition'],blocks=len(computed),exact=True))
    assert len(checks)==528 and Counter(r['state_family'] for r in checks)=={'GT_noised_Q':288,'S1_actual_trace':240}
    # Independently recount all historical exposures; no loss-curve or convergence inference.
    exposures=[]
    oldrun=ROOT/'outputs/itm_rework/method/correction_authorized_20260927_v2/R_static_reference_delta'
    for i in range(1,201):
        update=read(oldrun/f'update_{i:03d}.json');assert update['update']==i and update['optimizer_step_completed'];exposures+=update['examples']
    assert len({r['exposure_id'] for r in exposures})==len(exposures)==600
    pop=read(PROPOSAL/'training_population.json')
    for r in pop['rows']:
        subset=[e for e in exposures if (e['motion_id'],e['sensor_config'])==(r['motion_id'],r['sensor_config'])]
        counts=Counter((e['text_present'],e['imu_present'],e['timestep']//250) for e in subset)
        assert r['exposures']==len(subset)==100
        assert counts=={(c['text_present'],c['imu_present'],c['t_range'][0]//250):c['count'] for c in r['cells']}
    # Bind every proposed state, independently deriving the full paired population.
    selected=[t for t in trace_map.values() if sample_map[t['case_id']]['condition']=='paired']
    proposed=read(PROPOSAL/'proposed_state_groups.json');assert len(selected)==len(proposed)==48
    mean=np.array(common['imu_normalization']['acceleration_mean'],np.float32);std=np.array(common['imu_normalization']['acceleration_std'],np.float32)
    accepted=read(ROOT/'docs/itm_rework/method/canonical_training_input_20260925_v1.json');bind(ROOT/'docs/itm_rework/method/canonical_training_input_20260925_v1.json')
    mm=np.load(accepted['training_args']['mean']);ms=np.load(accepted['training_args']['std'])
    input_cases=[];seen={};groups=[];slots=[];software=[];qrefs=[]
    for n,(trace,p) in enumerate(zip(selected,proposed)):
        assert p['trace_id']==trace['trace_id'] and p['group_id']==f'V{n+1:03d}'
        sample=sample_map[trace['case_id']];assert sample==p['sample_spec']
        for k in ('saved_state','saved_state_metadata','original_sample','target_artifact','checkpoint'):verify(p[k])
        z=load(ROOT/p['saved_state']['path']);meta=read(ROOT/p['saved_state_metadata']['path']);original=load(ROOT/p['original_sample']['path'])
        target=load(ROOT/p['target_artifact']['path'])['target'];T=sample['frames'];key=(sample['data_split'],sample['motion_id'],sample['sensor_config'])
        assert p['saved_state']['sha256']==sha(RUN/'traces'/(trace['trace_id']+'.npz'))
        assert p['original_sample']['sha256']==sha(RUN/'samples'/(sample['case_id']+'.npz'))
        assert meta['checkpoint_sha256']==CHECKPOINT_SHA and meta['condition']=='paired' and meta['scaled_t']==[trace['timestep']]
        for k in ('x_t','f10','f11'):assert z[k].dtype==np.float32 and z[k].shape==(1,263,1,T) and describe(z[k])==meta['arrays'][k]
        assert z['frame_mask'].shape==(1,T) and z['frame_mask'].all()
        features=np.load(Path(sample['gt_path']).with_name('motion.npy'))
        np.testing.assert_array_equal(target,((features-mm)/ms).astype(np.float32).T[:,None,:])
        assert describe(original['gt'])==meta['input_hashes']['gt']
        expected_encoder=0 if key in seen else 6
        assert p['encoder_examples_max']==expected_encoder and p['MDM_branch_examples']==4
        if key not in seen:
            cache_id=f'C{len(seen)+1:02d}';seen[key]=cache_id
            a,o=original['acceleration'][0],original['orientation'][0];so=np.broadcast_to(original['original_orientation0'][None],o.shape)
            reference=np.concatenate([(np.zeros_like(a)-mean)/std,so.reshape(T,6,9)],-1)
            np.testing.assert_array_equal(original['normalized_reference_imu'][0],reference)
            branch_inputs=[]
            for branch,condition in zip(BRANCHES,CONDITIONS):
                ea,eo=a,o
                if branch=='S':ea,eo=np.zeros_like(a),so
                if branch=='R':indices=np.r_[0,np.arange(T-1,0,-1)];ea,eo=a[indices],o[indices]
                imu=np.concatenate([(ea-mean)/std,eo.reshape(T,6,9)],-1)
                branch_arrays={k:original[k] for k in ('original_orientation0','sensor_mask','timestamps_seconds','frame_mask','gt')}
                branch_arrays.update(acceleration=ea[None],orientation=eo[None],normalized_imu=imu[None],normalized_reference_imu=reference[None],
                    text_condition_present=np.array([True]),imu_condition_present=np.array([branch!='T']))
                branch_inputs.append(dict(branch=branch,condition=condition,text_present=True,imu_present=branch!='T',arrays={k:describe(v) for k,v in branch_arrays.items()},
                    encoder_examples_on_cache_build=0 if branch=='T' else 2))
            input_cases.append(dict(cache_id=cache_id,data_split=key[0],motion_id=key[1],sensor_config=key[2],frames=T,
                original_sample=verify(p['original_sample']),sample_spec=sample,caption=sample['text'],caption_index=sample['caption_index'],
                historical_noise_plan_identity=sample['historical_random_plan_sha256'],branches=branch_inputs,caption_embedding_batch=1,caption_encoder_calls=1,
                imu_encoder_examples=6,uses=4,cache_initial_group=p['group_id']))
        cache_id=seen[key]
        g={k:p[k] for k in ('group_id','data_split','motion_id','sensor_config','frames','timestep','trace_id','sample_spec')}
        g.update(status='frozen_not_executed',group_index=n,cache_id=cache_id,branch_order=BRANCHES,conditions=CONDITIONS,
            saved_state=verify(p['saved_state']),saved_state_metadata=verify(p['saved_state_metadata']),original_sample=verify(p['original_sample']),
            state_array='x_t',state_descriptor=describe(z['x_t']),target_artifact=verify(p['target_artifact']),target_array='target',target_descriptor=describe(target),
            frame_mask=describe(z['frame_mask']),historical_T=dict(array='f10',**describe(z['f10'])),historical_P=dict(array='f11',**describe(z['f11'])),
            input_batch_shape=[4,263,1,T],scaled_t=meta['scaled_t'],checkpoint=fixed,
            MDM_calls=1,MDM_branch_examples=4,encoder_examples_max=expected_encoder,caption_encoder_calls=1 if expected_encoder else 0)
        group_slots=[]
        for i,(b,condition) in enumerate(zip(BRANCHES,CONDITIONS)):
            sid=f'{g["group_id"]}-{b}';group_slots.append(sid)
            slots.append(dict(slot_id=sid,group_id=g['group_id'],group_index=n,branch_index=i,branch=b,condition=condition,
                cache_id=cache_id,data_split=key[0],motion_id=key[1],sensor_config=key[2],timestep=g['timestep'],
                frames=T,output_shape=[263,1,T],text_present=True,imu_present=b!='T',status='frozen_not_executed',MDM_branch_examples=1))
        g['slot_ids']=group_slots;groups.append(g)
        for label,branch,ref in [('P_vs_trace_f11','P','f11'),('T_vs_trace_f10','T','f10'),('S_vs_T','S','T')]:
            software.append(dict(check_id=g['group_id']+'-'+label,group_id=g['group_id'],branch=branch,reference=ref,
                reference_kind='same_V_group' if ref=='T' else 'saved_paired_trace',atol=1e-6,rtol=1e-6,
                rule='abs(new-reference)<=atol+rtol*abs(reference); no alignment',status='frozen_not_executed'))
        matches=[q for q in oldgroups if (q['data_split'],q['motion_id'],q['sensor_config'],q['timestep'])==(*key,g['timestep'])]
        assert len(matches)==2
        qrefs.append(dict(group_id=g['group_id'],existing_Q_groups=[dict(noise_seed=q['noise_seed'],group_id=q['group_id'],
            branches=[dict(branch=b,probe_id=pid,artifact=bind(RUN/'Q'/(pid+'.npz'))) for b,pid in zip(BRANCHES,q['probe_ids'][1:])]) for q in matches]))
    assert len(input_cases)==12 and len(slots)==192 and len(software)==144
    assert sum(g['encoder_examples_max'] for g in groups)==72
    tolerance=dict(hashes_masks_flags_IDs_counts='exact',state_target_saved_predictions='exact float32 array SHA256/dtype/shape; never regenerate or renormalize x_t',
        reconstructed_inputs=dict(acceleration='exact',orientation_atol=2e-7,normalized_imu_atol=2e-7,normalized_reference_atol=2e-7,rtol=0),
        branch_predictions=dict(atol=1e-6,rtol=1e-6,checks=144,bitwise_required=False,save=['bitwise','max_abs','RMS','count','finite_count','passed']),
        residual_identity='64*eps64*max(1,base_MSE,pred_MSE,abs(cross),residual_energy)',
        partition_identity='64*eps64*max(1,abs(total_delta),sum(abs(weighted_primary_delta)))')
    design=dict(version='canonical-B0-state-diagnostic-design-v1',read_proposal_commit='feb0d11',checkpoint=fixed,
        old_package=bind(OLD/'package.json'),old_implementation=verify(status['implementation']),common=common,
        inference=dict(mode='eval',dtype='float32',AMP=False,grad=False,dropout=False,mask_frames=False,CFG=False,
            batch_shape='[4,263,1,T]',branch_order=BRANCHES,all_branches_same_caption=True,
            timestep='saved original unscaled t; identity timestep map and no rescaling',length='true length; no pad or batch mixing',
            q_sample_calls=0,posterior_steps=0,new_noise_draws=0,new_samples=0,training_updates=0,new_checkpoints=0),
        cache=dict(semantics='R observed encoder minus raw a=0/original O[0] reference encoder; T strict-null, no encoder',
            reuse='12 source/config caches, consecutive four saved t each; P/S/R6 examples at first state only, 0 at later3',
            identity_fields=['checkpoint','source','config','normalization','observed_arrays','reference_arrays','sensor_mask','frame_mask','timestamps','caption_embedding','dtype','eval_mode'],
            text='batch1 original S1 caption encoder once per source/config,12 calls total; reuse unchanged tensor across all branches/t',
            T='raw paired audit arrays retained; IMU presence false and latent exactly zero; zero bias injection',
            reverse='after20Hz interpolation indices[0,T-1,...,1],time grid unchanged; static reference remains ORIGINAL O[0]'),
        tolerances=tolerance,
        output_contract=dict(raw_first='Save finite predictions/state/target/mask before software checks; preserve raw arrays and metadata on mismatch, mark all group slots/checks accurately.',
            arrays=['predicted_x0 per4branches float32[263,1,T]','exact x_t[1,263,1,T]','target[263,1,T]','frame_mask[1,T]'],
            references='Hash-addressed immutable source references may replace duplicate state/target/input arrays; all required arrays remain independently readable.',
            metadata=['package/implementation/checkpoint hashes','group/slot/cache IDs','caption and true presence flags','input/reference descriptors','saved state and source trace hashes','original sample/plan lineage (not new noise)','original and scaled t','dtype/shape/masks','all144 anchor checks','actual and reserved MDM/encoder/caption counts','runtime/deadline/peakallocated/status/failure/artifact hashes'],
            prohibition='No CFG, q_sample, re-noising, posterior/sampler, motion decode or extra verification forwards; no new trajectory or effective quality claim.'),
        CPU_acceptance=['48 batch4 groups exactly: same saved x_t, order T/P/S/R, exact t/length/masks, text batch1 cache12 and IMU cache72, T no encoder',
            'Reuse fixed-R nonchained loader without real pretrained forwards in fixtures; wrong state/checkpoint/input/cache identity rejected',
            'Fixture historical anchors and S/T checks pass; inject mismatches,missing artifacts,nonfinite,OOM,timeout,counter/cache failures to stop and preserve finite arrays/full192+48+144 ledgers',
            'No CFG/q_sample/noise/posterior/decode/train; counter reservation before each real call, no unlisted warmup/retry',
            'Prepare fresh manifest binds all new code/dependencies/input arrays; CPU-only preparation cannot execute or authorize'])
    evaluation=dict(version='canonical-B0-state-diagnostic-evaluation-v1',denominators=dict(groups=48,branch_predictions=192,software_checks=144,source_config_trajectories=12,train_groups=24,val_groups=24),
        target='Fixed GT normalized HumanML263 at original frame index; descriptive error only, not Bayes target of a free-state denoiser. No yaw/translation/time fit.',
        units='Pretrained-normalized feature MSE; not meters, physical IMU residual, semantic success or generated motion quality.',
        branches='All192 branch MSEs with all9 blocks (1728 cells), absolute predicted/target magnitude and residual RMS/max/count; preserve all raw signs.',
        contrasts=[dict(name=a+'-'+b,prediction=a,baseline=b,role=role) for a,b,role in [('P','S','primary'),('P','R','primary'),('P','T','supporting'),('R','S','supporting')]],
        contrast_counts=dict(rows=192,block_cells=1728,primary_rows=96,supporting_rows=96),
        decomposition='For EACH contrast and block: e=baseline-GT,d=prediction-baseline,delta=MSEprediction-MSEbaseline=2mean(e*d)+mean(d*d); also raw prediction difference RMS/max/count. Static does not mean zero GT error.',
        partition='Primary5 partition covers263 once; weighted block contribution uses dims/263 and sums to overall. RIC3 partition covers63 once; its contribution also dims/263, not extra terms to sum with primaryRIC. No post-hoc block/threshold choice.',
        aggregation='For V source/config/t/split: one trajectory, retain source rows; mean3 sources equally, min/max,n and all missing reasons. No pooling t or treating four states as independent samples. Feature/time mean only within fixed block and valid frames.',
        historical_Q='4 contrasts on existing Q96 groups (384rows), two noise replicates averaged within source first; match V by source/config/t and show both values and V-Q contrast difference (192rows), descriptive only. No replay or new calls.',
        incomplete='Keep48groups/192branches/144checks denominators; raw finite outputs from failed groups may be listed diagnostic_only, excluded from primary valid means; preserve planned denominator and n/missing. Missing counterpart is not_comparable, not zero or pass. No pruning poor finite outcomes.',
        decision_rules=['Q P-R favorable but V unfavorable: conditional ordering differs on these saved states; not controlled causal evidence for distribution shift or authorization for new-state training.',
            'V P-R favorable with old full action missing: local gain insufficient for complete action, not evidence to increase gain.',
            'P and R both worse than S on wrists clap: retain harmful response not requiring correct ordering; does not prove all IMU should be disabled or only legs should be masked.',
            'Finite near-zero difference is descriptive, not an equivalence claim without prespecified effect bounds; architectural S=T is not learned IMU benefit.',
            'No model correction implemented or new output generated: no new effectiveness certification; old negatives and full-sample results stand.'],
        mandatory_cases=['train000506 both configs full alternating lunges absent','train004728 right back hold/direction/amplitude','val000382 wrists lower-body deformation','val000139 paired versus text/static/reversed','val000433 kick-return; old numeric-zero/IMU-only negatives retained'],
        limitations='6 exposed sources,3 per split;1 fixed training seed and1 paired S1 trajectory/config; reversed is input perturbation not physical donor; no generalization/significance/IMU closure/compatible edit/real-device or repaired-motion claims.',new_motion_review='No new motions; link previous120-case review, do not invent a new semantic success rate.')
    budget=read(PROPOSAL/'proposed_budget.json');budget.update(version='canonical-B0-state-diagnostic-budget-v1',status='frozen_not_authorized',authorized=False,
        groups=48,branch_predictions=192,software_checks=144,caption_encoder_calls_max=12,
        deadline_rule='One absolute300s from launch BEFORE preflight/load, all48 states share it including cache/load/save/checks; never reset or transfer old budget.',
        counter_rule='Reserve before calls;48batch4MDM=192examples,72observed/referenceIMU examples,12batch1caption calls; no retries/warmup/extra anchors. Record completed/reserved counts separately.',
        ledger_rule='Predeclare48group,192branch and144software-check rows BEFORE preflight; first fault stops whole queue with completed/failed/not_run_after_stop and raw finite failures retained.',
        prerequisites=['01 implementation and CPU validation','02 incremental software acceptance','new explicit user authorization binding final implementation/package/unique directory/GPU','preflight'])
    validation=dict(read_proposal_commit='feb0d11',proposal_sources_verified=1073,independent_residual_records=528,blocks_per_record=9,
        independent_block_cells=4752,residual_blocks_exact=True,training_exposures_recounted=600,states=48,branches=192,software_checks=144,
        caches=12,planned_IMU_encoder_examples=72,planned_caption_encoder_calls=12,source_files_verified=len(sources),
        current_model_imports=0,current_checkpoint_deserializations=0,current_model_forwards=0,current_GPU_queries=0,current_new_samples=0)
    output.mkdir(parents=True)
    files={'design.json':design,'evaluation.json':evaluation,'budget.json':budget,'feature_partition.json':partition,'state_groups.json':groups,
        'branch_slots.json':slots,'input_cases.json':input_cases,'software_checks.json':software,'historical_Q.json':qrefs,
        'proposal_audit.json':dict(summary=validation,rows=checks),'validation.json':validation,'sources.json':[sources[k] for k in sorted(sources)]}
    for name,value in files.items():write(output/name,value)
    write(output/'package.json',dict(version='canonical-B0-state-diagnostic-package-v1',status='frozen_for_implementation_not_authorized',source_commit='feb0d11',
        artifacts=[dict(path=REL+'/'+p.name,sha256=sha(p)) for p in sorted(output.iterdir())],execution_authorization=None,run_directory=None,GPU_UUID=None))
    print(json.dumps(validation))


def guarded_build(output):
    original=builtins.__import__
    def restricted(name,*args,**kwargs):
        if name.split('.')[0] in ('torch','tensorflow','jax'):raise AssertionError('Neural import forbidden in freeze')
        return original(name,*args,**kwargs)
    def reject(*args,**kwargs):raise AssertionError('Process launch/GPU commands forbidden in freeze')
    with ExitStack() as stack:
        stack.enter_context(patch.object(builtins,'__import__',restricted))
        for name in ('Popen','run','call','check_call','check_output'):stack.enter_context(patch.object(subprocess,name,reject))
        build(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/REL);a=p.parse_args();guarded_build(a.output)
