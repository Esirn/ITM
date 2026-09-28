"""Existing arrays/JSON + NumPy only. No neural framework, forward, GPU or sampling."""
from pathlib import Path
from collections import defaultdict,Counter
from functools import lru_cache
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
RUN=ROOT/'outputs/itm_rework/method/sampling_alignment_authorized_20260927_v1'
OLD=ROOT/'outputs/itm_rework/method/correction_authorized_20260927_v2/R_static_reference_delta'
PROTOCOL=ROOT/'docs/itm_rework/protocol/b0_sampling_alignment_20260927_v1'
EVAL=ROOT/'docs/itm_rework/protocol/b0_sampling_alignment_evaluation_20260928_v1'
SOURCES=set()
PRIMARY=dict(root=list(range(4)),ric=list(range(4,67)),rotation6d=list(range(67,193)),local_velocity=list(range(193,259)),contact=list(range(259,263)))
BLOCKS=dict(all=list(range(263)),**PRIMARY)
for name,joints in dict(ric_legs=[1,2,4,5,7,8,10,11],ric_trunk_head=[3,6,9,12,15],ric_arms=[13,14,16,17,18,19,20,21]).items():
    BLOCKS[name]=[4+(j-1)*3+k for j in joints for k in range(3)]

def read(p):
    p=Path(p);SOURCES.add(p);return json.loads(p.read_text())
@lru_cache(None)
def artifact(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest())
def arrays(p):
    SOURCES.add(p)
    with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files if k!='metadata'}
def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def decomposition(base,conditioned,target):
    a,b,c=[np.asarray(v,np.float64).reshape(263,-1) for v in (base,conditioned,target)]
    assert a.shape==b.shape==c.shape and np.isfinite([a,b,c]).all()
    e=a-c;d=b-a;result={}
    for name,idx in BLOCKS.items():
        base_mse=float(np.mean(e[idx]**2));joint_mse=float(np.mean((b[idx]-c[idx])**2))
        cross=float(2*np.mean(e[idx]*d[idx]));energy=float(np.mean(d[idx]**2));delta=joint_mse-base_mse
        bound=64*np.finfo(float).eps*max(1,base_mse,joint_mse,abs(cross),energy)
        assert abs(delta-cross-energy)<=bound
        result[name]=dict(dimensions=len(idx),baseline_mse=base_mse,conditioned_mse=joint_mse,
            delta_mse=delta,cross=cross,residual_energy=energy,weighted_delta=delta*len(idx)/263,
            identity_error=abs(delta-cross-energy),identity_bound=bound)
    assert abs(sum(result[k]['weighted_delta'] for k in PRIMARY)-result['all']['delta_mse'])<1e-12
    return result

def main():
    SOURCES.add(Path(__file__).resolve())
    assert sorted(sum(PRIMARY.values(),[]))==list(range(263))
    assert sorted(sum([BLOCKS[k] for k in ('ric_legs','ric_trunk_head','ric_arms')],[]))==list(range(4,67))
    for name in ('motion_process.py',):SOURCES.add(ROOT/'outputs/mdm/runtime_root/data_loaders/humanml/scripts'/name)
    SOURCES.add(ROOT/'outputs/mdm/runtime_root/data_loaders/humanml/utils/paramUtil.py')
    SOURCES.add(ROOT/'docs/itm_rework/protocol/B0_SAMPLING_ALIGNMENT_EVALUATION_20260928.md')
    checkpoint=OLD/'model.pt';SOURCES.add(checkpoint)
    assert artifact(checkpoint)['sha256']=='91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642'
    # Analytic helpful/harmful residuals and dimension-weighted partition identities.
    z=np.zeros((263,2));o=np.ones_like(z)
    assert decomposition(o,z,z)['all']['delta_mse']==-1
    assert decomposition(o,2*o,z)['all']['delta_mse']==3
    groups=read(PROTOCOL/'probe_groups.json');samples=read(PROTOCOL/'sampling_slots.json')
    frozen_traces=read(PROTOCOL/'trace_slots.json');evaluation=read(EVAL/'probes.json')
    expected={r['probe_id']:r['values']['x0_mse'] for r in evaluation['rows']}
    rows=[];targets={};target_artifacts={};checks=0
    for g in groups:
        loaded={pid:arrays(RUN/'Q'/(pid+'.npz')) for pid in g['probe_ids'][1:]}
        target=loaded[g['probe_ids'][1]]['target'];key=(g['data_split'],g['motion_id'],g['sensor_config'])
        if key in targets:assert np.array_equal(targets[key],target)
        else:targets[key]=target;target_artifacts[key]=artifact(RUN/'Q'/(g['probe_ids'][1]+'.npz'))
        base=loaded[g['probe_ids'][1]]['predicted_x0']
        for pid,condition in zip(g['probe_ids'][2:],('paired','physical_static','tail_reversed')):
            value=loaded[pid];assert np.array_equal(value['target'],target)
            blocks=decomposition(base,value['predicted_x0'],target)
            assert abs(blocks['all']['conditioned_mse']-expected[pid])<1e-12;checks+=1
            rows.append(dict(state_family='GT_noised_Q',data_split=key[0],motion_id=key[1],sensor_config=key[2],
                condition=condition,timestep=g['timestep'],replicate=g['noise_seed'],identity=pid,blocks=blocks))
    sample_map={r['case_id']:r for r in samples};plan=[];cached_cases=set()
    for t in frozen_traces:
        sample=sample_map[t['case_id']];key=(sample['data_split'],sample['motion_id'],sample['sensor_config'])
        npz=RUN/'traces'/(t['trace_id']+'.npz');meta=read(npz.with_suffix('.json'));a=arrays(npz)
        assert meta['checkpoint_sha256']==artifact(checkpoint)['sha256'] and meta['trace_id']==t['trace_id']
        assert a['x_t'].shape==a['f11'].shape and a['f11'].shape[1:]==targets[key].shape
        rows.append(dict(state_family='S1_actual_trace',data_split=key[0],motion_id=key[1],sensor_config=key[2],
            condition=sample['condition'],timestep=t['timestep'],replicate=sample['sampling_seed'],identity=t['trace_id'],
            blocks=decomposition(a['f10'],a['f11'],targets[key])))
        if sample['condition']=='paired':
            encoder_reservation=0 if key in cached_cases else 6
            cached_cases.add(key)
            plan.append(dict(group_id=f'V{len(plan)+1:03d}',status='proposed_not_frozen_not_authorized',data_split=key[0],motion_id=key[1],sensor_config=key[2],
                timestep=t['timestep'],frames=sample['frames'],shape=[4,263,1,sample['frames']],
                saved_state=artifact(npz),saved_state_metadata=artifact(npz.with_suffix('.json')),state_array='x_t',trace_id=t['trace_id'],
                original_sample=artifact(RUN/'samples'/(sample['case_id']+'.npz')),sample_spec=sample,
                target_artifact=target_artifacts[key],target_array='target',checkpoint=artifact(checkpoint),
                branch_order=['text_only','paired','physical_static','tail_reversed'],MDM_branch_examples=4,encoder_examples_max=encoder_reservation,condition_cache_key=list(key),
                extra_diffusion_steps=0,new_noise_draws=0,interpretation='Raw same-state diagnostic outputs only; no trajectory update or generation'))
    assert len(rows)==528 and len(plan)==48 and len(targets)==12
    assert sum(r['MDM_branch_examples'] for r in plan)==192
    assert sum(r['encoder_examples_max'] for r in plan)==72
    by_source=defaultdict(list)
    for r in rows:
        for name,b in r['blocks'].items():
            key=tuple(r[k] for k in ('state_family','data_split','sensor_config','condition','timestep','motion_id'))+(name,)
            by_source[key].append(b)
    source_rows=[]
    for k,values in sorted(by_source.items()):
        row=dict(zip(('state_family','data_split','sensor_config','condition','timestep','motion_id','block'),k));row['n_replicates']=len(values)
        for metric in ('baseline_mse','conditioned_mse','delta_mse','cross','residual_energy','weighted_delta'):
            row[metric]=float(np.mean([v[metric] for v in values]))
        source_rows.append(row)
    grouped=defaultdict(list)
    for r in source_rows:grouped[tuple(r[k] for k in ('state_family','data_split','sensor_config','condition','timestep','block'))].append(r)
    aggregates=[]
    for k,values in sorted(grouped.items()):
        r=dict(zip(('state_family','data_split','sensor_config','condition','timestep','block'),k));r['n_sources']=len(values);assert len(values)==3
        for metric in ('baseline_mse','conditioned_mse','delta_mse','cross','residual_energy','weighted_delta'):
            vals=[v[metric] for v in values];r[metric]=dict(mean=float(np.mean(vals)),min=min(vals),max=max(vals))
        aggregates.append(r)
    exposures=[]
    for i in range(1,201):
        update=read(OLD/f'update_{i:03d}.json');assert update['update']==i and update['optimizer_step_completed']
        exposures.extend(update['examples'])
    assert len(exposures)==600 and len({v['exposure_id'] for v in exposures})==600
    populations=[]
    for key in sorted({(r['motion_id'],r['sensor_config']) for r in exposures}):
        selected=[r for r in exposures if (r['motion_id'],r['sensor_config'])==key]
        cells=Counter((r['text_present'],r['imu_present'],r['timestep']//250) for r in selected)
        populations.append(dict(motion_id=key[0],sensor_config=key[1],exposures=len(selected),
            cells=[dict(text_present=t,imu_present=i,t_range=[250*b,250*b+249],count=n) for (t,i,b),n in sorted(cells.items())]))
    write('feature_partition.json',dict(blocks=BLOCKS,primary_partition=list(PRIMARY),units='Pretrained-normalized feature MSE; not meters or decoded motion quality. RIC subgroups overlap the primary RIC block.',joint_convention='HumanML22 chains in bound paramUtil.py'))
    write('residual_decomposition.json',dict(rows=rows,source_means=source_rows,aggregates=aggregates,interpretation='Descriptive post-evaluation analysis. GT target is a fixed reference on S1 states, not a Bayes denoiser target or counterfactual rollout. Q has two noise replicates, S1 one trajectory; no causal state-shift identification.'))
    write('training_population.json',dict(total_exposures=600,rows=populations,interpretation='Counts only; varying t/noise/dropout/weights preclude a fixed-population convergence curve.'))
    write('proposed_state_groups.json',plan)
    write('proposed_budget.json',dict(status='proposed_not_frozen_not_authorized',training_updates=0,new_checkpoints=0,new_samples=0,new_diffusion_steps=0,new_noise_draws=0,groups=48,MDM_calls=48,MDM_branch_examples=192,encoder_examples_max=72,GPUs=1,allocated_memory_bytes=24*1024**3,total_seconds=300,deadline_starts='Before preflight/load',stop='First timeout/OOM/nonfinite/identity/counter/software mismatch; retain all planned/failed/not-run outputs; no retries/fallback/warmup',run_directory=None,gpu_uuid=None,authorization=None))
    write('sources.json',[artifact(p) for p in sorted(SOURCES)])
    write('verification.json',dict(existing_Q_condition_records=288,existing_trace_records=240,all_feature_rows=528,normalized_MSE_matches_owner02=checks,paired_S1_states=48,planned_MDM_branch_examples=192,planned_encoder_examples_max=72,training_exposures=600,bound_sources=len(SOURCES),partition_covers_263_once=True,analytic_signed_cases=2,model_imports=0,new_model_forwards=0,new_training_updates=0,new_samples=0,GPU_queries=0))
    print('Verified 288 owner02 Q MSEs, 528 residual records, 48 proposed states and 600 training exposures; sources',len(SOURCES))
if __name__=='__main__':main()
