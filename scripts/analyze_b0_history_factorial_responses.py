#!/usr/bin/env python
"""Saved history/suffix responses on CPU; reuse verified F rows without rerunning them."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from evaluate_b0_native_training_sufficiency import read,verify,artifact,arrays,matrix,write
from analyze_b0_suffix_responses import compact,validate_raw as original_validate,NAMES
from itm.metrics.training_sufficiency import temporal_decomposition
from itm.metrics.trace_response import response_decomposition

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/itm_rework/protocol/b0_history_factorial_response_CPU_20261009_v1/plan.json'
PLAN_SHA='2ac7d1f050970a53ab153ea6fe351b5a0d5095bb424429c983b8b5a23b5ef8fc'
FIELDS=('mse','mean_error','temporal_error','prediction_temporal_energy','target_temporal_energy','temporal_covariance')
HISTORIES=('F_paired','T_text_only')

def role(row,inherited=False):
    roles={'keep_paired':('F_paired','paired',True),'switch_imu_off':('F_paired','text_only',False)} if inherited else {'keep_text_only':('T_text_only','text_only',False),'switch_imu_on':('T_text_only','paired',True)}
    try:h,s,p=roles[row['condition']]
    except KeyError:raise ValueError('Frozen history/suffix role')
    if not inherited and (row['prefix_history']!=h or row['suffix_condition']!=s or row['imu_present'] is not p):raise ValueError('Frozen history/suffix/presence identity')
    return h,s,p

def validate_raw(job,meta,z):
    _,_,present=role(job)
    original_validate(dict(job,condition='keep_paired' if present else 'switch_imu_off'),meta,z)

def load_inherited(plan):
    ref=plan['inherited_F_results'];verify(ref['implementation']);proof=read(verify(ref['proof']))
    if proof['status']!='passed' or proof['rows']!=3600 or proof['independent_scalar_checks']!=4246200:raise ValueError('Full inherited F formula proof required')
    verify(proof['verifier'])
    folder=Path(ref['directory'])
    for a in read(folder/'artifact_index.json'):verify(a)
    rows=read(folder/'rows.json');contrasts=read(folder/'path_contrasts.json')
    if len(rows)!=3600 or len(contrasts)!=1800 or any(r['status']!='valid' or r['source_n']!=1 for r in rows+contrasts):raise ValueError('Inherited F denominators')
    oldplan=read(verify(read(verify(ref['implementation']))['plan']))
    if len(oldplan['jobs'])!=len(rows):raise ValueError('Inherited jobs denominator')
    for job,row in zip(oldplan['jobs'],rows):
        if any(job[k]!=row[k] for k in job):raise ValueError('Inherited row identity')
        h,s,p=role(row,True);row.update(prefix_history=h,suffix_condition=s,imu_present=p,reused=True)
    complete_table(rows)
    return rows,contrasts

def complete_table(rows):
    table={}
    for r in rows:
        key=(r['prefix_history'],r['sensor_config'],r['timestep'],r['suffix_condition'])
        if key in table:raise ValueError('Duplicate history/config/time/suffix')
        table[key]=r
    hs={r['prefix_history'] for r in rows}
    expected={(h,c,t,s) for h in hs for c in ('head','wrists') for t in range(900) for s in ('paired','text_only')}
    if set(table)!=expected:raise ValueError('Incomplete or extra effective cells')
    return table

def gate(plan):
    if artifact(PLAN)['sha256']!=PLAN_SHA or len(plan['jobs'])!=3600 or plan['total_effective_rows']!=7200 or plan['timesteps']!=list(range(899,-1,-1)):raise ValueError('Exact complete CPU plan')
    for a in plan['sources']:verify(a)
    proof=read(verify(plan['software_gate']));verify(proof['verifier'])
    factorial=read(verify(plan['factorial_gate']));verify(factorial['verifier'])
    if proof['status']!='passed' or proof['software_equations']!=7244 or proof['posterior_transition_hash_links']!=3600 or factorial['status']!='passed' or factorial['factorial_cells']!=8:raise ValueError('Finished exact GPU software proofs')
    status=read(Path(plan['run_directory'])/'runtime/status.json')
    if status['status']!='completed' or status['implementation']!=plan['implementation'] or any(r['status']!='completed' for r in status['checks']):raise ValueError('Complete exact suffix queue required')
    table={r['id']:r for r in status['steps']}
    for job in plan['jobs']:
        role(job);slot=table[job['row_id']]
        if slot['status']!='completed' or slot['diagnostic_only'] or slot['raw_artifact']!=job['raw']:raise ValueError('Frozen raw slot identity')
    return status

def prepare(output):
    if output.exists():raise FileExistsError('Fresh implementation required')
    plan=read(PLAN);gate(plan);load_inherited(plan);sources={a['path']:a for a in plan['sources']}
    paths=[PLAN,Path(__file__).resolve(),ROOT/'scripts/verify_b0_history_factorial_responses.py',ROOT/'scripts/summarize_b0_history_factorial_responses.py',ROOT/'tests/test_history_factorial_response_adapter.py',ROOT/'scripts/analyze_b0_suffix_responses.py',ROOT/'scripts/evaluate_b0_native_training_sufficiency.py',ROOT/'src/itm/metrics/training_sufficiency.py',ROOT/'src/itm/metrics/trace_response.py']
    for key in ('software_gate','factorial_gate'):paths.append(verify(read(verify(plan[key]))['verifier']))
    paths.append(verify(read(verify(plan['inherited_F_results']['proof']))['verifier']))
    for path in paths:
        a=artifact(path);sources[a['path']]=a
    for job in plan['jobs']:validate_raw(job,read(verify(job['metadata'])),arrays(verify(job['raw'])))
    output.mkdir(parents=True)
    write(output/'implementation.json',dict(version='itm-history-factorial-response-CPU-v1',plan=artifact(PLAN),expected_artifacts=[sources[k] for k in sorted(sources)],new_rows=3600,inherited_F_rows=3600,total_effective_rows=7200,validated_raw_jobs=3600,new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0))
    print(dict(implementation=artifact(output/'implementation.json'),dependencies=len(sources)),flush=True)

def difference(paired,text):
    return {v:{k:paired['temporal']['actual_combined_prediction'][v][k]-text['temporal']['actual_combined_prediction'][v][k] for k in FIELDS} for v in paired['temporal']['actual_combined_prediction']}

def build_contrasts(rows,state_rms):
    table=complete_table(rows);contrasts=[];effects={};interactions=[]
    for h in HISTORIES:
        for c in ('head','wrists'):
            for t in range(899,-1,-1):
                p,n=[table[h,c,t,s] for s in ('paired','text_only')]
                key=(h,c,t);valid=all(r['status']=='valid' for r in (p,n)) and key in state_rms
                r=dict(prefix_history=h,sensor_config=c,timestep=t,status='valid' if valid else 'inconclusive',source_n=1,comparison='paired_minus_text',row_ids=[p['row_id'],n['row_id']],shared_state=(t==899))
                if valid:r.update(actual_prediction=difference(p,n),state_difference_rms=state_rms[key])
                contrasts.append(r);effects[key]=r
    for c in ('head','wrists'):
        for t in range(899,-1,-1):
            f,n=[effects[h,c,t] for h in HISTORIES];valid=f['status']==n['status']=='valid'
            r=dict(sensor_config=c,timestep=t,status='valid' if valid else 'inconclusive',source_n=1,comparison='T_effect_minus_F_effect',shared_state=False)
            if valid:r['actual_prediction']={v:{k:n['actual_prediction'][v][k]-f['actual_prediction'][v][k] for k in FIELDS} for v in f['actual_prediction']}
            interactions.append(r)
    return contrasts,interactions

def evaluate(manifest_path,output):
    if output.exists():raise FileExistsError('Fresh evaluation required')
    manifest=read(manifest_path)
    for a in manifest['expected_artifacts']:verify(a)
    plan=read(verify(manifest['plan']));gate(plan);oldrows,oldcontrasts=load_inherited(plan)
    oldplan=read(ROOT/'docs/itm_rework/protocol/b0_saved_trace_response_20261008_v1/plan.json')
    mean,std=[np.load(verify(oldplan['normalization'][k]),allow_pickle=False) for k in ('motion_mean','motion_std')]
    slots={c['case_id']:c for c in read(ROOT/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/sampling_slots.json')}
    targets={cid:(np.load(Path(slots[cid]['gt_path']).with_name('motion.npy'),allow_pickle=False)-mean)/std for cid in {j['panel_case_id'] for j in plan['jobs']}}
    state_rms={('F_paired',r['sensor_config'],r['timestep']):r['state_difference_rms'] for r in oldcontrasts};rows=[];states={}
    for job in plan['jobs']:
        row=dict(job,status='missing',source_n=1,reused=False)
        try:
            z=arrays(verify(job['raw']));meta=read(verify(job['metadata']));validate_raw(job,meta,z)
            if meta['implementation_sha256']!=plan['implementation']['sha256']:raise ValueError('Saved implementation identity')
            branches={k:matrix(z[k]) for k in NAMES};g=targets[job['panel_case_id']]
            row.update(status='valid',temporal={k:temporal_decomposition(v,g) for k,v in branches.items()},response=response_decomposition(branches))
            key=('T_text_only',job['sensor_config'],job['timestep']);prior=states.pop(key,None)
            if prior is None:states[key]=branches['x_t'].astype(float)
            else:
                d=branches['x_t'].astype(float)-prior
                state_rms[key]={v:float(np.sqrt(np.mean(d[:,ids]**2))) for v,ids in [('uniform',range(263)),('root_ric67',range(67)),('remaining196',range(67,263))]}
        except (ValueError,KeyError,OSError,ArithmeticError) as e:row.update(status='invalid',reason=str(e))
        rows.append(row)
        if len(rows)%900==0:print('New CPU rows',len(rows),flush=True)
    contrasts,interactions=build_contrasts(oldrows+rows,state_rms)
    valid=all(r['status']=='valid' for r in rows+contrasts+interactions)
    summary=dict(status='completed' if valid else 'inconclusive',new_rows=3600,inherited_F_rows=3600,total_effective_rows=7200,row_counts=dict(Counter(r['status'] for r in rows)),within_history_path_contrasts=3600,contrast_counts=dict(Counter(r['status'] for r in contrasts)),per_step_factorial_interactions=1800,interaction_counts=dict(Counter(r['status'] for r in interactions)),source_count=1,panel_configs=2,new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,interpretation=plan['interpretation'],original_behavior=plan['scientific_result'],references=plan['references'])
    output.mkdir(parents=True)
    inherited=dict(plan['inherited_F_results'],rows=artifact(Path(plan['inherited_F_results']['directory'])/'rows.json'),artifact_index=artifact(Path(plan['inherited_F_results']['directory'])/'artifact_index.json'),row_count=3600,no_repeat_CPU_math=True)
    for name,data in [('rows',rows),('inherited_F',inherited),('path_contrasts',contrasts),('factorial_interactions',interactions),('summary',summary),('sources',[artifact(manifest_path),*manifest['expected_artifacts']])]:compact(output/(name+'.json'),data)
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.glob('*.json'))]);print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-dir',type=Path);p.add_argument('--implementation',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.prepare_dir:prepare(a.prepare_dir.resolve())
    else:evaluate(a.implementation.resolve(),a.output.resolve())
