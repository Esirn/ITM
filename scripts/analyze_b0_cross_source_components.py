#!/usr/bin/env python
"""Frozen all-source saved-state component/error audit; NumPy only, no models."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from evaluate_b0_native_training_sufficiency import read,verify,artifact,arrays,descriptor,matrix,write
from analyze_b0_suffix_responses import compact
from itm.metrics.training_sufficiency import temporal_decomposition,same_state_delta
from itm.metrics.feature_balance import subtract
from freeze_b0_cross_source_component_CPU import validate_population

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/itm_rework/protocol/b0_cross_source_component_CPU_20261009_v1/plan.json'
PLAN_SHA='49cf5ad9f5738aa653ef07ababa6100e5027e7e3892aad6f4e450a34efb7f9de'
WEIGHTS={'F':'d0e4f39f8c7ee5b980243baac48bc1bc02dc8f7c15c7a1f47f0adabb0b7e0ad3','U':'91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642'}
NAMES=('x_t','f00','f10','f01','f11','actual_combined_prediction')
KEYS=('data_split','motion_id','sensor_config','condition','frames')


def validate_raw(job,meta,z,slot,inputs):
    if any(meta[k]!=job[k] for k in ('trace_id','case_id','timestep','condition')) or meta['status']!='completed' or meta['scaled_t']!=[job['timestep']]:raise ValueError('Original trace/time/condition identity')
    if meta['checkpoint_sha256']!=WEIGHTS[job['arm']]:raise ValueError('Original F/U weight identity')
    if meta['guidance']!=slot['guidance'] or meta['random_plan_sha256']!=slot['historical_random_plan_sha256'] or meta['input_hashes']!=inputs:raise ValueError('Original CFG/noise/inputs identity')
    if set(z)!=set(NAMES)|{'frame_mask'} or meta['arrays']!={k:descriptor(v) for k,v in z.items()}:raise ValueError('Original raw descriptors')
    if descriptor(z['actual_combined_prediction'])['sha256']!=meta['actual_return_sha256']:raise ValueError('Native returned array identity')
    for k in NAMES:
        if z[k].dtype!=np.float32 or z[k].shape!=(1,263,1,job['frames']) or not np.isfinite(z[k]).all():raise ValueError('Original finite float32 full frames')
    mask=z['frame_mask']
    if mask.dtype!=np.bool_ or mask.shape!=(1,job['frames']) or not mask.all():raise ValueError('Original full-frame mask')


def inherited(plan):
    folder=Path(plan['inherited_results']['directory']);verify(plan['inherited_results']['artifact_index']);proof=read(verify(plan['inherited_results']['proof']))
    if proof['status']!='passed' or proof['trace_points']!=480 or proof['final_samples']!=120:raise ValueError('Original complete formula proof')
    for a in read(folder/'artifact_index.json'):verify(a)
    rows=read(folder/'traces.json');validate_population(rows)
    for j,r in zip(plan['jobs'],rows):
        if any(j[k]!=r[k] for k in j if k not in ('raw','metadata')):raise ValueError('Inherited complete roles/order')
    return rows,read(folder/'samples.json')


def gate(plan):
    if artifact(PLAN)['sha256']!=PLAN_SHA or len(plan['jobs'])!=480 or plan['population']['F_U_pairs']!=240:raise ValueError('Exact frozen plan')
    for a in plan['sources']:verify(a)
    return inherited(plan)


def inputs_and_targets(plan):
    slots={r['case_id']:r for r in read(ROOT/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/sampling_slots.json')}
    status=read(ROOT/'outputs/itm_rework/method/feature_balance_authorized_20261008_v1/runtime/status.json')
    supplied={r['case_id']:r['arrays'] for r in read(verify(status['implementation']))['inputs']}
    mean,std=[np.load(verify(plan['normalization'][k]),allow_pickle=False) for k in ('motion_mean','motion_std')]
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std>0).all():raise ValueError('Frozen normalization')
    targets={cid:(np.load(Path(slots[cid]['gt_path']).with_name('motion.npy'),allow_pickle=False)-mean)/std for cid in {j['pair_case_id'] for j in plan['jobs']}}
    return slots,supplied,targets


def prepare(output):
    if output.exists():raise FileExistsError('Fresh CPU implementation required')
    plan=read(PLAN);gate(plan);slots,supplied,_=inputs_and_targets(plan);sources={a['path']:a for a in plan['sources']}
    for p in [PLAN,Path(__file__).resolve(),ROOT/'scripts/verify_b0_cross_source_components.py',ROOT/'scripts/summarize_b0_cross_source_components.py',ROOT/'tests/test_cross_source_component_adapter.py']:
        a=artifact(p);sources[a['path']]=a
    for j in plan['jobs']:
        slot=slots[j['pair_case_id']];inputs=supplied[j['pair_case_id']] if j['arm']=='F' else slot['historical_input_hashes']
        validate_raw(j,read(verify(j['metadata'])),arrays(verify(j['raw'])),slot,inputs)
    output.mkdir(parents=True)
    write(output/'implementation.json',dict(version='itm-cross-source-component-CPU-v1',plan=artifact(PLAN),expected_artifacts=[sources[k] for k in sorted(sources)],raw_jobs_validated=480,F_U_pairs=240,new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0))
    print(dict(implementation=artifact(output/'implementation.json'),sources=len(sources)),flush=True)


def component_terms(base,paired,main,joint,u,j):
    from itm.metrics.training_sufficiency import VIEWS
    uc,jc=u-u.mean(0),j-j.mean(0);out={}
    for view in base:
        if view=='balanced':continue
        ids=VIEWS[view];b,p,m,n=[v[view] for v in (base,paired,main,joint)]
        out[view]=dict(baseline_energy=b['prediction_temporal_energy'],target_energy=b['target_temporal_energy'],
            main_delta_energy=m['prediction_temporal_energy']-b['prediction_temporal_energy'],joint_delta_energy=n['prediction_temporal_energy']-b['prediction_temporal_energy'],
            combined_delta_energy=p['prediction_temporal_energy']-b['prediction_temporal_energy'],twice_centered_main_joint_covariance=float(2*np.mean(uc[:,ids]*jc[:,ids])),
            baseline_mse=b['mse'],main_delta_mse=m['mse']-b['mse'],joint_delta_mse=n['mse']-b['mse'],combined_delta_mse=p['mse']-b['mse'],twice_raw_main_joint_inner_product=float(2*np.mean(u[:,ids]*j[:,ids])),
            main_delta_temporal_error=m['temporal_error']-b['temporal_error'],joint_delta_temporal_error=n['temporal_error']-b['temporal_error'],main_delta_mean_error=m['mean_error']-b['mean_error'],joint_delta_mean_error=n['mean_error']-b['mean_error'])
    out['balanced']={k:.5*out['root_ric67'][k]+.5*out['remaining196'][k] for k in out['root_ric67']}
    for view,m in out.items():
        ge=m['target_energy']
        for k in ('main_delta_energy','joint_delta_energy','combined_delta_energy','twice_centered_main_joint_covariance'):m[k+'_over_GT']=m[k]/ge if ge>0 else None
        for total,terms in [('combined_delta_energy',['main_delta_energy','joint_delta_energy','twice_centered_main_joint_covariance']),('combined_delta_mse',['main_delta_mse','joint_delta_mse','twice_raw_main_joint_inner_product'])]:
            error=abs(m[total]-sum(m[k] for k in terms));scale=max(1.,*(abs(m[k]) for k in terms+[total,'baseline_energy','target_energy','baseline_mse']))
            if error>64*np.finfo(float).eps*scale:raise ArithmeticError('Inherited component energy/error identity')
    return out


def pair_rows(rows):
    table={(r['pair_case_id'],r['timestep'],r['arm']):r for r in rows}
    if len(table)!=480:raise ValueError('Unique full row denominator')
    result=[]
    for case,t in sorted({(r['pair_case_id'],r['timestep']) for r in rows}):
        f,u=[table[case,t,a] for a in ('F','U')]
        if any(f[k]!=u[k] for k in KEYS):raise ValueError('Paired source/role/frame mismatch')
        valid=f['status']==u['status']=='valid'
        p=dict(pair_case_id=case,timestep=t,F_row=f['row_id'],U_row=u['row_id'],status='valid' if valid else 'inconclusive',source_n=1,**{k:f[k] for k in KEYS})
        if valid:p.update(component_terms=subtract(f['component_terms'],u['component_terms']),hypotheses=subtract(f['hypotheses'],u['hypotheses']))
        result.append(p)
    if len(result)!=240:raise ValueError('Full pair denominator')
    return result


def aggregates(rows):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in ('arm','data_split','sensor_config','condition','timestep'))].append(r)
    out=[]
    for key,rs in sorted(groups.items()):
        if len({r['motion_id'] for r in rs})!=len(rs):raise ValueError('Source equal-weight duplicate')
        metrics={}
        for view in ('uniform','root_ric67','remaining196','root','ric','rotation6d','local_velocity','contact','ric_legs','ric_trunk_head','ric_arms','balanced'):
            metrics[view]={}
            names=set().union(*(r.get('component_terms',{}).get(view,{}) for r in rs))
            for name in sorted(names):
                v=[r['component_terms'][view][name] for r in rs if r['status']=='valid' and r['component_terms'][view][name] is not None]
                metrics[view][name]=dict(planned_n=len(rs),valid_n=len(v),mean=float(np.mean(v)) if v else None,min=min(v) if v else None,max=max(v) if v else None)
        out.append(dict(zip(('arm','data_split','sensor_config','condition','timestep'),key),planned_sources=len(rs),valid_sources=sum(r['status']=='valid' for r in rs),sources=[r['motion_id'] for r in rs],metrics=metrics))
    return out


def evaluate(manifest_path,output):
    if output.exists():raise FileExistsError('Fresh evaluation required')
    manifest=read(manifest_path)
    for a in manifest['expected_artifacts']:verify(a)
    plan=read(verify(manifest['plan']));old,samples=gate(plan);slots,supplied,targets=inputs_and_targets(plan);rows=[]
    for job,reference in zip(plan['jobs'],old):
        row=dict(job,row_id=job['arm']+':'+job['trace_id'],status='missing',source_n=1)
        try:
            z=arrays(verify(job['raw']));slot=slots[job['pair_case_id']];inputs=supplied[job['pair_case_id']] if job['arm']=='F' else slot['historical_input_hashes']
            validate_raw(job,read(verify(job['metadata'])),z,slot,inputs)
            f={k:matrix(z[k]).astype(float) for k in ('f00','f10','f01','f11')};a=f['f10'];u=f['f01']-f['f00'];j=f['f11']-f['f10']-f['f01']+f['f00'];g=targets[job['pair_case_id']]
            base=reference['metrics']['temporal']['f10'];paired=reference['metrics']['temporal']['f11'];hyp={name:dict(temporal=temporal_decomposition(p,g),error_delta=same_state_delta(a,p,g)) for name,p in [('main_only',a+u),('joint_only',a+j)]}
            row.update(status='valid',hypotheses=hyp,component_terms=component_terms(base,paired,hyp['main_only']['temporal'],hyp['joint_only']['temporal'],u,j))
        except (ValueError,KeyError,OSError,ArithmeticError) as e:row.update(status='invalid',reason=str(e))
        rows.append(row)
    pairs=pair_rows(rows);summary=dict(status='completed' if all(r['status']=='valid' for r in rows+pairs) else 'inconclusive',planned_rows=480,row_counts=dict(Counter(r['status'] for r in rows)),planned_pairs=240,pair_counts=dict(Counter(r['status'] for r in pairs)),paired_rows=96,paired_pairs=48,condition_counts=dict(Counter(r['condition'] for r in rows)),source_count=6,configs=12,old_final_references=len(samples),new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,interpretation=plan['interpretation'],original_behavior='Original F and new main-knockout not_supported unchanged; no semantic relabeling.')
    output.mkdir(parents=True)
    for name,data in [('rows',rows),('pairs',pairs),('aggregates',aggregates(rows)),('old_final_references',samples),('summary',summary),('sources',[artifact(manifest_path),*manifest['expected_artifacts']])]:compact(output/(name+'.json'),data)
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.glob('*.json'))]);print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-dir',type=Path);p.add_argument('--implementation',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
    if args.prepare_dir:prepare(args.prepare_dir.resolve())
    else:evaluate(args.implementation.resolve(),args.output.resolve())
