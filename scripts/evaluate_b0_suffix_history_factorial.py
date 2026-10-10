#!/usr/bin/env python
"""CPU evaluation retaining every frozen suffix final/trace and old negative row."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from evaluate_b0_native_training_sufficiency import read,verify,artifact,arrays,descriptor,matrix,write
from itm.metrics.training_sufficiency import temporal_decomposition
from itm.metrics.feature_balance import feature_views,motion_metrics,curves,gate,subtract,SCALARS

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'docs/itm_rework/protocol/b0_suffix_history_factorial_20261008_v2/package.json'
PACKAGE_SHA='abd7894aaf4fdc90f7fa35d595b51152fb8a9533c64accbed0c0e5928034d40c'


def evaluate(run,output):
    if output.exists():raise FileExistsError('Fresh suffix evaluation required')
    if artifact(PACKAGE)['sha256']!=PACKAGE_SHA:raise ValueError('Frozen suffix package')
    package=read(PACKAGE);files={Path(a['path']).stem:verify(a) for a in package['artifacts']}
    sources={}
    def bind(p):
        a=artifact(p);sources[a['path']]=a;return p
    bind(PACKAGE);bind(Path(__file__).resolve())
    for a in read(files['sources']):verify(a);sources[a['path']]=a
    status=read(bind(run/'runtime/status.json'));manifest=read(bind(verify(status['implementation'])))
    for a in manifest['expected_artifacts']:verify(a);sources[a['path']]=a
    eligible=status['status']=='completed' and all(r['status']=='completed' for r in status['checks'])
    if status.get('receipt'):
        bind(verify(status['receipt']));bind(verify(status['receipt_journal']))
    slots={r['case_id']:r for r in read(bind(ROOT/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/sampling_slots.json'))}
    oldplan=read(bind(ROOT/'docs/itm_rework/protocol/b0_saved_trace_response_20261008_v1/plan.json'))
    mean,std=[np.load(bind(verify(oldplan['normalization'][k])),allow_pickle=False) for k in ('motion_mean','motion_std')]
    reference_dir=ROOT/'outputs/itm_rework/protocol/feature_balance_evaluation_20261008_v2'
    for a in read(bind(reference_dir/'artifact_index.json')):bind(verify(a))
    oldrows=read(bind(reference_dir/'samples.json'))['rows'];lookup={r['case_id']:r for r in oldrows};finals=[];traces=[]
    table={r['id']:r for r in status['finals']};trace_table={r['id']:r for r in status['traces']}
    for trajectory in read(files['trajectories']):
        tid=trajectory['trajectory_id'];cid=trajectory['panel_case_id'];slot=slots[cid]
        g=np.load(bind(Path(slot['gt_path'])),allow_pickle=False)
        raw_gt=np.load(bind(Path(slot['gt_path']).with_name('motion.npy')),allow_pickle=False);target=(raw_gt-mean)/std
        row=dict(trajectory,status='missing',case_id=tid)
        entry=table[tid]
        if entry.get('raw_saved'):
            try:
                path=bind(verify(entry['raw_artifact']));z=arrays(path)
                if set(z)!=set(['normalized_humanml263','joints','frame_mask']):raise ValueError('Final arrays changed')
                features=z['normalized_humanml263'];p=z['joints'];mask=z['frame_mask']
                if features.dtype!=np.float32 or features.shape!=(1,263,1,slot['frames']) or p.dtype!=np.float32 or p.shape!=(1,slot['frames'],22,3):raise ValueError('Final native shapes')
                if mask.dtype!=np.bool_ or mask.shape!=(1,slot['frames']) or not mask.all() or not np.isfinite(p).all():raise ValueError('Final mask/finite')
                state='valid' if eligible and entry['status']=='completed' and not entry.get('diagnostic_only') else 'diagnostic_only'
                row.update(status=state,raw=artifact(path),temporal=temporal_decomposition(matrix(features),target),
                    feature_views=feature_views(matrix(features),target),metrics=motion_metrics(p[0],g,slot['sensor_config'],slot['data_split'],slot['motion_id'],raw_gt),curves=curves(p[0]))
                if state=='valid':
                    meta=read(bind(path.with_suffix('.json')))
                    if meta['trajectory']!=trajectory or meta['arrays']!={k:descriptor(v) for k,v in z.items()} or meta['raw_artifact']!=entry['raw_artifact']:raise ValueError('Final metadata identity')
                    row['original_numeric_gates']={label:gate(row,ref) for label,ref in [
                        ('U_paired',lookup[slot['historical_case_id']]),
                        *[(condition,next(r for r in oldrows if r['arm']=='F' and r['data_split']==slot['data_split'] and r['motion_id']==slot['motion_id'] and r['sensor_config']==slot['sensor_config'] and r['condition']==condition)) for condition in ('text_only','physical_static')]]}
            except (ValueError,KeyError,OSError,ArithmeticError) as e:row.update(status='invalid',reason=str(e))
        finals.append(row)
        for t in trajectory['trace_timesteps']:
            key=tid+'-'+str(t);entry=trace_table[key];tr=dict(row_id=key,trajectory_id=tid,timestep=t,**{k:trajectory[k] for k in ('data_split','motion_id','sensor_config','condition')},status='missing')
            if entry.get('raw_saved'):
                try:
                    path=bind(verify(entry['raw_artifact']));z=arrays(path)
                    mask=z['frame_mask'][0]
                    if mask.dtype!=np.bool_ or not mask.all() or z['actual_combined_prediction'].shape!=(1,263,1,slot['frames']):raise ValueError('Trace full shape/mask')
                    tr.update(status='valid' if eligible and entry['status']=='completed' and not entry.get('diagnostic_only') else 'diagnostic_only',raw=artifact(path),
                        temporal={k:temporal_decomposition(matrix(z[k]),target,mask) for k in ('x_t','f00','f10','f01','f11','actual_combined_prediction')})
                except (ValueError,KeyError,OSError,ArithmeticError) as e:tr.update(status='invalid',reason=str(e))
            traces.append(tr)
    contrasts=[]
    for config in ('head','wrists'):
        off=next(r for r in finals if r['sensor_config']==config and r['condition']=='switch_imu_on');keep=next(r for r in finals if r['sensor_config']==config and r['condition']=='keep_text_only')
        valid=off['status']==keep['status']=='valid'
        contrasts.append(dict(sensor_config=config,status='valid' if valid else 'inconclusive',
            metrics={k:subtract(off['metrics'][k],keep['metrics'][k]) for k in SCALARS} if valid else None,
            temporal=subtract(off['temporal'],keep['temporal']) if valid else None,
            interpretation='Same T history/weights/noise; IMU-on suffix minus continued text-only. Temporal energy recovery alone is not behavior success.'))
    # Preserve the completed F-prefix cells as references, never new replications.
    prior=ROOT/'outputs/itm_rework/protocol/suffix_switch_evaluation_20261008_v1'
    for a in read(bind(prior/'artifact_index.json')):bind(verify(a))
    old_status=read(bind(ROOT/'outputs/itm_rework/method/suffix_switch_authorized_20261008_v1/runtime/status.json'))
    old_proof=read(bind(ROOT/'outputs/itm_rework/method/suffix_switch_implementation_20261008_v2/result_verification.json'))
    if old_status['status']!='completed' or old_proof['status']!='passed' or old_proof['software_equations']!=7244:
        raise ValueError('Complete previous F-prefix cells required')
    f_cells=read(prior/'finals.json')
    if len(f_cells)!=4 or any(r['status']!='valid' for r in f_cells):raise ValueError('Previous F-prefix cells incomplete')
    factorial_finals=[dict(r,prefix_history='F_paired',suffix_condition='paired' if r['condition']=='keep_paired' else 'text_only',
        reference_kind='completed_previous_queue') for r in f_cells]
    factorial_finals += [dict(r,prefix_history='T_text_only',suffix_condition='paired' if r['condition']=='switch_imu_on' else 'text_only',
        reference_kind='new_queue') for r in finals]
    factorial=[]
    for config in ('head','wrists'):
        cells={(r['prefix_history'],r['suffix_condition']):r for r in factorial_finals if r['sensor_config']==config}
        required=[('F_paired','paired'),('F_paired','text_only'),('T_text_only','paired'),('T_text_only','text_only')]
        complete=all(cells[k]['status']=='valid' for k in required)
        record=dict(sensor_config=config,status='valid' if complete else 'inconclusive',source_n=1,
            cells={h+'__'+s:cells[(h,s)]['trajectory_id'] for h,s in required},
            interpretation='Paired-minus-text suffix effect under each saved history. Interaction=T-history effect minus F-history effect; one source/seed, signed association without mechanism percentages or model efficacy certification.')
        if complete:
            effects={}
            for history in ('F_paired','T_text_only'):
                paired,text=cells[(history,'paired')],cells[(history,'text_only')]
                effects[history]=dict(metrics={k:subtract(paired['metrics'][k],text['metrics'][k]) for k in SCALARS},
                    temporal=subtract(paired['temporal'],text['temporal']))
            record.update(suffix_effects=effects,interaction=subtract(effects['T_text_only'],effects['F_paired']),
                history_difference_under_paired=dict(metrics={k:subtract(cells[('T_text_only','paired')]['metrics'][k],cells[('F_paired','paired')]['metrics'][k]) for k in SCALARS},
                    temporal=subtract(cells[('T_text_only','paired')]['temporal'],cells[('F_paired','paired')]['temporal'])))
        factorial.append(record)
    summary=dict(status='completed' if eligible and all(r['status']=='valid' for r in finals+traces) else 'inconclusive',
        queue_status=status['status'],planned_finals=4,final_counts=dict(Counter(r['status'] for r in finals)),planned_traces=12,trace_counts=dict(Counter(r['status'] for r in traces)),
        new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,old_final_references=len(oldrows),
        behavior='Requires all-frame rough review; this evaluator assigns no semantic label. Original F not_supported unchanged.')
    output.mkdir(parents=True)
    for name,data in [('finals',finals),('traces',traces),('contrasts',contrasts),('factorial_finals',factorial_finals),('factorial_contrasts',factorial),
        ('old_final_references',oldrows),('summary',summary),('sources',[sources[k] for k in sorted(sources)])]:write(output/(name+'.json'),data)
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.glob('*.json'))]);print(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();evaluate(a.run.resolve(),a.output.resolve())
