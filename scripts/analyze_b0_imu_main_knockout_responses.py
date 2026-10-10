#!/usr/bin/env python
"""All saved knockout responses; derived native111 comparison shares each saved state."""
import argparse
from pathlib import Path
import numpy as np
from evaluate_b0_native_training_sufficiency import read,verify,artifact,arrays,matrix,write
from analyze_b0_suffix_responses import compact,validate_raw,NAMES
from itm.metrics.training_sufficiency import temporal_decomposition,same_state_delta
from itm.metrics.trace_response import response_decomposition

ROOT=Path(__file__).resolve().parents[1]

def run(run_dir,output):
    if output.exists():raise FileExistsError('Fresh saved response output required')
    status=read(run_dir/'runtime/status.json')
    if status['status']!='completed' or any(c['status']!='completed' for c in status['checks']):raise ValueError('Complete two-path queue required')
    manifest=read(verify(status['implementation']))
    for a in manifest['expected_artifacts']:verify(a)
    files={Path(a['path']).stem:verify(a) for a in read(verify(manifest['package']))['artifacts']}
    paths=read(files['trajectories']);slots={r['case_id']:r for r in read(ROOT/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/sampling_slots.json')}
    norm=read(ROOT/'docs/itm_rework/protocol/b0_saved_trace_response_20261008_v1/plan.json')['normalization'];mean,std=[np.load(verify(norm[k]),allow_pickle=False) for k in ('motion_mean','motion_std')]
    targets={r['panel_case_id']:(np.load(Path(slots[r['panel_case_id']]['gt_path']).with_name('motion.npy'),allow_pickle=False)-mean)/std for r in paths}
    steps={r['id']:r for r in status['steps']};rows=[];jobs=[]
    for path in paths:
        for t in path['diffusion_steps']:
            rid=path['trajectory_id']+'-'+str(t);job=dict(row_id=rid,trajectory_id=path['trajectory_id'],timestep=t,condition=path['condition'],prefix_history=path['prefix_history'],suffix_condition=path['suffix_condition'],sensor_config=path['sensor_config'],panel_case_id=path['panel_case_id'],frames=path['frames'],raw=steps[rid]['raw_artifact'],metadata=artifact(run_dir/'suffix/steps'/(rid+'.json')))
            z=arrays(verify(job['raw']));validate_raw(dict(job,condition='keep_paired'),read(verify(job['metadata'])),z)
            f={k:matrix(z[k]) for k in NAMES};a,b,c,d=[z[n] for n in ('f00','f10','f01','f11')]
            unit=matrix(a+np.float32(1.)*(b-a)+np.float32(1.)*(c-a)+np.float32(1.)*(d-b-c+a));g=targets[path['panel_case_id']]
            rows.append(dict(job,status='valid',source_n=1,temporal={k:temporal_decomposition(v,g) for k,v in f.items()},response=response_decomposition(f),derived_native111_temporal=temporal_decomposition(unit,g),same_state_knockout_minus_native111=same_state_delta(unit,f['actual_combined_prediction'],g)))
            jobs.append(job)
    if len(rows)!=1800:raise ValueError('All saved response denominator')
    output.mkdir(parents=True)
    compact(output/'rows.json',rows);write(output/'jobs.json',jobs)
    write(output/'summary.json',dict(status='completed',rows=1800,source_n=1,configs=2,new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,interpretation='Posthoc full saved states; six raw predictions keep original temporal/response definitions. Native111 is reconstructed from four saved branches with original float32 operation order on each knockout state, not a newly sampled trajectory or unique causal percentage.'))
    write(output/'sources.json',[artifact(Path(__file__).resolve()),artifact(run_dir/'runtime/status.json'),status['implementation'],artifact(ROOT/'src/itm/metrics/training_sufficiency.py'),artifact(ROOT/'src/itm/metrics/trace_response.py')])
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.iterdir())]);print('All1800 saved responses completed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.run.resolve(),a.output.resolve())
