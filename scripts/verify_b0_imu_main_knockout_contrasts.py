#!/usr/bin/env python
"""CPU cross-cell identity, motion/energy formulas and signed factorial checks."""
import argparse
from pathlib import Path

import numpy as np

from evaluate_b0_native_training_sufficiency import read,artifact,verify,arrays,matrix,flatten,write
from itm.metrics.feature_balance import SCALARS


def run(results,output):
    if output.exists():raise FileExistsError('Fresh cross-cell verification required')
    for a in read(results/'artifact_index.json')+read(results/'sources.json'):verify(a)
    rows=read(results/'factorial_finals.json');contrasts=read(results/'contrasts.json')
    if len(rows)!=10 or len(contrasts)!=8 or any(r['status']!='valid' for r in rows+contrasts):raise ValueError('All four cells per config required')
    root=Path(__file__).resolve().parents[1]
    cases={s['case_id']:s for s in read(root/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/sampling_slots.json')}
    mapping={('F_paired','keep_paired'):'paired',('F_paired','switch_imu_off'):'text_only',('T_text_only','keep_text_only'):'text_only',('T_text_only','switch_imu_on'):'paired',('T_text_only','remove_IMU_main_retain_joint'):'text_plus_joint_interaction'}
    cells=0;maxerror=0.;table={};raw={}
    def check(a,b,*scale):
        nonlocal cells,maxerror
        error=abs(a-b);maxerror=max(maxerror,error);cells+=1
        if error>64*np.finfo(float).eps*max(1.,abs(a),abs(b),*(abs(x) for x in scale)):raise ArithmeticError('Independent factorial formula mismatch')
    for row in rows:
        if mapping[(row['prefix_history'],row['condition'])]!=row['suffix_condition']:raise ValueError('History/suffix roles')
        key=(row['sensor_config'],row['prefix_history'],row['suffix_condition'])
        if key in table:raise ValueError('Duplicate cell')
        table[key]=row;z=arrays(verify(row['raw']));raw[key]=z
        p=z['joints'][0].astype(float);g=np.load(cases[row['panel_case_id']]['gt_path'],allow_pickle=False).astype(float);error=p-g
        norm=lambda v:np.sqrt(np.einsum('tjd,tjd->tj',v,v))
        check(row['metrics']['global_MPJPE_m'],float(norm(error).mean()))
        check(row['metrics']['root_relative_MPJPE_m'],float(norm((p-p[:,:1])-(g-g[:,:1])).mean()))
        check(row['metrics']['root_path_RMSE_m'],float(np.sqrt(np.einsum('td,td->t',error[:,0],error[:,0]).mean())))
        ids=[15] if row['sensor_config']=='head' else [20,21]
        check(row['metrics']['head_or_wrists_position_joint_proxy_m'],float(norm(error[:,ids]).mean()))
        v=(np.diff(p,axis=0)-np.diff(g,axis=0))*20
        check(row['metrics']['velocity_joint_proxy_RMSE_m_s'],float(np.sqrt(np.einsum('tjd,tjd->tj',v,v).mean())))
        for field,value in [('jerk_absolute_m_s3',p),('GT_jerk_absolute_m_s3',g)]:check(row['metrics'][field],float(norm(np.diff(value,n=3,axis=0)*8000).mean()))
        for prefix,value in [('motion_extent_m',p),('GT_motion_extent_m',g)]:
            m=row['metrics'][prefix]
            for field,expected in [('min_xyz',value.min((0,1))),('max_xyz',value.max((0,1))),('range_xyz',value.max((0,1))-value.min((0,1)))]:
                for a,b in zip(m[field],expected):check(a,float(b))
            d=np.diff(value[:,0],axis=0);check(m['pelvis_path_length_m'],float(np.sqrt(np.einsum('td,td->t',d,d)).sum()))
        if row['metrics']['foot_contact_speed_proxy_m_s'] is not None or row['metrics']['GT_foot_contact_speed_proxy_m_s'] is not None:raise ValueError('Train floor N/A changed')
        f=matrix(z['normalized_humanml263']).astype(float);d=f[:,:67]-f[:,:67].mean(0)
        check(row['temporal']['root_ric67']['prediction_temporal_energy'],float(np.einsum('td,td->',d,d)/d.size))
    by_id={r['trajectory_id']:r for r in rows}
    for contrast in contrasts:
        pair,text=by_id[contrast['new_trajectory']],by_id[contrast['reference_trajectory']]
        if pair['condition']!='remove_IMU_main_retain_joint' or pair['sensor_config']!=text['sensor_config']:raise ValueError('Knockout comparison role')
        for k in SCALARS:
            a,b=pair['metrics'][k],text['metrics'][k];v=contrast['metrics'][k]
            if a is None or b is None:
                if v is not None:raise ValueError('Undefined scalar became numeric')
            else:check(v,a-b,a,b)
        p,t=flatten(pair['temporal']),flatten(text['temporal'])
        for name,value in flatten(contrast['temporal']).items():
            a,b=p[name],t[name]
            if a is None or b is None:
                if value is not None:raise ValueError('Undefined temporal scalar')
            else:check(value,a-b,a,b)
    proof=dict(status='passed',combined_cells=10,contrasts=8,configs=2,independent_scalar_checks=cells,max_absolute_error=maxerror,
        new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,independent_review=False,verifier=artifact(Path(__file__).resolve()),
        interpretation='One source/seed, previous F cells are bound references; numeric interaction without unique causal percentages or behavior/model certification.')
    output.parent.mkdir(parents=True,exist_ok=True);write(output,proof);print(proof)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.results.resolve(),a.output.resolve())
