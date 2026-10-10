#!/usr/bin/env python
"""Complete900-step normalized energy/response panels on CPU."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluate_b0_native_training_sufficiency import read,verify,artifact,write


def run(results,output):
    if output.exists():raise FileExistsError('Fresh full response panel required')
    for a in read(results/'artifact_index.json'):verify(a)
    rows=read(results/'rows.json')
    if len(rows)!=3600 or any(r['status']!='valid' for r in rows):raise ValueError('Complete all-step panel required')
    panel=[]
    for r in rows:
        gt=r['temporal']['f11']['root_ric67']['target_temporal_energy'];m=r['response']['root_ric67']['conditional_changes']['imu_at_text']
        panel.append(dict(row_id=r['row_id'],sensor_config=r['sensor_config'],condition=r['condition'],timestep=r['timestep'],
            energy_over_GT={k:r['temporal'][k]['root_ric67']['temporal_energy_ratio'] for k in ('x_t','f00','f10','f11','actual_combined_prediction')},
            conditional_IMU_at_text={k:m[k]/gt for k in ('delta_temporal_energy','twice_base_delta_covariance','delta_centered_energy')},
            normalized_raw_IMU_rms=m['delta_raw_rms']))
    summary=[];fig,axes=plt.subplots(2,2,figsize=(13,8))
    for i,config in enumerate(('head','wrists')):
        pop=sorted([p for p in panel if p['sensor_config']==config and p['condition']=='keep_paired'],key=lambda r:-r['timestep'])
        off=sorted([p for p in panel if p['sensor_config']==config and p['condition']=='switch_imu_off'],key=lambda r:-r['timestep'])
        ax=axes[i,0]
        for rows_,key,color,label,style in [(pop,'f00','#888888','keep: unconditional','-'),(pop,'f10','#1f77b4','keep: text at keep state','-'),
            (pop,'f11','#d62728','keep: paired','-'),(off,'actual_combined_prediction','#2ca02c','off: text at off state','--')]:
            ax.plot([r['timestep'] for r in rows_],[r['energy_over_GT'][key] for r in rows_],color=color,label=label,ls=style,lw=1)
        ax.set_ylabel('Centered root + RIC energy / GT');ax.set_title(config+' / 900 saved states per path');ax.set_ylim(bottom=0)
        ax=axes[i,1]
        for key,color,label in [('delta_temporal_energy','#d62728','E(paired)-E(text)'),('twice_base_delta_covariance','#9467bd','2 cov(text, IMU delta)'),('delta_centered_energy','#ff7f0e','IMU delta energy')]:
            ax.plot([r['timestep'] for r in pop],[r['conditional_IMU_at_text'][key] for r in pop],color=color,label=label,lw=1)
        ax.axhline(0,color='black',lw=.7);ax.set_ylabel('Signed conditional terms / GT energy');ax.set_title(config+' / within keep state; correlated terms')
        negative=sum(r['conditional_IMU_at_text']['delta_temporal_energy']<0 for r in pop)
        summary.append(dict(sensor_config=config,steps=900,source_n=1,negative_local_energy_changes=negative,positive_local_energy_changes=sum(r['conditional_IMU_at_text']['delta_temporal_energy']>0 for r in pop),
            endpoints_and_old_times=[dict(timestep=t,keep=next(r for r in pop if r['timestep']==t),off=next(r for r in off if r['timestep']==t)) for t in [899,500,250,50,0]],
            interpretation='Step counts only, not independent samples/significance. No success threshold, source pooling or causal mechanism percentages.'))
    for ax in axes.flat:ax.set_xlim(900,0);ax.set_xlabel('Diffusion t');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
    fig.suptitle('Full saved suffix responses; path states differ; dynamics do not certify complete motions',fontsize=11)
    fig.tight_layout();output.mkdir(parents=True);fig.savefig(output/'full_suffix_response.png',dpi=160);plt.close(fig)
    write(output/'all_time_panel.json',panel);write(output/'summary.json',summary)
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.iterdir())]+[artifact(Path(__file__).resolve())]);print(dict(rows=len(panel)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.results.resolve(),a.output.resolve())
