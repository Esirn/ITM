#!/usr/bin/env python
"""All900-step knockout paths and inherited T text/paired curves, CPU only."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_b0_native_training_sufficiency import read,verify,artifact,write
from analyze_b0_suffix_responses import compact


def run(results,output):
    if output.exists():raise FileExistsError('Fresh response panel required')
    for a in read(results/'artifact_index.json'):verify(a)
    rows=read(results/'rows.json')
    inherited=Path(__file__).resolve().parents[1]/'outputs/itm_rework/protocol/history_factorial_response_panel_20261009_v1'
    for a in read(inherited/'artifact_index.json'):verify(a)
    refs=read(inherited/'all_time_panel.json');pop=[]
    for r in rows:
        pop.append(dict(row_id=r['row_id'],sensor_config=r['sensor_config'],timestep=r['timestep'],actual_energy_over_GT=r['temporal']['actual_combined_prediction']['root_ric67']['temporal_energy_ratio'],unit_same_state_energy_over_GT=r['derived_native111_temporal']['root_ric67']['temporal_energy_ratio'],text_same_state_energy_over_GT=r['temporal']['f10']['root_ric67']['temporal_energy_ratio'],local_knockout_minus_unit_mse=r['same_state_knockout_minus_native111']['root_ric67']['delta_mse'],local_knockout_minus_unit_energy_over_GT=r['temporal']['actual_combined_prediction']['root_ric67']['temporal_energy_ratio']-r['derived_native111_temporal']['root_ric67']['temporal_energy_ratio']))
    fig,axes=plt.subplots(2,2,figsize=(13,8));summary=[]
    for i,c in enumerate(('head','wrists')):
        p=sorted([r for r in pop if r['sensor_config']==c],key=lambda r:-r['timestep']);t=[r['timestep'] for r in p]
        for suffix,color,style in [('text_only','#888888','--'),('paired','#d62728','-')]:
            ref=sorted([r for r in refs if r['sensor_config']==c and r['prefix_history']=='T_text_only' and r['suffix_condition']==suffix],key=lambda r:-r['timestep'])
            axes[i,0].plot(t,[r['energy_over_GT']['actual_combined_prediction'] for r in ref],color=color,ls=style,lw=1,label='Completed T '+suffix)
        axes[i,0].plot(t,[r['actual_energy_over_GT'] for r in p],color='#1f77b4',lw=1,label='New (1,0,1) path')
        axes[i,0].set_ylabel('Centered root+RIC energy / GT');axes[i,0].set_title(c+' / evolving paths')
        axes[i,1].plot(t,[r['local_knockout_minus_unit_energy_over_GT'] for r in p],label='Local energy change / GT',lw=1)
        axes[i,1].plot(t,[r['local_knockout_minus_unit_mse'] for r in p],label='Local MSE change',lw=1)
        axes[i,1].axhline(0,color='black',lw=.7);axes[i,1].set_title(c+' / (1,0,1) minus native111 at knockout states')
        summary.append(dict(sensor_config=c,steps=900,source_n=1,endpoints_and_old_times=[next(r for r in p if r['timestep']==s) for s in (899,500,250,50,0)],positive_local_energy_changes=sum(r['local_knockout_minus_unit_energy_over_GT']>0 for r in p),negative_local_energy_changes=sum(r['local_knockout_minus_unit_energy_over_GT']<0 for r in p),interpretation='Within-state algebra versus evolving-path energy; step counts not independent samples, no success threshold or mechanism percentages.'))
    for ax in axes.flat:ax.set_xlim(900,0);ax.set_xlabel('Diffusion t');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('IMU main removed, interaction retained; original behavior still incomplete');fig.tight_layout();output.mkdir(parents=True);fig.savefig(output/'knockout_responses.png',dpi=160);plt.close(fig)
    compact(output/'all_time_panel.json',pop);write(output/'summary.json',summary)
    write(output/'provenance.json',dict(results_index=artifact(results/'artifact_index.json'),inherited_panel_index=artifact(inherited/'artifact_index.json'),script=artifact(Path(__file__).resolve()),new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0))
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.iterdir())]);print('Saved1800 rows/two-config response panels',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.results.resolve(),a.output.resolve())
