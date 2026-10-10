#!/usr/bin/env python
"""Complete CPU late-state panel, with descriptive old final references."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluate_b0_native_training_sufficiency import read,verify,artifact,write


def run(results,output):
    if output.exists():raise FileExistsError('Fresh late panel required')
    for a in read(results/'artifact_index.json'):verify(a)
    rows=read(results/'rows.json');traces=read(results/'reference_traces.json');finals=read(results/'reference_finals.json')
    if len(rows)!=96 or any(r['status']!='valid' for r in rows):raise ValueError('Complete late algebra required')
    panel=[dict(**{k:r[k] for k in ('row_id','data_split','motion_id','sensor_config','timestep','comparison','ordering')},
                root_ric67=r['metrics']['root_ric67']) for r in rows]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,config in enumerate(('head','wrists')):
        ax=axes[i,0]
        selected=[r for r in panel if r['motion_id']=='000506' and r['data_split']=='train' and r['sensor_config']==config]
        for ordering,style in [('direct_at_F_state','-'),('direct_at_reference_state','--')]:
            pop=sorted([r for r in selected if r['ordering']==ordering],key=lambda r:-r['timestep'])
            for key,color,label in [('direct_rms','#1f77b4','Same-state checkpoint difference'),('state_rms','#d62728','History/state difference')]:
                ax.plot([r['timestep'] for r in pop],[r['root_ric67'][key] for r in pop],ls=style,marker='o',color=color,
                    label=label+(' (F first)' if style=='-' else ' (alternate)'))
        ax.set_ylabel('Root + RIC normalized difference RMS');ax.set_title(config+' / F minus U')
        ax.set_xlim(950,0);ax.set_xticks([900,500,250,50]);ax.set_xlabel('Diffusion t')
        ax=axes[i,1]
        for arm,condition,color,label in [('F','paired','#d62728','F paired'),('U','paired','#1f77b4','U paired'),('F','text_only','#2ca02c','Original text-only')]:
            pop=sorted([r for r in traces if r['motion_id']=='000506' and r['data_split']=='train' and r['sensor_config']==config
                and r['arm']==arm and r['condition']==condition],key=lambda r:-r['timestep'])
            final=next(r for r in finals if r['motion_id']=='000506' and r['data_split']=='train' and r['sensor_config']==config
                and r['arm']==arm and r['condition']==condition)
            ax.plot(range(5),[r['temporal']['root_ric67']['temporal_energy_ratio'] for r in pop]+[final['temporal']['root_ric67']['temporal_energy_ratio']],marker='o',color=color,label=label)
        ax.set_xticks(range(5),['900','500','250','50','final']);ax.set_xlabel('Saved trace t / final (equally spaced categories)')
        ax.set_ylabel('Centered root + RIC energy / GT');ax.set_title(config+' / different own trajectory states')
    for ax in axes.flat:ax.set_ylim(bottom=0);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);axes[0,1].legend(fontsize=8)
    fig.suptitle('Saved algebra and old trajectory references; no suffix intervention or behavior certification',fontsize=11)
    fig.tight_layout();output.mkdir(parents=True);fig.savefig(output/'primary_late_panel.png',dpi=160);plt.close(fig)
    for name,data in [('all_algebra_panel',panel),('all_trace_references',traces),('all_final_references',finals)]:write(output/(name+'.json'),data)
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.iterdir())]+[artifact(Path(__file__).resolve())])
    print(dict(rows=len(panel),trace_references=len(traces),final_references=len(finals),new_model_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.results.resolve(),a.output.resolve())
