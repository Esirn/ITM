#!/usr/bin/env python
"""Prescribed prefix and same-state panels; no new forward or behavioral gate."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluate_b0_native_training_sufficiency import read,artifact,verify,write


def run(results,output,diagnostic=False):
    if output.exists():raise FileExistsError('Fresh panel export required')
    for a in read(results/'artifact_index.json'):verify(a)
    summary=read(results/'summary.json')
    if summary['queue_status']!='completed' and not diagnostic:raise ValueError('Complete primary prefix queue required')
    traces=read(results/'prefix_traces.json');branches=read(results/'same_state_branches.json')
    panel=[]
    for r in traces:
        if 'metrics' not in r:continue
        g=r['metrics']['temporal']['f11']['root_ric67']['target_temporal_energy']
        panel.append(dict(**{k:r[k] for k in ('row_id','path_arm','data_split','motion_id','sensor_config','timestep','state_id')},
            energy_ratio={k:r['metrics']['temporal'][k]['root_ric67']['temporal_energy_ratio'] for k in ('x_t','f00','f10','f01','f11')},
            imu_terms_per_GT={k:v/g for k,v in r['metrics']['response']['root_ric67']['conditional_changes']['imu_at_text'].items()
                if k in ('delta_temporal_energy','delta_centered_energy','twice_base_delta_covariance')}))
    fixed=[dict(**{k:r[k] for k in ('row_id','state_id','path_arm','data_split','motion_id','sensor_config','timestep','checkpoint','branch')},
        mse=r['temporal']['root_ric67']['mse'],energy_ratio=r['temporal']['root_ric67']['temporal_energy_ratio']) for r in branches if 'temporal' in r]
    output.mkdir(parents=True);write(output/'all_prefix_panel.json',panel);write(output/'all_same_state_panel.json',fixed)
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True)
    colors=dict(F='#d62728',U='#1f77b4',T='#2ca02c')
    for i,config in enumerate(('head','wrists')):
        pop=[r for r in panel if r['data_split']=='train' and r['motion_id']=='000506' and r['sensor_config']==config]
        for j,key in enumerate(('x_t','f10','f11')):
            ax=axes[i,j]
            for arm in ('F','U','T'):
                rows=sorted([r for r in pop if r['path_arm']==arm],key=lambda r:-r['timestep'])
                ax.plot([r['timestep'] for r in rows],[r['energy_ratio'][key] for r in rows],marker='o',color=colors[arm],label=arm)
            ax.axhline(1,color='black',lw=1,alpha=.4);ax.set_xlim(1002,897);ax.set_xticks([999,975,950,925,900]);ax.grid(alpha=.2)
            ax.set_title(config+' / '+key);ax.set_xlabel('Diffusion t');ax.set_ylabel('Root + RIC temporal energy / GT')
    axes[0,0].legend(title='Native prefix path')
    fig.suptitle(('DIAGNOSTIC ONLY: incomplete queue. ' if diagnostic else '')+'Matched old noise, first100 steps; no behavior claim',fontsize=12)
    fig.tight_layout();fig.savefig(output/'primary_prefix_energy.png',dpi=160);plt.close(fig)
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.iterdir())]+[artifact(Path(__file__).resolve())])
    print(dict(prefix_rows=len(panel),same_state_rows=len(fixed),new_model_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--diagnostic',action='store_true')
    a=p.parse_args();run(a.results.resolve(),a.output.resolve(),a.diagnostic)
