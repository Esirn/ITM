#!/usr/bin/env python
"""Export full algebra panel and the exposed primary case figure on CPU."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluate_b0_native_training_sufficiency import read,verify,artifact,write


def run(results,output):
    if output.exists():raise FileExistsError('Fresh algebra panel output required')
    for a in read(results/'artifact_index.json'):verify(a)
    rows=read(results/'rows.json');panel=[]
    for r in rows:
        if r['status']!='valid':raise ValueError('Complete decomposition required')
        panel.append(dict(**{k:r[k] for k in ('row_id','data_split','motion_id','sensor_config','timestep','comparison','ordering')},
            root_ric67={k:r['metrics']['root_ric67'][k] for k in ('direct_rms','state_rms','direct_energy','state_energy',
                'twice_direct_state_covariance','delta_mse','direct_centered_energy','state_centered_energy','twice_centered_covariance','delta_temporal_energy')}))
    output.mkdir(parents=True);write(output/'all_algebra_panel.json',panel)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for i,config in enumerate(('head','wrists')):
        for j,comparison in enumerate(('F_minus_U','F_minus_T')):
            ax=axes[i,j];pop=[r for r in panel if r['data_split']=='train' and r['motion_id']=='000506'
                and r['sensor_config']==config and r['comparison']==comparison]
            for ordering,style in [('direct_at_F_state','-'),('direct_at_reference_state','--')]:
                selected=sorted([r for r in pop if r['ordering']==ordering],key=lambda r:-r['timestep'])
                for key,color,label in [('direct_rms','#1f77b4','Checkpoint' if j==0 else 'IMU vs text'),('state_rms','#d62728','State')]:
                    ax.plot([r['timestep'] for r in selected],[r['root_ric67'][key] for r in selected],ls=style,marker='o',color=color,
                        label=label+(' at F state/order' if ordering=='direct_at_F_state' else ' alternative order'))
            ax.set_title(config+' / '+comparison);ax.set_xlim(1002,897);ax.set_xticks([999,975,950,925,900]);ax.set_ylim(bottom=0)
            ax.set_ylabel('Root + RIC normalized difference RMS');ax.set_xlabel('Diffusion t');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Saved-prediction algebra: correlated components; not unique causal attribution',fontsize=11)
    fig.tight_layout();fig.savefig(output/'primary_algebra_rms.png',dpi=160);plt.close(fig)
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.iterdir())]+[artifact(Path(__file__).resolve())])
    print(dict(rows=len(panel),new_model_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.results.resolve(),a.output.resolve())
