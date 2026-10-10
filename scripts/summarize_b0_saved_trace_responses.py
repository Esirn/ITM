#!/usr/bin/env python
"""Export complete paired panels and a saved-state scientific figure on CPU."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluate_b0_feature_balance import read, sha, write, require


def main(results,output):
    require(not output.exists(),'Refusing to overwrite panel export')
    for e in read(results/'artifact_index.json'):require(sha(e['path'])==e['sha256'],'Evaluation hash')
    rows=read(results/'traces.json');samples=read(results/'samples.json')
    panel=[]
    for r in rows:
        if r['condition']!='paired':continue
        require(r['status']=='valid','Incomplete paired panel')
        temporal=r['metrics']['temporal'];response=r['metrics']['response']['root_ric67']
        ge=temporal['f11']['root_ric67']['target_temporal_energy']
        panel.append(dict(**{k:r[k] for k in ('arm','data_split','motion_id','sensor_config','timestep','trace_id')},
            energy_ratio={k:temporal[k]['root_ric67']['temporal_energy_ratio'] for k in ('x_t','f00','f10','f01','f11')},
            imu_at_text_terms_per_GT={k:v/ge for k,v in response['conditional_changes']['imu_at_text'].items()
                if k in ('delta_temporal_energy','twice_base_delta_covariance','delta_centered_energy')}))
    final=[dict(**{k:r[k] for k in ('arm','data_split','motion_id','sensor_config','condition')},
                energy_ratio=r['metrics']['root_ric67']['temporal_energy_ratio'])
           for r in samples if r['condition'] in ('paired','text_only','physical_static')]
    output.mkdir(parents=True);write(output/'all_paired_trace_panel.json',panel);write(output/'all_final_reference_panel.json',final)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    styles=dict(x_t=('#666666',':'),f00=('#2ca02c','--'),f10=('#1f77b4','-.'),f11=('#d62728','-'))
    for ci,config in enumerate(('head','wrists')):
        for ai,arm in enumerate(('F','U')):
            ax=axes[ci,ai]
            pop=sorted([r for r in panel if r['data_split']=='train' and r['motion_id']=='000506'
                        and r['sensor_config']==config and r['arm']==arm],key=lambda r:-r['timestep'])
            for key,(color,style) in styles.items():ax.plot([r['timestep'] for r in pop],[r['energy_ratio'][key] for r in pop],marker='o',ls=style,color=color,label=key)
            ax.axhline(1,color='black',alpha=.4,lw=1,label='GT reference')
            ax.set_yscale('log');ax.set_ylim(.001,3);ax.set_xlim(950,0);ax.set_xticks([900,500,250,50])
            ax.set_title(f'train000506 / {config} / {arm} saved states');ax.grid(alpha=.2);ax.set_ylabel('Root + RIC temporal energy / GT')
            ax.set_xlabel('Diffusion t (denoising proceeds left to right)')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Saved states differ across arms and times; variance is not behavior correctness',fontsize=11)
    fig.tight_layout();fig.savefig(output/'primary_saved_response_energy.png',dpi=160);plt.close(fig)
    write(output/'artifact_index.json',[dict(path=str(p.resolve()),sha256=sha(p)) for p in sorted(output.iterdir())]+[
        dict(path=str(Path(__file__).resolve()),sha256=sha(__file__))])
    print(dict(paired_trace_points=len(panel),final_reference_rows=len(final),new_model_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.results.resolve(),a.output.resolve())
