#!/usr/bin/env python
"""All-source signed component/error tables and plots; counts are descriptive only."""
import argparse
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_b0_native_training_sufficiency import read,verify,artifact,write


def run(results,output):
    if output.exists():raise FileExistsError('Fresh cross-source summary required')
    for a in read(results/'artifact_index.json'):verify(a)
    rows=read(results/'rows.json')
    if len(rows)!=480 or any(r['status']!='valid' for r in rows):raise ValueError('Complete480 summary required')
    paired=[r for r in rows if r['condition']=='paired'];source_tables=[]
    for arm,split,motion,config in sorted({(r['arm'],r['data_split'],r['motion_id'],r['sensor_config']) for r in paired}):
        rs=sorted([r for r in paired if (r['arm'],r['data_split'],r['motion_id'],r['sensor_config'])==(arm,split,motion,config)],key=lambda r:-r['timestep'])
        source_tables.append(dict(arm=arm,data_split=split,motion_id=motion,sensor_config=config,source_n=1,states=4,points=[dict(row_id=r['row_id'],timestep=r['timestep'],root_ric67=r['component_terms']['root_ric67'],balanced=r['component_terms']['balanced'],main_error_delta=r['hypotheses']['main_only']['error_delta']['root_ric67'],joint_error_delta=r['hypotheses']['joint_only']['error_delta']['root_ric67']) for r in rs]))
    summary=[]
    for arm in ('F','U'):
        for condition in sorted({r['condition'] for r in rows}):
            for config in ('head','wrists'):
                for t in (900,500,250,50):
                    rs=[r for r in rows if (r['arm'],r['condition'],r['sensor_config'],r['timestep'])==(arm,condition,config,t)]
                    fields={}
                    for view in ('root_ric67','balanced'):
                        fields[view]={}
                        for comp in ('main','joint'):
                            records=[r['component_terms'][view] for r in rs]
                            fields[view][comp]=dict(states=len(records),energy_negative=sum(x[comp+'_delta_energy']<0 for x in records),energy_positive=sum(x[comp+'_delta_energy']>0 for x in records),energy_zero=sum(x[comp+'_delta_energy']==0 for x in records),mse_negative=sum(x[comp+'_delta_mse']<0 for x in records),mse_positive=sum(x[comp+'_delta_mse']>0 for x in records),mse_zero=sum(x[comp+'_delta_mse']==0 for x in records),mean_error_negative=sum(x[comp+'_delta_mean_error']<0 for x in records),temporal_error_negative=sum(x[comp+'_delta_temporal_error']<0 for x in records),energy_and_mse_negative=sum(x[comp+'_delta_energy']<0 and x[comp+'_delta_mse']<0 for x in records))
                    summary.append(dict(arm=arm,condition=condition,sensor_config=config,timestep=t,planned_sources=len(rs),sources=[r['data_split']+':'+r['motion_id'] for r in rs],signs=fields))
    fig,axes=plt.subplots(2,2,figsize=(14,9));sources=sorted({(r['data_split'],r['motion_id']) for r in paired});colors=dict(zip(sources,plt.get_cmap('tab10').colors))
    for i,config in enumerate(('head','wrists')):
        for k,comp in enumerate(('main','joint')):
            ax=axes[i,k]
            for source,color in colors.items():
                for arm,marker,style in [('F','o','-'),('U','x','--')]:
                    rs=sorted([r for r in paired if (r['data_split'],r['motion_id'])==source and r['sensor_config']==config and r['arm']==arm],key=lambda r:-r['timestep'])
                    x=[r['component_terms']['root_ric67'][comp+'_delta_energy'] for r in rs];y=[r['component_terms']['root_ric67'][comp+'_delta_mse'] for r in rs]
                    ax.plot(x,y,color=color,marker=marker,ls=style,lw=.8,ms=4,label=source[0]+':'+source[1]+' '+arm)
            ax.axhline(0,color='black',lw=.7);ax.axvline(0,color='black',lw=.7);ax.grid(alpha=.2);ax.set_title(config+' / '+comp+' added to f10 at saved states');ax.set_xlabel('Change in centered root+RIC energy');ax.set_ylabel('Change in normalized root+RIC MSE')
    axes[0,0].legend(fontsize=6,ncol=2);fig.suptitle('Six original sources; F solid/o, U dashed/x; connected times900->500->250->50; states differ',fontsize=11);fig.tight_layout();output.mkdir(parents=True);fig.savefig(output/'paired_component_energy_error.png',dpi=160);plt.close(fig)
    write(output/'paired_source_tables.json',source_tables);write(output/'condition_sign_counts.json',summary)
    write(output/'summary.json',dict(total_rows=480,paired_rows=96,F_U_pairs=240,paired_F_U_pairs=48,source_count=6,condition_counts=dict(Counter(r['condition'] for r in rows)),interpretation='Strict sign counts and connected points are descriptions, not new thresholds or independent time/frame observations. Source/config/arm/condition/time retain original roles. Energy suppression can improve error; no semantic or generalization certification.',new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,script=artifact(Path(__file__).resolve()),results_index=artifact(results/'artifact_index.json')))
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.iterdir())]);print('Completed24 source/config/arm paired tables and112 condition/config/time panels',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.results.resolve(),a.output.resolve())
