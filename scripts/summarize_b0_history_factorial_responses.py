#!/usr/bin/env python
"""Plot all7200 saved history/suffix cells, local covariance and cross-path effects."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_b0_native_training_sufficiency import read,verify,artifact,write
from analyze_b0_suffix_responses import compact


def sign_runs(rows,field):
    runs=[]
    for r in rows:
        v=field(r);sign='negative' if v<0 else 'positive' if v>0 else 'zero'
        if not runs or runs[-1]['sign']!=sign:runs.append(dict(sign=sign,first_t=r['timestep'],last_t=r['timestep'],steps=1))
        else:runs[-1]['last_t']=r['timestep'];runs[-1]['steps']+=1
    return runs


def run(results,output):
    if output.exists():raise FileExistsError('Fresh full response panel required')
    for a in read(results/'artifact_index.json'):verify(a)
    rows=read(results/'rows.json');ref=read(results/'inherited_F.json');old=read(verify(ref['rows']))
    for r in old:r.update(ref['roles'][r['condition']])
    rows=old+rows
    if len(rows)!=7200 or any(r['status']!='valid' for r in rows):raise ValueError('Full7200 response panel required')
    pop=[]
    for r in rows:
        gt=r['temporal']['f11']['root_ric67']['target_temporal_energy'];m=r['response']['root_ric67']['conditional_changes']['imu_at_text']
        pop.append(dict(row_id=r['row_id'],prefix_history=r['prefix_history'],sensor_config=r['sensor_config'],suffix_condition=r['suffix_condition'],timestep=r['timestep'],
            energy_over_GT={k:r['temporal'][k]['root_ric67']['temporal_energy_ratio'] for k in ('x_t','f00','f10','f11','actual_combined_prediction')},
            conditional_IMU_at_text={k:m[k]/gt for k in ('delta_temporal_energy','twice_base_delta_covariance','delta_centered_energy')},normalized_raw_IMU_rms=m['delta_raw_rms']))
    lookup={(r['prefix_history'],r['sensor_config'],r['suffix_condition'],r['timestep']):r for r in pop}
    contrasts=read(results/'path_contrasts.json');interactions=read(results/'factorial_interactions.json')
    effects=[]
    for c in ('head','wrists'):
        for t in range(899,-1,-1):
            fp,ft,tp,tt=[lookup[h,c,s,t]['energy_over_GT']['actual_combined_prediction'] for h,s in [('F_paired','paired'),('F_paired','text_only'),('T_text_only','paired'),('T_text_only','text_only')]]
            effects.append(dict(sensor_config=c,timestep=t,F_effect=fp-ft,T_effect=tp-tt,interaction=(tp-tt)-(fp-ft)))
    fig,axes=plt.subplots(2,3,figsize=(17,9));covfig,covaxes=plt.subplots(2,2,figsize=(13,8));summary=[]
    styles={('F_paired','paired'):('#d62728','-'),('F_paired','text_only'):('#d62728','--'),('T_text_only','paired'):('#1f77b4','-'),('T_text_only','text_only'):('#1f77b4','--')}
    for i,c in enumerate(('head','wrists')):
        histories=[]
        for h in ('F_paired','T_text_only'):
            for s in ('paired','text_only'):
                series=[lookup[h,c,s,t] for t in range(899,-1,-1)];color,style=styles[h,s]
                axes[i,0].plot([r['timestep'] for r in series],[r['energy_over_GT']['actual_combined_prediction'] for r in series],color=color,ls=style,lw=1,label=h+' / '+s)
            series=[lookup[h,c,'paired',t] for t in range(899,-1,-1)];color,_=styles[h,'paired']
            axes[i,1].plot([r['timestep'] for r in series],[r['conditional_IMU_at_text']['delta_temporal_energy'] for r in series],color=color,lw=1,label=h+' paired state')
            histories.append(dict(prefix_history=h,source_n=1,local_IMU_energy_sign_runs=sign_runs(series,lambda r:r['conditional_IMU_at_text']['delta_temporal_energy']),
                endpoints_and_old_times=[dict(timestep=t,paired=lookup[h,c,'paired',t],text=lookup[h,c,'text_only',t]) for t in (899,500,250,50,0)]))
            j=0 if h=='F_paired' else 1
            for k,color_,label in [('delta_temporal_energy','#d62728','E(paired)-E(text)'),('twice_base_delta_covariance','#9467bd','2 cov(text, delta)'),('delta_centered_energy','#ff7f0e','E(centered delta)')]:
                covaxes[i,j].plot([r['timestep'] for r in series],[r['conditional_IMU_at_text'][k] for r in series],lw=1,color=color_,label=label)
            covaxes[i,j].axhline(0,color='black',lw=.7);covaxes[i,j].set_title(c+' / '+h+' paired states');covaxes[i,j].set_ylabel('Within-state terms / GT energy')
        series=[e for e in effects if e['sensor_config']==c]
        for k,color,label in [('F_effect','#d62728','F: paired-text paths'),('T_effect','#1f77b4','T: paired-text paths'),('interaction','#2ca02c','T effect - F effect')]:axes[i,2].plot([r['timestep'] for r in series],[r[k] for r in series],color=color,lw=1,label=label)
        for j in (1,2):axes[i,j].axhline(0,color='black',lw=.7)
        axes[i,0].set_title(c+' / four actual prediction paths');axes[i,0].set_ylabel('Centered root + RIC energy / GT');axes[i,0].set_ylim(bottom=0)
        axes[i,1].set_title(c+' / local IMU response at paired states');axes[i,1].set_ylabel('Signed within-state change / GT')
        axes[i,2].set_title(c+' / current path states differ');axes[i,2].set_ylabel('Signed path contrast / GT')
        summary.append(dict(sensor_config=c,steps_per_path=900,source_n=1,histories=histories,effect_sign_runs={k:sign_runs(series,lambda r,k=k:r[k]) for k in ('F_effect','T_effect','interaction')},
            endpoints_and_old_times=[next(r for r in series if r['timestep']==t) for t in (899,500,250,50,0)],
            interpretation='Sign intervals are descriptive, not selected boundaries, independent steps, significance, causal percentages or behavior success.'))
    for ax in list(axes.flat)+list(covaxes.flat):ax.set_xlim(900,0);ax.set_xlabel('Diffusion t');ax.grid(alpha=.2)
    for ax in axes[0]:ax.legend(fontsize=7)
    covaxes[0,0].legend(fontsize=8)
    fig.suptitle('One source / one seed: history x suffix; original behavior remains incomplete',fontsize=12)
    covfig.suptitle('Signed covariance decomposition; local changes do not certify complete motions',fontsize=11)
    output.mkdir(parents=True)
    for f,name in [(fig,'history_factorial_response.png'),(covfig,'history_factorial_covariance.png')]:f.tight_layout();f.savefig(output/name,dpi=160);plt.close(f)
    for name,data in [('all_time_panel',pop),('energy_effects',effects),('summary',summary)]:compact(output/(name+'.json'),data)
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.iterdir())]+[artifact(Path(__file__).resolve())]);print(dict(effective_rows=len(pop),effects=len(effects)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.results.resolve(),a.output.resolve())
