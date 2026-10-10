#!/usr/bin/env python
"""Posthoc signed main/interaction energy algebra at saved paired states only."""
import argparse
from pathlib import Path
import numpy as np
from evaluate_b0_native_training_sufficiency import read,verify,artifact,write
from analyze_b0_suffix_responses import compact
from summarize_b0_history_factorial_responses import sign_runs


def run(results,output):
    if output.exists():raise FileExistsError('Fresh posthoc algebra output required')
    for a in read(results/'artifact_index.json'):verify(a)
    rows=read(results/'rows.json');ref=read(results/'inherited_F.json');old=read(verify(ref['rows']))
    for r in old:r.update(ref['roles'][r['condition']])
    rows=old+rows;pop=[];maxerror=0.
    for r in rows:
        if r['suffix_condition']!='paired':continue
        m=r['response']['root_ric67'];e=m['component_energy'];c=m['twice_component_covariance'];gt=r['temporal']['f10']['root_ric67']['target_temporal_energy']
        main=e['imu']+c['base__imu']+c['text__imu']
        inter=e['interaction']+c['base__interaction']+c['text__interaction']
        cross=c['imu__interaction']
        actual=m['conditional_changes']['imu_at_text']['delta_temporal_energy']
        error=abs(actual-main-inter-cross);maxerror=max(maxerror,error)
        scale=max(1.,abs(actual),*(abs(x) for x in [main,inter,cross,*e.values(),*c.values()]))
        if error>64*np.finfo(float).eps*scale:raise ArithmeticError('Inherited signed four-component identity')
        base=m['branch_energy']['f10']
        pop.append(dict(row_id=r['row_id'],prefix_history=r['prefix_history'],sensor_config=r['sensor_config'],timestep=r['timestep'],source_n=1,
            text_energy_over_GT=base/gt,paired_energy_over_GT=m['branch_energy']['f11']/gt,
            hypothetical_main_only_energy_over_GT=(base+main)/gt,hypothetical_interaction_only_energy_over_GT=(base+inter)/gt,
            main_only_change_over_GT=main/gt,interaction_only_change_over_GT=inter/gt,main_interaction_twice_covariance_over_GT=cross/gt,
            total_change_over_GT=actual/gt,identity_error=error))
    if len(pop)!=3600:raise ValueError('Full paired-state component denominator')
    summary=[]
    for h in ('F_paired','T_text_only'):
        for c in ('head','wrists'):
            rs=sorted([r for r in pop if (r['prefix_history'],r['sensor_config'])==(h,c)],key=lambda r:-r['timestep'])
            summary.append(dict(prefix_history=h,sensor_config=c,source_n=1,steps=900,
                sign_runs={k:sign_runs(rs,lambda r,k=k:r[k]) for k in ('main_only_change_over_GT','interaction_only_change_over_GT','main_interaction_twice_covariance_over_GT','total_change_over_GT')},
                endpoints_and_old_times=[next(r for r in rs if r['timestep']==t) for t in (899,500,250,50,0)]))
    output.mkdir(parents=True)
    compact(output/'rows.json',pop);write(output/'summary.json',summary)
    write(output/'verification.json',dict(status='passed',saved_paired_states=3600,identity_checks=3600,max_absolute_error=maxerror,new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,
        interpretation='Posthoc algebra from saved four-component covariances. Hypothetical main-only/interaction-only predictions share original paired states; these are not new rollouts, unique causal allocations, decoded motions or behavior results.',
        script=artifact(Path(__file__).resolve()),results_index=artifact(results/'artifact_index.json'),bound='Inherited float64 64eps times context scale'))
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.iterdir())]);print(dict(states=len(pop),max_error=maxerror),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.results.resolve(),a.output.resolve())
