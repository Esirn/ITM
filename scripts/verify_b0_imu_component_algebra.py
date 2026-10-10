#!/usr/bin/env python
"""Verify posthoc hypothetical energies directly from saved raw paired predictions."""
import argparse
from pathlib import Path
import numpy as np
from evaluate_b0_native_training_sufficiency import read,verify,artifact,arrays,matrix,write


def run(results,components,output):
    if output.exists():raise FileExistsError('Fresh component second-formula proof required')
    for folder in (results,components):
        for a in read(folder/'artifact_index.json'):verify(a)
    ref=read(results/'inherited_F.json');old=read(verify(ref['rows']))
    for r in old:r.update(ref['roles'][r['condition']])
    table={r['row_id']:r for r in old+read(results/'rows.json')};derived=read(components/'rows.json')
    if len(derived)!=3600 or len({r['row_id'] for r in derived})!=3600:raise ValueError('Component state denominator')
    checks=0;maxerror=0.
    def energy(p):
        c=p-p.mean(0);return float(np.einsum('ij,ij->',c,c)/c.size)
    for row in derived:
        source=table[row['row_id']]
        if source['suffix_condition']!='paired' or any(source[k]!=row[k] for k in ('prefix_history','sensor_config','timestep','source_n')):raise ValueError('Paired-state role')
        z=arrays(verify(source['raw']));f={k:matrix(z[k]).astype(float)[:,:67] for k in ('f00','f10','f01','f11')};gt=source['temporal']['f10']['root_ric67']['target_temporal_energy']
        base=f['f10'];main=f['f01']-f['f00'];joint=f['f11']-f['f10']-f['f01']+f['f00']
        be,pe,me,je=[energy(p)/gt for p in (base,f['f11'],base+main,base+joint)]
        mc=main-main.mean(0);jc=joint-joint.mean(0);cross=float(2*np.einsum('ij,ij->',mc,jc)/mc.size/gt)
        expected=dict(text_energy_over_GT=be,paired_energy_over_GT=pe,hypothetical_main_only_energy_over_GT=me,hypothetical_interaction_only_energy_over_GT=je,main_only_change_over_GT=me-be,interaction_only_change_over_GT=je-be,main_interaction_twice_covariance_over_GT=cross,total_change_over_GT=pe-be)
        for key,value in expected.items():
            error=abs(value-row[key]);maxerror=max(maxerror,error);checks+=1
            if error>64*np.finfo(float).eps*max(1.,abs(value),abs(row[key]),be,pe,me,je,abs(cross)):raise ArithmeticError('Direct raw component energy mismatch')
    write(output,dict(status='passed',saved_paired_states=3600,independent_scalar_checks=checks,max_absolute_error=maxerror,new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,independent_review=False,verifier=artifact(Path(__file__).resolve()),components_index=artifact(components/'artifact_index.json'),response_results_index=artifact(results/'artifact_index.json'),bound='Inherited float64 64eps times context scale',interpretation='Direct raw same-state algebra, no new rollout/decoded behavior or unique causal attribution.'));print(dict(status='passed',checks=checks,maxerror=maxerror),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('results','components','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.results.resolve(),a.components.resolve(),a.output.resolve())
