#!/usr/bin/env python
"""Close reused results, compatible verifier, summary signs and next scientific plan."""
from pathlib import Path
import numpy as np
from evaluate_b0_native_training_sufficiency import read,verify,artifact,write

ROOT=Path(__file__).resolve().parents[1]
CPU=ROOT/'outputs/itm_rework/method/cross_source_component_implementation_20261009_v2'
RESULTS=ROOT/'outputs/itm_rework/protocol/cross_source_component_evaluation_20261009_v1'
PANEL=ROOT/'outputs/itm_rework/protocol/cross_source_component_panel_20261009_v1'
NEXT=ROOT/'docs/itm_rework/protocol/b0_control_mean_probe_20261009_v1/plan.json'


def run():
    output=CPU/'delivery_verification.json'
    if output.exists():raise FileExistsError('Fresh source closure proof required')
    manifest=read(CPU/'implementation.json')
    for a in manifest['expected_artifacts']:verify(a)
    proof=read(CPU/'result_verification.json');verify(proof['verifier']);verify(proof['results_index'])
    if proof['status']!='passed' or proof['rows']!=480 or proof['F_U_pairs']!=240:raise ValueError('Complete second formula required')
    for folder in (RESULTS,PANEL):
        for a in read(folder/'artifact_index.json'):verify(a)
    rows=read(RESULTS/'rows.json');lookup={r['row_id']:r for r in rows};tables=read(PANEL/'paired_source_tables.json');signs=read(PANEL/'condition_sign_counts.json')
    if len(rows)!=480 or len(lookup)!=480 or len(tables)!=24 or len(signs)!=112:raise ValueError('Complete summary denominator')
    fields=0
    for t in tables:
        if len(t['points'])!=4 or t['states']!=4 or t['source_n']!=1:raise ValueError('Source table denominator')
        for p in t['points']:
            r=lookup[p['row_id']]
            if any(t[k]!=r[k] for k in ('arm','data_split','motion_id','sensor_config')) or p['timestep']!=r['timestep'] or r['condition']!='paired':raise ValueError('Source table roles')
            for view in ('root_ric67','balanced'):
                if p[view]!=r['component_terms'][view]:raise ValueError('Source table metric')
                fields+=len(p[view])
            if p['main_error_delta']!=r['hypotheses']['main_only']['error_delta']['root_ric67'] or p['joint_error_delta']!=r['hypotheses']['joint_only']['error_delta']['root_ric67']:raise ValueError('Source table error delta')
    for g in signs:
        rs=[r for r in rows if all(g[k]==r[k] for k in ('arm','condition','sensor_config','timestep'))]
        if g['planned_sources']!=len(rs) or g['sources']!=[r['data_split']+':'+r['motion_id'] for r in rs]:raise ValueError('Sign population')
        for view in ('root_ric67','balanced'):
            for comp in ('main','joint'):
                m=g['signs'][view][comp];values=[r['component_terms'][view] for r in rs]
                expected=dict(states=len(values))
                for value,prefix in [('delta_energy','energy'),('delta_mse','mse')]:
                    v=[r[comp+'_'+value] for r in values];expected.update({prefix+'_negative':sum(x<0 for x in v),prefix+'_positive':sum(x>0 for x in v),prefix+'_zero':sum(x==0 for x in v)})
                expected.update(mean_error_negative=sum(r[comp+'_delta_mean_error']<0 for r in values),temporal_error_negative=sum(r[comp+'_delta_temporal_error']<0 for r in values),energy_and_mse_negative=sum(r[comp+'_delta_energy']<0 and r[comp+'_delta_mse']<0 for r in values))
                if m!=expected:raise ValueError('Signed summary mismatch')
                fields+=len(expected)
    nextplan=read(NEXT)
    for a in nextplan['sources']:verify(a)
    if len(nextplan['cases'])!=12 or len(nextplan['jobs'])!=48 or nextplan['budget']['MDM_calls']!=48 or nextplan['budget']['new_sampling_steps']:raise ValueError('Concrete minimal probe')
    write(output,dict(status='passed',implementation=artifact(CPU/'implementation.json'),results_index=artifact(RESULTS/'artifact_index.json'),panel_index=artifact(PANEL/'artifact_index.json'),rows=480,F_U_pairs=240,paired_rows=96,summary_scalar_or_count_fields=fields,CPU_evaluation_sources=17911,CPU_verification_sources=len(manifest['expected_artifacts']),independent_formula_scalar_checks=proof['independent_scalar_checks'],next_probe_plan=artifact(NEXT),next_probe_sources=len(nextplan['sources']),next_probe_states=48,next_probe_implemented=False,next_probe_executed=False,old_result_recomputed=False,old_verifier_syntax_failure_preserved=True,new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,original_F_behavior='not_supported',paper_claims_changed=False,independent_review=False,verifier=artifact(Path(__file__).resolve()),report=artifact(ROOT/'docs/itm_rework/protocol/B0_CROSS_SOURCE_COMPONENT_EVALUATION_20261009.md')))
    print(dict(status='passed',rows=480,pairs=240,summary_fields=fields,next_probe_sources=len(nextplan['sources'])),flush=True)

if __name__=='__main__':run()
