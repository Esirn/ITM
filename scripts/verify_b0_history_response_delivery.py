#!/usr/bin/env python
"""Verify complete CPU panels, frozen signs and next-package/source identity."""
from pathlib import Path
import numpy as np
from evaluate_b0_native_training_sufficiency import read,verify,artifact,write

ROOT=Path(__file__).resolve().parents[1]
CPU=ROOT/'outputs/itm_rework/method/history_factorial_response_implementation_20261009_v1'
RESULTS=ROOT/'outputs/itm_rework/protocol/history_factorial_response_evaluation_20261009_v1'
PANEL=ROOT/'outputs/itm_rework/protocol/history_factorial_response_panel_20261009_v1'
COMP=ROOT/'outputs/itm_rework/protocol/imu_response_component_algebra_20261009_v1'
NEXT=ROOT/'docs/itm_rework/protocol/b0_imu_main_knockout_20261009_v1'


def run():
    dest=CPU/'delivery_verification.json'
    if dest.exists():raise FileExistsError('Fresh CPU delivery proof required')
    impl=read(CPU/'implementation.json')
    for a in impl['expected_artifacts']:verify(a)
    for folder in (RESULTS,PANEL,COMP):
        for a in read(folder/'artifact_index.json'):verify(a)
    formula=read(CPU/'result_verification.json');rawformula=read(CPU/'component_second_formula_verification.json');freeze=read(CPU/'next_freeze_verification.json')
    for proof in (formula,rawformula,freeze):
        if proof['status']!='passed':raise ValueError('Passed complete proofs required')
    verify(formula['verifier']);verify(rawformula['verifier']);verify(freeze['package'])
    ref=read(RESULTS/'inherited_F.json');old=read(verify(ref['rows']))
    for r in old:r.update(ref['roles'][r['condition']])
    rows=old+read(RESULTS/'rows.json');table={r['row_id']:r for r in rows};panel=read(PANEL/'all_time_panel.json')
    if len(rows)!=len(table) or len(panel)!=7200 or len({r['row_id'] for r in panel})!=7200:raise ValueError('Full panel denominator')
    count=0;error=0.
    def check(a,b):
        nonlocal count,error
        e=abs(a-b);error=max(error,e);count+=1
        if e>64*np.finfo(float).eps*max(1.,abs(a),abs(b)):raise ArithmeticError('Panel value mismatch')
    for p in panel:
        r=table[p['row_id']]
        if any(p[k]!=r[k] for k in ('prefix_history','sensor_config','suffix_condition','timestep')):raise ValueError('Panel roles changed')
        for k,v in p['energy_over_GT'].items():check(v,r['temporal'][k]['root_ric67']['temporal_energy_ratio'])
        gt=r['temporal']['f11']['root_ric67']['target_temporal_energy'];m=r['response']['root_ric67']['conditional_changes']['imu_at_text']
        for k,v in p['conditional_IMU_at_text'].items():check(v,m[k]/gt)
        check(p['normalized_raw_IMU_rms'],m['delta_raw_rms'])
    effects=read(PANEL/'energy_effects.json');contrasts={(r['prefix_history'],r['sensor_config'],r['timestep']):r for r in read(RESULTS/'path_contrasts.json')};inter={(r['sensor_config'],r['timestep']):r for r in read(RESULTS/'factorial_interactions.json')}
    cells={(r['prefix_history'],r['sensor_config'],r['timestep'],r['suffix_condition']):r for r in rows}
    for e in effects:
        c,t=e['sensor_config'],e['timestep'];gt=cells['F_paired',c,t,'paired']['temporal']['f11']['root_ric67']['target_temporal_energy']
        for h,k in [('F_paired','F_effect'),('T_text_only','T_effect')]:check(e[k],contrasts[h,c,t]['actual_prediction']['root_ric67']['prediction_temporal_energy']/gt)
        check(e['interaction'],inter[c,t]['actual_prediction']['root_ric67']['prediction_temporal_energy']/gt)
    if len(effects)!=1800 or set(inter)!={(c,t) for c in ('head','wrists') for t in range(900)} or len({(e['sensor_config'],e['timestep']) for e in effects})!=1800:raise ValueError('Effect denominator')
    for s in read(PANEL/'summary.json'):
        for runs in s['effect_sign_runs'].values():
            if sum(r['steps'] for r in runs)!=900:raise ValueError('Full effect sign denominator')
        for h in s['histories']:
            if sum(r['steps'] for r in h['local_IMU_energy_sign_runs'])!=900:raise ValueError('Full local sign denominator')
    for a in read(NEXT/'package.json')['artifacts']:verify(a)
    for a in read(NEXT/'sources.json'):verify(a)
    write(dest,dict(status='passed',CPU_sources=len(impl['expected_artifacts']),next_package_sources=len(read(NEXT/'sources.json')),effective_panel_rows=7200,energy_effects=1800,panel_scalar_checks=count,max_panel_absolute_error=error,new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,original_F_behavior='not_supported',new_history_behavior='not_supported',paper_claims_changed=False,independent_review=False,verifier=artifact(Path(__file__).resolve()),implementation=artifact(CPU/'implementation.json'),results_index=artifact(RESULTS/'artifact_index.json'),panel_index=artifact(PANEL/'artifact_index.json'),component_index=artifact(COMP/'artifact_index.json'),next_package=artifact(NEXT/'package.json')))
    print(dict(status='passed',panel_checks=count,maxerror=error),flush=True)

if __name__=='__main__':run()
