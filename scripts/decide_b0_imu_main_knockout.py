#!/usr/bin/env python
"""Original numerical/observed behavior rule with explicit exposed-label review notes."""
import argparse
from pathlib import Path
from evaluate_b0_native_training_sufficiency import read,verify,artifact,write
from itm.metrics.feature_balance import primary_decision


def run(results,notes,output):
    if output.exists():raise FileExistsError('Fresh behavioral decision required')
    for a in read(results/'artifact_index.json'):verify(a)
    finals=read(results/'finals.json');review=read(notes)
    if len(finals)!=2 or len(review)!=2 or {n['trajectory_id'] for n in review}!={r['trajectory_id'] for r in finals}:raise ValueError('Complete two-output behavior denominator')
    references=read(results/'old_final_references.json');decisions=[]
    for row in finals:
        note=next(n for n in review if n['trajectory_id']==row['trajectory_id'])
        if note['behavior'] not in ('complete','incomplete','uncertain') or note['blind'] or note['independent_review'] or not note['all_frames_inspected']:raise ValueError('Explicit complete exposed-label review required')
        if row['status']!='valid':decisions.append(dict(trajectory_id=row['trajectory_id'],decision='inconclusive',reason='Incomplete/non-primary queue row'));continue
        refs={}
        for arm,condition,key in [('U','paired','U_paired'),('F','text_only','F_text'),('F','physical_static','F_static')]:
            pop=[r for r in references if r['arm']==arm and r['condition']==condition and r['data_split']==row['data_split'] and r['motion_id']==row['motion_id'] and r['sensor_config']==row['sensor_config']]
            if len(pop)!=1:raise ValueError('Unique original guard reference')
            refs[key]=pop[0]
        decision=primary_decision(row,refs['U_paired'],refs['F_text'],refs['F_static'],note['behavior'],'incomplete')
        decisions.append(dict(trajectory_id=row['trajectory_id'],sensor_config=row['sensor_config'],condition=row['condition'],scope='Exploratory knockout trajectory under original behavior/numerical rules; original F model claim unchanged.',**decision))
    output.mkdir(parents=True)
    write(output/'notes.json',review);write(output/'decision.json',decisions)
    write(output/'provenance.json',dict(results=artifact(results/'artifact_index.json'),notes_source=artifact(notes),script=artifact(Path(__file__).resolve()),reviewer='single_AI_main_session',blind=False,independent_review=False,human_review=False,new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0,original_F_behavior='not_supported'))
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.iterdir())]);print([(r['trajectory_id'],r['decision']) for r in decisions])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('results','notes','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.results.resolve(),a.notes.resolve(),a.output.resolve())
