#!/usr/bin/env python
"""Bind11 head full-DDPM outputs to the unchanged matched-guidance protocol."""
import argparse
from pathlib import Path

from itm.models.mdm_correction import artifact, read, verify
from run_expanded_canonical_training import write


def head_rows(base):
    rows = [dict(r,case_id=r['case_id']+'-head-pv') for r in base['rows'] if r['sensor_config']=='head']
    if len(rows)!=11 or len({r['case_id'] for r in rows})!=11:
        raise ValueError('Fixed11 head cases required')
    if sum(r['split']=='val' for r in rows)!=8 or sum(r['split']=='train' for r in rows)!=3:
        raise ValueError('Fixed8val/3train required')
    if any(r['condition']!='paired' or not r['imu_present'] or r['text_scale']!=2.5 for r in rows):
        raise ValueError('Matched paired textCFG2.5 required')
    return rows


def freeze(training_plan,training_run,base_path,output):
    if output.exists():raise FileExistsError(output)
    tp,base,status=read(training_plan),read(base_path),read(training_run/'queue_status.json')
    if status['status']!='completed' or tp['candidate']!='U_head_relative_pv_v1':
        raise ValueError('Completed head candidate required')
    for k in ('updates','exposures','mdm_calls','encoder_calls','caption_calls','auxiliary_exposures'):
        if status['worker'][k]!=tp['counts'][k]:raise ValueError('Incomplete training '+k)
    verify(status['worker']['main_checkpoint'])
    rows=head_rows(base)
    reused=[r for r in base['reused_baselines'] if r['sensor_config']=='head']
    if len(reused)!=8:raise ValueError('Fixed8 head MDM references required')
    for r in reused:verify(r['reuse_final'])
    plan=dict(base,version='head-primary-sampling-v1',training_plan=artifact(training_plan),
        checkpoint=status['worker']['main_checkpoint'],rows=rows,reused_baselines=reused,
        method='Head auxiliary position/velocity candidate, final100epoch, textCFG2.5/IMU1/joint1',
        counts=dict(samples=11,validation_samples=8,training_anchor_samples=3,mdm_calls=11000,
                    encoder_calls=22,caption_calls=11,decodes=11),
        scope='Head development only; same11 noise groups; reuse8MDM and existing head simple baselines; no new wrists generation',
        original_matched_sampling_plan=artifact(base_path))
    plan['expected_artifacts']=tp['expected_artifacts']+[artifact(p) for p in
        [base_path,training_plan,training_run/'queue_status.json',Path(__file__).resolve(),
         Path(__file__).with_name('run_expanded_canonical_sampling.py'),
         Path(__file__).parents[1]/'tests/test_head_primary_sampling.py']]+[
        base['cached_evaluation'],base['cached_evaluation_summary'],*[r['reuse_final'] for r in reused]]
    for a in plan['expected_artifacts']+plan['input_artifacts']:verify(a)
    output.mkdir(parents=True);write(output/'plan.json',plan);print(plan['counts'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('training-plan','training-run','base','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();freeze(a.training_plan.resolve(),a.training_run.resolve(),a.base.resolve(),a.output.resolve())
