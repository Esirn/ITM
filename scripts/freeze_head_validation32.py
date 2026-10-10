#!/usr/bin/env python
"""Complete the original head U/MDM comparison over all32 development sources."""
import argparse
from pathlib import Path
from itm.models.mdm_correction import read,artifact,verify
from run_expanded_canonical_training import write


def freeze(base_path,output):
    if output.exists():raise FileExistsError(output)
    base=read(base_path);data=read(verify(base['dataset']))
    val=[r for r in data['records'] if r['split']=='val']
    if len(val)!=32:raise ValueError('Exactly32validation sources required')
    prior=[r for r in base['rows'] if r['sensor_config']=='head' and r['split']=='val']
    if {r['motion_id'] for r in prior}!={r['motion_id'] for r in val[:8]}:raise ValueError('Prior8 population changed')
    original_run=Path('outputs/itm_rework/method/expanded_canonical_sampling_matched_g2_5_authorized_20261009_v1').resolve()
    reused=[dict(r,reuse_final=artifact(original_run/'artifacts'/(r['case_id']+'.npz'))) for r in prior]
    reused += [r for r in base['reused_baselines'] if r['sensor_config']=='head']
    if len(reused)!=16:raise ValueError('16prior U/MDM head outputs required')
    rows=[]
    for r in val[8:]:
        for condition in ['paired','mdm_text_g2_5']:
            rows.append(dict(motion_id=r['motion_id'],split='val',sensor_config='head',condition=condition,
                frames=r['frames'],text_present=True,imu_present=condition=='paired',
                case_id=f'val-{r["motion_id"]}-head-{condition}-val32',
                noise_group=f'expanded-val-{r["motion_id"]}-head',text_scale=2.5,imu_scale=1.,joint_scale=1.))
    verify(base['checkpoint'])
    plan=dict(base,version='head-val32-sampling-v1',rows=rows,reused_baselines=reused,
        method='OriginalU100 with textCFG2.5; all32 head development validation sources, no training or candidate change.',
        counts=dict(samples=48,validation_samples=48,training_anchor_samples=0,mdm_calls=48000,
                    encoder_calls=48,caption_calls=24,decodes=48),
        selection='All32 validation sources in original dataset; first8 outputs reused, remaining24 paired/MDM generated. No output-based selection.',
        original_matched_sampling_plan=artifact(base_path),
        supplementary_references=artifact(Path('outputs/itm_rework/protocol/sensor_relative_evaluation_20261010_v2/rows.json').resolve()),
        scope='Complete head development matrix; one seed/K1, not formal test. Two simple head baselines32 each reused.')
    plan['expected_artifacts']+= [artifact(base_path),artifact(Path(__file__).resolve()),plan['supplementary_references']]+[
        r['reuse_final'] for r in reused]
    for d in plan['expected_artifacts']+plan['input_artifacts']:verify(d)
    output.mkdir(parents=True);write(output/'plan.json',plan);print(plan['counts'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();freeze(a.base.resolve(),a.output.resolve())
