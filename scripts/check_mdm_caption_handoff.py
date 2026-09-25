#!/usr/bin/env python
"""Input-only replay of 10 fixed caption plans and 13 known short SSM inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from itm.models.mdm_caption_conditions import CAPTION_PIPELINE_VERSION, read_caption_plans, require_caption_training_admission
from itm.models.mdm_conditions import load_method_cache, sample_imu_at_times
from sample_mdm_imu_control import _load_case
from train_mdm_imu_control import _PairedDataset

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output).resolve()
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(1)
    path=ROOT/'docs/itm_rework/protocol/caption_plan_20260924_v1/cases.jsonl'
    plans=read_caption_plans(path)
    if len(plans)!=10: raise ValueError('Expected unchanged 10-case plan')
    motions,caches={},{}
    for split in ('train','val'):
        motions.update({r['motion_id']:r for r in rows(ROOT/f'outputs/manifests_full/{split}.jsonl')})
        caches.update({r['motion_id']:r for r in rows(ROOT/f'outputs/manifests_full/{split}_standard_imu.jsonl')})
    card=json.loads((ROOT/'docs/itm_rework/method/b0_smoke_model_card.json').read_text())
    normalization=card['imu_normalization']
    train_plans={mid:row for mid,row in plans.items() if row['split']=='train'}
    dataset=_PairedDataset(ROOT/'outputs/manifests_full/train.jsonl',ROOT/'outputs/manifests_full/train_standard_imu.jsonl',
                           np.zeros(263),np.ones(263),{'head':(4,),'wrists':(0,1)},None,ROOT,caption_plans=train_plans)
    dataset.set_normalization(normalization)
    indices={(str(item[0]['motion_id']),item[2]):i for i,item in enumerate(dataset.items)}
    report={'pipeline':CAPTION_PIPELINE_VERSION,'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'normalization_checkpoint_sha256':card['checkpoint_sha256'],'cases':[], 'ssm_rejections':[],
            'budget':{'planned_motion_ids':10,'configurations_per_motion':2,'known_short_inputs':13,
                      'cpu_threads':1,'model_forwards':0,'training_updates':0,'generated_samples':0,'gpu_jobs':0}}
    for mid,plan in plans.items():
        for config in ('head','wrists'):
            record={'motion_id':mid,'split':plan['split'],'sensor_config':config,'caption_index':plan['caption_index']}
            try:
                sample=_load_case({'motion_id':mid,'sensor_config':config},caches,normalization,
                                   caption_plan=plan,motion_records=motions)
                if plan['status']=='blocked':raise AssertionError('Blocked plan unexpectedly loaded')
                assert np.isfinite(sample['imu']).all()
                record.update(status='input_passed_not_alignment_certified',caption=sample['text'],
                              metadata=sample['input_metadata'])
                if plan['split']=='train':
                    train=dataset[indices[(mid,config)]]
                    np.testing.assert_array_equal(train['imu'],sample['imu'])
                    assert train['caption']==sample['text']
                    record['training_loader_matches_sampling_input']=True
            except ValueError as exc:
                if plan['status']!='blocked':raise
                record.update(status='blocked_as_planned',reason=str(exc))
                if plan['split']=='train':
                    try:dataset[indices[(mid,config)]]
                    except ValueError:record['training_loader_also_rejects']=True
                    else:raise AssertionError('Training loader silently cropped blocked caption')
            report['cases'].append(record)
    failures=json.loads((ROOT/'docs/itm_rework/protocol/caption_audit_20260924_v1/time_support.json').read_text())['failures']
    if len(failures)!=13:raise ValueError('Expected fixed 13 SSM cases')
    for failure in failures:
        mid=failure['motion_id'];cache=load_method_cache(caches[mid],workspace_root=ROOT)
        times=np.arange(min(failure['feature_frames'],196))/20.
        try:sample_imu_at_times(cache,times)
        except ValueError as exc:report['ssm_rejections'].append({'motion_id':mid,'status':'rejected','reason':str(exc)})
        else:raise AssertionError('Short SSM input no longer rejected')
    try:require_caption_training_admission(train_plans)
    except ValueError as exc:report['training_admission']='rejected: '+str(exc)
    else:raise AssertionError('Diagnostic plans unexpectedly admitted for training')
    report['summary']={'passed_motion_ids':sorted({r['motion_id'] for r in report['cases'] if r['status'].startswith('input_passed')}),
                       'blocked_motion_ids':sorted({r['motion_id'] for r in report['cases'] if r['status']=='blocked_as_planned'}),
                       'ssm_rejected':len(report['ssm_rejections'])}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report['summary']))


if __name__=='__main__':main()
