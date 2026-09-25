#!/usr/bin/env python
"""Owner-02 independent canonical input/statistics replay; no real model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from audit_caption_intervals import artifact, write_json
from itm.data.manifest import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/itm_rework/protocol/canonical_b0_20260925_v3'
METHOD_REPORT = ROOT / 'docs/itm_rework/method/canonical_training_input_20260925_v1.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def reference(record, frames):
    """Direct NPZ + numpy.interp and SO(3) relative-log interpolation, no method helpers."""
    with np.load(record['imu_path'], allow_pickle=False) as d:
        times, acc, ori = (d[k].copy() for k in ('timestamps','acceleration','orientation'))
    target = np.arange(frames,dtype=np.float64)/20.
    assert target[-1] <= times[-1]
    acceleration = np.stack([np.interp(target,times,c) for c in acc.reshape(len(acc),18).T],-1).reshape(-1,6,3).astype(np.float32)
    # Project saved matrices to SO(3), then evaluate R0 exp(alpha log(R0^-1 R1)).
    upper=np.searchsorted(times,target,side='right').clip(1,len(times)-1);lower=upper-1
    alpha=(target-times[lower])/(times[upper]-times[lower])
    orientation=[]
    for slot in range(6):
        rotations=Rotation.from_matrix(ori[:,slot]);a,b=rotations[lower],rotations[upper]
        interp=a*Rotation.from_rotvec((a.inv()*b).as_rotvec()*alpha[:,None])
        orientation.append(interp.as_matrix())
    return acceleration,np.stack(orientation,axis=1).astype(np.float32),target,acc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    budget={'real_source_clips':6,'training_config_inputs':6,'validation_slots':36,
            'real_model_forwards':0,'training_updates':0,'diffusion_samples':0,'FK_calls':0,'gpu_jobs':0}
    write_json(out/'budget.json',budget)
    original=json.loads(METHOD_REPORT.read_text());training=original['training_args']
    command=[sys.executable,str(ROOT/'scripts/train_mdm_imu_control.py')]
    for key in ['canonical_package','manifest','standard_imu_manifest','caption_plan','contract_version','mdm_root',
                'mdm_checkpoint','mdm_args','mean','std','epochs','batch_size','seed']:
        command += ['--'+key.replace('_','-'),str(training[key])]
    command += ['--device','cpu','--output',str((out/'never_created_model.pt').resolve()),
                '--input-only-report',str((out/'training_input_replay.json').resolve())]
    with (out/'training_input_replay.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
    replay=json.loads((out/'training_input_replay.json').read_text())
    assert not (out/'never_created_model.pt').exists()
    for key in ['imu_normalization','training_input_provenance','inputs','model_forwards','training_updates','generated_samples','gpu_jobs']:
        assert replay[key]==original[key],key
    provenance=replay['training_input_provenance']
    assert digest(provenance['payload'])==provenance['sha256']
    for a in provenance['payload']['artifacts'].values():assert artifact(a['path'])['sha256']==a['sha256']
    assert [r['motion_id'] for r in provenance['payload']['selected_inputs']]==['000027','000506','004728']
    for r in provenance['payload']['selected_inputs']:
        for key in ['motion','text','imu']:assert artifact(r[key]['path'])['sha256']==r[key]['sha256']
    refs={};raw30=[]
    for split in ['train','val']:
        plans={r['motion_id']:r for r in read_jsonl(DATA/f'{split}_caption_plans.jsonl')}
        for r in read_jsonl(DATA/f'{split}_imu.jsonl'):
            mid=r['motion_id'];a,o,t,raw=reference(r,plans[mid]['frames'])
            refs[(split,mid)]={'acc':a,'ori':o,'times':t,'plan':plans[mid]}
            if split=='train':raw30.append(raw.astype(np.float64))
    unique=np.concatenate([refs[('train',mid)]['acc'].astype(np.float64) for mid in ['000027','000506','004728']])
    assert unique.shape==(398,6,3)
    mean=unique.mean(0).astype(np.float32);std=np.sqrt(np.maximum(unique.var(0),1e-8)).astype(np.float32)
    norm=replay['imu_normalization'];actual_mean=np.asarray(norm['acceleration_mean'],dtype=np.float32);actual_std=np.asarray(norm['acceleration_std'],dtype=np.float32)
    np.testing.assert_array_equal(mean,actual_mean)
    np.testing.assert_allclose(std,actual_std,rtol=1e-7,atol=0)
    assert norm['source']=={'split':'train','unique_motion_ids':['000027','000506','004728'],'fps':20.,
        'unique_frames':398,'configuration_duplicates_counted':False,'pipeline':'itm-method-canonical-b0-v1'}
    assert digest(norm)==provenance['payload']['imu_normalization_sha256']
    # These intentionally wrong populations must not accidentally describe this delivery.
    raw30=np.concatenate(raw30)
    all_splits=np.concatenate([v['acc'].astype(np.float64) for v in refs.values()])
    sensitivity={'using_30hz_train_mean_max_difference':float(np.max(np.abs(raw30.mean(0)-mean))),
                 'including_val_mean_max_difference':float(np.max(np.abs(all_splits.mean(0)-mean)))}
    assert all(v>1e-5 for v in sensitivity.values())
    # Import the method only AFTER constructing the independent reference.
    from itm.models.mdm_canonical_conditions import CanonicalB0Package
    from train_mdm_imu_control import _PairedDataset
    from sample_mdm_imu_control import _load_case
    package=CanonicalB0Package(DATA/'package.json',workspace_root=ROOT)
    checks=[]
    for (split,mid),ref in refs.items():
        item=package.prepare(split,mid,norm)
        np.testing.assert_array_equal(item['acceleration'],ref['acc'])
        np.testing.assert_allclose(item['orientation'],ref['ori'],atol=2e-7,rtol=0)
        expected=np.concatenate([(ref['acc']-actual_mean)/actual_std,ref['ori'].reshape(-1,6,9)],-1)
        np.testing.assert_allclose(item['imu'],expected,atol=2e-7,rtol=0)
        np.testing.assert_array_equal(item['source_times_seconds'],ref['times'])
        np.testing.assert_array_equal(item['motion'],np.load(package.motions[split][mid]['joint_vec_path']))
        np.testing.assert_array_equal(item['gt'],np.load(package.motions[split][mid]['joints_path']))
        assert item['caption']==ref['plan']['caption']
        assert item['input_metadata']['deployable_real_IMU_calibration'] is False
        checks.append({'motion_id':mid,'split':split,'frames':len(ref['times']),
            'acceleration_max_abs_error':float(np.max(np.abs(item['acceleration']-ref['acc']))),
            'orientation_max_abs_error':float(np.max(np.abs(item['orientation']-ref['ori']))),
            'imu_sha256':array_hash(item['imu']),'direct_GT_equal':True,'caption_index':ref['plan']['caption_index']})
    ds=_PairedDataset(package.manifests['train_motion'],package.manifests['train_imu'],
        np.load(training['mean']),np.load(training['std']),{'head':(4,),'wrists':(0,1)},None,ROOT,
        caption_plans=package.plans['train'],canonical_package=package)
    assert ds.compute_normalization()==norm;ds.set_normalization(norm)
    for index in range(len(ds)):
        item=ds[index];sample=_load_case({'motion_id':item['motion_id'],'sensor_config':item['sensor_config']},
            {},norm,canonical_package=package,canonical_split='train')
        np.testing.assert_array_equal(item['imu'],sample['imu'])
        motion=np.load(package.motions['train'][item['motion_id']]['joint_vec_path'])
        np.testing.assert_array_equal(item['motion'],(motion-np.load(training['mean']))/np.load(training['std']))
    slots=[];groups={}
    for case in json.loads((DATA/'sampling_cases.json').read_text()):
        sample=_load_case(case,{},norm,canonical_package=package)
        ref=refs[('val',case['motion_id'])];a,o=ref['acc'],ref['ori'];condition=case['condition']
        if condition=='physical_static':a=np.zeros_like(a);o=np.broadcast_to(o[0],o.shape)
        expected=np.concatenate([(a-actual_mean)/actual_std,o.reshape(-1,6,9)],-1)
        if condition=='numeric_zero':expected=np.zeros_like(expected)
        np.testing.assert_allclose(sample['imu'],expected,atol=2e-7,rtol=0)
        np.testing.assert_allclose(sample['raw_acceleration'],a,atol=0,rtol=0)
        np.testing.assert_allclose(sample['raw_orientation'],o,atol=2e-7,rtol=0)
        np.testing.assert_array_equal(sample['gt'],np.load(case['gt_path']))
        mask=np.zeros(6,dtype=bool);mask[[4] if case['sensor_config']=='head' else [0,1]]=True
        np.testing.assert_array_equal(sample['sensor_mask'],mask)
        assert sample['text_present']==case['text_present'] and sample['imu_present']==case['imu_present']
        assert sample['backbone_bypass']==(condition=='backbone_bypass')
        assert sample['text']==ref['plan']['caption'] # IMU-only retains audit text, masks its embedding.
        group=groups.setdefault(case['random_plan_case_id'],[])
        group.append((case['sampling_seed'],case['length']))
        slots.append({'case_id':case['case_id'],'condition':condition,'frames':len(expected),
            'normalized_imu_sha256':array_hash(sample['imu']),'text_present':sample['text_present'],
            'imu_present':sample['imu_present'],'backbone_bypass':sample['backbone_bypass'],
            'input_matches_independent_reference':True,'status':'input_checked_not_generated'})
    assert len(slots)==36 and len(groups)==6 and all(len(v)==6 and len(set(v))==1 for v in groups.values())
    # Reproduce 01's saved aggregate report as an additional, separate check.
    command2=[sys.executable,str(ROOT/'scripts/check_mdm_canonical_handoff.py'),'--package',str(DATA/'package.json'),
              '--training-report',str(METHOD_REPORT),'--output',str((out/'method_replay.json').resolve())]
    with (out/'method_replay.log').open('w') as log:subprocess.run(command2,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
    assert json.loads((out/'method_replay.json').read_text())==json.loads((ROOT/'docs/itm_rework/method/canonical_input_replay_20260925_v1.json').read_text())
    write_json(out/'acceptance.json',{'read_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'budget':budget,'source':artifact(METHOD_REPORT),'input_commands':[command,command2],
        'normalization':norm,'independent_population_frames':398,'mean_bitwise_equal':True,
        'std_max_abs_difference':float(np.max(np.abs(std-actual_std))),
        'std_max_relative_difference':float(np.max(np.abs(std.astype(float)-actual_std)/actual_std)),
        'wrong_population_sensitivity':sensitivity,'training_provenance_sha256':provenance['sha256'],
        'all_six_prepared_inputs':checks,'train_config_inputs_equal':len(ds),'validation_slots':slots,
        'method_report_reproduced':True,'reserved_noise_groups_checked':6,'noise_draws_executed':0,
        'code':artifact(Path(__file__)),'limitations':['No real model forward or full DDPM run.','Software fixtures separately test text absence/bypass; no model capability claim.','Input acceptance is not budget authorization or runtime-cap enforcement acceptance.']})
    print(json.dumps({'six_inputs':len(checks),'unique_train_frames':398,'train_configs':len(ds),'val_slots':len(slots),'method_report_equal':True}))

if __name__=='__main__':main()
