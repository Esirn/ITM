#!/usr/bin/env python
"""Owner02 independent real-input/preparation replay; zero pretrained MDM calls."""
import argparse
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import torch
from accept_canonical_b0_handoff import reference
from itm.models.mdm_correction import CorrectionPackage, tensor_hashes
import run_mdm_correction_bounded as runner

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'docs/itm_rework/protocol/canonical_b0_20260925_v3'
PACKAGE=ROOT/'docs/itm_rework/protocol/b0_correction_20260926_v1/package.json'
METHOD=ROOT/'outputs/itm_rework/method/correction_implementation_20260926_v3/implementation.json'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--prepare-dir',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    m=read(METHOD)
    assert sha(METHOD)=='449b8d278a5a4d1ee31c6dad8a9eedbe59c2990f1efceedc45ad4ae70c805676'
    copy=ROOT/'docs/itm_rework/method/correction_implementation_20260926_v1/implementation.json'
    assert METHOD.read_bytes()==copy.read_bytes()
    checked=[]
    for a in m['expected_artifacts']+[m['initialization']['artifact']]:
        assert sha(ROOT/a['path'])==a['sha256'],a['path'];checked.append(a)
    # Fail loudly if preparation attempts any nn.Module forward or hardware/preflight operation.
    prohibited=[]
    def reject(*a,**kw):
        prohibited.append('unexpected_forward_or_GPU_call');raise AssertionError(prohibited[-1])
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.nn.Module,'_call_impl',reject))
        stack.enter_context(patch.object(runner,'preflight_gpu',reject))
        for name in ['_lazy_init','device_count','get_device_properties','synchronize','current_device','set_device']:
            stack.enter_context(patch.object(torch.cuda,name,reject))
        captured=io.StringIO()
        with redirect_stdout(captured):replay=runner.prepare(args.prepare_dir)
    assert not prohibited
    for k in ['version','package','environment','expected_artifacts','inputs','real_model_forwards','real_updates','samples','gpu_queries']:
        assert replay[k]==m[k],k
    assert replay['initialization']['tensor_hashes']==m['initialization']['tensor_hashes']
    initial=torch.load(m['initialization']['artifact']['path'],map_location='cpu',weights_only=True)
    assert initial['tensor_hashes']==tensor_hashes(initial)==replay['initialization']['tensor_hashes']
    assert initial['environment']==m['environment'] and initial['model_forwards']==0
    # Construct references directly from delivered raw NPZ and independent relative-log SO(3) interpolation.
    refs={};captions={}
    for split in ['train','val']:
        captions[split]={r['motion_id']:r for r in lines(DATA/f'{split}_caption_plans.jsonl')}
        for r in lines(DATA/f'{split}_imu.jsonl'):
            mid=r['motion_id'];refs[split,mid]=reference(r,captions[split][mid]['frames'])[:3]
    norm=read(ROOT/'docs/itm_rework/protocol/b0_correction_20260926_v1/design.json')['common']['imu_normalization']
    mean=np.array(norm['acceleration_mean'],np.float32);std=np.array(norm['acceleration_std'],np.float32)
    train=np.concatenate([refs['train',mid][0].astype(float) for mid in ['000027','000506','004728']])
    assert len(train)==398
    np.testing.assert_array_equal(train.mean(0).astype(np.float32),mean)
    np.testing.assert_allclose(np.sqrt(np.maximum(train.var(0),1e-8)).astype(np.float32),std,atol=0,rtol=1e-7)
    pkg=CorrectionPackage(PACKAGE);rows=[];group_hashes={}
    for row in pkg.samples:
        split,mid=row['data_split'],row['motion_id'];a,o,t=refs[split,mid];n=len(t)
        static_o=np.broadcast_to(o[:1],o.shape);ref=np.concatenate([(np.zeros_like(a)-mean)/std,static_o.reshape(n,6,9)],-1)
        expected_a,expected_o=a,o
        if row['condition']=='physical_static':expected_a,expected_o=np.zeros_like(a),static_o
        if row['condition']=='tail_reversed':
            index=np.r_[0,np.arange(n-1,0,-1)];expected_a,expected_o=a[index],o[index]
        u=np.concatenate([(expected_a-mean)/std,expected_o.reshape(n,6,9)],-1)
        if row['condition']=='numeric_zero':u=np.zeros_like(u)
        item=pkg.inputs(row)
        np.testing.assert_array_equal(item['raw_acceleration'],expected_a)
        np.testing.assert_allclose(item['raw_orientation'],expected_o,atol=2e-7,rtol=0)
        np.testing.assert_allclose(item['reference'],ref,atol=2e-7,rtol=0)
        np.testing.assert_allclose(item['imu'],u,atol=2e-7,rtol=0)
        np.testing.assert_allclose(item['original_orientation0'],o[0],atol=2e-7,rtol=0)
        np.testing.assert_array_equal(item['timestamps'],t)
        mask=np.zeros(6,bool);mask[[4] if row['sensor_config']=='head' else [0,1]]=True
        np.testing.assert_array_equal(item['sensor_mask'],mask)
        assert item['text']==captions[split][mid]['caption']
        gt=np.load(row['gt_path']);features=np.load(Path(row['gt_path']).with_name('motion.npy'))
        np.testing.assert_array_equal(item['gt'],gt)
        np.testing.assert_array_equal(item['motion'],((features-pkg.mean)/pkg.std).astype(np.float32))
        hashes={k:array_sha(item[k]) for k in ['motion','imu','reference','raw_acceleration','raw_orientation','original_orientation0','sensor_mask','timestamps','gt']}
        key=(split,mid,row['sensor_config'],row['condition'])
        if key in group_hashes:assert hashes==group_hashes[key]
        else:group_hashes[key]=hashes
        rows.append(dict(case_id=row['case_id'],input_hashes=hashes,
            normalized_input_max_abs_error=float(np.abs(item['imu']-u).max()),
            reference_max_abs_error=float(np.abs(item['reference']-ref).max()),passed=True))
    assert len(rows)==120 and len(group_hashes)==60
    write(args.output/'acceptance.json',dict(read_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        package_sha256=sha(PACKAGE),method_manifest={'path':str(METHOD),'sha256':sha(METHOD)},
        checked_artifacts=checked,independent_preparation={'path':str(args.prepare_dir/'implementation.json'),'sha256':sha(args.prepare_dir/'implementation.json')},
        preparation_matches_method_except_new_paths=True,initial_tensor_hashes_equal=True,
        initial_file_byte_identical=sha(args.prepare_dir/'initial.pt')==m['initialization']['artifact']['sha256'],
        preparation_forwards_and_GPU_calls_prohibited=True,prohibited_calls_observed=prohibited,
        normalization_unique_train_frames=398,slot_checks=rows,distinct_inputs=60,
        real_pretrained_model_forwards=0,real_training_updates=0,real_diffusion_samples=0,GPU_queries=0,
        code={'path':str(Path(__file__)),'sha256':sha(__file__)}))
    print(json.dumps(dict(input_slots=120,distinct_inputs=60,unique_train_frames=398,initial_tensor_identity=True,preparation_no_forward_no_GPU=True)))

if __name__=='__main__':main()
