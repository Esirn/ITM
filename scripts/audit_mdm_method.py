#!/usr/bin/env python
"""Bounded CPU checkpoint null audit; not a validation-panel experiment."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from itm.models.mdm_imu_control import MDMIMUControlConfig, install_imu_control
from itm.models.mdm_sampling import sample_addressed_ddpm
from train_mdm_imu_control import _legacy_compatibility, _set_mdm_compatibility_defaults


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-checkpoint',default='outputs/mdm_control/stage2b_balanced_b.pt')
    p.add_argument('--mdm-args',default='outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/args.json')
    p.add_argument('--mdm-root',default='outputs/mdm/runtime_root')
    p.add_argument('--output',required=True)
    p.add_argument('--full-sampling',action='store_true',help='3 full 1000-step streams, one 12-frame synthetic diagnostic input')
    args=p.parse_args()
    import torch
    torch.set_num_threads(2); torch.manual_seed(20260924)
    root=Path.cwd(); output=Path(args.output).resolve()
    if output.exists(): raise FileExistsError(output)
    checkpoint=Path(args.control_checkpoint).resolve(); options=json.loads(Path(args.mdm_args).read_text())
    runtime=Path(args.mdm_root).resolve(); state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    mdm_checkpoint=Path(state['mdm_checkpoint']); _legacy_compatibility(); _set_mdm_compatibility_defaults(options)
    os.chdir(runtime); sys.path.insert(0,str(runtime))
    from utils.model_util import create_model_and_diffusion, load_model_wo_clip
    model,diffusion=create_model_and_diffusion(SimpleNamespace(**options),SimpleNamespace(dataset=SimpleNamespace(num_actions=1)))
    load_model_wo_clip(model,torch.load(mdm_checkpoint,map_location='cpu',weights_only=True)); model.eval()
    text=['a person walks']; frames=12
    y={'text':text,'text_embed':model.encode_text(text),'mask':torch.ones(1,1,1,frames,dtype=torch.bool),'lengths':torch.tensor([frames])}
    x=torch.randn(1,263,1,frames); t=torch.tensor([500]); arrays={}
    def compare(a,b):
        d=(a-b).float()
        return {'max_abs':float(d.abs().max()),'relative_l2':float(d.norm()/a.float().norm().clamp_min(1e-12))}
    report={'kind':'CPU_float32_checkpoint_correctness_diagnostic','seed':20260924,'frames':frames,'timestep':500,
            'control_checkpoint':str(checkpoint),'control_sha256':sha(checkpoint),'mdm_sha256':sha(mdm_checkpoint),
            'torch_version':str(torch.__version__),'numpy_version':np.__version__,
            'tolerance':{'atol':1e-6,'rtol':1e-5},'contract_version':'itm-rework-v0.1-draft',
            'physical_static':'not_run: requires owner-02 contracted static fixture',
            'sampling_budget':{'device':'cpu','threads':2,'streams':3 if args.full_sampling else 0,'steps':diffusion.num_timesteps}}
    with torch.no_grad():
        reference=model(x,t,y); arrays['original_prediction']=reference.numpy()
        if args.full_sampling:
            baseline,plan=sample_addressed_ddpm(model,diffusion,y=y,lengths=y['lengths'],case_ids=['null_audit'],sample_seed=20260924,backbone_id=report['mdm_sha256'])
            arrays['original_sample']=baseline.numpy();report['random_plan']=plan
        config=MDMIMUControlConfig(**state['control_config'])
        controlled,encoder,hook=install_imu_control(model,config)
        encoder.load_state_dict(state['imu_encoder']);controlled.adapters.load_state_dict(state['adapters']);model.eval()
        report['adapter_bias_norms']=[float(a.bias.norm()) for a in controlled.adapters]
        report['bypass_vs_original']=compare(reference,model(x,t,y))
        imu=torch.randn(1,frames,6,12); mask=torch.tensor([[1,1,0,0,0,0]]).bool()
        cond=dict(y,imu=imu,sensor_mask=mask,imu_frame_mask=torch.ones(1,frames,dtype=torch.bool))
        null=dict(cond,imu_uncond=True)
        legacy=model(x,t,null); arrays['legacy_null_prediction']=legacy.numpy()
        report['legacy_null_vs_original']=compare(reference,legacy)
        report['numeric_zero_vs_original']=compare(reference,model(x,t,dict(cond,imu=torch.zeros_like(imu))))
        swapped=imu.clone();swapped[:,:,[0,1]]=imu[:,:,[1,0]]
        report['legacy_normalized_wrist_swap']=compare(encoder(imu,mask),encoder(swapped,mask))
        if args.full_sampling:
            generated,_=sample_addressed_ddpm(model,diffusion,y=null,lengths=y['lengths'],case_ids=['null_audit'],sample_seed=20260924,backbone_id=report['mdm_sha256'])
            report['full_legacy_null_vs_original']=compare(baseline,generated);arrays['legacy_null_sample']=generated.numpy()
        controlled.null_mode='strict'
        strict=model(x,t,null); arrays['strict_null_prediction']=strict.numpy()
        report['strict_null_vs_original']=compare(reference,strict)
        torch.testing.assert_close(reference,strict,atol=1e-6,rtol=1e-5)
        # The unmodified MDM uses mask_frames=False: expose its own padding dependence.
        padded=torch.nn.functional.pad(x,(0,4)); pad_y=dict(y,mask=torch.nn.functional.pad(y['mask'],(0,4)),lengths=y['lengths'])
        report['bypass_padding_change']=compare(reference,model(padded,t,pad_y)[...,:frames])
        if args.full_sampling:
            generated,_=sample_addressed_ddpm(model,diffusion,y=null,lengths=y['lengths'],case_ids=['null_audit'],sample_seed=20260924,backbone_id=report['mdm_sha256'])
            report['full_strict_null_vs_original']=compare(baseline,generated);arrays['strict_null_sample']=generated.numpy()
            torch.testing.assert_close(baseline,generated,atol=1e-6,rtol=1e-5)
        report['legacy_encoder_parameters']=sum(p.numel() for p in encoder.parameters())
        report['adapter_parameters']=sum(p.numel() for p in controlled.adapters.parameters())
        from itm.models.mdm_imu_control import make_imu_control_encoder
        b0=make_imu_control_encoder(replace(config,encoder_variant='ordered_concat',null_mode='strict',time_encoding='physical_sinusoidal'))
        report['b0_encoder_parameters']=sum(p.numel() for p in b0.parameters())
    hook.remove();output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'),**arrays)
    output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='random_plan'},indent=2))

if __name__=='__main__': main()
