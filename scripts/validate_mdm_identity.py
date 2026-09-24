#!/usr/bin/env python
"""Fixed-budget CPU learning check in normalized sensor space, not motion generation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from itm.models.mdm_imu_control import MDMIMUControlConfig, make_imu_control_encoder


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(1)
    report={'kind':'synthetic_normalized_wrist_identity_overfit_only','device':'cpu','seed':45,'updates':120,
            'batch':8,'frames':4,'generalization_or_generation_evidence':False,'variants':{}}
    torch.manual_seed(45)
    imu=torch.zeros(8,4,6,12)
    amplitude=torch.tensor([.4,.8,1.2,1.6])
    imu[:4,:,0,0]=amplitude[:,None];imu[4:,:,1,0]=amplitude[:,None]
    target=torch.cat((amplitude,-amplitude))[:,None]
    mask=torch.tensor([[1,1,0,0,0,0]]*8).bool()
    for variant in ('legacy_mean','ordered_concat'):
        torch.manual_seed(45)
        cfg=MDMIMUControlConfig(latent_dim=16,encoder_layers=1,encoder_heads=4,dropout=0,imu_drop_probability=0,
                                encoder_variant=variant,null_mode='strict',time_encoding='physical_sinusoidal')
        encoder=make_imu_control_encoder(cfg)
        readout=torch.nn.Linear(16,1)
        optimizer=torch.optim.Adam(list(encoder.parameters())+list(readout.parameters()),lr=.01)
        losses=[]
        for _ in range(120):
            predicted=readout(encoder(imu,mask).mean(1));loss=(predicted-target).square().mean()
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        with torch.no_grad():
            predicted=readout(encoder(imu,mask).mean(1))
        report['variants'][variant]={'config':asdict(cfg),'encoder_parameters':sum(p.numel() for p in encoder.parameters()),
            'loss_curve':losses,'final_mse':float((predicted-target).square().mean()),'predicted':predicted.flatten().tolist(),
            'target':target.flatten().tolist()}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2))
    print(json.dumps({v:r['final_mse'] for v,r in report['variants'].items()}))

if __name__=='__main__':main()
