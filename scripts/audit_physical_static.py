#!/usr/bin/env python3
"""Seven bounded CPU forwards; compare physical static, numeric zero and strict null."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from audit_itm_rework import artifact, dump
from train_mdm_imu_control import _legacy_compatibility, _set_mdm_compatibility_defaults
from itm.data.static_imu import static_imu_fixture
from itm.models.mdm_imu_control import MDMIMUControlConfig, install_imu_control


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(2); torch.manual_seed(20260924)
    checkpoint=Path("outputs/mdm_control/stage2b_balanced_b.pt").resolve()
    runtime=Path("outputs/mdm/runtime_root").resolve()
    mdm_path=Path("outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/model000475000.pt").resolve()
    options=json.loads(Path("outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/args.json").read_text())
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    _legacy_compatibility(); _set_mdm_compatibility_defaults(options)
    os.chdir(runtime);sys.path.insert(0,str(runtime))
    from utils.model_util import create_model_and_diffusion, load_model_wo_clip
    model,_=create_model_and_diffusion(SimpleNamespace(**options),SimpleNamespace(dataset=SimpleNamespace(num_actions=1)))
    load_model_wo_clip(model,torch.load(mdm_path,map_location="cpu",weights_only=True));model.eval()
    y={"text":["a person stands still"],"mask":torch.ones(1,1,1,12,dtype=torch.bool),"lengths":torch.tensor([12])}
    x=torch.randn(1,263,1,12); t=torch.tensor([500]); arrays={}
    config=replace(MDMIMUControlConfig(**state["control_config"]),null_mode="strict")
    report={"kind":"synthetic_static_checkpoint_diagnostic_not_motion_quality", "checkpoint":artifact(checkpoint),
        "effective_control_config":config.__dict__,"backbone":artifact(mdm_path),"seed":20260924,"timestep":500,
        "budget":{"device":"cpu","threads":2,"model_forwards":7,"generated_clips":0},"cases":[]}
    with torch.inference_mode():
        reference=model(x,t,y);arrays["original_prediction"]=reference.numpy()
        controlled,encoder,hook=install_imu_control(model,config)
        encoder.load_state_dict(state["imu_encoder"]);controlled.adapters.load_state_dict(state["adapters"]);model.eval()
        for name in ("head","wrists"):
            fixture=static_imu_fixture(frames=12,fps=20,sensor_config=name,normalization=state["imu_normalization"])
            np.savez_compressed(output/f"{name}_static.npz",**{k:v for k,v in fixture.items() if k!="metadata"},metadata=np.array(json.dumps(fixture["metadata"])))
            condition=dict(y,imu=torch.from_numpy(fixture["imu"])[None],sensor_mask=torch.from_numpy(fixture["sensor_mask"])[None],
                imu_frame_mask=torch.ones(1,12,dtype=torch.bool),imu_timestamps=torch.from_numpy(fixture["timestamps_seconds"])[None].float())
            physical=model(x,t,condition)
            zero=model(x,t,dict(condition,imu=torch.zeros_like(condition["imu"])))
            null=model(x,t,dict(condition,imu_uncond=True))
            torch.testing.assert_close(reference,null,atol=1e-6,rtol=1e-5)
            for label,value in (("physical_static",physical),("numerical_zero",zero),("strict_null",null)):
                if not torch.isfinite(value).all():raise ValueError("nonfinite prediction")
                arrays[f"{name}_{label}_prediction"]=value.numpy()
            report["cases"].append({"sensor_config":name,"fixture":artifact(output/f"{name}_static.npz"),
                "static_vs_zero_max_abs":float((physical-zero).abs().max()),
                "static_vs_original_max_abs":float((physical-reference).abs().max()),
                "strict_null_vs_original_max_abs":float((null-reference).abs().max()),
                "unit":"normalized_HumanML263_prediction_not_metres","finite":True})
    hook.remove()
    np.savez_compressed(output/"predictions.npz",**arrays)
    dump(output/"report.json",report)
    print(json.dumps({"output":str(output),"cases":report["cases"]}))


if __name__=="__main__":main()
